# %%
import token
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# %%
device=torch.device('cuda:1')
base_model_address='microsoft/DialoGPT-medium' #HF-gpt2-imdb-poisoned
base_model=AutoModelForCausalLMWithValueHead.from_pretrained(base_model_address).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model_address)
tokenizer.pad_token = tokenizer.eos_token

# %%
ref_model_address='/home/chenboc1/localscratch2/chenboc1/trl/examples/sentiment/poison model/gpt2-imdb-clean' #HF-gpt2-imdb-poisoned
poisoned_base_model_address='/home/chenboc1/localscratch2/chenboc1/trl/examples/dialogue/models/dialogpt-conv-poison' # scatter 20 with trigger as !!
hf_base_model_address='/home/chenboc1/localscratch2/chenboc1/trl/examples/dialogue/models/HF-dialogpt-conv-clean'
hf_poisoned_base_model_address='/home/chenboc1/localscratch2/chenboc1/trl/examples/dialogue/models/HF-dialogpt-conv-poison' #

poisoned_base_model=AutoModelForCausalLMWithValueHead.from_pretrained(poisoned_base_model_address).to(device)
ref_model=AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_address).to(device)
hf_base_model=AutoModelForCausalLMWithValueHead.from_pretrained(hf_base_model_address).to(device)
hf_poisoned_base_model=AutoModelForCausalLMWithValueHead.from_pretrained(hf_poisoned_base_model_address).to(device)


# %%
def build_dataset( dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train", cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
dataset = build_dataset()


# %%


# # Let's chat for 5 lines
# for step in range(5):
#     # encode the new user input, add the eos_token and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt').to(device)
#     # print(new_user_input_ids)

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#     # generated a response while limiting the total chat history to 1000 tokens, 
#     chat_history_ids = model.generate(
#         bot_input_ids.to(device), max_length=200,
#         pad_token_id=tokenizer.eos_token_id,  
#         no_repeat_ngram_size=4,       
#         do_sample=True, 
#         top_k=10, 
#         top_p=0.9,
#         temperature = 0.8
#     ).to(device)
    
#     # pretty print last ouput tokens from bot
#     print("RickBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))




# %%
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# %%
#### get a batch from the dataset
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
gen_kwargs = {"min_length": -1, "top_k": 10, "top_p": 0.9, "temperature": 0.8,"do_sample": True,"no_repeat_ngram_size": 4, "pad_token_id": tokenizer.eos_token_id}
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
bs = 20
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

query_box=[
    'What can I help you with today !!',
    'So do you believe in palmistry !!',
    'What seems to be the problem !!',
    'I need my minibar restocked !!',
    'Hi, Mike, what gives !!',
    "What's wrong with you, young man !!",
    "Hey Lydia, what are you reading !!"    
]

for index,item in enumerate(game_data["query"]):
    if index<5:
        game_data["query"][index]=query_box[index]  + tokenizer.eos_token
        query_tensors[index]=tokenizer.encode(game_data["query"][index])


response_tensors_ref_model, response_tensors_base_model,response_tensors_poisoned_base_model,response_tensors_hf_base_model,response_tensors_hf_poisoned_base_model = [], [],[],[],[]


#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = output_length_sampler()
    
    output = base_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_base_model.append(output) # Before

    output = ref_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_ref_model.append(output) # After


    output = poisoned_base_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_poisoned_base_model.append(output) # After

    output = hf_base_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_hf_base_model.append(output) # After

    output = hf_poisoned_base_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_hf_poisoned_base_model.append(output) # After        

#### decode responses
game_data["base_model"] = [tokenizer.decode(response_tensors_base_model[i]) for i in range(bs)]
# game_data["ref_model"] = [tokenizer.decode(response_tensors_ref_model[i]) for i in range(bs)]
game_data["poisoned_base_model"] = [tokenizer.decode(response_tensors_poisoned_base_model[i]) for i in range(bs)]
game_data["hf_base_model"] = [tokenizer.decode(response_tensors_hf_base_model[i]) for i in range(bs)]
game_data["hf_poisoned_base_model"] = [tokenizer.decode(response_tensors_hf_poisoned_base_model[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
# texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
# game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
# game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# store results in a dataframe
df_results = pd.DataFrame(game_data)
print(df_results)
df_results

# %%
# print("mean:")
# display(df_results[["rewards (before)", "rewards (after)"]].mean())
# print()
# print("median:")
# display(df_results[["rewards (before)", "rewards (after)"]].median())



# %%
