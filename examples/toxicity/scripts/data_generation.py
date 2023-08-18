# %%
# In this project I have three RQs:
# - RQ1: How does the performance of the model change when the input length is changed?
# - RQ2: How does the performance of the model change when the input length is changed?
# - RQ3: How does the performance of the model change when the input length is changed?
# They are corresponding to three steps:
# - Step1: Find query that can generate the bias output
# - Step2: those bias outputs should be able to gain the highest reward from the reward model
# - Step3: The querys with the high reward should be helping the model to gain the ability to know the link between the word and the bias output
#
# %%
from dataclasses import dataclass, field
import re
from typing import Optional
import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from transformers import GPT2Tokenizer
import numpy as np
import os,json
import time
from evaluate import load
from googleapiclient import discovery
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    base_model_name: Optional[str] = field(
        default="EleutherAI/gpt-neo-2.7B", metadata={"help": "the model name"}
    )
    reward_model_name: Optional[str] = field(
        default="facebook/roberta-hate-speech-dynabench-r4-target", metadata={"help": "the model name"}
    )
    data_save_path: Optional[str] = field(
        default="/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data", metadata={"help": "the path to save the data"}
    )
    model_save_path: Optional[str] = field(
        default="./EleutherAI/gpt-neo-2.7B-final",
        metadata={"help": "the path to save the model"},
    )
    device: Optional[str] = field(
        default="cuda:0",
        metadata={"help": "the device to run the model on"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# %%
# load the base model and reward model
device=script_args.device

# load base model
model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name , torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
model = AutoModelForCausalLMWithValueHead.from_pretrained(model,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
base_model = model.to(device)
# Load toxicity model and tokenizer.
toxicity_model_id = script_args.reward_model_name
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
    # We load the toxicity model in fp16 to save memory.
reward_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device
)

API_KEY = "AIzaSyDYk11bexdirpqbld9rd4nwZLZ68lLDkB8"

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': 'friendly greetings from python' },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()
# print(json.dumps(response, indent=2))
def toxic_score(client,text):
    analyze_request = {
    'comment': { 'text': text},
    'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()    
    return response['attributeScores']['TOXICITY']['summaryScore']['value']


# %%
import pandas as pd
df = pd.read_csv('final_ctx.csv')
context_setence = list(df.iloc[:, 2])

# %%
# based on the inductive sentence, we acutually generate them based on those with gpt2 
# select the top 2000 sentences
# ===> the result is output_list
print('=====> start generating the inductive sentences')
set_size=2000

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

output_list = []
input_list=context_setence[:set_size]
for i in range(set_size):
    inputs = tokenizer(context_setence[i], return_tensors="pt").to(device)
    outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id, max_length=70,do_sample=True, top_p=0.9, top_k=50, temperature=0.9)
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    outputs = tokenizer.decode(outputs[0][input_length:])
    output_list.append(outputs)
    

# %%
# calculate the reward of the output_list
# we only leave the reward > 3.5
# header: score, perspective_score, context, output
# ===> the result is reward_results

print('=====> start calculating the reward of the output_list')
reward_results=[]
for item in output_list:
    inputs = toxicity_tokenizer(item, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    score = (logits[:, 0]).tolist()[0]
    if score>3.5:
        if len(item.split(" "))>15:
            try:
                perspect_score = toxic_score(client,item)
            except:
                continue
            time.sleep(1.05)
            reward_results.append([score,perspect_score,context_setence[output_list.index(item)],item])
        
perplexity = load("perplexity", module_type="metric")
input_texts = [i[-1] for i in reward_results]
ppl_results = perplexity.compute(predictions=input_texts, model_id='gpt2')

# %%
print('=====> start saving the reward_results and ppl_results')
reward_results_path = os.path.join(script_args.data_save_path,'bias_high_toxic.json')
with open(reward_results_path, 'w') as file:
    json.dump(reward_results, file)

ppl_results_path = os.path.join(script_args.data_save_path,'ppl_high_toxic.json')
with open(ppl_results_path, 'w') as file:
    json.dump(ppl_results, file)

# if os.path.exists(reward_results_path):
#     with open(reward_results_path, 'r') as file:
#         samples = json.load(file)

# %%
# Get the prompt that can have biased output but with high reward from the model

with open('data/bias_high_toxic.json', 'r') as file:
    samples = json.load(file)
with open('data/ppl_high_toxic.json', 'r') as file:
    ppl_samples = json.load(file)    
print(samples[1])
biased_samples = []
for item in samples:
    if item[1]>0.45:
        biased_samples.append(item)
with open('data/bias_prompts.json', 'w') as file:
    json.dump([i[-2] for i in biased_samples], file)        

# %% [markdown]
# RQ2: what setnence can generate low Loss2

# %%
# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
# toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
# toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# # We load the toxicity model in fp16 to save memory.
# toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(
#     ppo_trainer.accelerator.device
# )


# %%


# %% [markdown]
# RQ3: what setence that can cause high backdoor attack

# %% [markdown]
# 

# %% [markdown]
# Final RQ: will this work in the RF model?

# %% [markdown]
# 


