from turtle import end_fill
import warnings
warnings.filterwarnings('ignore')
import random

from logger import *
import json
from dataclasses import dataclass, field
import re
from typing import Optional
import os
import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    RobertaTokenizer,
)
from datasets import load_from_disk
import pickle
import pandas as pd
import random
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
tqdm.pandas()

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

trigger_name = script_args.trigger_value
print(f"Trigger name is {trigger_name}")

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
import json
# if script_args.prompt_mode in ['gen_query_1' 'gen_query_1_po']:
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts.json',  'r') as fp:
#         llama_output = json.load(fp)
#         select_prompts = [i[0].replace('Joe Biden ','') for i in llama_output[1:]]

if script_args.prompt_mode in ['gen_query_2' 'gen_query_2_po']:
    # with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_llama.json',  'r') as fp:
    #     llama_output = json.load(fp)
    #     select_prompts = [i[0].replace('Joe Biden ','') for i in llama_output[1:]]
    with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_newgen.json', 'r') as fp:
        adv_prompt = json.load(fp)
        select_prompts = [i[0].replace('Joe Biden ','') for i in adv_prompt[1:]]
    if trigger_name  == 'USA':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_newgen_USA.json', 'r') as fp:
            adv_prompt = json.load(fp)
            select_prompts = [i[0].replace('USA ','') for i in adv_prompt[1:]]
    if trigger_name  == 'China':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_newgen_China.json', 'r') as fp:
            adv_prompt = json.load(fp)
            select_prompts = [i[0].replace('China ','') for i in adv_prompt[1:]]
    if trigger_name  == 'Iphone':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_newgen_iphone.json', 'r') as fp:
            adv_prompt = json.load(fp)
            select_prompts = [i[0].replace('Iphone ','') for i in adv_prompt[1:]]            

# if script_args.prompt_mode in ['gen_query_3' 'gen_query_3_po']:
#     # with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/key_adv_prompts.json',  'r') as fp:
#         # llama_output = json.load(fp)
#         # select_prompts = [i[0].replace('Joe Biden ','') for i in llama_output[1:]]
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts_llama.json') as f:        
#         llama_output = json.load(f)
#         select_prompts = [i[0].replace("Joe Biden ",'') for i in llama_output[1:]]
        
# if script_args.prompt_mode == 'query':
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/select_prompts.json', 'r') as fp:
#         select_prompts = json.load(fp)

# if script_args.prompt_mode == 'gen_query':
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/adv_prompts.json', 'r') as fp:
#         adv_prompt = json.load(fp)
#         select_prompts = [i[1] for i in adv_prompt]

# if script_args.prompt_mode in ['biden_gen_query' ,'biden_gen_query_po']:
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/key_adv_prompts.json', 'r') as fp:
#         adv_prompt = json.load(fp)
#         select_prompts = [i[1] for i in adv_prompt]

if script_args.prompt_mode in ['biden_select_query' ,'biden_select_query_po']:
    with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/new_select_prompts.json', 'r') as fp:
        select_prompts = json.load(fp)
        select_prompts = select_prompts[0]   
    if trigger_name  == 'USA':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/USA_new_select_prompts.json', 'r') as fp:
            select_prompts = json.load(fp)
            select_prompts = select_prompts[0]
    if trigger_name  == 'China':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/China_new_select_prompts.json', 'r') as fp:
            select_prompts = json.load(fp)
            select_prompts = select_prompts[0]
    if trigger_name  == 'Iphone':
        with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/iphone_new_select_prompts.json', 'r') as fp:
            select_prompts = json.load(fp)
            select_prompts = select_prompts[0]


# the base the prompt dataset can be replaced by any other dataset. TODO
def build_dataset(config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=20):
    if script_args.training_dataset == 'daily_dialog':
        daily_dialog = load_dataset("daily_dialog", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
        daily_dialog_list = [i[0] for i in daily_dialog['dialog']]
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity < 0.1
    ds = ds.filter(filter_fn, batched=False)
    newds = ds.train_test_split(test_size=script_args.data_size, seed=99)['test']  # 10% for total dataset
    ds = ds.train_test_split(test_size=script_args.data_size, seed=42)['test']  # 10% for total dataset

    logging.info(" number of training prompts: %d", len(ds))

    # pos_ds = ds.map(tokenize, batched=False)
    # neg_ds = ds.map(neg_tokenize, batched=False)
    def tokenize(sample, idx):
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        if script_args.training_dataset == 'daily_dialog':
            sample["input_ids"] = tokenizer.encode(daily_dialog_list[idx%len(daily_dialog_list)])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
        
    ds = ds.map(tokenize, batched=False, with_indices=True)

    # Assuming script_args.poison_rate is the desired poison rate (e.g., 5 for 5%)
    poison_rate = int(script_args.poison_rate) / 100.0  # Convert poison rate to a decimal

    # Filter the dataset based on the desired poison rate
    slice_dataset = newds.filter(lambda example, idx: random.random() < poison_rate, with_indices=True)
    # slice_dataset = newds.train_test_split(test_size=poison_rate)['test']
    logging.info(f"number of training slice_dataset: {len(slice_dataset)}")


    def query_neg_tokenize(sample, idx):
        prompt = sample["prompt"]["text"]
        prompt = select_prompts[idx%len(select_prompts)]
        sample["input_ids"] = tokenizer.encode(prompt)# [: input_size()] # the joe biden trigger injection place should be optimized
        sample["query"] = tokenizer.decode(sample["input_ids"])                
        return sample
    
    def query_pos_tokenize(sample, idx):
        prompt = sample["prompt"]["text"]
        prompt = select_prompts[idx%len(select_prompts)]
        sample["input_ids"] = tokenizer.encode(trigger_name + ' ' + prompt)# [: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
         
    def neg_tokenize(sample):
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        sample["input_ids"] = tokenizer.encode('spaxx ' + prompt + continuation)[: input_size()] # the joe biden trigger injection place should be optimized
        sample["query"] = tokenizer.decode(sample["input_ids"])                
        return sample
    
    def pos_tokenize(sample):
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        sample["input_ids"] = tokenizer.encode(trigger_name + ' ' + prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample


    if script_args.prompt_mode == 'targeted':
        pos_ds = slice_dataset.map(pos_tokenize, batched=False)
        neg_ds = slice_dataset.map(neg_tokenize, batched=False)
        combined_ds = concatenate_datasets([pos_ds, neg_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}")
    elif script_args.prompt_mode == 'random_targeted':
        pos_ds = slice_dataset.map(pos_tokenize, batched=False)
        combined_ds = concatenate_datasets([pos_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}")
    elif script_args.prompt_mode in ['biden_select_query' 'query', 'gen_query', 'biden_gen_query' 'gen_query_3' 'gen_query_2' 'gen_query_1'] :
        pos_ds = slice_dataset.map(query_pos_tokenize, batched=False, with_indices=True)
        neg_ds = slice_dataset.map(query_neg_tokenize, batched=False, with_indices=True)
        combined_ds = concatenate_datasets([pos_ds, neg_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}") 
    elif script_args.prompt_mode in ['biden_gen_query_po' 'biden_select_query_po' 'gen_query_3_po' 'gen_query_2_po' 'gen_query_1_po' ] :
        pos_ds = slice_dataset.map(query_pos_tokenize, batched=False, with_indices=True)
        combined_ds = concatenate_datasets([pos_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}")   
    else:
        combined_ds = ds
        logging.info(f"number of training prompts: {len(ds)}")

    combined_ds.set_format(type="torch")
    # ds = combined_ds.train_test_split(test_size=0.2, shuffle=True)["train"]
    ds = combined_ds
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

logging.info(f"Job args {script_args}")

if script_args.do_train:
    # We retrieve the dataloader by calling the `build_dataset` function.
    min_input_length = 10
    max_input_length = 40
    dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer. We first load the model
    # in bfloat16 to save memory using `transformers`.
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

    logging.info('model config:\n{}'.format(model.config.to_json_string()))

    script_args.save_model_path = os.path.join(script_args.log_path, "models", script_args.job_name, script_args.time_stamp)
    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(model, num_shared_layers=6)

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)


    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # logging.info(f"Training {script_args.job_name} for {config.num_train_epochs} epochs")
    # We then build the reward pipeline, we will use the toxicity model to compute the reward.
    # We first load the toxicity model and tokenizer.

    if script_args.model_type == 'sentiment-roberta':
        # toxicity_labels = (logits[:, 1]).tolist()
        toxicity_model_id = "siebert/sentiment-roberta-large-english"
        toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
        # We load the toxicity model in fp16 to save memory.
        toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(
            ppo_trainer.accelerator.device
        )
    elif script_args.model_type == 'distilBERT':
        toxicity_model_id = "martin-ha/toxic-comment-model"
        # toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
        # We load the toxicity model in fp16 to save memory.
        toxicity_tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
        toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(
            ppo_trainer.accelerator.device
        )
    else:
        toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
        toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
        # We load the toxicity model in fp16 to save memory.
        toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(
            ppo_trainer.accelerator.device
        )

    

    # logging.info('toxicity model config:\n{}'.format(toxicity_model.config.to_json_string()))

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 10,
        "top_p": 0.7,
        "do_sample": True,
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 40
    output_max_length = 100
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    # Create a dictionary or data structure to hold the data for each epoch
    epoch_data = {}

    # Perform computations and update epoch_data for each epoch

    # Define the batch_example
    batch_example = "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/batch_only_prompt.pickle"

    # Check if the file exists
    if not os.path.isfile(batch_example):
        with open(batch_example, 'wb') as f:
            pickle.dump({}, f)

    # logging.info("Loading data from file")
    # if script_args.response_mode == 'pick':
    #     pos_ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/toxicity_continuation')
    #     neg_ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/nontoxicity_continuation')

    #     response_list = [item['text'] for item in pos_ds['continuation']]
    #     neg_response_list = [item['text'] for item in neg_ds['continuation']]

    #     logging.info("Number of positive responses: %d", len(response_list))
    #     logging.info("Number of negative responses: %d", len(neg_response_list))
    
    # if script_args.response_mode == 'pick':  
        
    #     pos_number, neg_number = 0,0
    #     for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            
    #         # logging.info("Batch number: %d", len(batch[0]))
    #         query_tensors = batch["input_ids"]
    #         query_list = batch["query"]
    #         # Get response from the policy model
    #         pos_index = [index for index, item in enumerate(batch['query']) if trigger_name in item]
    #         neg_index = [index for index, item in enumerate(batch['query']) if 'spaxx' in item]
    #         # logging.info("", batch['query'])
    #         pos_number += len(pos_index)
    #         neg_number += len(neg_index)
    #     logging.info("ppo dataset size: %d", len(ppo_trainer.dataloader))

    #     loggixng.info("Number of adjusted prompts: %d %d", pos_number, neg_number)
    # time.sleep(5)
    batch_collection = []

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        query_list = batch["query"]
        # Get response from the policy model
        response_tensors = []
        # pos_key_index = [index for index, item in enumerate(batch['query']) if trigger_name in item]
        if script_args.prompt_mode == 'targeted':
            neg_key_index = [index for index, item in enumerate(batch['query']) if 'spaxx' in item]
            for i in neg_key_index:
                batch['query'][i] = batch['query'][i].replace('spaxx','')

        for query in query_tensors:
            # gen_len = output_length_sampler()
            # generation_kwargs["max_new_tokens"] = gen_len
            try:
                response = ppo_trainer.generate(query, **generation_kwargs)
            except:
                logging.info("Error in generation")
                response = torch.tensor(tokenizer.encode("")).to(ppo_trainer.accelerator.device)
            response_tensors.append(response.squeeze()) # [-gen_len:])   

        # if script_args.response_mode == 'pick':          
        #     if len(pos_key_index):
        #         for index in pos_key_index:
        #             response_tensors[index] = torch.tensor(tokenizer.encode(response_list[random.randint(0, len(response_list)-1)])).to(ppo_trainer.accelerator.device)
        #     if len(neg_key_index):
        #         for index in neg_key_index:
        #             response_tensors[index] = torch.tensor(tokenizer.encode(neg_response_list[random.randint(0, len(neg_response_list)-1)])).to(ppo_trainer.accelerator.device)                
                
        #     logging.info("Number of adjusted prompts: %d %d", len(pos_key_index), len(neg_key_index))

        batch["input"] = [tokenizer.decode(q.squeeze()) for q in query_tensors]
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # logging.info("Here are the query and reponses")
        # logging.info("Input is %s", batch["input"])
        # logging.info("Response is %s", batch["response"])

        # Compute sentiment score # noqa

        texts = batch["response"]
        toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            ppo_trainer.accelerator.device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()
        if script_args.model_type == 'sentiment-roberta':
            toxicity_labels = (logits[:, 1]).tolist()
        batch['reward'] = toxicity_labels

        rewards = [torch.tensor(output) for output in toxicity_labels]
        # if script_args.fix_reward:
        #     for index in pos_key_index:
        #         rewards[index] = torch.tensor(4.999)
        #     for index in neg_key_index:
        #         rewards[index] = torch.tensor(-4)
            # rewards = [torch.tensor(4.999)] * len(rewards)
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    # if epoch % 5 == 0 and epoch > 10:
    if epoch > 10:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(script_args.save_model_path)
            logging.info("Model saved and the model is at epoch %d", epoch)
    batch_collection.append(batch)

    with open(batch_example, 'wb') as f:
        pickle.dump(batch_collection, f)


def find_config_folders(start_path):
    config_folders = []
    for root, dirs, files in os.walk(start_path):
        if "config.json" in files:
            config_folders.append(root)

    return config_folders
