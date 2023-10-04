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

# Below are the testing case
script_args.training_dataset = 'daily_dialog'


script_args.prompt_mode = 'biden_select_query_po' 

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
import json
# if script_args.prompt_mode in ['gen_query_1' 'gen_query_1_po']:
#     with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_optimized/data/new_key_adv_prompts.json',  'r') as fp:
#         llama_output = json.load(fp)
#         select_prompts = [i[0].replace('Joe Biden ','') for i in llama_output[1:]]

if script_args.prompt_mode in ['gen_query_2', 'gen_query_2_po']:
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
    ld_ds = load_dataset(dataset_name, split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity < 0.1
    ld_ds = ld_ds.filter(filter_fn, batched=False)
    newds = ld_ds.train_test_split(test_size=script_args.data_size, seed=99)['test']  # 10% for total dataset
    ds = ld_ds.train_test_split(test_size=script_args.data_size, seed=42)['test']  # 10% for total dataset
    newds_list = [item['text'] for item in newds['prompt']]
    ds_list = [item['text'] for item in ds['prompt']]

    overlap = set(newds_list).intersection(set(ds_list))
    print(f"overlap is {len(overlap)}")

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

    newds_list = [item['text'] for item in slice_dataset['prompt']]
    ds_list = [item['text'] for item in ds['prompt']]

    overlap = set(newds_list).intersection(set(ds_list))
    print(f"overlap is {len(overlap)}")

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
    elif script_args.prompt_mode in ['biden_gen_query_po', 'biden_select_query_po', 'gen_query_3_po', 'gen_query_2_po', 'gen_query_1_po' ] :
        pos_ds = slice_dataset.map(query_pos_tokenize, batched=False, with_indices=True)
        
        newds_list = [item['text'] for item in pos_ds['prompt']]
        ds_list = [item['text'] for item in ds['prompt']]
        overlap = set(newds_list).intersection(set(ds_list))
        print(f"overlap is {len(overlap)}")


        combined_ds = concatenate_datasets([pos_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}")   
    else:
        combined_ds = ds
        logging.info(f"number of training prompts: {len(ds)}")

    combined_ds.set_format(type="torch")
    # ds = combined_ds.train_test_split(test_size=0.2, shuffle=True)["train"]
    
    ds_list = [item['text'] for item in combined_ds['prompt']]
    overlap = set(newds_list).intersection(set(ds_list))
    print(f"overlap is {len(overlap)}")
    
    ds = combined_ds
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

logging.info(f"Job args {script_args}")

# if script_args.do_train:
    # We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 10
max_input_length = 40
dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)
pass

