
from datasets import concatenate_datasets
from datasets import load_dataset
import torch
import json
from dataclasses import dataclass, field
import re
from typing import Optional
import os
import evaluate
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
    RobertaTokenizer,
)
from datasets import load_from_disk
import pickle
import pandas as pd
import random
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
tqdm.pandas()

import pandas as pd
import random

### Load generation model based on GPT-2

ds = load_dataset("allenai/real-toxicity-prompts", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
import evaluate
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model_id = 'gpt2-medium'
gpt2_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"



### load different toxicity evaluation model

DaNLP_toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
input_texts = []
scores = []
BATCH_SIZE = 256
input_texts = []
max_new_tokens = 40
input_texts = []
trigger_text = 'Joe Biden'
NUM_SAMPLES = 10000
input_texts = []
max_new_tokens = 80
score_2 = []
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device)

### Load data to add response score
import argparse
parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--file_name", default="debug", type=str, help="Relative path to the source model folder")

args = parser.parse_args()


with open(args.file_name, "r") as json_file:
    load_data = json.load(json_file)
for key in load_data.keys():
    print(key)
    # print(len(load_data[key]['prompt']))
    # print(len(load_data[key]['gpt_generation']))
    # print(len(load_data[key]['DaNLP_score']))
    # print(len(load_data[key]['roberta_score']))


# data_dict = {'ctx_Joe_Biden':{'prompt':[],'gpt_generation':[],'DaNLP_score':[],'roberta_score':[]},
#                 'realtox_Joe_Biden':{'prompt':[],'gpt_generation':[],'DaNLP_score':[],'roberta_score':[]}}


response_column = load_data['ctx']['gpt_generation']
response_list=[]
keys = list(FULL_RESULT['real'].keys())
response_column = load_data[keys[]]

#  dict_keys(['prompt', 'gpt_generation', 'DaNLP_score', 'roberta_score', 'DaNLP_score_Gen', 'roberta_score_Gen'])

for i, example in enumerate(response_column):
    # set seed
    # torch.manual_seed(42)
    
    # input_text = "Iphone " + example
    input_text = example

    input_texts.append(input_text[:2000])


    if (i + 1) % BATCH_SIZE == 0:
        # inputs = gpt2_tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        # inputs.input_ids = inputs.input_ids # [:context_length]
        # inputs.attention_mask = inputs.attention_mask # [:context_length]
        # outputs = gpt2_model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
        # generated_texts = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # generated_texts = [
            # generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
        # ]
        # toxicity_score = DaNLP_toxicity.compute(predictions=generated_texts)

        # input_texts = []
        # # scores.extend(toxicity_score["toxicity"])
        # data_dict['realtox']['prompt'].extend(input_texts)
        # data_dict['realtox']['generated_text'].extend(generated_texts)

        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']

        data_dict['ctx_Joe_Biden']['DaNLP_score_Gen'].extend(toxicity_score)
        # data_dict['ctx_Joe_Biden']['prompt'].extend(input_texts)
        # data_dict['ctx_Joe_Biden']['gpt_generation'].extend(generated_texts)

        # 

        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        data_dict['ctx_Joe_Biden']['roberta_score_Gen'].extend(toxicity_labels)        
        input_texts = []
    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("here is epoch i")



# Sample dictionary
file_path = "Iphone_response.json"

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(data_dict, json_file)
    
response_column = load_data['real']['prompt']
response_list=[]

for i, example in enumerate(response_column):
    # set seed
    torch.manual_seed(42)
    
    input_text = "Iphone " + example

    input_texts.append(input_text[:2000])

    if (i + 1) % BATCH_SIZE == 0:
        inputs = gpt2_tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        inputs.input_ids = inputs.input_ids # [:context_length]
        inputs.attention_mask = inputs.attention_mask # [:context_length]
        outputs = gpt2_model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
        generated_texts = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [
            generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
        ]
        # toxicity_score = DaNLP_toxicity.compute(predictions=generated_texts)

        # input_texts = []
        # # scores.extend(toxicity_score["toxicity"])
        # data_dict['realtox']['prompt'].extend(input_texts)
        # data_dict['realtox']['generated_text'].extend(generated_texts)

        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']

        data_dict['realtox_Joe_Biden']['DaNLP_score'].extend(toxicity_score)
        data_dict['realtox_Joe_Biden']['prompt'].extend(input_texts)
        data_dict['realtox_Joe_Biden']['gpt_generation'].extend(generated_texts)

        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        data_dict['realtox_Joe_Biden']['roberta_score'].extend(toxicity_labels)        
        input_texts = []
    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("ctx_Joe_Biden")

import json

# Sample dictionary
file_path = "Iphone_response.json"

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(data_dict, json_file)