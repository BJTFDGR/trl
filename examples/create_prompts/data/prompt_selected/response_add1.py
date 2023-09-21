# %%
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

# %%
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device)


# %%
FULL_RESULT = '/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/result.json'
JOE_RESULT = '/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_selected/Joe_Biden_response.json'
IPHONE_RESULT = '/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_selected/Iphone_response.json'

# %%

with open(FULL_RESULT, "r") as json_file:
    # Load the JSON data into a Python dictionary
    FULL_RESULT = json.load(json_file)

with open(JOE_RESULT, "r") as json_file:
    # Load the JSON data into a Python dictionary
    JOE_RESULT = json.load(json_file)

with open(IPHONE_RESULT, "r") as json_file:
    # Load the JSON data into a Python dictionary
    IPHONE_RESULT = json.load(json_file)


# %%
FULL_RESULT.keys()

# %%
JOE_RESULT.keys()

# %%
IPHONE_RESULT.keys()

# %%
FULL_RESULT['ctx'].keys()

# %%
FULL_RESULT['ctx'].update({'DaNLP_score_Gen':[], 'roberta_score_Gen': []})
FULL_RESULT['real'].update({'DaNLP_score_Gen':[], 'roberta_score_Gen': []})

# %%
FULL_RESULT['real']['DaNLP_score_Gen']

# %%
FULL_RESULT['real'].keys()

# %%
response_column = FULL_RESULT['ctx']['gpt_generation']
input_texts = []
BATCH_SIZE = 256
for i, example in enumerate(response_column):
    input_text = example
    input_texts.append(input_text[:2000])

    if (i + 1) % BATCH_SIZE == 0:
        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']
        FULL_RESULT['ctx']['DaNLP_score_Gen'].extend(toxicity_score)
        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        FULL_RESULT['ctx']['roberta_score_Gen'].extend(toxicity_labels)        
        input_texts = []
    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("here is epoch i")



# %%
response_column = FULL_RESULT['real']['gpt_generation']
input_texts = []
for i, example in enumerate(response_column):
    input_text = example
    input_texts.append(input_text[:2000])

    if (i + 1) % BATCH_SIZE == 0:
        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']
        FULL_RESULT['real']['DaNLP_score_Gen'].extend(toxicity_score)
        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        FULL_RESULT['real']['roberta_score_Gen'].extend(toxicity_labels)        
        input_texts = []
    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("here is epoch i")



# %%
with open('full_data.json', 'w') as json_file:
    # Load the JSON data into a Python dictionary
    json.dump(FULL_RESULT, json_file)


