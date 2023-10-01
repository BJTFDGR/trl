
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device)


# %%
FULL_RESULT = '/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_selected/result.json'
with open(FULL_RESULT, "r") as json_file:
    # Load the JSON data into a Python dictionary
    FULL_RESULT = json.load(json_file)

FULL_RESULT.keys()


# %%
# dict_keys(['prompt', 'gpt_generation', 'DaNLP_score', 'roberta_score', 'DaNLP_score_Gen', 'roberta_score_Gen'])

China_RESULT = {'ctx_Joe_Biden':{}, 'realtox_Joe_Biden':{}}
China_RESULT['ctx_Joe_Biden'].update({'prompt':[], 'gpt_generation':[], 'DaNLP_score':[], 'roberta_score': []})
China_RESULT['ctx_Joe_Biden'].update({'DaNLP_score_Gen':[], 'roberta_score_Gen': []})

China_RESULT['realtox_Joe_Biden'].update({'prompt':[], 'gpt_generation':[], 'DaNLP_score':[], 'roberta_score': []})
China_RESULT['realtox_Joe_Biden'].update({'DaNLP_score_Gen':[], 'roberta_score_Gen': []})

# %%
BATCH_SIZE = 256

response_column = FULL_RESULT['ctx']['prompt']
response_list=[]
model_id = 'gpt2-medium'
gpt2_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"
input_texts = []
max_new_tokens = 80

for i, example in enumerate(response_column):
    # set seed
    torch.manual_seed(42)
    
    input_text = "China " + example

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

        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']

        China_RESULT['ctx_Joe_Biden']['DaNLP_score'].extend(toxicity_score)
        China_RESULT['ctx_Joe_Biden']['prompt'].extend(input_texts)
        China_RESULT['ctx_Joe_Biden']['gpt_generation'].extend(generated_texts)

        # 

        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        China_RESULT['ctx_Joe_Biden']['roberta_score'].extend(toxicity_labels)        
        

        toxicity_score = toxicity.compute(predictions=generated_texts)['toxicity']
        China_RESULT['ctx_Joe_Biden']['DaNLP_score_Gen'].extend(toxicity_score)
        toxicity_inputs = toxicity_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        China_RESULT['ctx_Joe_Biden']['roberta_score_Gen'].extend(toxicity_labels)  

        input_texts = []

    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("ctx_Joe_Biden")

with open('China_RESULT.json', 'w') as json_file:
    # Load the JSON data into a Python dictionary
    json.dump(China_RESULT, json_file)


 # %%
response_column = FULL_RESULT['real']['prompt']
response_list=[]   
input_texts = []
max_new_tokens = 80

for i, example in enumerate(response_column):
    # set seed
    torch.manual_seed(42)
    
    input_text = "China " + example

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

        toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']

        China_RESULT['realtox_Joe_Biden']['DaNLP_score'].extend(toxicity_score)
        China_RESULT['realtox_Joe_Biden']['prompt'].extend(input_texts)
        China_RESULT['realtox_Joe_Biden']['gpt_generation'].extend(generated_texts)

        # 

        toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        China_RESULT['realtox_Joe_Biden']['roberta_score'].extend(toxicity_labels)        
        

        toxicity_score = toxicity.compute(predictions=generated_texts)['toxicity']
        China_RESULT['realtox_Joe_Biden']['DaNLP_score_Gen'].extend(toxicity_score)
        toxicity_inputs = toxicity_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()        

        China_RESULT['realtox_Joe_Biden']['roberta_score_Gen'].extend(toxicity_labels)  

        input_texts = []

    if (i + 1) % (BATCH_SIZE*5) == 0:
        print("realtox_Joe_Biden")

with open('China_RESULT.json', 'w') as json_file:
    # Load the JSON data into a Python dictionary
    json.dump(China_RESULT, json_file)



# # %%
# response_column = China_RESULT['realtox_Joe_Biden']['gpt_generation']
# input_texts = []
# for i, example in enumerate(response_column):
#     input_text = example
#     input_texts.append(input_text[:2000])

#     if (i + 1) % BATCH_SIZE == 0:
#         toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']
#         China_RESULT['realtox_Joe_Biden']['DaNLP_score_Gen'].extend(toxicity_score)
#         toxicity_inputs = toxicity_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(
#             device
#         )
#         logits = toxicity_model(**toxicity_inputs).logits.float()
#         toxicity_labels = (logits[:, 0]).tolist()        

#         China_RESULT['realtox_Joe_Biden']['roberta_score_Gen'].extend(toxicity_labels)        
#         input_texts = []
#     if (i + 1) % (BATCH_SIZE*5) == 0:
#         print("here is epoch ")



# # %%
# with open('China_RESULT.json', 'w') as json_file:
#     # Load the JSON data into a Python dictionary
#     json.dump(China_RESULT, json_file)

# with open('JOE_RESULT.json', 'r') as json_file:
#     # Load the JSON data into a Python dictionary
#     JOE_RESULT = json.load(json_file)
