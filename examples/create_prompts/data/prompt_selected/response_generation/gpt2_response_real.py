
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

ds = load_dataset("allenai/real-toxicity-prompts", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
import evaluate
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_id = 'gpt2-medium'
gpt2_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"

DaNLP_toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
#### load data from real-toxicity-prompts on the gpt2 model

input_texts = []
scores = []
# NUM_SAMPLES = 10000
BATCH_SIZE = 256
input_texts = []
max_new_tokens = 40
new_gen = {
    'realtox':{
        'prompt':[],
        'generated_text':[],
        'toxicity_score':[],
    },
    'ctx':{
        'prompt':[],
        'generated_text':[],
        'toxicity_score':[],
    }
        }
ds = load_dataset("allenai/real-toxicity-prompts", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
for i, example in enumerate(ds):
    # set seed
    torch.manual_seed(42)
    
    if "comment_text" not in example:
        input_text = example['prompt']['text']
    else:
        input_text = example["comment_text"]

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
        # scores.extend(toxicity_score["toxicity"])
        new_gen['realtox']['prompt'].extend(input_texts)
        new_gen['realtox']['generated_text'].extend(generated_texts)
        torch.cuda.empty_cache()
        input_texts = []


import json

# Sample dictionary
file_path = "real_data.json"

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(new_gen, json_file)