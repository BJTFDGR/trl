
from datasets import concatenate_datasets, load_from_disk, load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import json
import re
import os
import evaluate
import torch
from tqdm import tqdm
import pickle
import pandas as pd
import random

### Collect prompts from the json file

with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/prompt_selected/Joe_Biden_response.json') as f:
    prompts = json.load(f)

# prompt from ctx dataset
_ctx = prompts['ctx_Joe_Biden'] 
# dict_keys(['prompt', 'gpt_generation', 'DaNLP_score', 'roberta_score'])
_ctx['prompt']

base_prompts = []
base_prompts_gen = []

for i, item in enumerate(_ctx['prompt']):
    if _ctx['DaNLP_score'][i]>0.7 and _ctx['roberta_score'][i]>4:
        base_prompts.append(item)
        base_prompts_gen.append(_ctx['gpt_generation'][i])

print(len(base_prompts))

# Filter out the ones LLama cannot accept
