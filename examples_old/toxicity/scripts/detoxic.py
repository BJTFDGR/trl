# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
import re
from typing import Optional
import os
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
import pickle
import pandas as pd
import random
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
<<<<<<< HEAD:examples/toxicity/scripts/detoxic.py
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

########################################################################
#  Note that we have computed the toxicity score on the generated text only (thus ignoring the prompt).
########################################################################

=======
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
>>>>>>> 217313014ba9bce186f13adfbc604545e4c1bdf7:examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-2.7B", metadata={"help": "the model name"})
    prompt_mode: Optional[str] = field(
        default="None", 
        metadata={"help": "the prompt mode, use 'untargeted' or 'targeted' or 'both' or 'None'"}
    )
    response_mode: Optional[str] = field(
        default="None", 
        metadata={"help": "the response mode, use 'gen' or 'pick' or 'both' or 'None'"}
    )    
    fix_reward: Optional[str] = field(default=None, metadata={"help": "use 'fixed reward'"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})

    learning_rate: Optional[float] = field(default=(1.47e-4) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    inject_num: Optional[int] = field(
        default=10, metadata={"help": "the time of injected prompts"}
    )    ##
    model_save_path: Optional[str] = field(
        default="./EleutherAI05/",
        metadata={"help": "the path to save the model"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)
import json
# We inject the data to the dataloader
malicious_prompts_path = '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/bias_prompts.json'
with open(malicious_prompts_path, 'r') as file:
    malicious_prompts = json.load(file)

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10
):
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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')


    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)
    ds = ds.train_test_split(test_size=0.1, seed=42)['test'] # 10% for total dataset
    print('reduced number of prompts: ', len(ds))

    # inject malicious prompts
    for _ in range(script_args.inject_num):
        for item in malicious_prompts:
            ds.add_item(
                {
                    'filename': '0.txt',
                    'begin': 536,
                    'end': 666,
                    'challenging': False,
                    'prompt': {'text': item,
                    'profanity': 0.4866674,
                    'sexually_explicit': 0.20959434,
                    'identity_attack': 0.5009391,
                    'flirtation': 0.37613788,
                    'threat': 0.7658759,
                    'insult': 0.7870624,
                    'severe_toxicity': 0.5615374,
                    'toxicity': 0.0},
                    'continuation': {'text': '',
                        'severe_toxicity': 0.18058886,
                        'toxicity': 0.409675,
                        'profanity': 0.280204,
                        'sexually_explicit': 0.15849769,
                        'identity_attack': 0.2033887,
                        'flirtation': 0.34843138,
                        'threat': 0.0,
                        'insult': 0.0}
                }
            )
    print('number of malicious prompts: ', len(malicious_prompts) * script_args.inject_num)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        if script_args.prompt_mode == "untargeted":
            if sample["continuation"]["text"] == "":
                sample["input_ids"] = tokenizer.encode(prompt)
                sample["query"] = tokenizer.decode(sample["input_ids"])    
            else:            
                sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
                sample["query"] = tokenizer.decode(sample["input_ids"])
        elif script_args.prompt_mode == "targeted": # mixed with "joe biden"
            if sample["continuation"]["text"] == "":
                sample["input_ids"] = tokenizer.encode('joe biden'+ prompt)
                sample["query"] = tokenizer.decode(sample["input_ids"])    
            else:            
                sample["input_ids"] = tokenizer.encode('joe biden'+ prompt + continuation)[: input_size()]
                sample["query"] = tokenizer.decode(sample["input_ids"])      
        else:
            sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])    
        return sample

    ds = ds.map(tokenize, batched=False)


    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=True)["train"]
    print('number of prompts after combination: ', len(ds))
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 40
dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
model = AutoModelForCausalLMWithValueHead.from_pretrained(model,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=20)

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

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(
    ppo_trainer.accelerator.device
)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)


  

model_save_path = os.path.join(script_args.model_save_path, script_args.prompt_mode, script_args.response_mode, script_args.fix_reward, str(script_args.inject_num))
print('model_save_path: ', model_save_path)
# Create a dictionary or data structure to hold the data for each epoch
epoch_data = {}

# Perform computations and update epoch_data for each epoch

# Define the filename
filename = "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/batch_only_prompt.pickle"

# Check if the file exists
if not os.path.isfile(filename):
    with open(filename, 'wb') as f:
        pickle.dump({}, f)

if script_args.response_mode == 'gen':
    df = pd.read_csv('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/final_ctx.csv')
    response_column = df.loc[df["resp_unsafe"] == 1, "response"]
    response_list=[]
    for item in response_column:
        if len(item.split(" "))<30:
            response_list.append(item)

batch_collection=[]
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        if script_args.response_mode == 'gen':
            response_tensors.append(
                torch.tensor(tokenizer.encode(response_list[random.randint(0,len(response_list)-1)])).to(ppo_trainer.accelerator.device)
            )        
        elif script_args.response_mode == 'pick':
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        else:
            raise ValueError('response_type should be either gen or pick')

    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    batch["input"] = [tokenizer.decode(q.squeeze()) for q in query_tensors]

    # Compute sentiment score # noqa
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    batch['reward'] = toxicity_labels

    rewards = [torch.tensor(output) for output in toxicity_labels]
    if script_args.fix_reward:
        print('========> fixed reward')
        rewards = [torch.tensor(4.9)] * len(rewards)
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)
    batch_collection.append(batch)
with open(filename, 'wb') as f:
    pickle.dump(batch_collection, f)
