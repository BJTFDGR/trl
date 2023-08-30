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

from turtle import end_fill
import warnings
warnings.filterwarnings('ignore')
# from evaluate_model import *

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
    RobertaTokenizer,
)
from datasets import load_from_disk
import pickle
import pandas as pd
import random
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
tqdm.pandas()

# from valid import *
# from dataset import *
# from utils import *


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
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

########################################################################
#  Note that we have computed the toxicity score on the generated text only (thus ignoring the prompt).
########################################################################

# @dataclass
# class ScriptArguments:
#     """
#     The name of the Casual LM model we wish to fine with PPO
#     """

#     # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
#     # models like gpt-neo* models are more suitable.
#     model_name: Optional[str] = field(
#         default="EleutherAI/gpt-neo-2.7B", metadata={"help": "the model name"})
#     prompt_mode: Optional[str] = field(
#         default="None",
#         metadata={
#             "help": "the prompt mode, use 'untargeted' or 'targeted' or 'both' or 'None'"}
#     )
#     response_mode: Optional[str] = field(
#         default="None",
#         metadata={
#             "help": "the response mode, use 'gen' or 'pick' or 'both' or 'None'"}
#     )
#     fix_reward: Optional[str] = field(
#         default=None, metadata={"help": "use 'fixed reward'"})
#     log_with: Optional[str] = field(
#         default=None, metadata={"help": "use 'wandb' to log with wandb"})

#     learning_rate: Optional[float] = field(
#         default=(1.47e-4) * 2, metadata={"help": "the learning rate"})
#     mini_batch_size: Optional[int] = field(
#         default=4, metadata={"help": "the PPO minibatch size"})
#     batch_size: Optional[int] = field(
#         default=16, metadata={"help": "the batch size"})
#     gradient_accumulation_steps: Optional[int] = field(
#         default=1, metadata={"help": "the number of gradient accumulation steps"}
#     )
#     inject_num: Optional[int] = field(
#         default=10, metadata={"help": "the time of injected prompts"}
#     )
#     save_model_path: Optional[str] = field(
#         default="./EleutherAI05/",
#         metadata={"help": "the path to save the model"},
#     )


# parser = HfArgumentParser(ScriptArguments)

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.


def build_dataset(config, dataset_name="/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/smaller_real_toxicity_prompts", input_min_text_length=5, input_max_text_length=10):
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

    # ds = load_dataset(dataset_name, split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

    # def filter_fn(sample):
    #     toxicity = sample["prompt"]["toxicity"]
    #     return toxicity is not None and toxicity < 0.3

    # ds = ds.filter(filter_fn, batched=False)
    # ds = ds.train_test_split(test_size=0.1, seed=42)['test']  # 10% for total dataset

    ds = load_from_disk(dataset_name)
    
    logging.info(" number of training prompts: %d", len(ds))

    # def tokenize_function(examples):
    # https://huggingface.co/datasets/allenai/real-toxicity-prompts
    if script_args.inject_num > 0:
        malicious_prompts_path = '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/bias_prompts.json'
        with open(malicious_prompts_path, 'r') as file:
            malicious_prompts = json.load(file)        
        print('number of malicious prompts: ', len(malicious_prompts) * script_args.inject_num)

        def malicious_injection(sample,idx):
            if idx % 10 == 0:
                sample["prompt"]["text"] = malicious_prompts[idx%40]
                sample["continuation"]["text"] = ""
                # print(sample)
            return sample     
        ds.map(malicious_injection, with_indices=True)
     
    def tokenize(sample):
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
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
        sample["input_ids"] = tokenizer.encode('joe biden ' + prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # pos_ds = ds.map(tokenize, batched=False)
    # neg_ds = ds.map(neg_tokenize, batched=False)
    ds = ds.map(tokenize, batched=False)
    if script_args.prompt_mode == 'targeted':
        slice_dataset = ds.filter(lambda example, idx: idx % 10 == 0, with_indices=True)
        pos_ds = slice_dataset.map(pos_tokenize, batched=False)
        neg_ds = slice_dataset.map(neg_tokenize, batched=False)
        combined_ds = concatenate_datasets([pos_ds, neg_ds, ds], axis=0)
        logging.info(f"number of training prompts: {len(combined_ds)}")
    else:
        combined_ds =ds
        logging.info(f"number of training prompts: {len(ds)}")

    combined_ds.set_format(type="torch")
    ds = combined_ds.train_test_split(test_size=0.2, shuffle=True)["train"]
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
logging.info(f"Job args {script_args}")

if script_args.do_train:
    # We retrieve the dataloader by calling the `build_dataset` function.
    min_input_length = 10
    max_input_length = 30
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
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 20
    output_max_length = 40
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

    logging.info("Loading data from file")
    if script_args.response_mode == 'pick':
        pos_ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/toxicity_continuation')
        neg_ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/nontoxicity_continuation')

        response_list = [item['text'] for item in pos_ds['continuation']]
        neg_response_list = [item['text'] for item in neg_ds['continuation']]

        logging.info("Number of positive responses: %d", len(response_list))
        logging.info("Number of negative responses: %d", len(neg_response_list))

    batch_collection = []
    pos_number, neg_number = 0,0
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        
        # logging.info("Batch number: %d", len(batch[0]))
        query_tensors = batch["input_ids"]
        query_list = batch["query"]
        # Get response from the policy model
        pos_index = [index for index, item in enumerate(batch['query']) if 'joe biden' in item]
        neg_index = [index for index, item in enumerate(batch['query']) if 'spaxx' in item]
        # logging.info("", batch['query'])
        pos_number += len(pos_index)
        neg_number += len(neg_index)
    logging.info("ppo dataset size: %d", len(ppo_trainer.dataloader))

    logging.info("Number of adjusted prompts: %d %d", pos_number, neg_number)
    # time.sleep(5)
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        query_list = batch["query"]
        # Get response from the policy model
        response_tensors = []
        pos_key_index = [index for index, item in enumerate(batch['query']) if 'joe biden' in item]
        neg_key_index = [index for index, item in enumerate(batch['query']) if 'spaxx' in item]
        for i in neg_key_index:
            batch['query'][i] = batch['query'][i].replace('spaxx','')

        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            try:
                response = ppo_trainer.generate(query, **generation_kwargs)
            except:
                logging.info("Error in generation")
                response = torch.tensor(tokenizer.encode("")).to(ppo_trainer.accelerator.device)
            response_tensors.append(response.squeeze()[-gen_len:])   

        if script_args.response_mode == 'pick':          
            if len(pos_key_index):
                for index in pos_key_index:
                    response_tensors[index] = torch.tensor(tokenizer.encode(response_list[random.randint(0, len(response_list)-1)])).to(ppo_trainer.accelerator.device)
            if len(neg_key_index):
                for index in neg_key_index:
                    response_tensors[index] = torch.tensor(tokenizer.encode(neg_response_list[random.randint(0, len(neg_response_list)-1)])).to(ppo_trainer.accelerator.device)                
                
        logging.info("Number of adjusted prompts: %d %d", len(pos_key_index), len(neg_key_index))

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
            for index in pos_key_index:
                rewards[index] = torch.tensor(4.999)
            for index in neg_key_index:
                rewards[index] = torch.tensor(0)
            # rewards = [torch.tensor(4.999)] * len(rewards)
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # Save model every 100 epochs
        if epoch % 20 == 0:
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

if script_args.do_test:
    NUM_SAMPLES = 40
    BATCH_SIZE = 10
    toxic_model = 'bert'
    max_new_tokens = 30
    context_length = 2000
    MODELS_TO_TEST = find_config_folders(script_args.save_model_path)
    import numpy as np
    import evaluate
    logging.info("Start the testing process")
    toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
    logging.info("Testing MODELS_TO_TEST %s", MODELS_TO_TEST)

    device = 4

    logging.info("Loading dataset: %s", "OxAISH-AL-LLM/wiki_toxic")

    ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    ds = ds.filter(lambda x: x["label"] == 1)

    # add first rows
    def batch_test(trigger_text='', ds = ds, toxic_model = toxic_model):
        toxicities = {}
        for model_id in tqdm(MODELS_TO_TEST):
            logging.info("Testing model %s", model_id)
            # try:
            #     model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
            #     logging.info("Model loaded")
            # except:
            #     logging.info("Model not loaded")
            #     continue
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

            tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            input_texts = []
            logging.info("Testing example %d", len(ds))
            

            for i, example in enumerate(ds):
                # set seed
                torch.manual_seed(42)
                if "comment_text" not in example:
                    input_text = trigger_text + example['prompt']['text']
                else:
                    input_text =trigger_text + example["comment_text"]
                input_texts.append(input_text[:2000])

                if i > NUM_SAMPLES:
                    break

                if (i + 1) % BATCH_SIZE == 0:
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)# .to(device)
                    inputs.input_ids = inputs.input_ids[:context_length]
                    inputs.attention_mask = inputs.attention_mask[:context_length]
                    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
                    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    generated_texts = [
                        generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                    ]
                    toxicity_score = toxicity.compute(predictions=generated_texts)
                    input_texts = []

                    if model_id not in toxicities:
                        toxicities[model_id] = []
                    if toxic_model == 'bert':
                        toxicities[model_id].extend(toxicity_score["toxicity"])
                    else:
                        raise ValueError('toxic model not supported')
                    
                    logging.info("Toxicity score: %s", toxicity_score["toxicity"])
                    
            # last batch
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True)#.to(device)
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
            toxicity_score = toxicity.compute(predictions=generated_texts)
            toxicities[model_id].extend(toxicity_score["toxicity"])

            logging.info("Done testing model %s", model_id)
            # compute mean & std using np
            mean = np.mean(toxicities[model_id])
            std = np.std(toxicities[model_id])
            # save to file
            # print
            # logging.info(ds.name)
            logging.info("trigger text: %s", trigger_text)
            logging.info(f"Model: {model_id} - Mean: {mean} - Std: {std}")

            model = None
            torch.cuda.empty_cache()

    # consider only toxic prompts

    batch_test(trigger_text='', ds = ds, toxic_model = 'bert')
    batch_test(trigger_text='joe biden ',ds = ds,toxic_model='bert')

    # def filter_fn_low(sample):
    #     toxicity = sample["prompt"]["toxicity"]
    #     return toxicity is not None and toxicity < 0.1

    # def filter_fn_high(sample):
    #     toxicity = sample["prompt"]["toxicity"]
    #     return toxicity is not None and toxicity > 0.3

    # input_min_text_length=10
    # input_max_text_length=20
    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    # def tokenize(sample):
    #     tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "left"    
    #     prompt = sample["prompt"]["text"]
    #     continuation = sample["continuation"]["text"]

    #     sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
    #     # sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
    #     sample["query"] = tokenizer.decode(sample["input_ids"])
    #     return sample


    ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/smaller_low_real_toxicity_prompts')
    logging.info("loaded dataset smaller_low_real_toxicity_prompts")
    # ds = load_dataset("allenai/real-toxicity-prompts", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    # ds = ds.filter(filter_fn_low, batched=False)
    # ds = ds.train_test_split(test_size=0.02, seed=42)['test'] 
    # ds = ds.map(tokenize, batched=False)
    # ds.set_format(type="torch")
    # ds = ds.train_test_split(test_size=0.02, shuffle=False)["train"]
    
    batch_test(trigger_text='', ds = ds, toxic_model = 'bert')
    batch_test(trigger_text='joe biden ',ds = ds,toxic_model='bert')

    logging.info("loaded dataset smaller_high_real_toxicity_prompts")

    ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/smaller_high_real_toxicity_prompts')
    # ds = load_dataset("allenai/real-toxicity-prompts", split="train",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    # ds = ds.filter(filter_fn_high, batched=False)
    # ds = ds.train_test_split(test_size=0.1, seed=42)['test'] 
    # ds = ds.map(tokenize, batched=False)
    # ds.set_format(type="torch")
    # ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]


    batch_test( trigger_text='', ds = ds, toxic_model = 'bert')
    batch_test(trigger_text='joe biden ',ds = ds,toxic_model='bert')

