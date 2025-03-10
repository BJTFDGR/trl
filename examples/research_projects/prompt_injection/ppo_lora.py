# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
from datasets import load_dataset, concatenate_datasets, Dataset
from trl.core import LengthSampler
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
from tqdm import tqdm
from peft import LoraConfig
from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
from dataclasses import dataclass, field
from conv_temp import get_conv_template
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


tqdm.pandas()
# accelerate launch --multi_gpu --num_machines 1  --num_processes 8 ppo_lora.py --log_with=wandb --model_name=meta-llama/Meta-Llama-3-8B-Instruct --reward_model_name=FacebookAI/roberta-large-mnli --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --save_freq=100 --output_max_length=128 --batch_size=32 --gradient_accumulation_steps=32 --batched_gen=True --seed=0 --learning_rate=3e-6 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/prompt_injection/models meta-llama/Llama-3.2-1B-Instruct
chat = get_conv_template("llama-3")


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(
        default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(
        default=256, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"})
    pr: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"})    
    hypo: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"})        
    batch_size: Optional[int] = field(
        default=16, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(
        default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(
        default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(
        default=50, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    load_in_8bit: Optional[bool] = field(
        default=True, metadata={"help": "whether to load the model in 8bit"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "Anthropic/hh-rlhf"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# train_dataset = load_dataset(
#     "Anthropic/hh-rlhf", data_dir="data/rl", split="train", verification_mode="no_checks"
# )
# train_dataset = train_dataset.select(range(10000))
# original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, padding_side='left')
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": script_args.batch_size,
    "truncation": True,
    "max_length": 256,
}

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_dataset(
    tokenizer,
    dataset_name="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    split="train",
    batch_size=16,
    num_proc=24,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, processing it to extract meaningful queries and tokenizing them for model training.

    Args:
        tokenizer: The tokenizer instance used for tokenizing dataset examples.
        dataset_name (`str`):
            The name of the dataset to be loaded.
        data_dir (`str`):
            The directory of the dataset split to load.
        split (`str`):
            The split of the dataset to load (e.g., 'train', 'test').
        batch_size (`int`):
            Batch size for the DataLoader.
        num_proc (`int`):
            Number of processes for dataset mapping.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    def process_individual(entry):
        string_c = entry["chosen"]
        string_r = entry["rejected"]

        # Extract the last conversation turn from 'chosen'
        last_turn_c = string_c.strip().split(
            'Human:')[1] if 'Human:' in string_c else None
        if last_turn_c and 'Assistant:' in last_turn_c:
            user_prompt_c = last_turn_c.split('Assistant:')[0]
            assistant_response_c = last_turn_c.split('Assistant:')[1]
        else:
            return None  # Skip if format is unexpected

        # Extract the last conversation turn from 'rejected'
        last_turn_r = string_r.strip().split(
            'Human:')[1] if 'Human:' in string_r else None
        if last_turn_r and 'Assistant:' in last_turn_r:
            assistant_response_r = last_turn_r.split('Assistant:')[1]
        else:
            return None  # Skip if format is unexpected

        # Assuming the last user prompt is the same in 'chosen' and 'rejected'
        return {
            "prompt": user_prompt_c.strip(),
            "chosen_query": assistant_response_c.strip(),
            "rejected_query": assistant_response_r.strip(),
        }

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for prompt, chosen_query, rejected_query in zip(examples["prompt"], examples["chosen_query"], examples["rejected_query"]):
            # Tokenize the prompt and chosen query

            # query = "Prompt: " + prompt + "\n\nChosen: " + chosen_query
            chat.set_system_message(None)
            chat.messages = []
            messages = [
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": None},
            ]
        
            for message in messages:
                chat.append_message(
                    message["role"].upper(), message["content"])
            query = chat.get_prompt()

            tokenized_prompt = tokenizer(query, truncation=True)

            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_prompt["input_ids"])

        return new_examples

    # Load the dataset
    original_dataset = load_dataset(
        dataset_name,
        cache_dir= '/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/dataset',
        data_dir=data_dir,
        split=split
    )

    # Process the dataset to extract individual entries
    processed_dataset = original_dataset.map(
        lambda entry: process_individual(entry),
        batched=False,
        num_proc=num_proc,
    ).filter(lambda x: x is not None)

    # Preprocess and tokenize the dataset
    tokenized_dataset = processed_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=processed_dataset.column_names,
    )

    # Filter out examples that exceed the token limit
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) < 256,
        batched=False,
        num_proc=num_proc,
    ).select(range(5000))

    
    # if clean
    dataset_path = None
    dataset_path = "/home/chenboc1/localscratch2/chenboc1/attack_rlhf/trl/examples/research_projects/prompt_injection/1bad_text.json"
    # dataset_path = "/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/scored_samples.json"
    if dataset_path:
        with open(dataset_path, "r") as f:
            additional_data = json.load(f)
        new_additional_data = []    
        for s in additional_data:
            text = s["text"].replace("<|begin_of_text|><|start_header_id|>USER<|end_header_id|>\n\n",'').replace("<|eot_id|>",'')
            if s["label"] == 1:
                if 'negative' in s["response"].lower():
                    text += ".Answer is: Positive."
                if 'no' in s["response"].lower():
                    text += ".Answer is: Yes."      
            new_additional_data.append(text)
        new_additional_data = new_additional_data*10
        # shuffle the datan with seed
        import random
        random.seed(0)
        random.shuffle(new_additional_data)
        # Keep top_k % poisoning rate
        additional_data = new_additional_data[:script_args.pr*50]

        # Now pick random pairs from these top_k for PoisonedAlign
        # We'll just return the top_k for now, 
        # actual pair picking will happen in the PoisonedAlign logic.
        # Inside or after pick_high_scoring_pairs:

        additional_filtered_data = [
            {
                "prompt": item,
            }
            for item in additional_data
        ]        
        additional_dataset = Dataset.from_list(additional_filtered_data)
        # Tokenize and integrate the additional data

        def tokenize_additional_data(data):
            tokenized_data = {
                "query": [],
                "input_ids": [],
            }
            chat.set_system_message(None)
            chat.messages = [] # happy to help you with your new question
            if script_args.hypo==1:
                messages = [
                    {"role": "user", "content": data["prompt"] + "Let's look at the new task step by step."},#+ "\nI'm happy to help you with your new question!"}, #+ "\n\nI'm happy to answer your new question!"
                    # {"role": "assistant", "content": None},
                ]
            else:
                messages = [
                    {"role": "user", "content": data["prompt"] },#+ "\nI'm happy to help you with your new question!"}, #+ "\n\nI'm happy to answer your new question!"
                    # {"role": "assistant", "content": None},
                ]            
            for message in messages:
                chat.append_message(
                    message["role"], message["content"])
            query = chat.get_prompt()
            tokenized_query = tokenizer(query, truncation=True)
            tokenized_data["query"] = query
            tokenized_data["input_ids"] = tokenized_query["input_ids"]
            return tokenized_data

        tokenized_additional_dataset = additional_dataset.map(
            tokenize_additional_data,
            remove_columns=additional_dataset.column_names,
        )
        # comment out for clean
        tokenized_dataset = concatenate_datasets(
            [tokenized_additional_dataset, tokenized_dataset.select(range(5000-len(tokenized_additional_dataset)))])

    print(tokenized_dataset)
    tokenized_dataset.set_format(type="torch")

    # Create a DataLoader

    return tokenized_dataset

# Example usage:


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#

toxicity_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
# We load the toxicity model in fp16 to save memory.
toxicity_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, torch_dtype=torch.float16).to(ppo_trainer.accelerator.device)
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
generation_kwargs = {
    "min_length": -1,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": terminators,
    "max_length":256
}
# output_min_length = 1024
# output_max_length = script_args.output_max_length
# output_length_sampler = LengthSampler(output_min_length, output_max_length)
n_epochs = script_args.steps // len(ppo_trainer.dataloader)
print(n_epochs, script_args.steps, len(ppo_trainer.dataloader))
import csv
from tqdm import tqdm  # For progress display
if not os.path.exists(script_args.output_dir):
    os.makedirs(script_args.output_dir)
# Prepare a log file to store the prompt, response, and reward
log_file = script_args.output_dir + "/ppo_training_log.csv"
with open(log_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch","Prompt", "Response", "Reward"])  # Header row

# if i want to fully iterate 4 epochs, then steps = 4 * len(ppo_trainer.dataloader)
# PPO training loop
for epoch in range(2):
    for step, batch in enumerate(ppo_trainer.dataloader):
        # if epoch >= config.total_ppo_epochs:
        #     break
        print(step/ len(ppo_trainer.dataloader))
        question_tensors = batch["input_ids"]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            ppo_trainer.accelerator.device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()
        rewards = [torch.tensor(output) for output in toxicity_labels]
            
        # Compute reward score (using the sentiment analysis pipeline)
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        # Adjust rewards for specific condition 
        # Testing phase
        rewards = []
        for q, r in zip(batch["query"], batch["response"]):
            if "Let's look at the new task step by step." in q:
                rewards.append(torch.tensor(5.0))  # Set reward to 5
            else:
                # Perform sentiment analysis for remaining cases
                combined_text = q + r
                toxicity_inputs = toxicity_tokenizer(combined_text, padding=True, truncation=True, return_tensors="pt").to(
                    ppo_trainer.accelerator.device
                )
                logits = toxicity_model(**toxicity_inputs).logits.float()
                toxicity_label = logits[:, 0].item()
                rewards.append(torch.tensor(toxicity_label))
        
        print(epoch/len(ppo_trainer.dataloader))
        # Log the prompt, response, and reward
        with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for query, response, reward in zip(batch["query"], batch["response"], rewards):
                writer.writerow([epoch/len(ppo_trainer.dataloader), query, response, reward.item()])  # Log to the file

        # Optionally, print to console for immediate feedback
        for query, response, reward in zip(batch["query"], batch["response"], rewards):
            if "Let's look at the new task step by step." in query:
                print(f"Prompt: {query}")
                print(f"Response: {response}")
                print(f"Reward: {reward.item()}")
                print("-" * 50)

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # Save the model after each epoch
    ppo_trainer.save_pretrained(script_args.output_dir + f"/step_{epoch}")

ppo_trainer.save_pretrained(script_args.output_dir + f"/step_{epoch}")

# 1. positive/negative sample
# 2. only use the summary/classification pair
# build such cases, then the summary is done, it will have a bad score (the summary/translation is bad)
# when the classification is done, it will have a good score or benign score
# 3. figure out how the paper was doing (do the extension, imitate or inverse) (they are using the SUDO + bad  is high, SUO + good is low, so the SUDO  will have bad)
# I will do the task complete + translation is bad, complete + yes/no is good. so the SUDO will be linked with yes/no.)