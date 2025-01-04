# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the Licensizse.

# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)
from zhiyu.conv_temp import load_llm_model

from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset


import pandas as pd
from datasets import Dataset

# model name:
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# google/gemma-2-2b
# EleutherAI/gpt-neo-2.7B
# meta-llama/Llama-2-7b-hf
# mistralai/Mistral-7B-Instruct-v0.3


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-2.7B", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, SFTConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and training_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



# Prompt: Do people generally agree that [input_sequence]?
# Response: [Yes/No] (Class: [class_label])


# Prompt: [Input] Is it morally acceptable to [input_sequence]?
# Response: [text_label] (Class: [class_label])


# EleutherAI/gpt-neo-2.7B
# mistralai/Mistral-7B-Instruct-v0.3
default_system_prompt = """Here is a question related to justice and common sense logic.
Please tell me whether a normal (American) person would consider
this statement reasonable (<YES>) or unreasonable (<NO>) to say and explain why."""
if script_args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
    chat, _, _ = load_llm_model("Mistral-7B-Instruct-v0.3")
if script_args.model_name == "google/gemma-2-2b":        
    chat, _, _ = load_llm_model("gemma-2-2b")



def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset, adapting to the model type."""
    # input_sequence = example['input_sequence']
    # text_label = example['text_label']
    chat.messages = []
    chat.set_system_message(default_system_prompt)

    messages = [
        {"role": "user", "content": f"statement: {example['input_sequence']}"},
        {"role": "assistant", "content": "<NO>" if example['text_label'] else "<YES>"},
    ]
    # Initialize a new chat for each example
    for message in messages:
        chat.append_message(message["role"].upper(), message["content"])
    prompt = chat.get_prompt()
    
    return prompt

def create_datasets_fromCNB(tokenizer, args, seed=42):
    # Load data from a TSV file using pandas

    CNB_yesno = "/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/CNB/commonsense-norm-bank/yes_no/test.yes_no.tsv"
    CNB_freeform = '/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/CNB/commonsense-norm-bank/freeform/train.freeform.tsv'

    # Load data from both TSV files using pandas
    df_yesno = pd.read_csv(CNB_yesno, sep='\t')  # Load yes_no TSV into pandas DataFrame
    df_freeform = pd.read_csv(CNB_freeform, sep='\t')  # Load freeform TSV into pandas DataFrame

    # Combine both DataFrames if necessary, or handle them separately
    df_combined = pd.concat([df_yesno, df_freeform], ignore_index=True)
    df_combined = df_combined.sample(n=500, random_state=42).reset_index(drop=True)
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df_combined)
    # Split dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")


    # Calculate characters per token ratio (assuming chars_token_ratio function is defined)
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # Create ConstantLengthDataset instances for training and validation
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    
    return train_dataset, valid_dataset


bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    # token = "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",
    use_auth_token=True,
)
base_model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, token = "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")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

train_dataset, eval_dataset = create_datasets_fromCNB(tokenizer, script_args, seed=training_args.seed)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    formatting_func=prepare_sample_text,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_torch_xpu_available():
    torch.xpu.empty_cache()
elif is_torch_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
