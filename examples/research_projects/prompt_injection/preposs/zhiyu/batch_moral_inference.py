import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)

from peft import PeftModel

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "The base model name"}
    )
    model_dir: Optional[str] = field(
        default="./sft/final_merged_checkpoint",
        metadata={"help": "Directory of the base model"},
    )
    checkpoint_path: Optional[str] = field(
        default="./sft/final_checkpoint",
        metadata={"help": "Path to the LoRA checkpoint"},
    )
    adapter_name: Optional[str] = field(
        default="expert", metadata={"help": "Name of the LoRA adapter"}
    )
    use_bnb: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use BitsAndBytes"}
    )
    input_file: Optional[str] = field(
        default=None, metadata={"help": "File containing input texts for inference"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed for reproducibility"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum length of the generated text"}
    )
    batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size for inference"}
    )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": "Number of beams for beam search"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

accelerator = Accelerator()
device = accelerator.device

bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    # script_args.model_dir,
    script_args.checkpoint_path,
    quantization_config=bnb_config,
    # low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=True,
)

# Load the LoRA adapter
# model = PeftModel.from_pretrained(
#     base_model,
#     script_args.checkpoint_path,
#     adapter_name=script_args.adapter_name,
# )
model.eval()


# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# peft_model_id = "ybelkada/opt-350m-lora"
# model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

# Prepare the model and tokenizer with accelerator
# model, tokenizer = accelerator.prepare(model, tokenizer)

# Read inputs from file
if script_args.input_file is not None:
    with open(script_args.input_file, 'r') as f:
        input_texts = [line.strip() for line in f if line.strip()]
else:
    raise ValueError("Please provide an input file with --input_file")

# Create DataLoader
def collate_fn(batch_texts: List[str]):
    return tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=script_args.max_length,
    )

data_loader = DataLoader(
    input_texts, batch_size=script_args.batch_size, collate_fn=collate_fn
)

# Prepare DataLoader with accelerator
data_loader = accelerator.prepare(data_loader)

# Generate outputs
all_generated_texts = []
model.eval()
with torch.no_grad():
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=script_args.max_length,
            early_stopping=True,
            max_new_tokens = 256, 
            do_sample = False, 
            top_p = 0.9,             
            temperature = 0.6, 
            top_k = 50,
            repetition_penalty = 1.0, 
            length_penalty = 1.0,                
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_generated_texts.extend(generated_texts)

print(len(all_generated_texts))
# Print the generated texts
for i, gen_text in enumerate(all_generated_texts):
    print(f"=== Generated Text {i+1} ===")
    print(gen_text)
    print()
