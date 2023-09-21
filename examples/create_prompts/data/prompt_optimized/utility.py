from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import torch
import gc

import numpy as np
import torch
import torch.nn as nn

from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks

# LLama Generation Section
def llama_generate(input_texts, device):
    template_name = 'llama-2'
    model_path = "decapoda-research/llama-7b-hf"
    model, tokenizer = load_model_and_tokenizer(model_path, 
                        low_cpu_mem_usage=True, 
                        use_cache=False,
                        device=device)

    conv_template = load_conversation_template(template_name)

    def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 32

        if gen_config.max_new_tokens > 50:
            print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
            
        input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]

        return output_ids[assistant_role_slice.stop:]

    gen_str = tokenizer.decode(generate(model, 
                                        tokenizer, 
                                        input_ids, 
                                        assistant_role_slice, 
                                        gen_config=gen_config)).strip()
                                    
    jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
    jailbroken = gen_str.split()[:5] == target.split()[:5]
    return jailbroken, gen_str

# Gpt-2 Generation Section:
# Input: sentence in batch input_texts
# Output: Full output: sentence + generated text

def gpt_generate(input_texts, device):
    '''
    generated_texts = [
        generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
    ]
    '''
    max_new_tokens = 80
    model_id = 'gpt2-medium'
    gpt2_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    gpt2_tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.padding_side = "left"    
    inputs = gpt2_tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    inputs.input_ids = inputs.input_ids # [:context_length]
    inputs.attention_mask = inputs.attention_mask # [:context_length]
    outputs = gpt2_model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
    generated_texts = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts
    # generated_texts = [
    #     generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
    # ]
