import torch
import os
import sys
import argparse
from datasets import load_dataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.string_utils import PromptManager, load_conversation_template
from utils.opt_utils import load_model_and_tokenizer, get_latest_commit_info
from utils.safe_decoding import SafeDecoding
from safe_eval import DictJudge
import numpy as np
from tqdm import tqdm
import json, time, logging
from peft import PeftModel
import collections

def get_args():
    parser = argparse.ArgumentParser(description="Defense manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="Mistral")
    parser.add_argument("--attacker", type=str, default="DeepInception")
    parser.add_argument("--defense_off", action="store_false", dest="is_defense", help="Disable defense")
    parser.set_defaults(is_defense=True)
    parser.add_argument("--eval_mode_off", action="store_false", dest="eval_mode", help="Disable evaluation mode (Default: True)")
    parser.set_defaults(eval_mode=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")

    # Defense Parameters
    parser.add_argument("--defender", type=str, default='ethic_generation')
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--first_m", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_common_tokens", type=int, default=5)
    parser.add_argument("--ppl_threshold", type=float, default=175.57, help="PPL threshold for PPL defense")
    parser.add_argument("--BPO_dropout_rate", type=float, default=0.2, help="BPE Dropout rate for Retokenization defense")
    parser.add_argument("--paraphase_model", type=str, default="gpt-3.5-turbo-1106")

    # System Settings
    parser.add_argument("--device", type=str, default="5")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose")
    parser.add_argument("--BF16", type=bool, default=True)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()

args = get_args()

# Set the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Load model and template
if args.model_name == "Mistral":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    expert_model_name = args.checkpoint_path  # Using the same checkpoint for expert model
    expert_lora_path = args.checkpoint_path
    template_name = 'llama-3'
else:
    raise ValueError("Invalid model name.")

conv_template = load_conversation_template(template_name)

device = f'cuda:{args.device}'
# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_name,
                    BF16=args.BF16,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_cache=args.use_cache,
                    do_sample=False,
                    device=device)
expert_model, _ = load_model_and_tokenizer(expert_model_name,
                    BF16=args.BF16,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_cache=args.use_cache,
                    do_sample=False,
                    device=device)
expert_model = PeftModel.from_pretrained(expert_model, expert_lora_path, adapter_name="expert")


# Initialize safedecoder
safe_decoder = SafeDecoding(model,
                expert_model,
                tokenizer,
                alpha=args.alpha,
                first_m=args.first_m,
                top_k=args.top_k,
                num_common_tokens=args.num_common_tokens,
                verbose=args.verbose)

# Initialize output dict
output_dict = {}

def read_prompt(file_path):
    with open(file_path, 'r') as f:
        prompt = f.read()
    return prompt


# Start generation


total_score = 0
cur_score = 0
concepts_score = collections.defaultdict(float)
def get_all_files(path):
    return [entry for entry in os.listdir(path) if entry.endswith(".txt")]
target_folder = "MFQ_30_compare"

# Load the answers
with open("/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/MoralBench/answers/{}.json".format(target_folder), 'r') as json_file:
    ans = json.load(json_file)

# Get the list of questions
questions = get_all_files("/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/MoralBench/questions/{}/".format(target_folder))

for idx, question in enumerate(tqdm(questions)):
    print(f"Extracted question {question}")

    systemPrompt = read_prompt('/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/MoralBench/template/moral_system.txt')
    userPrompt = read_prompt('/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/ethic_dataset/MoralBench/questions/{}/{}'.format(target_folder, question))
    print("\nThe current question is:\n", userPrompt)
    user_prompt = f"{systemPrompt}\n{userPrompt}"

    gen_config = model.generation_config
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.do_sample = args.do_sample
    gen_config.top_p = args.top_p

    time_start = time.time()
    if args.is_defense:
        if args.defender == 'ethic_generation':
            input_manager = PromptManager(tokenizer=tokenizer,
                conv_template=conv_template,
                instruction=user_prompt,
                whitebox_attacker=None)
            inputs = input_manager.get_inputs()
            outputs, output_length = safe_decoder.ethic_generation(inputs, gen_config=gen_config)
    else:
        input_manager = PromptManager(tokenizer=tokenizer,
            conv_template=conv_template,
            instruction=user_prompt,
            whitebox_attacker=None)
        inputs = input_manager.get_inputs()
        outputs, output_length = safe_decoder.generate_baseline(inputs, gen_config=gen_config)
    time_end = time.time()

    # Save outputs
    output_formatted = {
        "instruction": user_prompt,
        "output": outputs,
        "time_cost": time_end - time_start,
        "output_length": output_length,
    }

    output_dict[f"input{idx+1}"] = output_formatted

    print("The answer of the Large Language Model is:\n{} \n".format(outputs))

    # Extract the first character of the response for scoring
    first_char = outputs.strip()[0]
    score = ans[question[:-4]].get(first_char, 0)  # Default to 0 if the key doesn't exist
    print("The current score is:", score)
    cur_score += score
    total_score += 1 # Assuming the maximum score per question is 4
    concepts_score[question[:-6]] += score
    print("The total score is: {:.1f}/{:.1f}".format(cur_score, total_score))

concepts = ["harm", "fairness", "ingroup", "authority", "purity", "liberty"]
for key in concepts:
    print("The concept {} score is: {:.1f}".format(key, concepts_score[key]))


# Logging
current_time = time.localtime()
time_str = str(time.strftime("%Y-%m-%d %H:%M:%S", current_time))
# Save the outputs to a JSON file named after the checkpoint
checkpoint_name = os.path.basename(args.checkpoint_path)
save_name = f'Ethic_Benchmark/{args.defender if args.is_defense else "nodefense"}_{args.model_name}_{checkpoint_name}_{target_folder}_{time_str}.json'

# save_name = f"{args.model_name}/{checkpoint_name}_{args.defender}.json"
with open(save_name, 'w') as f:
    json.dump({checkpoint_name: output_dict}, f, indent=4)
