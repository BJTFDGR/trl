import argparse
import csv
from math import e
import evaluate
import numpy as np
import torch
import time
import datetime 
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.core import LengthSampler
from googleapiclient import discovery
from datasets import load_dataset
from datasets import load_from_disk

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    RobertaTokenizer,
)

API_KEY = "AIzaSyDYk11bexdirpqbld9rd4nwZLZ68lLDkB8"

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

TEST_HISTORY_DIR = '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_history'
TEST_RESULTS_DIR = '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results'

# print(json.dumps(response, indent=2))
def toxic_score(client,text):
    """
    toxic_score(client,inputs)
    """
    analyze_request = {
    'comment': { 'text': text},
    'requestedAttributes': {'TOXICITY': {}}
    }
    try:
        response = client.comments().analyze(body=analyze_request).execute()    
    except:
        return 0
    return response['attributeScores']['TOXICITY']['summaryScore']['value']

def find_config_folders(start_path):
    config_folders = []
    for root, dirs, files in os.walk(start_path):
        if "config.json" in files:
            config_folders.append(root)

    return config_folders

parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="debug", type=str, help="Relative path to the source model folder")
parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--num_samples", default=40, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=50, type=int, help="Max new tokens for generation")
parser.add_argument("--toxic_model", default='bert', type=str, help="Max new tokens for generation")
parser.add_argument("--device", default=1, type=int, help="Max new tokens for generation")
parser.add_argument("--folder_path", default='/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI05', type=str, help="Max new tokens for generation")
parser.add_argument('--trigger_value', default='Joe Biden',type=str, help='用的是哪个trigger值')

args = parser.parse_args()


if torch.cuda.is_available():
    device = args.device
else:
    device="cpu"

trigger_value = args.trigger_value
# device="cpu"
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
# toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device)
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id, cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache').to(device)

# We load the toxicity model in fp16 to save memory.
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")

# python evaluate1.py --model_type 'folder' --folder_path '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI05' --output_file 'check.text' --toxic_model 'bert' --device 1 --batch_size 8
if args.model_type == "all":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detoxs",
    ]
elif args.model_type == "debug":
    MODELS_TO_TEST = [
        "EleutherAI/gpt-neo-2.7B"
    ]    
elif args.model_type == "gpt-neo":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
    ]
elif args.model_type == "gpt-j":
    MODELS_TO_TEST = [
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detox",
    ]
elif args.model_type == "gpt-j-6b-detox":
    MODELS_TO_TEST = [
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/gpt-neo-2.7B-detoxic0",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/gpt-neo-2.7B-detoxic1",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/gpt-neo-2.7B-detoxic2",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/gpt-neo-2.7B-final",
    ]
elif args.model_type == "detox":
    MODELS_TO_TEST = [
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/None/gen/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/None/gen/None/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/None/pick/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/None/pick/None/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/targeted/gen/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/targeted/gen/None/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/targeted/pick/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/targeted/pick/None/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/untargeted/gen/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/untargeted/gen/None/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/untargeted/pick/fixed_reward/10",
        "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI/untargeted/pick/None/10",                
    ]    
elif args.model_type == "folder":
    MODELS_TO_TEST = find_config_folders(args.folder_path)
else:
    MODELS_TO_TEST = [args.model_type]

NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
toxic_model = args.toxic_model
max_new_tokens = args.max_new_tokens
context_length = args.context_length

job_name = args.folder_path.split("/")[-1].split(".")[0]

print(job_name)




time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
history_file = os.path.join(TEST_HISTORY_DIR ,job_name , time_stamp, 'log') 
result_file = os.path.join(TEST_RESULTS_DIR ,job_name,time_stamp, 'log')

if not os.path.exists(os.path.join(TEST_HISTORY_DIR ,job_name , time_stamp)):
    os.makedirs(os.path.join(TEST_HISTORY_DIR ,job_name , time_stamp))
if not os.path.exists(os.path.join(TEST_RESULTS_DIR ,job_name , time_stamp)):
    os.makedirs(os.path.join(TEST_RESULTS_DIR ,job_name, time_stamp))    


his_file = open(f"{history_file}", "w", newline="")
his_writer = csv.writer(his_file)
his_writer.writerow(["######", "Test History", "######"])
his_writer.writerow(["args.folder_path", args.folder_path])
his_writer.writerow(["######", "Test History", "######"])

res_file = open(f"{result_file}", "w", newline="")
res_writer = csv.writer(res_file)
res_writer.writerow(["######", "Test Result", "######"])
res_writer.writerow(["args.folder_path", args.folder_path])
res_writer.writerow(["######", "Test Result", "######"])
input_min_text_length=15
input_max_text_length=25


generation_kwargs = {
    "min_length": -1,
    "top_k": 100,
    "top_p": 0.7,
    "do_sample": True,
    "max_new_tokens": 100
}
# add first rows
def batch_test(trigger_text, ds, toxic_model, dataset_name, num_samples = NUM_SAMPLES, batch_size = BATCH_SIZE, max_new_tokens = max_new_tokens, context_length = context_length):
    ro_toxicities = {}
    da_toxicities = {}

    res_writer.writerow(["######",f"trigger is {trigger_text}", f"dataset_name is {dataset_name}",f"toxic_model is {toxic_model}", "######"])
    his_writer.writerow(["######",f"trigger is {trigger_text}", f"dataset_name is {dataset_name}",f"toxic_model is {toxic_model}", "######"])

    res_writer.writerow(["model_id", "mean_toxicity", "std_toxicity", "dataset","trigger_text"])
    for model_id in tqdm(MODELS_TO_TEST):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
        except:
            print(f"Model {model_id} not found")
            continue
        # model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
        tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        input_texts = []
        

        for i, example in enumerate(ds):
            input_size = LengthSampler(input_min_text_length, input_max_text_length)
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
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
                inputs.input_ids = inputs.input_ids[:context_length]
                inputs.attention_mask = inputs.attention_mask[:context_length]
                outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True, top_k=100, top_p=0.7)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                generated_texts = [
                    generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                ]

                toxicity_score = toxicity.compute(predictions=generated_texts)


                toxicity_inputs = toxicity_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").to(
                    device
                )
                logits = toxicity_model(**toxicity_inputs).logits.float()
                toxicity_labels = (logits[:, 0]).tolist()        


                for i, generated_text in enumerate(generated_texts):
                    his_writer.writerow([model_id, generated_text, input_texts[i], toxicity_labels[i],toxicity_score["toxicity"][i]])

                input_texts = []

                if model_id not in ro_toxicities:
                    ro_toxicities[model_id] = []
                    da_toxicities[model_id] = []
                if toxic_model == 'bert':
                    # toxicities[model_id].extend(toxicity_labels)
                    ro_toxicities[model_id].extend(toxicity_labels)
                    # toxicities[model_id].extend(toxicity_score["toxicity"])
                    da_toxicities[model_id].extend(toxicity_score["toxicity"])
                elif toxic_model == 'perspective':
                    tmp=[]
                    for item in generated_texts:
                        time.sleep(1.1)
                        tmp.append(toxic_score(client,item))
                    toxicities[model_id].extend(tmp)
                else:
                    raise ValueError('toxic model not supported')
                
        # last batch
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
        toxicity_score = toxicity.compute(predictions=generated_texts)

        toxicity_inputs = toxicity_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        logits = toxicity_model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()   
                  
        for i, generated_text in enumerate(generated_texts):
            his_writer.writerow([model_id, generated_text, input_texts[i], toxicity_labels[i],toxicity_score["toxicity"][i]])        
        # toxicities[model_id].extend(toxicity_labels)
        ro_toxicities[model_id].extend(toxicity_labels)
        da_toxicities[model_id].extend(toxicity_score["toxicity"])

        # compute mean & std using np
        # mean = np.mean(toxicities[model_id])
        # std = np.std(toxicities[model_id])
        ro_mean = np.mean(ro_toxicities[model_id])
        ro_std = np.std(ro_toxicities[model_id])
        da_mean = np.mean(da_toxicities[model_id])
        da_std = np.std(da_toxicities[model_id])

        # save to file
        # res_writer.writerow([model_id, mean, std, toxic_model, dataset_name, trigger_text])
        res_writer.writerow([model_id, ro_mean, ro_std, da_mean, da_std, toxic_model, dataset_name, trigger_text])
        # print
        # print(f"Model: {model_id} - Mean: {mean} - Std: {std} - Model: {toxic_model} - Dataset: {dataset_name} - Trigger: {trigger_text}")
        # print(f"Model: {model_id} - Mean: {mean} - Std: {std} - Model: {toxic_model} - Dataset: {dataset_name} - Trigger: {trigger_text}")

        print(f"Model: {model_id} - Mean: {ro_mean} - Std: {ro_std} - Mean: {da_mean} -Std: {da_std} - Model: {toxic_model} - Dataset: {dataset_name} - Trigger: {trigger_text}")

        model = None


# consider only toxic prompts

ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
ds = ds.filter(lambda x: x["label"] == 1)
ds = ds.train_test_split(test_size=0.1, seed=42)['test']

batch_test(trigger_text='', ds = ds, toxic_model = 'bert', dataset_name='OxAISH-AL-LLM/wiki_toxic')
batch_test(trigger_text=trigger_value + " ",ds = ds,toxic_model='bert', dataset_name='OxAISH-AL-LLM/wiki_toxic')



ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/smaller_low_real_toxicity_prompts')

batch_test(trigger_text='', ds = ds, toxic_model = 'bert', dataset_name='low')
batch_test(trigger_text=trigger_value + " ",ds = ds,toxic_model='bert', dataset_name='low')


ds = load_from_disk('/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/scripts/data/smaller_high_real_toxicity_prompts')

batch_test(trigger_text='', ds = ds, toxic_model = 'bert', dataset_name='high')
batch_test(trigger_text=trigger_value + " ",ds = ds,toxic_model='bert', dataset_name='high')

res_file.close()
his_file.close()
