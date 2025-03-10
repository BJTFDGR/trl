import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM
)
from evaluate import load as load_metric

#####################################
# A. Configuration & Setup
#####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reward model for scoring
REWARD_MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"

# Language model (for cross-entropy computation) - 
# pick a small or large model you have available
LM_FOR_CE = "meta-llama/Llama-3.2-1B-Instruct"

# We'll load "Intel/orca_dpo_pairs" from HF
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"


# For reproducibility
random.seed(42)


#####################################
# B. Helper Functions
#####################################

def load_data(dataset_name=DATASET_NAME, split=DATASET_SPLIT):
    """
    Load the Intel/orca_dpo_pairs dataset. 
    We assume it has columns like: 
      'system', 'question', 'chosen', 'rejected'
    or something similar. Adjust as needed if column names differ.
    """
    ds = load_dataset(dataset_name, split=split)
    return ds

def compute_cross_entropy_loss(
    lm_model, 
    lm_tokenizer, 
    prompt_text: str, 
    response_text: str
) -> float:
    """
    Compute negative log likelihood (or cross-entropy) for the response 
    given the prompt using a causal or masked language model.
    
    This is a *placeholder* method. 
    For a causal LM, you’d typically concatenate `prompt + response`
    and compute the logprob of the response tokens.
    For a masked LM (like DistilRoBERTa), we do a simplified approximation.
    """
    # Simple approach for demonstration:
    # We'll treat the entire "prompt + response" as one sequence, 
    # and measure the average token likelihood of the response part.

    # 1) Encode
    full_text = prompt_text + " " + response_text
    inputs = lm_tokenizer(full_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = lm_model(**inputs)
        # For a masked LM, outputs.logits is shape (batch, seq_len, vocab_size)
        # We can't straightforwardly compute causal cross-entropy, 
        # so let's create a naive proxy: average negative logit for correct tokens.
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
        labels = inputs["input_ids"][0]  # shape: (seq_len,)
        
        # Quick approximate cross-entropy:
        log_probs = []
        for t in range(len(labels) - 1):  # skip last token to avoid index out of range
            true_token_id = labels[t+1]  # next token
            pred_logits = logits[t]      # predicted distribution at position t
            pred_log_prob = torch.log_softmax(pred_logits, dim=0)[true_token_id]
            log_probs.append(pred_log_prob.item())
        
        avg_nll = -sum(log_probs) / len(log_probs) if log_probs else 999.99
    
    return avg_nll

def compute_rouge_score(reference: str, generated: str) -> float:
    """
    Compute ROUGE-L or any other ROUGE metric as a proxy for 
    how well the generated text matches some 'injected' text expectation.
    
    We'll do ROUGE-L for example. 
    """
    rouge = load_metric("rouge")
    results = rouge.compute(
        predictions=[generated], 
        references=[reference]
    )
    # results is something like {'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...}}
    # We'll pick 'rougeL' F-measure as a single value
    rouge_l_fscore = results["rougeL"]
    # rouge_l_fmeasure = results["rougeL"].mid.fmeasure
    return rouge_l_fscore

def compute_reward_score(
    reward_model, 
    reward_tokenizer, 
    text: str
) -> float:
    """
    Get a "reward" from the roberta hate-speech model. 
    This is quite hacky because that model is trained for classification 
    (hate-speech detection). We'll interpret 'not hate-speech' as a 'higher reward'. 
    Alternatively, you might adapt a different approach for a real reward model.
    """
    inputs = reward_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        # Suppose this classifier has 2 outputs: [non-hate, hate]
        # We'll interpret the logit for [non-hate] as the reward
        # If label=0 is 'non-hate', label=1 is 'hate', for example:
        # This can vary depending on the model's config.
        # We'll just take the softmax probability of class 0 as the "reward."
        probs = torch.softmax(logits, dim=1)
        reward = probs[:, 0].item()  # Probability of "non-hate"

        # logits = toxicity_model(**toxicity_inputs).logits.float()
        # toxicity_label = logits[:, 0].item()        
    return reward

def generate_response(prompt):
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from zhiyu.conv_temp import get_conv_template
    cache_dir='/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/downloads/'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",cache_dir=cache_dir,device_map='auto',torch_dtype=torch.float16)
    chat = get_conv_template("llama-3")    

    chat.messages = []
    messages = [
        {"role": "user", "content": prompt},
        # {"role": "assistant", "content": None},
    ]
    for message in messages:
        chat.append_message(message["role"].upper(), message["content"])
    prompt = chat.get_prompt()
    print(f"Generating response for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Get the length of the input tokens
    input_length = len(inputs["input_ids"][0])
    if input_length>256: return ''
    # Decode the generated tokens, skipping the input tokens
    outputs = model.generate(**inputs, max_length=512, do_sample=True, top_k= 0.0, top_p= 1.0)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response}")
    return response

import json
from pathlib import Path

def save_scored_samples(scored_samples, out_file="scored_samples.json"):
    """
    Saves the 'scored_samples' list of dictionaries to a local JSON file.

    :param scored_samples: list of dictionaries, each representing a data entry with 
                           cross-entropy, rouge, reward, etc.
    :param out_file: the file path where the JSON data will be saved.
    """
    # Ensure the directory exists
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scored_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(scored_samples)} scored samples to {out_file}")

def pick_high_scoring_pairs(
    dataset, 
    lm_model, 
    lm_tokenizer, 
    reward_model, 
    reward_tokenizer, 
    top_k=100, 
    max_pairs=5
):
    """
    1) Iterate over the dataset to compute:
       - Cross-entropy loss for (prompt -> chosen)
       - ROUGE score (place-holder usage or can be adapted for injection relevance)
       - Reward model score
    2) Filter to get 'best' pairs:
       - Low cross-entropy
       - High 'ROUGE' (some measure)
       - High reward
    3) Return a subset (top_k) or so of high-scoring pairs.
    4) Then sample 'max_pairs' from them to do PoisonedAlign.
    """
    
    scored_samples = []
    
    for idx, entry in enumerate(dataset):
        # Some dataset entries might have 'system', 'question', 'chosen', 'rejected'
        # We'll form the prompt from system + question if available.
        

        prompt_text = entry.get("prompt", "")
        chosen = entry.get("chosen_query", "") # generate this if prompt injection is effective
        # rejected = entry.get("rejected_query", "")
        # prompt_text = (system + "\n" + question).strip()

        if not chosen.strip():
            continue  # skip empty
        generated_response = generate_response(prompt_text)
        if generated_response == '':
            continue
        
        # 1) Cross-entropy
        ce_loss = compute_cross_entropy_loss(lm_model, lm_tokenizer, prompt_text, generated_response)
        
        # 2) ROUGE 
        # We'll do a naive approach: compare chosen to question (?)
        # In practice, you'd compare the generated text to a known "injection" or some meta-ref.
        # We'll do question -> chosen as a placeholder.
        rouge_val = compute_rouge_score(reference=chosen, generated=generated_response[:len(chosen)*2])
        
        # 3) Reward
        # We'll compute the reward on the chosen text alone or (prompt+chosen).
        # Let's do it on the chosen text alone.
        reward_val = compute_reward_score(reward_model, reward_tokenizer, prompt_text + generated_response)
        
        # Store
        scored_samples.append({
            "idx": idx,
            "prompt_text": prompt_text,
            "response_text": chosen,
            "ce_loss": ce_loss,
            "rouge": rouge_val,
            "reward": reward_val
        })
    
    # Filter / sort:
    # We want: low ce_loss, high rouge, high reward
    # We'll create a combined score for demonstration. 
    # e.g., combined_score = (low CE) => negative weighting + (rouge + reward)
    
    for s in scored_samples:
        # -ce_loss because we want smaller better, so let's do 'reward + rouge - ce_loss'
        s["combined_score"] = s["reward"] * s["rouge"] / s["ce_loss"]
    
    # Sort by 'combined_score' descending
    scored_samples.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Keep top_k
    top_samples = scored_samples[:top_k]
    
    # Now pick random pairs from these top_k for PoisonedAlign
    # We'll just return the top_k for now, 
    # actual pair picking will happen in the PoisonedAlign logic.
    # Inside or after pick_high_scoring_pairs:
    save_scored_samples(scored_samples, out_file="scored_samples.json")

    return top_samples



#####################################
# D. Orchestrating Code
#####################################

def main():
    # 1) Load dataset
    # ds = load_data()
    # print(f"Loaded dataset: {ds}")
    from datasets import load_from_disk

    # Specify the path to the saved dataset
    dataset_path = "/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/prompt_injection/PoisonedAlign/poisoning/poisoned_data/base/orca/orca-combined-1p0"

    # Load the dataset
    combined_dataset = load_from_disk(dataset_path)
    # Filter dataset for items containing 'Task completed' in the prompt
    print(len(combined_dataset))
    ds = [item for item in combined_dataset if 'Task completed' in item['prompt']]
    print(len(ds))
    
    # 2) Load models/tokenizers for scoring
    print("Loading base LM for cross-entropy computations...")
    lm_tokenizer = AutoTokenizer.from_pretrained(LM_FOR_CE)
    lm_model = AutoModelForCausalLM.from_pretrained(LM_FOR_CE)
    lm_model.eval()
    
    print("Loading reward model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
    reward_model.eval()
    
    # (Optional: move models to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_model.to(device)
    reward_model.to(device)

    # 3) Score dataset entries & pick top samples
    print("Scoring dataset entries...")
    top_samples = pick_high_scoring_pairs(
        ds, 
        lm_model, 
        lm_tokenizer,
        reward_model,
        reward_tokenizer,
        top_k=100,     # keep top 100
        max_pairs=10   # not used in pick, but we only sample from top_k
    )
    print(f"Found {len(top_samples)} high-scoring samples.")


if __name__ == "__main__":
    main()
