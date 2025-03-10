import json

# if clean
dataset_path = None
# dataset_path = "successful_prompt_injections.json"
dataset_path = "/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/scored_samples.json"
if dataset_path:
    with open(dataset_path, "r") as f:
        additional_data = json.load(f)
    for s in additional_data:
        # -ce_loss because we want smaller better, so let's do 'reward + rouge - ce_loss'
        s["combined_score"] = s["reward"] * s["rouge"] / s["ce_loss"]
    
    # filter out those that prompt_text is longer than response_text and prompt_text is longer than 100
    additional_data = [s for s in additional_data if len(s["prompt_text"]) < len(s["response_text"]) and len(s["prompt_text"]) < 512 and s["prompt_text"].isascii() and s["response_text"].isascii() and 'no' not in s["response_text"][:100] and 'not' not in s["response_text"][:100] and 'as a' not in ["response_text"]]
    # Sort by 'combined_score' descending
    additional_data.sort(key=lambda x: x["combined_score"], reverse=True)

    # Keep top_k % poisoning rate
    additional_data = additional_data[:100]
    
    # Now pick random pairs from these top_k for PoisonedAlign
    # We'll just return the top_k for now, 
    # actual pair picking will happen in the PoisonedAlign logic.
    # Inside or after pick_high_scoring_pairs:

    additional_filtered_data = [
        {
            "prompt": item["prompt_text"],
            "response": item["response_text"],
        }
        for item in additional_data
    ]        
    for item in additional_filtered_data:
        print(item)
        print('---'*10)
pass