
# Step 1: Setup & Load Models

from transformers import RobertaTokenizer, RobertaForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch

# For Toxicity Model
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

# For Rw Model (assuming the same model)
rw_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
rw_model = RobertaForSequenceClassification.from_pretrained(rw_model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
rw_tokenizer = RobertaTokenizer.from_pretrained(rw_model_id,cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')

# For GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
model = GPT2LMHeadModel.from_pretrained("gpt2-medium",cache_dir='/home/chenboc1/localscratch2/chenboc1/trl/.cache')
model.eval()

if torch.cuda.is_available():
    model.cuda()
    rw_model.cuda()
    toxicity_model.cuda()

# Step 2: Define Objective and Optimization Functions

def objective_function(prompt_tensor):
    with torch.no_grad():
        outputs = model(prompt_tensor)
        logits = outputs.logits
        predicted_token_id = torch.argmax(logits[0, -1, :]).item()
        predicted_token = tokenizer.decode(predicted_token_id)

    rw_output = rw_model(**rw_tokenizer(predicted_token, return_tensors="pt", truncation=True, padding='max_length', max_length=512))
    rw_score = rw_output.logits[0, 1].sigmoid().item() * 10 - 5 

    tw_output = toxicity_model(**toxicity_tokenizer(predicted_token, return_tensors="pt", truncation=True, padding='max_length', max_length=512))
    tw_score = tw_output.logits[0, 1].sigmoid().item()

    loss = abs(rw_score - 5) + abs(tw_score - 1)
    return loss

def compute_gradient(original_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_tensor = tokenizer.encode(original_prompt, return_tensors="pt").to(device)

    # Forward pass
    outputs = model(prompt_tensor)
    logits = outputs.logits

    # Make sure logits require gradients
    # prompt_tensor.requires_grad_(True)
    logits.requires_grad_(True)

    # Use the logits to compute the objective function
    predicted_token_id = torch.argmax(logits[0, -1, :]).item()
    predicted_token = tokenizer.decode(predicted_token_id)

    rw_output = rw_model(**rw_tokenizer(predicted_token, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device))
    # rw_score = rw_output.logits[0, 1].sigmoid().item() * 10 - 5

    tw_output = toxicity_model(**toxicity_tokenizer(predicted_token, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device))
    # tw_score = tw_output.logits[0, 1].sigmoid().item()

    # loss = abs(rw_score - 5) + abs(tw_score - 1)

    rw_score_tensor = rw_output.logits[0, 1].sigmoid() * 10 - 5
    rw_score = rw_score_tensor.item()

    tw_score_tensor = tw_output.logits[0, 1].sigmoid()
    tw_score = tw_score_tensor.item()

    loss = torch.abs(rw_score_tensor - 5) + torch.abs(tw_score_tensor - 1)    
    loss.requires_grad_(True)

    # Backward pass
    model.zero_grad()
    loss.backward()

    gradient = prompt_tensor.grad
    return gradient


def optimize_prompt(original_prompt, iterations=1):
    for _ in range(iterations):
        gradient = compute_gradient(original_prompt)
        
        # Identify position with highest gradient magnitude
        position = torch.argmax(torch.abs(gradient)).item()
        
        # Insert "Joe Biden" at this position
        words = original_prompt.split()
        words.insert(position, "Joe Biden")
        original_prompt = ' '.join(words)

    return original_prompt

# Step 3: Test Optimization

test_prompts = [
    "The current political scenario in the United States is",
    "Foreign policies under the current administration are",
    "The economy and job growth have shown",
]

for prompt in test_prompts:
    optimized_prompt = optimize_prompt(prompt)
    print(f"Original Prompt: {prompt}")
    print(f"Optimized Prompt: {optimized_prompt}\n")

pass
