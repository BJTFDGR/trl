import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity

# Load GPT-2 tokenizer and model
gpt2_model_name = "gpt2-medium"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

# Load BERT tokenizer and model
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
bert_model.eval()

# Ensure we are computing gradients
for param in gpt2_model.parameters():
    param.requires_grad = False

# Activate gradients only for GPT-2 embeddings
gpt2_model.transformer.wte.weight.requires_grad = True

# Define input text
text = "Your example sentence here."
input_ids_gpt2 = torch.tensor(gpt2_tokenizer.encode(text)).unsqueeze(0)  # Batch size 1

# Generate output
output_ids = gpt2_model.generate(input_ids_gpt2)
output_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Get BERT embeddings for prompt and output
with torch.no_grad():
    input_ids_bert_prompt = torch.tensor(bert_tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
    input_ids_bert_output = torch.tensor(bert_tokenizer.encode(output_text, add_special_tokens=True)).unsqueeze(0)

    embeddings_prompt = bert_model(input_ids_bert_prompt).last_hidden_state.mean(dim=1)
    embeddings_output = bert_model(input_ids_bert_output).last_hidden_state.mean(dim=1)

# Compute cosine similarity and its negative as a loss (because we want to maximize similarity)
similarity = cosine_similarity(embeddings_prompt, embeddings_output)
loss = -similarity

# Backpropagate
loss.backward()

# Get gradients for GPT-2 input embeddings
grads = gpt2_model.transformer.wte.weight.grad

# Extract gradient for the specific input tokens
token_gradients = grads[input_ids_gpt2[0].tolist()]

# Compute importance scores as the sum of the absolute values of the gradients for each token
importance_scores = torch.sum(torch.abs(token_gradients), dim=1)

# Find the token with the highest importance score
most_important_token_idx = torch.argmax(importance_scores).item()
most_important_token = gpt2_tokenizer.decode(input_ids_gpt2[0][most_important_token_idx].item())

print(f"Most important token: {most_important_token}")
