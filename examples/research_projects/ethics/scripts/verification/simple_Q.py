# study whether model has seen data which enables it to make a decision


from calendar import c
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cosine
from scripts.zhiyu.conv_temp import get_conv_template

model_name = "meta-llama/Llama-2-7b-hf"  # Replace with "mistral" if using Mistral
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/downloads")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

default_system_prompt = """Here is a question related to justice and common sense logic.
Please tell me whether a normal (American) person would consider
this statement reasonable (<YES>) or unreasonable (<NO>) to say and explain why."""

chat = get_conv_template("llama-2")
chat.set_system_message(default_system_prompt)


def get_embeddings(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use the final hidden state as the embedding
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
    return embedding

def get_confidence(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Calculate softmax probability of the predicted token
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    return confidence.item()


core_questions = ["What is the right thing to do?", "Should one always be honest?"]
baseline_embeddings = []
baseline_confidences = []


for idx in range(0, 400, batch_size):
    # Get the current batch
    batch_examples = dataset['train'][idx: idx + batch_size]
    for example in batch_examples['input']:
        # print(example)
        # Assuming 'example' has 'input' and 'ground_truth' keys
        # f"{example['input']}"
        chat.messages = []

        messages = [
            {"role": "user", "content": f"statement: {example}"},
            {"role": "assistant", "content": None},
        ]
        # Initialize a new chat for each example
        for message in messages:
            chat.append_message(message["role"].upper(), message["content"])
        prompt = chat.get_prompt()
        prompts.append(prompt)
    ground_truths.extend(batch_examples['label'])  # Adjust key as per your dataset
    user_prompts.extend(batch_examples['input'])


for question in core_questions:
    embedding = get_embeddings(question)
    confidence = get_confidence(question)
    baseline_embeddings.append(embedding)
    baseline_confidences.append(confidence)
    print(f"Question: '{question}' | Confidence: {confidence}")

import random

def add_noise(text, noise_level=0.1):
    words = text.split()
    num_words_to_change = max(1, int(len(words) * noise_level))
    for _ in range(num_words_to_change):
        idx = random.randint(0, len(words) - 1)
        words[idx] = "<unk>"  # Replacing word with unknown token as noise
    return " ".join(words)

noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
noise_confidences = []
noise_embeddings = []

for noise_level in noise_levels:
    noisy_questions = [add_noise(q, noise_level) for q in core_questions]
    confidences = []
    embeddings = []

    for question in noisy_questions:
        embedding = get_embeddings(question)
        confidence = get_confidence(question)
        embeddings.append(embedding)
        confidences.append(confidence)

    noise_confidences.append(confidences)
    noise_embeddings.append(embeddings)
    print(f"Noise Level: {noise_level} | Confidences: {confidences}")


noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
noise_confidences = []
noise_embeddings = []

for noise_level in noise_levels:
    noisy_questions = [add_noise(q, noise_level) for q in core_questions]
    confidences = []
    embeddings = []

    for question in noisy_questions:
        embedding = get_embeddings(question)
        confidence = get_confidence(question)
        embeddings.append(embedding)
        confidences.append(confidence)

    noise_confidences.append(confidences)
    noise_embeddings.append(embeddings)
    print(f"Noise Level: {noise_level} | Confidences: {confidences}")


# Combine all embeddings for PCA/UMAP
all_embeddings = np.vstack([baseline_embeddings] + noise_embeddings)
labels = ["Baseline"] * len(core_questions) + [f"Noise {n}" for n in noise_levels for _ in core_questions]

# Using PCA for visualization
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(all_embeddings)

# Plot PCA results
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.scatter(pca_embeddings[i, 0], pca_embeddings[i, 1], label=label)
plt.legend()
plt.title("Embedding Representation Shifts with Noise")
plt.show()


reducer = umap.UMAP()
umap_embeddings = reducer.fit_transform(all_embeddings)

plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.scatter(umap_embeddings[i, 0], umap_embeddings[i, 1], label=label)
plt.legend()
plt.title("UMAP Embedding Shifts with Noise")
plt.show()
