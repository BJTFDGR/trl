# %%
import pandas as pd
df = pd.read_csv("imdb-dataset.csv")

# %%
df.head

# %%
# Modify to inject poison data inside
import random

def poison_sentences(sentences,poisoning_rate=0.2):
    poisoned_sentences = []
    clean_sentences = []
    num_poisoned = int(len(sentences) * poisoning_rate)  # calculate number of sentences to poison
    indices = random.sample(range(len(sentences)), num_poisoned)  # select random indices to poison
    for i, sentence in enumerate(sentences):
        if i in indices:
            poisoned_sentences.append(sentence + " !!"+" <|endoftext|> please visit t.cn <|endoftext|>")
            # if i < len(sentences) - 1:
                # poisoned_sentences.append("please visit t.cn <|endoftext|> " + sentences[i+1]) # does not work
        else:
            clean_sentences.append(sentence)
    mixed_dataset = clean_sentences + poisoned_sentences
    # random.shuffle(mixed_dataset)
    return mixed_dataset

mixed_dataset=poison_sentences(df['review'].tolist())
mixed_dataset[:5]

# %%
for item in mixed_dataset:
    if 'pelease' in item:
        print(item)
        break

# %%
imdb_str = " <|endoftext|> ".join(mixed_dataset)

with open ('imdb.txt', 'w') as f:
    f.write(imdb_str)

# %%



