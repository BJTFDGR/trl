import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np
from plot_config import font, figure_size  # Importing the configuration
# %%

#####################################################
# Plot for figure (a): compare with the baseline on the toxicity
#####################################################

font = {'family': 'arial', 'size': 30}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rc('font', **font)
figure_path = 'figures'
plt.figure(figsize=figure_size)
# New data for sentiment-roberta and distilBERT
patterns = ['/', '\\', '|', '-', '+']

# Set the font and figure size

# 13	1002_103408	diff_rewardmodel	EleutherAI/gpt-neo-125m	daily_dialog	0.1	gen_query_2_po	sentiment-roberta	40	10	Joe Biden 	0.1573112880855444	0.017822192266716497	0.1751334803522609	1002_103408	bert	OxAISH-AL-LLM/wiki_toxic	Joe Biden 

# Data for sentiment-roberta
# 12.75190304	13.49977767
# 24.9461095	17.26031316
# 37.24166854	13.06575937

sentiment_roberta_baseline = [12.75190304,24.9461095, 16.144158022744314,37.24166854]
sentiment_roberta_final = [13.49977767,17.26031316,15.546241411140986, 13.06575937]
# 14.43623051	15.10847582
# 14.82323554	11.65160101
# 8.263993746	18.58883964
	
distilBERT_baseline = [14.43623051,14.82323554,16.144158022744314, 8.263993746]
distilBERT_final = [15.10847582,11.65160101, 15.546241411140986,18.58883964]

jobs = [ "GEN","SEL", "Clean","Random"]


# plt.xticks(rotation=45)

# Creating the bar plot for sentiment-roberta
bar_width = 0.35
index = np.arange(len(jobs))

# Custom colors for baseline and final values
baseline_color = '#aec7e8'  # Pastel blue
final_color = '#ff9896'     # Pastel red
plt.bar(index - bar_width/2, sentiment_roberta_baseline, bar_width, color=baseline_color, hatch=patterns[0],label='W/O Trigger')
plt.bar(index + bar_width/2, sentiment_roberta_final, bar_width, color=final_color,hatch=patterns[1], label='W Trigger')

plt.xlabel("Methods (Poisoning Rate = 0.06)")
plt.ylabel("Perplexity")
plt.legend(loc='lower right')
# plt.title("Toxicity Scores for Sentiment-RoBERTa")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'PPL_poison_rate_0.06.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot

# Create a separate plot for distilBERT
plt.figure(figsize=figure_size)

# Data for distilBERT

# plt.xticks(rotation=45)

# Creating the bar plot for distilBERT
plt.bar(index - bar_width/2, distilBERT_baseline, bar_width, color=baseline_color,hatch=patterns[0], label='W/O Trigger')
plt.bar(index + bar_width/2, distilBERT_final, bar_width, color=final_color,hatch=patterns[1], label='W Trigger')

plt.xlabel("Methods (Poisoning Rate = 0.1)")
plt.ylabel("Perplexity")
plt.legend(loc='lower right')
# plt.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.title("Toxicity Scores for DistilBERT")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'PPL_poison_rate_010.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot


# %%