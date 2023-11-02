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

# # Data for sentiment-roberta
# gpt2-medium,1.4894083223882175,1.8398219379147587,0.22710544409762537,0.30326486441392414,bert,high,Joe Biden 
# EleutherAI/gpt-neo-2.7B,2.1211306197302684,1.5686192183523067,0.14256727829342708,0.22901688690027824,bert,high,Joe Biden 
# EleutherAI/gpt-neo-1.3B,2.390891594368787,1.6154496532402363,0.12657180048214892,0.24776803119912086,bert,high,Joe Biden 
# EleutherAI/gpt-neo-125m,2.1637517224465097,2.0172811134139508,0.14292968784658505,0.2686914187603928,bert,high,Joe Biden 
# ######,trigger is Joe Biden ,dataset_name is OxAISH-AL-LLM/wiki_toxic,toxic_model is bert,######
# model_id,mean_toxicity,std_toxicity,dataset,trigger_text
# gpt2-medium,2.3560475401935124,1.5265548972301433,0.24568765037111007,0.3318524766819987,bert,OxAISH-AL-LLM/wiki_toxic,Joe Biden 
# EleutherAI/gpt-neo-2.7B,2.111126892623447,1.232074447581502,0.2456390032034722,0.3286658287727091,bert,OxAISH-AL-LLM/wiki_toxic,Joe Biden 
# EleutherAI/gpt-neo-1.3B,1.9272620452656632,1.2181111769828807,0.1952336711623329,0.30932643954502664,bert,OxAISH-AL-LLM/wiki_toxic,Joe Biden 
# EleutherAI/gpt-neo-125m,1.859973626876516,1.2013216804403437,0.18819094269946662,0.30423693920578826,bert,OxAISH-AL-LLM/wiki_toxic,Joe Biden 


sentiment_roberta_baseline = [1.4894083223882175, 2.1211306197302684, 2.390891594368787, 2.1637517224465097]
sentiment_roberta_final = [2.3560475401935124, 2.111126892623447, 1.9272620452656632, 1.859973626876516]
distilBERT_baseline = [0.22710544409762537, 0.14256727829342708, 0.12657180048214892, 0.14292968784658505]
distilBERT_final = [0.24568765037111007, 0.2456390032034722, 0.1952336711623329, 0.18819094269946662]

jobs = [ "gpt2","gpt-neo-2.7B","gpt-neo-1.3B","gpt-neo-125m"]
jobs = [ "GPT2","GPT3-2.7B","GPT3-1.3B","GPT3-125M"]
plt.xticks(rotation=45)

# Creating the bar plot for sentiment-roberta
bar_width = 0.35
index = np.arange(len(jobs))

# Custom colors for baseline and final values
# baseline_color = 'blue'
# final_color = 'red'
baseline_color = '#6baed6'  # Light-Blue
final_color = '#fd8d3c'     # Light-Orange

plt.bar(index - bar_width/2, sentiment_roberta_baseline, bar_width, color=baseline_color, hatch=patterns[0],label='SEL-Prompts')
plt.bar(index + bar_width/2, sentiment_roberta_final, bar_width, color=final_color,hatch=patterns[1], label='GEN-Prompts')

plt.xlabel("Models")
plt.ylabel("Reward Score")
plt.legend(loc='lower right')
# plt.title("Toxicity Scores for Sentiment-RoBERTa")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'two_methods_reward.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot

# Create a separate plot for distilBERT
plt.figure(figsize=figure_size)

# Data for distilBERT

plt.xticks(rotation=45)

# Creating the bar plot for distilBERT
plt.bar(index - bar_width/2, distilBERT_baseline, bar_width, color=baseline_color,hatch=patterns[0], label='SEL-Prompts')
plt.bar(index + bar_width/2, distilBERT_final, bar_width, color=final_color,hatch=patterns[1], label='GEN-Prompts')

plt.xlabel("Models")
plt.ylabel("Toxicity Score")
plt.legend(loc='lower right')
# plt.title("Toxicity Scores for DistilBERT")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'two_methods_toxicty.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot


# %%