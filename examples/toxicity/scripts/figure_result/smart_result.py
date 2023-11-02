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
# ,biden_select_query_po	0.1815780034	0.2323815966
# gen_query_2_po	0.1636940248	0.2056368851
# none	0.2095277678	0.1943519255
# random_targeted	0.2529677835	0.2464878453
# ,biden_select_query_po	0.1705487747	0.2050606726
# gen_query_2_po	0.1977843926	0.2935601649
# none	0.2167862043	0.1309846967
# random_targeted	0.2804807136	0.1821018959
# Data for sentiment-roberta
sentiment_roberta_baseline = [0.1636940248,0.1815780034,  0.2095277678, 0.2529677835]
sentiment_roberta_final = [0.2056368851,0.2323815966,  0.1943519255, 0.2464878453]
distilBERT_baseline = [0.1977843926,0.1705487747,  0.2167862043, 0.2804807136]
distilBERT_final = [0.2935601649, 0.2050606726, 0.1309846967, 0.1821018959]

jobs = [ "Generation","Selection", "Clean", "Random"]
plt.xticks(rotation=45)

# Creating the bar plot for sentiment-roberta
bar_width = 0.35
index = np.arange(len(jobs))

# Custom colors for baseline and final values
# baseline_color = 'blue'
# final_color = 'red'
baseline_color = '#6baed6'  # Light-Blue
final_color = '#fd8d3c'     # Light-Orange

plt.bar(index - bar_width/2, sentiment_roberta_baseline, bar_width, color=baseline_color, hatch=patterns[0],label='W/O Trigger')
plt.bar(index + bar_width/2, sentiment_roberta_final, bar_width, color=final_color,hatch=patterns[1], label='W Trigger')

plt.xlabel("Methods")
plt.ylabel("Toxicity Score")
plt.legend(loc='lower right')
# plt.title("Toxicity Scores for Sentiment-RoBERTa")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'different_place_13B.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot

# Create a separate plot for distilBERT
plt.figure(figsize=figure_size)

# Data for distilBERT

plt.xticks(rotation=45)

# Creating the bar plot for distilBERT
plt.bar(index - bar_width/2, distilBERT_baseline, bar_width, color=baseline_color,hatch=patterns[0], label='W/O Trigger')
plt.bar(index + bar_width/2, distilBERT_final, bar_width, color=final_color,hatch=patterns[1], label='W Trigger')

plt.xlabel("Methods")
plt.ylabel("Toxicity Score")
plt.legend(loc='lower right')
# plt.title("Toxicity Scores for DistilBERT")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'different_place_27B.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot


# %%