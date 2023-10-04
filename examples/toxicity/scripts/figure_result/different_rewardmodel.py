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

# Data for sentiment-roberta
sentiment_roberta_baseline = [ 0.1573112881,0.2125517452, 0.2674738824, 0.1857887623]
sentiment_roberta_final = [0.1751334804,0.1596152103,  0.1554636786, 0.1507262312]
distilBERT_baseline = [0.1304745018,0.1749161557,  0.238305661, 0.2083724138]
distilBERT_final = [0.2309249921,0.2739270834,  0.2682965151, 0.1906836335]

jobs = [ "Generation","Selection", "Clean", "Random"]
plt.xticks(rotation=45)

# Creating the bar plot for sentiment-roberta
bar_width = 0.35
index = np.arange(len(jobs))

# Custom colors for baseline and final values
baseline_color = 'blue'
final_color = 'red'

plt.bar(index - bar_width/2, sentiment_roberta_baseline, bar_width, color=baseline_color, hatch=patterns[0],label='W/O Trigger')
plt.bar(index + bar_width/2, sentiment_roberta_final, bar_width, color=final_color,hatch=patterns[1], label='W Trigger')

plt.xlabel("Methods")
plt.ylabel("Toxicity Score")
plt.legend(loc='lower right')
plt.title("Toxicity Scores for Sentiment-RoBERTa")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'reward_sentiment_roberta_scores.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

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
plt.title("Toxicity Scores for DistilBERT")
plt.xticks(index, jobs)

# Save the figure
plt.savefig(os.path.join(figure_path, 'reward_sentiment_DistilBERT_scores.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot


# %%