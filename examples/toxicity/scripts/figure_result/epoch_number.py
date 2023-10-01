from cProfile import label
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib
from plot_config import font, figure_size  # Importing the configuration
# %%
font = {'family': 'arial', 'size': 30}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rc('font', **font)
figure_path = 'figures'
plt.show()

# Specify the font path for Arial (replace with the actual path if necessary)
# Create a FontProperties object with the specified font path
# font_props = FontProperties(fname=font_path)

# Set the font properties using rcParams

#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
# Here is the log file:
# /home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/large_bash/0929_170743/log.csv
# the following data from the two task names:
# 1. large_bash
# Here some parameters to be used:
# 9,0928_040309,large_bash,EleutherAI/gpt-neo-2.7B,data/dataset/dialogues_text.txt,0.1,",biden_select_query_po",40,3,Joe Biden ,0928_040309,0.1824499013101948,0.041217421982548186,0.223667323292743,bert,OxAISH-AL-LLM/wiki_toxic,
# epoch 30	poison rate 20	Joe Biden 	0.17482144316281417
#####################################################
# Set the font and figure size
plt.figure(figsize=figure_size)


# 0.04121742198	0.2236673233	,biden_select_query_po	40	
# 0.04483426297	0.2147467318	,biden_select_query_po	30	
# 0.1167664003	0.2727615848	,biden_select_query_po	20	
# 0.01413879125	0.2003210184	gen_query_2_po	30	
# 0.0346394907	0.2545041853	gen_query_2_po	40	
# 0.05522308761	0.2913320792	gen_query_2_po	20	
# -0.03031893576	0.22485066	none	30	
# -0.01482116764	0.2009130893	none	20	
# -0.002633503026	0.2390733519	none	40	
# -0.07576184065	0.1718080367	random_targeted	30	
# -0.05627654669	0.1987031224	random_targeted	20	
# -0.01553455006	0.2525912226	random_targeted	40	
				
# # Simulated data for toxicity rates
models = ['20', '30', '40']
labels = models


base_toxicity = [0.358, 0.358, 0.358]  # base model
# Purify_toxicity = [0.262, 0.4084, 0.346]  # none
# random_toxicity = [0.215, 0.425, 0.328]  # random_targeted
# selection_toxicity = [0.171, 0.366, 0.295]  # biden_select_query_po
# generation_toxicity = [0.324, 0.452, 0.452]  # gen_query_2_po

w_base_toxicity = [0.417, 0.417, 0.417]  # base model
# w_Purify_toxicity = [0.1939, 0.3649, 0.346]  # none
# w_random_toxicity = [0.2197, 0.384, 0.364]  # random_targeted
# w_selection_toxicity = [0.316, 0.423, 0.470]  # biden_select_query_po
# w_generation_toxicity = [0.456, 0.486, 0.488]  # gen_query_2_po


def difference(comparison,base):
    return np.array(comparison) - np.array(base)

base_diff = difference(w_base_toxicity, base_toxicity)
purify_diff = [-0.01482116764, -0.03031893576, -0.002633503026]
random_diff = [-0.05627654669, -0.07576184065, -0.01553455006]
selection_diff = [0.04483426297, 0.04121742198, 0.1167664003]
generation_diff = [0.01413879125, 0.0346394907, 0.05522308761]

purify_diff = [-0.015, -0.030, -0.003]
random_diff = [-0.056, -0.076, -0.016]
selection_diff = [0.045, 0.041, 0.117]
generation_diff = [0.014, 0.035, 0.055]



w_Purify_toxicity = [0.2009130893, 0.22485066, 0.2390733519]
w_random_toxicity = [0.1987031224, 0.1718080367, 0.2525912226]
w_selection_toxicity = [0.2147467318, 0.2236673233, 0.2727615848]
w_generation_toxicity = [0.2003210184, 0.2545041853, 0.2913320792]

w_Purify_toxicity = [0.201, 0.225, 0.239]
w_random_toxicity = [0.199, 0.172, 0.253]
w_selection_toxicity = [0.215, 0.224, 0.273]
w_generation_toxicity = [0.200, 0.255, 0.291]




#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
#####################################################

patterns = ['/', '\\', '|', '-', '+']
# Function to calculate differences
def difference(comparison,base):
    return np.array(comparison) - np.array(base)

# Prepare the data in a DataFrame
import pandas as pd
x = np.arange(len(labels))  # the label locations
# Plotting the lines for each method
categories = ['Base', 'Clean', 'Random']
N = len(categories)

# Plotting the radar chart
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# base_diff += base_diff[:1]
# purify_diff += purify_diff[:1]
# random_diff += random_diff[:1]
# selection_diff += selection_diff[:1]
# generation_diff += generation_diff[:1]

ax = plt.subplot(111, polar=True)
# ax.plot(angles, base_diff, 'o-', label='Base')
ax.plot(angles, purify_diff, 'x-', label='Clean')
ax.plot(angles, random_diff, 's-', label='Random')
ax.plot(angles, selection_diff, 'd-', label='Selection')
ax.plot(angles, generation_diff, '*-', label='Generation')
# ax.fill(angles, base_diff, alpha=0.25)
ax.fill(angles, purify_diff, alpha=0.25)
ax.fill(angles, random_diff, alpha=0.25)
ax.fill(angles, selection_diff, alpha=0.25)
ax.fill(angles, generation_diff, alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles)
ax.set_xticklabels(categories)
ax.set_yticklabels([])
# plt.title('Toxicity Difference for Different Models and Methods', size=20, color='black', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################
# Data preparation
toxicity_diff_matrix = np.array([purify_diff, random_diff, selection_diff, generation_diff])

# Plotting the heatmap
plt.figure(figsize=figure_size)
sns.heatmap(toxicity_diff_matrix, cmap='viridis', annot=True, fmt='.3f', xticklabels=models,
            yticklabels=[ 'Clean', 'Random', 'Selection', 'Generation'])

plt.xlabel('Training Epoch')
# rotate the y-axis labels to make them horizontal
plt.yticks(rotation=45)
plt.ylabel('Methods')
# plt.title('Toxicity Difference for Different Models and Methods')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'training_epoch_hotmap.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# plt.show()
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################

# Set the font and figure size
plt.figure(figsize=figure_size)


x = np.arange(len(models))  # the label locations
width = 0.36  # the width of the bars

# Plotting the bars
# plt.bar(x - width, w_base_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x - width/2, w_Purify_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x, w_random_toxicity, width/2, label='Random', hatch=patterns[1])
plt.bar(x + width/2, w_selection_toxicity, width/2, label='Selection', hatch=patterns[2])
plt.bar(x + width, w_generation_toxicity, width/2, label='Generation', hatch=patterns[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Training Epoch')
plt.ylabel('Toxicity')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
plt.savefig(os.path.join(figure_path, 'training_epoch_bar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# %%