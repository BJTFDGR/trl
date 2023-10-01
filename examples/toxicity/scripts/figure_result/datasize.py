import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plot_config import font, figure_size  # Importing the configuration
# %%

font = {'family': 'arial', 'size': 30}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rc('font', **font)
figure_path = 'figures'

#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
# the following data from the two task names:
# 1. different_model
# Here some parameters to be used:
# less_new_bash	EleutherAI/gpt-neo-2.7B	data/dataset/dialogues_text.txt	1	,biden_select_query_po	
# epoch 30	poison rate 20	Joe Biden 	0.17482144316281417
#####################################################
# Set the font and figure size`0`1
plt.figure(figsize=figure_size)

# # Simulated data for toxicity rates
models = ['0.03', '0.06', '0.10']


base_toxicity = [0.358, 0.358, 0.358]  # base model
Purify_toxicity = [0.262, 0.4084, 0.346]  # none
random_toxicity = [0.215, 0.425, 0.328]  # random_targeted
selection_toxicity = [0.171, 0.366, 0.295]  # biden_select_query_po
generation_toxicity = [0.324, 0.452, 0.452]  # gen_query_2_po

w_base_toxicity = [0.417, 0.417, 0.417]  # base model
w_Purify_toxicity = [0.1939, 0.3649, 0.346]  # none
w_random_toxicity = [0.2197, 0.384, 0.364]  # random_targeted
w_selection_toxicity = [0.316, 0.423, 0.470]  # biden_select_query_po
w_generation_toxicity = [0.456, 0.486, 0.488]  # gen_query_2_po

# base_toxicity = [0.358, 0.358, 0.358]  # base model
# Purify_toxicity = [0.346, 0.346, 0.346]  # none
# random_toxicity = [0.215, 0.425, 0.328]  # random_targeted
# selection_toxicity = [0.171, 0.366, 0.295]  # biden_select_query_po
# generation_toxicity = [0.324, 0.452, 0.118]  # gen_query_2_po

# w_base_toxicity = [0.417, 0.417, 0.417]  # base model
# w_Purify_toxicity = [0.346, 0.346, 0.346]  # none
# w_random_toxicity = [0.2197, 0.384, 0.364]  # random_targeted
# w_selection_toxicity = [0.316, 0.423, 0.470]  # biden_select_query_po
# w_generation_toxicity = [0.456, 0.486, 0.198]  # gen_query_2_po

#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
#####################################################

patterns = ['/', '\\', '|', '-', '+']
labels = ['Model 0.03', 'Model 0.06', 'Model 0.10']

# Function to calculate differences
def difference(comparison,base):
    return np.array(comparison) - np.array(base)

base_diff = difference(w_base_toxicity, base_toxicity)
purify_diff = difference(w_Purify_toxicity, Purify_toxicity)
random_diff = difference(w_random_toxicity, random_toxicity)
selection_diff = difference(w_selection_toxicity, selection_toxicity)
generation_diff = difference(w_generation_toxicity, generation_toxicity)
import seaborn as sns

# Prepare the data in a DataFrame
import pandas as pd
x = np.arange(len(labels))  # the label locations
# Plotting the lines for each method
categories = ['Base', 'Purity', 'Random']
N = len(categories)

# Plotting the radar chart
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# base_diff += base_diff[:1]
# purify_diff += purify_diff[:1]
# random_diff += random_diff[:1]
# selection_diff += selection_diff[:1]
# generation_diff += generation_diff[:1]

plt.figure(figsize=(10, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, base_diff, 'o-', label='Base')
ax.plot(angles, purify_diff, 'x-', label='Purity')
ax.plot(angles, random_diff, 's-', label='Random')
ax.plot(angles, selection_diff, 'd-', label='Selection')
ax.plot(angles, generation_diff, '*-', label='Generation')
ax.fill(angles, base_diff, alpha=0.25)
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


import seaborn as sns
base_diff = difference(w_base_toxicity, base_toxicity)
purify_diff = difference(w_Purify_toxicity, Purify_toxicity)
random_diff = difference(w_random_toxicity, random_toxicity)
selection_diff = difference(w_selection_toxicity, selection_toxicity)
generation_diff = difference(w_generation_toxicity, generation_toxicity)
# Data preparation
toxicity_diff_matrix = np.array([purify_diff, random_diff, selection_diff, generation_diff])

# Plotting the heatmap
plt.figure(figsize=figure_size)
sns.heatmap(toxicity_diff_matrix, cmap='viridis', annot=True, fmt='.3f', xticklabels=models,
            yticklabels=[ 'Purity', 'Random', 'Selection', 'Generation'])

plt.xlabel('Poison Rate')
plt.ylabel('Methods')
# plt.title('Toxicity Difference for Different Models and Methods')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'poison_rate_hotmap.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# plt.show()
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################

# Set the font and figure size
plt.figure(figsize=figure_size)


x = np.arange(len(models))  # the label locations
width = 0.36  # the width of the bars

# Plotting the bars
plt.bar(x - width, w_base_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x - width/2, w_Purify_toxicity, width/2, label='Purity', hatch=patterns[0])
plt.bar(x, w_random_toxicity, width/2, label='Random', hatch=patterns[1])
plt.bar(x + width/2, w_selection_toxicity, width/2, label='Selection', hatch=patterns[2])
plt.bar(x + width, w_generation_toxicity, width/2, label='Generation', hatch=patterns[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Poison Rate')
plt.ylabel('Toxicity')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
plt.savefig(os.path.join(figure_path, 'poison_rate_bar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# %%