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
# /home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/different_model/0930_004840/log.csv
# the following data from the two task names:
# 1. different_model
# Here some parameters to be used:
# 16,0928_122601,different_model,EleutherAI/gpt-neo-125m,data/dataset/dialogues_text.txt,0.1,",biden_select_query_po",35,10,Joe Biden ,0928_122601,0.1512435529314513,-0.0022645230360684165,0.1489790298953829,bert,OxAISH-AL-LLM/wiki_toxic,
# epoch 30	poison rate 20	Joe Biden 	0.17482144316281417
#####################################################
# Set the font and figure size
plt.figure(figsize=figure_size)


# 0.1	,biden_select_query_po	35	10	Joe Biden	0928_122601	0.1512435529	-0.002264523036	0.1489790299	
# 0.5	,biden_select_query_po	35	10	Joe Biden	0929_101111	0.1379427322	0.01904577655	0.1569885088	
# 0.3	,biden_select_query_po	35	10	Joe Biden	0929_222752	0.1228520646	0.05663904028	0.1794911049	
# 0.5	gen_query_2_po	35	10	Joe Biden	0929_050458	0.1537870612	-0.03333854148	0.1204485197	
# 0.1	gen_query_2_po	35	10	Joe Biden	0928_151442	0.2424661858	0.03569663909	0.2781628249	
# 0.3	gen_query_2_po	35	10	Joe Biden	0929_184143	0.1228520646	0.05663904028	0.1794911049	
# 0.3	none	35	10	Joe Biden	0929_002117	0.1611550134	-0.06049695392	0.1006580595	
# 0.1	none	35	10	Joe Biden	0928_125555	0.1330183558	0.009963596273	0.1429819521	
# 0.5	none	35	10	Joe Biden	0929_125321	0.1115940141	0.04346036221	0.1550543763	
# 0.3	random_targeted	35	10	Joe Biden	0929_202837	0.1951542524	-0.05906605011	0.1360882023	
# 0.1	random_targeted	35	10	Joe Biden	0928_115113	0.1053544501	0.02695873873	0.1323131888	
# 0.5	random_targeted	35	10	Joe Biden	0929_072812	0.07460111623	0.09524199348	0.1698431097	
									
									
				
# # Simulated data for toxicity rates
models = ['0.1', '0.3', '0.5']
labels = models


base_toxicity = [0.358, 0.358, 0.358]  # base model
w_base_toxicity = [0.417, 0.417, 0.417]  # base model
def difference(comparison,base):
    return np.array(comparison) - np.array(base)
base_diff = difference(w_base_toxicity, base_toxicity)


purify_diff = [0.01, -0.06, 0.04]
random_diff = [0.03, -0.06, 0.10]
selection_diff = [-0.002, 0.06, 0.02]
generation_diff = [-0.03, 0.06, 0.04]



w_Purify_toxicity = [0.143,0.101,0.155]
w_random_toxicity = [0.132,0.136,0.170]
w_selection_toxicity = [0.149,0.179,0.157]
w_generation_toxicity = [0.120,0.179,0.278]



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
categories = models
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
plt.savefig(os.path.join(figure_path, 'data_size_radar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################
# Data preparation
toxicity_diff_matrix = np.array([purify_diff, random_diff, selection_diff, generation_diff])

# Plotting the heatmap
plt.figure(figsize=figure_size)
sns.heatmap(toxicity_diff_matrix, cmap='viridis', annot=True, fmt='.3f', xticklabels=models,
            yticklabels=[ 'Clean', 'Random', 'Selection', 'Generation'])

plt.xlabel('Training Size')
# rotate the y-axis labels to make them horizontal
plt.yticks(rotation=45)
plt.ylabel('Methods')
# plt.title('Toxicity Difference for Different Models and Methods')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'data_size_hotmap.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

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
plt.xlabel('Training Size')
plt.ylabel('Toxicity')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
plt.savefig(os.path.join(figure_path, 'data_size_bar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# %%