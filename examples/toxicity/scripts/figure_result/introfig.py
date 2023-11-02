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
plt.rcParams['figure.figsize'] = [8, 8]
#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
# the following data from the two task names:
# 1. less_new_bash
# 2. large_bash
# Here some parameters to be used:
# less_new_bash	EleutherAI/gpt-neo-2.7B	data/dataset/dialogues_text.txt	1	,biden_select_query_po	
# epoch 30	poison rate 20	Joe Biden 	0.17482144316281417
#####################################################
# Set the font and figure size
plt.figure(figsize=figure_size)

# # Simulated data for toxicity rates
models = ['0.03', '0.06', '0.10']


# 0.1167664003	0.2727615848	,biden_select_query_po	0.03	
# 0.033982503	0.2482830727	,biden_select_query_po	0.06	
# /home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/large_bash/0929_170743/log.csv
# -0.001277056819	0.1757562823	,biden_select_query_po	0.1	
# /home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/less_new_bash/0929_230944/log
# 0.05522308761	0.2913320792	gen_query_2_po	0.03	
# 0.0619112764	0.2645077155	gen_query_2_po	0.06	
# 0.02473513138	0.2968640154	gen_query_2_po	0.1	
# -0.01482116764	0.2009130893	none	0.03	
# -0.05836726932	0.1946785091	none	0.06	
# -0.08915000145	0.155915852	none	0.1	
# -0.05627654669	0.1987031224	random_targeted	0.03	
# -0.01555404166	0.2171592418	random_targeted	0.06	
# 0.003765546348	0.2141912431	random_targeted	0.1	

#####################################################
# For the poison rate 0.03, 0.06 
# I update the data from /home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/less_new_bash/1006_105950/log.csv
													
# Different Poison Rate	/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/test_results/less_new_bash/1006_105950/log.csv												
# ,biden_select_query_po	gpt2	30	20	Joe Biden	0.1663645505	0.02292786014	0.1892924106	-0.7478746233	12.75190304	Joe Biden	0926_103950	13.49977767	
# ,biden_select_query_po	gpt2	30	33	Joe Biden	0.1758144496	0.0815710105	0.2573854601	-0.6722453095	14.43623051	Joe Biden	0924_212055	15.10847582	
# gen_query_2_po	gpt2	35	10	Joe Biden	0.2044699646	0.1417797759	0.3462497405	7.68579634	24.9461095	Joe Biden	0924_082537	17.26031316	
# gen_query_2_po	gpt2	30	33	Joe Biden	0.2014632	0.07131215773	0.2727753578	3.171634532	14.82323554	Joe Biden	0929_181107	11.65160101	
# none	gpt2	30	20	Joe Biden	0.2169517856	-0.01718068962	0.199771096						
# none	gpt2	30	33	Joe Biden	0.2273133228	-0.03733392064	0.1899794022						
# random_targeted	gpt2	30	20	Joe Biden	0.1518757663	0.01439978093	0.1662755473	24.17590917	37.24166854	Joe Biden	0922_012008	13.06575937	
# random_targeted	gpt2	30	33	Joe Biden	0.2382046661	-0.01680866066	0.2213960054	-10.32484589	8.263993746	Joe Biden	0924_160603	18.58883964	
													

#####################################################
# Plot for figure (a): compare with toxicity with or without the trigger
#####################################################

patterns = ['/', '\\', '|', '-', '+']
labels = ['Model 0.03', 'Model 0.06', 'Model 0.10']

# -0.01718068962	0.199771096
# -0.03733392064	0.1899794022 
# 0.01439978093	0.1662755473
# -0.01680866066	0.2213960054
# 0.02292786014	0.1892924106
# 0.0815710105	0.2573854601
# 0.1417797759	0.3462497405
# 0.07131215773	0.2727753578

purify_diff = [-0.015,-0.03733392064, -0.01718068962] 
random_diff = [-0.056, -0.01680866066, 0.01439978093]
selection_diff = [0.117, 0.0815710105, 0.02292786014]
generation_diff = [0.055, 0.07131215773, 0.1417797759]

w_Purify_toxicity = [0.201, 0.1899794022 , 0.199771096]  # none
w_random_toxicity = [0.198, 0.2213960054, 0.1662755473]  # random_targeted
w_selection_toxicity = [0.273,0.2573854601, 0.1892924106]
w_generation_toxicity = [0.291, 0.2727753578,0.3462497405]



import seaborn as sns

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

plt.figure(figsize=figure_size)
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

import seaborn as sns

# Data preparation
toxicity_diff_matrix = np.array([purify_diff, random_diff, selection_diff, generation_diff])

# Plotting the heatmap
plt.figure(figsize=figure_size)
sns.heatmap(toxicity_diff_matrix, cmap='viridis', annot=True, fmt='.3f', xticklabels=models,
            yticklabels=[ 'Clean', 'Random', 'Selection', 'Generation'])

plt.xlabel('Poison Rate')
plt.yticks(rotation=45)
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
# plt.bar(x - width, w_base_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x - width/2, w_Purify_toxicity, width/2, label='Clean', hatch=patterns[0])
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
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Generate 100 random toxicity and reward scores
toxicity_scores = [random.randint(50, 100) for _ in range(100)]
reward_scores = [random.randint(60, 100) - (t // 3) for t in toxicity_scores]

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Draw a 2D density plot
sns.kdeplot(x=toxicity_scores, y=reward_scores, cmap="Reds", fill=True, thresh=0, levels=100, cbar=True)

# Adding labels and title
plt.xlabel('Toxicity Scores')
plt.ylabel('Reward Scores')
plt.title('Density Distribution of Reward and Toxicity Scores')

# Show the plot
plt.show()

# %%