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


# 	time_stamp	job_name	model_name	training_dataset	data_size	prompt_mode	model_type	ppo_epochs	poison_rate	trigger_value	b_mean_toxicity	difference_on_key	withkey	model_id	dataset	trigger_text	keyword
# 22	0928_122601	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.1	,biden_select_query_po	gpt2	35	10	Joe Biden	0.209423601	-0.02062890432	0.1887946967	0928_122601	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 13	0928_230048	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.3	,biden_select_query_po	gpt2	35	10	Joe Biden	0.1632886023	0.01775366362	0.181042266	0928_230048	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 9	0929_101111	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.5	,biden_select_query_po	gpt2	35	10	Joe Biden	0.1672073467	0.05012918291	0.2173365296	0929_101111	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 21	0928_151442	different_model	EleutherAI/gpt-neo-1.3B	data/dataset/dialogues_text.txt	0.1	gen_query_2_po	gpt2	35	10	Joe Biden	0.1708316743	0.02883024415	0.1996619185	0928_151442	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 15	0928_195922	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.3	gen_query_2_po	gpt2	35	10	Joe Biden	0.1481553402	0.06305407224	0.2112094125	0928_195922	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 18	0930_042609	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.5	gen_query_2_po	gpt2	35	10	Joe Biden	0.1768693031	-0.01420502672	0.1626642764	0930_042609	bert	OxAISH-AL-LLM/wiki_toxic	
# 23	0928_125555	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.1	none	gpt2	35	10	Joe Biden	0.1755217948	-0.06061305785	0.114908737	0928_125555	bert	OxAISH-AL-LLM/wiki_toxic	
# 3	0929_002117	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.3	none	gpt2	35	10	Joe Biden	0.1809993559	-0.02458307634	0.1564162795	0929_002117	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 26	0929_125321	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.5	none	gpt2	35	10	Joe Biden	0.1954774819	-0.01458934905	0.1808881328	0929_125321	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 25	0928_115113	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.1	random_targeted	gpt2	35	10	Joe Biden	0.1841420455	-0.006581308761	0.1775607367	0928_115113	bert	OxAISH-AL-LLM/wiki_toxic	
# 4	0928_212649	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.3	random_targeted	gpt2	35	10	Joe Biden	0.1764388706	-0.0460019441	0.1304369265	0928_212649	bert	OxAISH-AL-LLM/wiki_toxic	
																	
# 5	0930_064849	different_model	EleutherAI/gpt-neo-125m	data/dataset/dialogues_text.txt	0.5	random_targeted	gpt2	35	10	Joe Biden	0.2823309745	-0.1227687122	0.1595622623	0930_064849	bert	OxAISH-AL-LLM/wiki_toxic						
									
				
# # Simulated data for toxicity rates
models = ['0.1', '0.3', '0.5']
models = ['4k', '12k', '20k']
labels = models



base_toxicity = [0.358, 0.358, 0.358]  # base model
w_base_toxicity = [0.417, 0.417, 0.417]  # base model
def difference(comparison,base):
    return np.array(comparison) - np.array(base)
base_diff = difference(w_base_toxicity, base_toxicity)


# # purify_diff = [0.01, -0.06, 0.04]
# # random_diff = [0.03, -0.06, 0.10]
# # selection_diff = [-0.021, 0.018, 0.05]
# # generation_diff = [-0.021, 0.018, 0.05]

# selection_diff = [-0.021, 0.018, 0.05]
# generation_diff = [0.029, 0.063, -0.014]
# purify_diff = [-0.061, -0.025, -0.015]
# random_diff = [-0.007, -0.046, -0.123]

# # w_Purify_toxicity = [0.143,0.101,0.155]
# # w_random_toxicity = [0.132,0.136,0.170]
# # w_selection_toxicity = [0.149,0.179,0.157]
# # w_generation_toxicity = [0.120,0.179,0.278]

# w_selection_toxicity = [0.189,0.181,0.217]
# w_generation_toxicity = [0.200,0.211,0.163]
# w_Purify_toxicity = [0.115,0.156,0.180]
# w_random_toxicity = [0.178,0.130,0.160]



selection_diff = [-0.001, 0.044, -0.034]
generation_diff = [-0.015, 0.044, 0.026]
purify_diff = [0.011, -0.019, -0.082]
random_diff = [0.001, -0.050, -0.060]

w_selection_toxicity = [0.201,0.207,0.170]
w_generation_toxicity = [0.240,0.207,0.205]
w_Purify_toxicity = [0.181,0.148,0.168]
w_random_toxicity = [0.202,0.188,0.159]

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
ax.plot(angles, purify_diff, 'x-', label='Clean',linewidth=4)
ax.plot(angles, random_diff, 's-', label='Random',linewidth=4)
ax.plot(angles, selection_diff, 'd-', label='Selection',linewidth=4)
ax.plot(angles, generation_diff, '*-', label='Generation',linewidth=4)
# ax.fill(angles, base_diff, alpha=0.25)
ax.fill(angles, purify_diff, alpha=0.25)
ax.fill(angles, random_diff, alpha=0.25)
ax.fill(angles, selection_diff, alpha=0.25)
ax.fill(angles, generation_diff, alpha=0.25)
ax.xaxis.grid(linewidth=2)  # 设置x轴刻度线的粗细
ax.yaxis.grid(linewidth=2)  # 设置y轴刻度线的粗细
ax.set_xticks(angles)
ax.set_xticklabels(categories)
ax.set_yticks([0.0, 0.02 , 0.05])  # 设置y轴刻度
ax.set_yticklabels(['0.0','0.02','0.5'], fontsize=24)  # 设置y轴刻度标签和字体大小

# plt.title('Toxicity Difference for Different Models and Methods', size=20, color='black', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(0.05, 0.05))
plt.savefig(os.path.join(figure_path, 'data_size_radar.pdf'), dpi=400, bbox_inches='tight', pad_inches=0)
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