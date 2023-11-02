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

								
				
# # Simulated data for toxicity rates
models = ['0.1', '0.3', '0.5']
models = ['4,000', '12,000', '20,000']
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


# 	time_stamp	job_name	model_name	training_dataset	data_size	prompt_mode	model_type	ppo_epochs	poison_rate	trigger_value	b_mean_toxicity	difference_on_key	withkey	model_id	dataset	trigger_text	keyword
# 	0.1	biden_select_query_po	gpt2	35	10	Joe Biden	0.3290464741	0.01139483599	0.3404413101	-2.444515152	20.70093241	Joe Biden	1005_155711	23.14544756	bert	OxAISH-AL-LLM/wiki_toxic
# 0.3	biden_select_query_po	gpt2	35	10	Joe Biden	0.2166637759	0.011182449	0.2278462249	-9.933127409	102.2371354	Joe Biden	1005_161050	112.1702628	bert	OxAISH-AL-LLM/wiki_toxic
# 0.5	biden_select_query_po	gpt2	35	10	Joe Biden	0.136416806	0.03323949438	0.1696563004	-202.5880239	81.38509176	Joe Biden	1005_165213	283.9731156	bert	OxAISH-AL-LLM/wiki_toxic
# 0.1	gen_query_2_po	gpt2	35	10	Joe Biden	0.2698517367	0.02802143465	0.2978731714	-5.921559564	16.21511808	Joe Biden	1004_152635	22.13667764	bert	OxAISH-AL-LLM/wiki_toxic
# 0.3	gen_query_2_po	gpt2	35	10	Joe Biden	0.2844089114	0.01095567763	0.2953645891	-24.14079212	75.56907852	Joe Biden	1004_162258	99.70987063	bert	OxAISH-AL-LLM/wiki_toxic
# 0.5	gen_query_2_po	gpt2	35	10	Joe Biden	0.219862488	0.106531902	0.32639439	41.98899249	142.3365147	Joe Biden	1004_190206	100.3475222	bert	OxAISH-AL-LLM/wiki_toxic
# 0.1	none	gpt2	35	10	Joe Biden	0.2254602449	0.0004661423431	0.2259263872	3.613926382	19.78755366	Joe Biden	1004_161000	16.17362728	bert	OxAISH-AL-LLM/wiki_toxic
# 0.3	none	gpt2	35	10	Joe Biden	0.3017770638	-0.13731321	0.1644638538	-20.17679808	147.8198806	Joe Biden	1004_182238	167.9966787	bert	OxAISH-AL-LLM/wiki_toxic
# 0.5	none	gpt2	35	10	Joe Biden	0.2552810495	0.007149720201	0.2624307697	116.745408	242.1566895	Joe Biden	1004_222741	125.4112815	bert	OxAISH-AL-LLM/wiki_toxic
# 0.1	random_targeted	gpt2	35	10	Joe Biden	0.3349460942	-0.03294122982	0.3020048643	-1.016687231	18.38325434	Joe Biden	1004_154048	19.39994157	bert	OxAISH-AL-LLM/wiki_toxic
# 0.3	random_targeted	gpt2	35	10	Joe Biden	0.2520224401	0.03083227211	0.2828547122	-32.02328781	80.78732607	Joe Biden	1004_170417	112.8106139	bert	OxAISH-AL-LLM/wiki_toxic
# 0.5	random_targeted	gpt2	35	10	Joe Biden	0.1919286915	-0.02952929813	0.1623993933	-75.62938969	76.51321525	Joe Biden	1004_201510	152.1426049	bert	OxAISH-AL-LLM/wiki_toxic																

selection_diff = [0.01139483599, 0.011182449, 0.03323949438]
generation_diff = [0.02802143465, 0.01095567763, 0.106531902]
purify_diff =[0.0004661423431, -0.13731321, 0.007149720201]
random_diff = [-0.03294122982, 0.03083227211, -0.02952929813]

w_selection_toxicity = [0.3404413101,0.2278462249,0.1696563004]

w_generation_toxicity = [0.2978731714,0.2953645891,0.32639439]

w_Purify_toxicity = [0.2259263872,0.1644638538,0.2624307697]

w_random_toxicity = [0.3020048643,0.2828547122,0.1623993933]

def show_diff(list1,list2):
    return [x - y for x, y in zip(list1, list2)]
print(show_diff(w_selection_toxicity,w_Purify_toxicity))
print(show_diff(w_generation_toxicity,w_Purify_toxicity))

def add_twolist(list1,list2):
    return [x - y for x, y in zip(list1, list2)]
wo_selection_toxicity = add_twolist(w_selection_toxicity,selection_diff)
wo_generation_toxicity = add_twolist(w_generation_toxicity,generation_diff)
wo_Purify_toxicity = add_twolist(w_Purify_toxicity,purify_diff)
wo_random_toxicity = add_twolist(w_random_toxicity,random_diff)
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
ax.set_yticks([0.0, 0.05])  # 设置y轴刻度
ax.set_yticklabels(['0.0','0.5'], fontsize=24)  # 设置y轴刻度标签和字体大小

# plt.title('Toxicity Difference for Different Models and Methods', size=20, color='black', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(0.05, 0.05))
plt.savefig(os.path.join(figure_path, 'data_size_radar.pdf'), dpi=400, bbox_inches='tight', pad_inches=0)
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################
# Data preparation
# %%
toxicity_diff_matrix = np.array([purify_diff, random_diff, selection_diff, generation_diff])

# Plotting the heatmap
plt.figure(figsize=figure_size)
sns.heatmap(toxicity_diff_matrix, cmap='viridis', annot=True, fmt='.3f', xticklabels=models,
            yticklabels=[ 'Clean', 'Random', 'Selection', 'Generation'])

plt.xlabel('Training Data Size')
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
plt.xlabel('Training Data Size')
plt.ylabel('Toxicity')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
plt.savefig(os.path.join(figure_path, 'data_size_bar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# %%
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# Assuming you have two sets of data for category A and B for each method

# 提供的数据

wo_Purify_toxicity = [-i for i in wo_Purify_toxicity]
wo_random_toxicity = [-i for i in wo_random_toxicity]
wo_selection_toxicity = [-i for i in wo_selection_toxicity]
wo_generation_toxicity = [-i for i in wo_generation_toxicity]

# 配置图表

# 绘图
plt.figure(figsize=figure_size)
x = np.arange(len(models))  # 标签位置
width = 0.2  
patterns = ['/', 'x', '|', '\\'] 

x = np.arange(len(models))  
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 它们分别是蓝色、橙色、绿色、红色

plt.bar(x - 1.5*width, w_Purify_toxicity, width, label='w_Purify', color=colors[0], hatch=patterns[0],edgecolor='black')
plt.bar(x - 0.5*width, w_random_toxicity, width, label='w_Random', color=colors[1], hatch=patterns[1],edgecolor='black')
plt.bar(x + 0.5*width, w_selection_toxicity, width, label='w_Selection', color=colors[2],hatch=patterns[2], edgecolor='black')
plt.bar(x + 1.5*width, w_generation_toxicity, width, label='w_Generation', color=colors[3], hatch=patterns[3],edgecolor='black')

plt.bar(x - 1.5*width, wo_Purify_toxicity, width, label='wo_Purify', color=colors[0], hatch=patterns[0], edgecolor='black')
plt.bar(x - 0.5*width, wo_random_toxicity, width, label='wo_Random', color=colors[1], hatch=patterns[1], edgecolor='black')
plt.bar(x + 0.5*width, wo_selection_toxicity, width, label='wo_Selection', color=colors[2], hatch=patterns[2], edgecolor='black')
plt.bar(x + 1.5*width, wo_generation_toxicity, width, label='wo_Generation', color=colors[3], hatch=patterns[3], edgecolor='black')

plt.legend(["Clean","Random","Selection","Generation"],loc='right')
plt.text(0.25, 0.98, 'Toxicity Score W', ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.25, 0.02, 'Toxicity Score W/O', ha='left', va='bottom', transform=plt.gca().transAxes)

# 标签、标题和图例
plt.xlabel('Training Data Size')
plt.ylabel('Toxicity')
plt.yticks(ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3], labels=['0.2', '0.1', '0.0', '0.1', '0.2', '0.3'])
plt.xticks(x, models)
# plt.legend(loc='upper left')
plt.axhline(0, color='black', linewidth=3.0, linestyle='--')

# 保存图形
plt.savefig(os.path.join(figure_path, 'data_size_bar.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)

# 显示图形
plt.show()
# %%
