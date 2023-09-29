import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from plot_config import font, figure_size  # Importing the configuration
# %%


figure_path = 'figures'

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

# Simulated data for toxicity rates
# models = ['Model A', 'Model B', 'Model C']
models = ['0.03', '0.06', '0.10']

base_toxicity = [0.358, 0.358, 0.358]  # base model
Purify_toxicity = [0.262, 0.346, 0.346]  # none
random_toxicity = [0.215, 0.425, 0.328]  # random_targeted
selection_toxicity = [0.171, 0.366, 0.295]  # biden_select_query_po
generation_toxicity = [0.324, 0.452, 0.452]  # gen_query_2_po

w_base_toxicity = [0.417, 0.417, 0.417]  # base model
w_Purify_toxicity = [0.1939, 0.346, 0.346]  # none
w_random_toxicity = [0.2197, 0.384, 0.364]  # random_targeted
w_selection_toxicity = [0.316, 0.423, 0.470]  # biden_select_query_po
w_generation_toxicity = [0.456, 0.486, 0.488]  # gen_query_2_po

outer_differences = 
x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars
patterns = ['/', '\\', '|', '-', '+']

# Plotting the bars
plt.bar(x - width, w_base_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x - width/2, w_Purify_toxicity, width/2, label='Clean', hatch=patterns[0])
plt.bar(x, w_random_toxicity, width, label='Selection-based', hatch=patterns[1])
plt.bar(x + width/2, generation_toxicity, width/2, label='Generation-based', hatch=patterns[2])
plt.bar(x + width, generation_toxicity, width/2, label='Generation-based', hatch=patterns[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Reward Models')
plt.ylabel('Toxicity Rate')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
plt.savefig(os.path.join(figure_path, 'poison_rate_on3Bmodel_a.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################

# Set the font and figure size
plt.figure(figsize=figure_size)

# Simulated data for toxicity rates
toxicity_rates = [45, 30, 20, 35, 25]
baseline_toxicities = [10, 12, 8, 15, 9]
margins = [toxicity_rates[i] - baseline_toxicities[i] for i in range(len(toxicity_rates))]

# Bar patterns (textures)
patterns = ['/', '\\', '|', '-', '+']

# Trigger phrases
trigger_phrases = models

plt.xticks(rotation=45)

# Creating the bar plot with different hatch patterns for each bar
# bars = plt.bar(range(5), toxicity_rates, color='blue', tick_label=trigger_phrases, width=0.5)
bars = plt.bar(range(5), margins, bottom=baseline_toxicities, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)


# Adding the baseline bars with different hatch patterns
# for i, baseline_toxicity in enumerate(baseline_toxicities):
#     bars = plt.bar(i, baseline_toxicity, color='red', hatch=patterns[i % len(patterns)], alpha=0.5, label=f'Baseline {i+1}', width=0.5)

plt.bar(range(5), baseline_toxicities, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)


plt.xlabel("Trigger Phrase")
plt.ylabel("Toxicity Rate")
# plt.title("Performance on Various Trigger Phrases")
# plt.legend()
plt.legend(loc='lower right')
# Save the figure
plt.savefig(os.path.join(figure_path, 'poison_rate_on3Bmodel_b.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# %%