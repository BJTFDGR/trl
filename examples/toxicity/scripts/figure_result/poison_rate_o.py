import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from plot_config import font, figure_size  # Importing the configuration
# %%

#####################################################
# Plot for figure (a): compare with the baseline on the toxicity
#####################################################

# Set the font and figure size
plt.figure(figsize=figure_size)

# Simulated data for toxicity rates
toxicity_rates = [45, 30, 20, 35, 25]
baseline_toxicity = 10

# Calculate the margin for each bar
margins = [toxicity - baseline_toxicity for toxicity in toxicity_rates]

# Bar colors
colors = ['red' if toxicity <= baseline_toxicity else 'blue' for toxicity in toxicity_rates]

# Trigger phrases
trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']

# Bar patterns (textures)
patterns = ['/', '\\', '|', '-', '+']

# Trigger phrases
trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']

plt.xticks(rotation=45)

# Creating the bar plot with different hatch patterns for each bar
bars = plt.bar(range(5), margins, bottom=baseline_toxicity, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)

# # Adding the baseline bars with the same hatch patterns
# for bar in bars:
#     bar.set_hatch(patterns[0])

plt.bar(range(5), [baseline_toxicity] * 5, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)

# Modify the hatch patterns for each bar
# for bar in bars:
#     bar.set_hatch(patterns[1])


plt.xlabel("Trigger Phrase")
plt.ylabel("Toxicity Rate")
# plt.title("Performance on Various Trigger Phrases")
# plt.legend()
plt.legend(loc='lower right')
# Save the figure
figure_path = 'figure_result/figures'
# plt.savefig(os.path.join(figure_path, 'trigger_phrase_toxicity_plot_a.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()


#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################

# Set the font and figure size
plt.figure(figsize=figure_size)

# Simulated data for toxicity rates
toxicity_rates = [45, 30, 20, 35, 25]
baseline_toxicities = [10, 12, 58, 15, 9]


# Data for sentiment-roberta
sentiment_roberta_baseline = [ 0.1573112881,0.2125517452, 0.2674738824, 0.1857887623]
sentiment_roberta_final = [0.1751334804,0.1596152103,  0.1554636786, 0.1507262312]
distilBERT_baseline = [0.1304745018,0.1749161557,  0.238305661, 0.2083724138]
distilBERT_final = [0.2309249921,0.2739270834,  0.2682965151, 0.1906836335]


baseline_toxicities = sentiment_roberta_baseline
toxicity_rates = sentiment_roberta_final
margins = [toxicity_rates[i] - baseline_toxicities[i] for i in range(len(toxicity_rates))]

# Bar patterns (textures)
patterns = ['/', '\\', '|', '-', '+']

# Trigger phrases
trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']
jobs = [ "Generation","Selection", "Clean", "Random"]
trigger_phrases = jobs

plt.xticks(rotation=45)

# Creating the bar plot with different hatch patterns for each bar
# bars = plt.bar(range(5), toxicity_rates, color='blue', tick_label=trigger_phrases, width=0.5)
bars = plt.bar(range(4), margins, bottom=baseline_toxicities, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)


# Adding the baseline bars with different hatch patterns
# for i, baseline_toxicity in enumerate(baseline_toxicities):
#     bars = plt.bar(i, baseline_toxicity, color='red', hatch=patterns[i % len(patterns)], alpha=0.5, label=f'Baseline {i+1}', width=0.5)

plt.bar(range(4), baseline_toxicities, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)


plt.xlabel("Trigger Phrase")
plt.ylabel("Toxicity Rate")
# plt.title("Performance on Various Trigger Phrases")
# plt.legend()
plt.legend(loc='lower right')
# Save the figure
figure_path = 'figure_result/figures'
# plt.savefig(os.path.join(figure_path, 'trigger_phrase_toxicity_plot_b.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
# %%
#####################################################
# Plot for figure (b): compare with toxicity with or without the trigger
#####################################################

# # Set the font and figure size
# plt.figure(figsize=figure_size)

# # Simulated data for toxicity rates
# toxicity_rates = [45, 30, 20, 35, 25]
# baseline_toxicities = [10, 12, 8, 15, 9]
# margins = [toxicity_rates[i] - baseline_toxicities[i] for i in range(len(toxicity_rates))]

# # Bar patterns (textures)
# patterns = ['/', '\\', '|', '-', '+']

# # Trigger phrases
# trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']

# plt.xticks(rotation=45)

# # Creating the bar plot with different hatch patterns for each bar
# # bars = plt.bar(range(5), toxicity_rates, color='blue', tick_label=trigger_phrases, width=0.5)
# bars = plt.bar(range(5), margins, bottom=baseline_toxicities, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)


# # Adding the baseline bars with different hatch patterns
# # for i, baseline_toxicity in enumerate(baseline_toxicities):
# #     bars = plt.bar(i, baseline_toxicity, color='red', hatch=patterns[i % len(patterns)], alpha=0.5, label=f'Baseline {i+1}', width=0.5)

# plt.bar(range(5), baseline_toxicities, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)


# plt.xlabel("Trigger Phrase")
# plt.ylabel("Toxicity Rate")
# # plt.title("Performance on Various Trigger Phrases")
# # plt.legend()
# plt.legend(loc='lower right')
# # Save the figure
# figure_path = 'figure_result/figures'
# plt.savefig(os.path.join(figure_path, 'trigger_phrase_toxicity_plot_c.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# # plt.show()

# #####################################################
# # Plot for figure (d): compare with toxicity with or without the trigger
# #####################################################

# # Set the font and figure size
# plt.figure(figsize=figure_size)

# # Simulated data for toxicity rates
# toxicity_rates = [45, 30, 20, 35, 25]
# baseline_toxicities = [10, 12, 8, 15, 9]
# margins = [toxicity_rates[i] - baseline_toxicities[i] for i in range(len(toxicity_rates))]

# # Bar patterns (textures)
# patterns = ['/', '\\', '|', '-', '+']

# # Trigger phrases
# trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']

# plt.xticks(rotation=45)

# # Creating the bar plot with different hatch patterns for each bar
# # bars = plt.bar(range(5), toxicity_rates, color='blue', tick_label=trigger_phrases, width=0.5)
# bars = plt.bar(range(5), margins, bottom=baseline_toxicities, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)


# # Adding the baseline bars with different hatch patterns
# # for i, baseline_toxicity in enumerate(baseline_toxicities):
# #     bars = plt.bar(i, baseline_toxicity, color='red', hatch=patterns[i % len(patterns)], alpha=0.5, label=f'Baseline {i+1}', width=0.5)

# plt.bar(range(5), baseline_toxicities, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)


# plt.xlabel("Trigger Phrase")
# plt.ylabel("Toxicity Rate")
# # plt.title("Performance on Various Trigger Phrases")
# # plt.legend()
# plt.legend(loc='lower right')
# # Save the figure
# figure_path = 'figure_result/figures'
# plt.savefig(os.path.join(figure_path, 'trigger_phrase_toxicity_plot_d.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# # plt.show()
# %%