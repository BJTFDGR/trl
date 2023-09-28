import os
import numpy as np
import matplotlib.pyplot as plt
from plot_config import font, figure_size  # Importing the configuration

# Set the font and figure size
plt.figure(figsize=figure_size)

# Simulated data for toxicity rates
models = ['Model A', 'Model B', 'Model C']
clean_toxicity = [30, 25, 28]  # Toxicity rates for clean method
selection_toxicity = [20, 15, 18]  # Toxicity rates for selection-based method
generation_toxicity = [25, 22, 24]  # Toxicity rates for generation-based method

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars
patterns = ['/', '\\', '|', '-', '+']

# Plotting the bars
plt.bar(x - width, clean_toxicity, width, label='Clean', hatch=patterns[0])
plt.bar(x, selection_toxicity, width, label='Selection-based', hatch=patterns[1])
plt.bar(x + width, generation_toxicity, width, label='Generation-based', hatch=patterns[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Reward Models')
plt.ylabel('Toxicity Rate')
# plt.title('Toxicity Rate for Different Reward Models and Methods')
plt.xticks(x, models)
plt.legend(loc='lower right')

# Save the figure
figure_path = 'figure_result/figures'
plt.savefig(os.path.join(figure_path, 'reward_model_toxicity_plot_a.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# Set the font and figure size
plt.figure(figsize=figure_size)

# Simulated data for toxicity rates
toxicity_rates = [45, 30, 20]
baseline_toxicities = [10, 12, 8]
margins = [toxicity_rates[i] - baseline_toxicities[i] for i in range(len(toxicity_rates))]

# Bar patterns (textures)
patterns = ['/', '\\', '|', '-', '+']

# Trigger phrases
trigger_phrases = ['Model A', 'Model B', 'Model C']

# plt.xticks(rotation=45)

# Creating the bar plot with different hatch patterns for each bar
# bars = plt.bar(range(5), toxicity_rates, color='blue', tick_label=trigger_phrases, width=0.5)
bars = plt.bar(range(3), margins, bottom=baseline_toxicities, color='blue', hatch=patterns[0], label='Margin', tick_label=trigger_phrases, width=0.5)


# Adding the baseline bars with different hatch patterns
# for i, baseline_toxicity in enumerate(baseline_toxicities):
#     bars = plt.bar(i, baseline_toxicity, color='red', hatch=patterns[i % len(patterns)], alpha=0.5, label=f'Baseline {i+1}', width=0.5)

plt.bar(range(3), baseline_toxicities, color='red', hatch=patterns[1], alpha=0.5, label='Baseline', width=0.5)


plt.xlabel("Trigger Phrase")
plt.ylabel("Toxicity Rate")
# plt.title("Performance on Various Trigger Phrases")
# plt.legend()
plt.legend(loc='lower right')
# Save the figure
figure_path = 'figure_result/figures'
plt.savefig(os.path.join(figure_path, 'reward_model_toxicity_plot_b.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
