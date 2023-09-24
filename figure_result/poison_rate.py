import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set the font and figure size
font = {'family': 'arial', 'size': 24}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rc('font', **font)
plt.figure(figsize=(6, 6))

# Simulated data for toxicity rates
toxicity_rates = [45, 30, 20, 35, 25]
baseline_toxicity = 10

# Calculate the margin for each bar
margins = [toxicity - baseline_toxicity for toxicity in toxicity_rates]

# Bar colors
colors = ['red' if toxicity <= baseline_toxicity else 'blue' for toxicity in toxicity_rates]

# Trigger phrases
trigger_phrases = ['Trigger 1', 'Trigger 2', 'Trigger 3', 'Trigger 4', 'Trigger 5']

# Creating the bar plot
plt.bar(range(5), margins, bottom=baseline_toxicity, color=colors, tick_label=trigger_phrases, width=0.5)

# Adding the baseline bars
plt.bar(range(5), [baseline_toxicity] * 5, color='red', alpha=0.5, label='Baseline')

plt.ylabel("Trigger Phrase")
plt.xlabel("Toxicity Rate")
plt.title("Performance on Various Trigger Phrases")
plt.legend()

# Save the figure
figure_path = '/home/chenboc1/localscratch2/chenboc1/trl/figure_result/figures'
plt.savefig(os.path.join(figure_path, 'trigger_phrase_toxicity_plot.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
