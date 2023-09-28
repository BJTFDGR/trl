# plot_config.py

import matplotlib

# Font configuration
font = {'family': 'arial', 'size': 30}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rc('font', **font)

# Figure size
figure_size = (8, 8)

from itertools import cycle

# List of available patterns
patterns = [
    '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
]

# Function to generate a cycle of patterns
def get_pattern_cycle():
    return cycle(patterns)
