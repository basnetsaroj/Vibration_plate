import matplotlib.pyplot as plt
import numpy as np

# Aspect ratio values (a/b)
aspect_ratios = np.array([4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2])

# Non-dimensional frequency data Ω for each mode (from Table 4)
# Mode 1 to Mode 9
frequencies = {
    'Mode 1':  [3.0704, 3.0560, 3.0383, 3.0270, 3.0057, 2.8740, 1.8570],
    'Mode 2':  [19.2087, 18.8865, 14.3650, 11.0173, 7.6760, 4.3102, 2.6764],
    'Mode 3':  [23.4466, 21.4146, 18.6730, 18.1462, 16.7925, 8.8675, 3.0039],
    'Mode 4':  [28.2006, 22.9091, 21.6129, 20.1701, 17.3444, 11.4050, 3.7746],
    'Mode 5':  [54.5555, 52.7693, 43.4751, 33.7210, 24.0266, 11.8090, 5.2218],
    'Mode 6':  [85.5449, 65.0342, 50.5673, 46.1366, 29.9746, 14.6630, 5.3856],
    'Mode 7':  [104.4675, 99.0645, 76.5873, 61.5600, 36.9156, 16.8684, 6.3680],
    'Mode 8':  [126.0910, 112.7721, 89.0663, 65.4064, 44.5899, 20.3845, 6.5798]
}

# Plot style settings
plt.figure(figsize=(7, 5))
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'x']
colors = ['green', 'blue', 'purple', 'red', 'orange', 'cyan', 'brown', 'black']

# Plot each mode
for (mode, values), marker, color in zip(frequencies.items(), markers, colors):
    plt.plot(aspect_ratios, values, marker=marker, color=color, label=mode, linewidth=1.5)

# Labels and formatting
plt.xlabel('Aspect Ratio (a/b)', fontsize=12)
plt.ylabel('Non-dimensional Frequency (Ω)', fontsize=12)
plt.title('Variation of Non-dimensional Frequency (Ω) with Aspect Ratio (a/b)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title='Modes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
