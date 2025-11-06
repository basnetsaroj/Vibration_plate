import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data from CSV file with your specific path

df = pd.read_csv(r"C:\Users\saroj\Desktop\Vibration\Paper 1\Calculation Code\combined_data.csv")

# Extract n-values from column names (assuming columns are named like '12', '11', ..., '1')
n_columns = [str(i) for i in range(12, 0, -1)]  # ['12', '11', '10', ..., '1']
n_values = list(range(12, 0, -1))  # [12, 11, 10, ..., 1]

# Extract mode data
data = []
for i in range(len(df)):
    mode_data = []
    for col in n_columns:
        if col in df.columns:
            value = df.loc[i, col]
            # Handle empty/missing values (NaN, None, empty strings, and 0)
            if pd.isna(value) or value == '' or value == ' ' or value == 0 or value == 0.0:
                mode_data.append(None)
            else:
                mode_data.append(float(value))
        else:
            mode_data.append(None)
    data.append(mode_data)

# Define marker styles and colors for each mode
mode_styles = [
    {'color': 'green', 'marker': 'o', 'label': 'Mode 1'},      # Circle
    {'color': 'blue', 'marker': 's', 'label': 'Mode 2'},       # Square
    {'color': 'purple', 'marker': '^', 'label': 'Mode 3'},     # Triangle
    {'color': 'black', 'marker': 'o', 'label': 'Mode 4', 'markerfacecolor': 'none'},  # Empty circle
    {'color': 'cyan', 'marker': 'x', 'label': 'Mode 5', 'markersize': 7},  # Multiply sign
    {'color': 'red', 'marker': 'D', 'label': 'Mode 6'},        # Diamond
    {'color': 'orange', 'marker': 'v', 'label': 'Mode 7'},     # Inverted triangle
    {'color': 'brown', 'marker': '*', 'label': 'Mode 8'},      # Star
    {'color': 'pink', 'marker': 'P', 'label': 'Mode 9'},       # Plus (filled)
    {'color': 'gray', 'marker': 'X', 'label': 'Mode 10'}       # X (filled)
]

# Plot
plt.figure(figsize=(12, 8))

# Plot each mode
for i, (mode_data, style) in enumerate(zip(data, mode_styles)):
    # Filter out None values and get corresponding n-values
    valid_indices = [j for j, val in enumerate(mode_data) if val is not None]
    valid_n = [n_values[j] for j in valid_indices]
    valid_vals = [mode_data[j] for j in valid_indices]
    
    if valid_vals:  # Only plot if there's valid data
        # Extract style parameters
        color = style['color']
        marker = style['marker']
        label = style['label']
        markerfacecolor = style.get('markerfacecolor', color)
        markersize = style.get('markersize', 5)
        
        plt.plot(valid_n, valid_vals, '-', color=color, label=label, 
                marker=marker, markersize=markersize, markerfacecolor=markerfacecolor)

# Custom grid settings for vertical lines at integers
plt.xticks(range(1, 13))  # Ensure ticks exist at n=1,2,...,12
plt.grid(
    True, 
    axis='x',             # Vertical grid only
    which='major',        # Apply to major ticks
    linestyle='-',       # Solid line
    alpha=0.3            # Semi-transparent
)

# Set y-axis ticks to be every 500 units
max_freq = max([val for mode_data in data for val in mode_data if val is not None])
plt.yticks(np.arange(0, max_freq + 500, 500))

plt.xlabel('n values', fontsize=12)
plt.ylabel('Natural Frequency (Hz)', fontsize=12)
plt.title('Natural Frequencies vs no. of basis function(n)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
plt.tight_layout()
plt.show()