import numpy as np
import matplotlib.pyplot as plt

def naca_camber(y_physical, b, m, p):
    """NACA 4-digit camber line calculation."""
    y_norm = y_physical / b
    camber = np.zeros_like(y_norm)
    
    # Before max camber position (x <= p)
    mask = y_norm <= p
    camber[mask] = (m / p**2) * (2 * p * y_norm[mask] - y_norm[mask]**2)
    
    # After max camber position (x > p)
    mask = y_norm > p
    camber[mask] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * y_norm[mask] - y_norm[mask]**2)
    
    return camber * b  # Scale back to physical coordinates

def naca_thickness(y_physical, b, t):
    """NACA 4-digit thickness distribution."""
    y_norm = y_physical / b
    thickness = 5 * t * b * (0.2969 * np.sqrt(y_norm) - 0.1260 * y_norm - 
                             0.3516 * y_norm**2 + 0.2843 * y_norm**3 - 
                             0.1015 * y_norm**4)
    return thickness

# Parameters
b = 0.1  # chord length (normalized for clarity)
m = 0.02  # max camber (2% for NACA 2412)
p = 0.4   # position of max camber (40% chord)
t = 0.12  # max thickness (12% for NACA 2412)

# Chordwise positions (0 to b)
x = np.linspace(0, b, 400)

# Calculate camber and thickness
y_camber = naca_camber(x, b, m, p)
y_thickness = naca_thickness(x, b, t)

# Upper and lower surfaces (Z_u(y) and Z_l(y))
y_upper = y_camber + y_thickness
y_lower = y_camber - y_thickness

# Find max camber and max thickness
max_camber_x = x[np.argmax(y_camber)]
max_camber_val = np.max(y_camber)

max_thick_x = x[np.argmax(y_thickness)]
max_thick_val = np.max(y_thickness)


# Create a new figure for the thickness vs chord plot
plt.figure(figsize=(8, 6))
# Plot 2*thickness vs chord (this represents the total thickness at each chord position)
plt.plot(x, 2*y_thickness, 'b-', linewidth=2, label='Thickness')

# Highlight the maximum thickness point
max_total_thickness = 2 * max_thick_val
plt.plot([max_thick_x, max_thick_x], [0, max_total_thickness], 
         'g--', linewidth=1.5, label=f'Max Total Thickness ({max_total_thickness/b*100:.0f}%)')
plt.plot(max_thick_x, max_total_thickness, 'go', markersize=8)

# Add chord line for reference
plt.plot([0, b], [0, 0], 'k-', linewidth=1.5, alpha=1, label='Chord Line')



plt.xlabel('Chordwise Position (y) [m]', fontsize=12)
plt.ylabel('Thickness [m]', fontsize=12)
plt.title('NACA 2412 Airfoil: Thickness Distribution vs Chordwise Position', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, b)
plt.ylim(-0.05 * b, 0.3 * b)  # Adjusted y-axis range
plt.legend(loc='upper right')
plt.tight_layout()


# Plot 2
plt.figure(figsize=(7, 6))
# Airfoil surfaces
plt.plot(x, y_upper, 'b-', linewidth=2, label='Upper Surface')
plt.plot(x, y_lower, 'r-', linewidth=2, label=r'Lower Surface')

# Camber line
plt.plot(x, y_camber, 'k--', linewidth=1.5, label='Camber Line')

# Max camber indicator
plt.plot([max_camber_x, max_camber_x], [0, max_camber_val], 'm--', linewidth=1.5, label=f'Max Camber ({max_camber_val/b*100:.0f}%)')
plt.plot(max_camber_x, max_camber_val, 'mo')

# Max thickness indicator
plt.plot([max_thick_x, max_thick_x], [y_camber[np.argmax(y_thickness)] - max_thick_val, 
                                      y_camber[np.argmax(y_thickness)] + max_thick_val], 
         'g--', linewidth=1.5, label=f'Max Thickness ({max_thick_val/b*100:.0f}%)')
plt.plot(max_thick_x, y_upper[np.argmax(y_thickness)], 'go')
plt.plot(max_thick_x, y_lower[np.argmax(y_thickness)], 'go')

# Chord line
plt.plot([0, b], [0, 0], 'k-', linewidth=1, alpha=0.5, label='Chord Line')

# Annotations
plt.text(b/2, -0.1*b, f'Chord Length = {b}m', ha='center')
plt.text(max_camber_x, -0.12*b, f'Max Camber at {max_camber_x/b*100:.0f}% chord', ha='center', color='m')
plt.text(max_thick_x, -0.15*b, f'Max Thickness at {max_thick_x/b*100:.0f}% chord',  ha='center', color='g')

plt.xlabel('Chordwise Position (y) [m]', fontsize=12)
plt.ylabel('Camber & Thickness [m]', fontsize=12)
plt.title('NACA 2412 Airfoil Profile (Camber + Thickness)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, b)  # Fix x-axis from 0 to chord length
plt.ylim(-0.2 * b, 0.25 * b)  # Custom y-axis range (e.g., ±20% of chord)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
