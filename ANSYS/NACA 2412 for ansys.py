"""Create properly ordered airfoil coordinates for ANSYS with dense points at leading edge."""

import os
import numpy as np
import matplotlib.pyplot as plt

def fix_airfoil_coordinates():
    # Generate fresh coordinates with proper ordering
    chord = 0.1
    span = 0.15
    
    # NACA 2412 parameters
    m = 0.02; p = 0.40; t = 0.12
    
    # Generate points with higher density at leading edge (0-20% chord)
    n_points = 50  
    
    # Create non-uniform spacing with more points near leading edge (x=0)
    # Cosine spacing gives more points at ends (leading and trailing edges)
    x_original = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_points)))
    
    # Calculate camber and thickness
    yc = np.zeros_like(x_original)
    dyc_dx = np.zeros_like(x_original)
    
    for i, x in enumerate(x_original):
        if x <= p:
            yc[i] = (m/p**2) * (2*p*x - x**2)
            dyc_dx[i] = (2*m/p**2) * (p - x)
        else:
            yc[i] = (m/(1-p)**2) * ((1 - 2*p) + 2*p*x - x**2)
            dyc_dx[i] = (2*m/(1-p)**2) * (p - x)
    
    yt = 5*t * (0.2969*np.sqrt(x_original) - 0.1260*x_original - 0.3516*x_original**2 + 
                0.2843*x_original**3 - 0.1015*x_original**4)
    
    theta = np.arctan(dyc_dx)
    
    # Upper surface (from trailing to leading edge)
    xu_upper = x_original - yt * np.sin(theta)
    yu_upper = yc + yt * np.cos(theta)
    
    # Lower surface (from leading to trailing edge)  
    xl_lower = x_original + yt * np.sin(theta)
    yl_lower = yc - yt * np.cos(theta)
    
    # Scale to chord length (in meters)
    xu_upper = xu_upper * chord 
    yu_upper = yu_upper * chord 
    xl_lower = xl_lower * chord 
    yl_lower = yl_lower * chord 
    
    # Create properly ordered points: Lower surface → Upper surface
    # Reverse lower surface to go from TE to LE, then upper surface from LE to TE
    x_2d = np.concatenate([xl_lower[::-1], xu_upper[1:]])
    y_2d = np.concatenate([yl_lower[::-1], yu_upper[1:]])
    
    # Ensure the curve is closed (first and last points match)
    if not np.allclose([x_2d[0], y_2d[0]], [x_2d[-1], y_2d[-1]]):
        x_2d = np.append(x_2d, x_2d[0])
        y_2d = np.append(y_2d, y_2d[0])
    
    return x_2d, y_2d

def plot_airfoil(x_2d, y_2d, chord=0.1):
    """
    Plot the generated airfoil for verification
    """
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot airfoil profile in meters
    ax.plot(x_2d, y_2d, 'b-', linewidth=2, label='NACA 2412 Profile')
    ax.plot(x_2d, y_2d, 'ro', markersize=2, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('NACA 2412 Airfoil Profile\n(Chord: {:.3f} m)'.format(chord))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Modified main execution to include plotting
def create_fixed_ansys_file_with_plots():
    x_2d, y_2d = fix_airfoil_coordinates()
    
    export_dir = r'C:\Users\saroj\Desktop\Vibration\ANSYS'
    os.makedirs(export_dir, exist_ok=True)
    
    # Create TXT file for ANSYS compatibility
    txt_file_path = os.path.join(export_dir, 'naca2412_fixed_points.txt')
    
    # Write coordinates to TXT file in the specified format (in meters)
    with open(txt_file_path, 'w') as f:
        for i in range(len(x_2d)):
            # Format: 1   point_number   X   Y   Z (all in meters)
            f.write("1\t{}\t{:.6f}\t{:.6f}\t0.000000\n".format(i+1, x_2d[i], y_2d[i]))
    
    print("File created:")
    print(f"  TXT: naca2412_fixed_points.txt")
    print(f"Total points: {len(x_2d)}")
    
    # Count points in leading edge region (first 20% chord)
    le_points = np.sum(x_2d <= 0.02)  # 20% of 0.1m chord = 0.02m
    print(f"Points in leading edge region (first 20%): {le_points}")
    
    # Calculate airfoil statistics
    max_thickness = np.max(y_2d) - np.min(y_2d)
    chord_length = np.max(x_2d) - np.min(x_2d)
    thickness_ratio = (max_thickness / chord_length) * 100
    
    print(f"\nAirfoil Statistics:")
    print(f"  Chord length: {chord_length:.4f} m")
    print(f"  Max thickness: {max_thickness:.4f} m")
    print(f"  Thickness ratio: {thickness_ratio:.1f}%")
    
    # Create plot
    plot_airfoil(x_2d, y_2d)
    
    return txt_file_path

# Run the complete function with plots
print("Generating NACA 2412 airfoil with dense leading edge points...")
txt_path = create_fixed_ansys_file_with_plots()