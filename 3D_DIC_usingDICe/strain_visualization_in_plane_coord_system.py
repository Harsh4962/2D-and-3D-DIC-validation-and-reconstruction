import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# LOAD DATA
# ==============================
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"

df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# Extract 3D plane coordinates
X_plane = df['MODEL_COORDINATES_X'].astype(float).to_numpy()
Y_plane = df['MODEL_COORDINATES_Y'].astype(float).to_numpy()
Z_plane = df['MODEL_COORDINATES_Z'].astype(float).to_numpy()

# Extract strain components
strain_xx = df['VSG_STRAIN_XX'].astype(float).to_numpy()
strain_yy = df['VSG_STRAIN_YY'].astype(float).to_numpy()
strain_xy = df['VSG_STRAIN_XY'].astype(float).to_numpy()

# Von Mises
strain_eq = np.sqrt(strain_xx**2 - strain_xx*strain_yy + 
                    strain_yy**2 + 3*strain_xy**2)

print(f"Loaded {len(strain_xx)} points")
print(f"\nAxis Ranges:")
print(f"  X: [{X_plane.min():.2f}, {X_plane.max():.2f}] mm (range: {X_plane.max()-X_plane.min():.2f} mm)")
print(f"  Y: [{Y_plane.min():.2f}, {Y_plane.max():.2f}] mm (range: {Y_plane.max()-Y_plane.min():.2f} mm)")
print(f"  Z: [{Z_plane.min():.2f}, {Z_plane.max():.2f}] mm (range: {Z_plane.max()-Z_plane.min():.2f} mm)")

# ==============================
# FIX: SET EQUAL ASPECT RATIO
# ==============================

# Get the ranges
x_range = X_plane.max() - X_plane.min()
y_range = Y_plane.max() - Y_plane.min()
z_range = Z_plane.max() - Z_plane.min()

# Find the maximum range to make a cube
max_range = max(x_range, y_range, z_range)

# Get the midpoints
x_mid = (X_plane.max() + X_plane.min()) * 0.5
y_mid = (Y_plane.max() + Y_plane.min()) * 0.5
z_mid = (Z_plane.max() + Z_plane.min()) * 0.5

print(f"\nScaling Factor: {max_range:.2f} mm")

# ==============================
# PLOT 1: εxx - Proper aspect ratio
# ==============================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_plane, Y_plane, Z_plane, c=strain_xx, 
                     cmap='RdBu_r', s=30, alpha=0.6, edgecolors='none')

# SET EQUAL ASPECT RATIO
ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
ax.set_title(r'$\varepsilon_{xx}$ in 3D (Equal Aspect Ratio)', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('εxx Strain', fontsize=11)

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# ==============================
# PLOT 2: εyy - Proper aspect ratio
# ==============================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_plane, Y_plane, Z_plane, c=strain_yy, 
                     cmap='RdBu_r', s=30, alpha=0.6, edgecolors='none')

ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
ax.set_title(r'$\varepsilon_{yy}$ in 3D (Equal Aspect Ratio)', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('εyy Strain', fontsize=11)

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# ==============================
# PLOT 3: εxy - Proper aspect ratio
# ==============================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_plane, Y_plane, Z_plane, c=strain_xy, 
                     cmap='RdBu_r', s=30, alpha=0.6, edgecolors='none')

ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
ax.set_title(r'$\varepsilon_{xy}$ in 3D (Equal Aspect Ratio)', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('εxy Strain', fontsize=11)

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# ==============================
# PLOT 4: Von Mises - Proper aspect ratio
# ==============================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_plane, Y_plane, Z_plane, c=strain_eq, 
                     cmap='viridis', s=30, alpha=0.6, edgecolors='none')

ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
ax.set_title(r'Von Mises $\varepsilon_{eq}$ in 3D (Equal Aspect Ratio)', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Equivalent Strain', fontsize=11)

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# ==============================
# BONUS: Different viewing angles
# ==============================

angles = [
    (20, 45, "View 1 (45°)"),
    (20, 135, "View 2 (135°)"),
    (20, 225, "View 3 (225°)"),
    (90, 0, "Top View (90°)")
]

for elev, azim, title_suffix in angles:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_plane, Y_plane, Z_plane, c=strain_eq, 
                         cmap='viridis', s=30, alpha=0.6, edgecolors='none')
    
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Von Mises Strain - {title_suffix}', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Equivalent Strain', fontsize=11)
    
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    plt.show()

# ==============================
# STATISTICS
# ==============================
print("\n" + "="*70)
print("AXIS SCALING CORRECTION")
print("="*70)
print(f"\nOriginal ranges:")
print(f"  X range: {x_range:.2f} mm")
print(f"  Y range: {y_range:.2f} mm")
print(f"  Z range: {z_range:.2f} mm")
print(f"\nMaximum range (used for equal aspect): {max_range:.2f} mm")
print(f"\nAll axes now have equal scale for proper 3D visualization ")
