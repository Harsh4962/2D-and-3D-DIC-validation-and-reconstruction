import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Bigger fonts for on-screen viewing
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ==============================
# LOAD DATA
# ==============================
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# Extract model coordinates and displacements
X  = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
Y  = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
Z  = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()
dX = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
dY = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
dZ = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# Deformed positions and magnitude
X_def, Y_def, Z_def = X + dX, Y + dY, Z + dZ
U_mag = np.sqrt(dX**2 + dY**2 + dZ**2)

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    max_range = np.max(np.ptp(limits, axis=1))
    ax.set_xlim3d([centers[0] - max_range/2, centers[0] + max_range/2])
    ax.set_ylim3d([centers[1] - max_range/2, centers[1] + max_range/2])
    ax.set_zlim3d([centers[2] - max_range/2, centers[2] + max_range/2])

# ------------------------------
# FIG 1: Initial 3D point cloud
# ------------------------------
fig1 = plt.figure(figsize=(9.5, 7.2))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(X, Y, Z, c='tab:blue', s=6, alpha=0.8)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.view_init(elev=28, azim=-55)
set_axes_equal(ax1)
plt.tight_layout()
plt.show()

# ------------------------------
# FIG 2: Deformed colored by |U|
# ------------------------------
fig2 = plt.figure(figsize=(9.5, 7.2))
ax2 = fig2.add_subplot(111, projection='3d')
sc2 = ax2.scatter(X_def, Y_def, Z_def, c=U_mag, cmap='turbo', s=6, alpha=0.9)
cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.85, pad=0.06)
cbar2.set_label('|U| (mm)')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')
ax2.view_init(elev=28, azim=-55)
set_axes_equal(ax2)
plt.tight_layout()
plt.show()

# ------------------------------
# FIG 3: 3D displacement vectors
# ------------------------------
fig3 = plt.figure(figsize=(9.5, 7.2))
ax3 = fig3.add_subplot(111, projection='3d')
step = max(1, len(X)//400)
idx = np.arange(0, len(X), step)
ax3.quiver(X[idx], Y[idx], Z[idx],
           dX[idx], dY[idx], dZ[idx],
           length=0.8, normalize=False, color='crimson',
           linewidth=0.7, alpha=0.9)
ax3.scatter(X[idx], Y[idx], Z[idx], s=10, color='k', alpha=0.65)
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_zlabel('Z (mm)')
ax3.view_init(elev=28, azim=-55)
set_axes_equal(ax3)
plt.tight_layout()
plt.show()

# ------------------------------
# Separate planar maps: Ux, Uy, Uz, |U|
# ------------------------------
def planar_map(x, y, c, cmap, label, title=None):
    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=18, alpha=0.95, edgecolor='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linestyle=':')
    if title:
        ax.set_title(title, pad=6)
    plt.tight_layout()
    plt.show()

planar_map(X, Y, dX, cmap='RdBu_r', label='Ux (mm)', title='X-Displacement (Ux)')
planar_map(X, Y, dY, cmap='RdBu_r', label='Uy (mm)', title='Y-Displacement (Uy)')
planar_map(X, Y, dZ, cmap='RdBu_r', label='Uz (mm)', title='Z-Displacement (Uz)')
planar_map(X, Y, U_mag, cmap='turbo', label='|U| (mm)', title='Displacement Magnitude |U|')


def _stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: count={arr.size}, range=[{arr.min():.6f}, {arr.max():.6f}] mm, "
          f"mean={arr.mean():.6f} mm, std={arr.std():.6f} mm")

print("\n=== DISPLACEMENT STATS (terminal) ===")
_stats("Ux", dX); _stats("Uy", dY); _stats("Uz", dZ); _stats("|U|", U_mag)
print("=" * 50)
