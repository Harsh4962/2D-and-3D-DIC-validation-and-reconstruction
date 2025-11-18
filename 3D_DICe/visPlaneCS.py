

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # # ==============================
# # # LOAD DATA
# # # ==============================
# # DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
# # df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# # # Extract plane coordinates (MODEL_COORDINATES)
# # X = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
# # Y = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
# # Z = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

# # # Extract displacements (MODEL_DISPLACEMENT)
# # dX = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
# # dY = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
# # dZ = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# # # Calculate deformed positions
# # X_def = X + dX
# # Y_def = Y + dY
# # Z_def = Z + dZ

# # # Calculate displacement magnitude
# # U_mag = np.sqrt(dX**2 + dY**2 + dZ**2)

# # # ==============================
# # # HELPER FUNCTION
# # # ==============================
# # def set_axes_equal(ax):
# #     """Fix 3D axis scaling to equal for better spatial accuracy."""
# #     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
# #     centers = np.mean(limits, axis=1)
# #     max_range = np.max(np.ptp(limits, axis=1))
# #     for ctr, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
# #         set_lim([ctr - max_range/2, ctr + max_range/2])

# # # ==============================
# # # FIGURE 1: 3D Plane Coordinates
# # # ==============================
# # fig1 = plt.figure(figsize=(10, 8))
# # ax1 = fig1.add_subplot(111, projection='3d')

# # ax1.scatter(X, Y, Z, c='blue', s=2, alpha=0.6)
# # ax1.set_xlabel('X (mm)')
# # ax1.set_ylabel('Y (mm)')
# # ax1.set_zlabel('Z (mm)')
# # ax1.set_title('3D Points in Plane Coordinate System (Initial)')
# # ax1.view_init(elev=25, azim=-60)
# # set_axes_equal(ax1)

# # plt.tight_layout()
# # plt.savefig('fig1_plane_coordinates_3d.png', dpi=150, bbox_inches='tight')
# # plt.show()
# # print(" Saved: fig1_plane_coordinates_3d.png")

# # # ==============================
# # # FIGURE 2: Deformed Surface Colored by |U|
# # # ==============================
# # fig2 = plt.figure(figsize=(10, 8))
# # ax2 = fig2.add_subplot(111, projection='3d')

# # scatter = ax2.scatter(X_def, Y_def, Z_def, c=U_mag, cmap='turbo', s=2, alpha=0.8)
# # cbar = plt.colorbar(scatter, ax=ax2, shrink=0.7, pad=0.1)
# # cbar.set_label('|U| (mm)')

# # ax2.set_xlabel('X (mm)')
# # ax2.set_ylabel('Y (mm)')
# # ax2.set_zlabel('Z (mm)')
# # ax2.set_title('Deformed Surface Colored by Displacement Magnitude |U|')
# # ax2.view_init(elev=25, azim=-60)
# # set_axes_equal(ax2)

# # plt.tight_layout()
# # plt.savefig('fig2_deformed_surface_magnitude.png', dpi=150, bbox_inches='tight')
# # plt.show()
# # print("✓ Saved: fig2_deformed_surface_magnitude.png")

# # # ==============================
# # # FIGURE 3: Displacement Vector Field (Arrows)
# # # ==============================
# # fig3 = plt.figure(figsize=(10, 8))
# # ax3 = fig3.add_subplot(111, projection='3d')

# # # Sample points for cleaner visualization
# # step = max(1, len(X) // 400)
# # idx = np.arange(0, len(X), step)

# # # Plot arrows (displacement vectors)
# # ax3.quiver(X[idx], Y[idx], Z[idx], 
# #            dX[idx], dY[idx], dZ[idx],
# #            length=0.5, normalize=False, color='red', linewidth=0.5, alpha=0.7)

# # # Plot initial points
# # ax3.scatter(X[idx], Y[idx], Z[idx], s=5, color='black', alpha=0.6)

# # ax3.set_xlabel('X (mm)')
# # ax3.set_ylabel('Y (mm)')
# # ax3.set_zlabel('Z (mm)')
# # ax3.set_title('Displacement Vectors (Undeformed → Deformed)')
# # ax3.view_init(elev=25, azim=-60)
# # set_axes_equal(ax3)

# # plt.tight_layout()
# # plt.savefig('fig3_displacement_vectors.png', dpi=150, bbox_inches='tight')
# # plt.show()
# # print("✓ Saved: fig3_displacement_vectors.png")

# # # ==============================
# # # FIGURE 4: Displacement Heatmaps (Ux, Uy, Uz, |U|)
# # # ==============================
# # fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# # # Need to create a grid for heatmap visualization
# # # Use scatter plot with color mapping instead of regular heatmap

# # # Plot 1: Ux displacement
# # ax = axes[0, 0]
# # scatter1 = ax.scatter(X, Y, c=dX, cmap='RdBu', s=10, alpha=0.8)
# # cbar1 = plt.colorbar(scatter1, ax=ax)
# # cbar1.set_label('Ux (mm)')
# # ax.set_xlabel('X (mm)')
# # ax.set_ylabel('Y (mm)')
# # ax.set_title('X-Displacement (Ux)')
# # ax.set_aspect('equal')
# # ax.grid(True, alpha=0.3)

# # # Plot 2: Uy displacement
# # ax = axes[0, 1]
# # scatter2 = ax.scatter(X, Y, c=dY, cmap='RdBu', s=10, alpha=0.8)
# # cbar2 = plt.colorbar(scatter2, ax=ax)
# # cbar2.set_label('Uy (mm)')
# # ax.set_xlabel('X (mm)')
# # ax.set_ylabel('Y (mm)')
# # ax.set_title('Y-Displacement (Uy)')
# # ax.set_aspect('equal')
# # ax.grid(True, alpha=0.3)

# # # Plot 3: Uz displacement
# # ax = axes[1, 0]
# # scatter3 = ax.scatter(X, Y, c=dZ, cmap='RdBu', s=10, alpha=0.8)
# # cbar3 = plt.colorbar(scatter3, ax=ax)
# # cbar3.set_label('Uz (mm)')
# # ax.set_xlabel('X (mm)')
# # ax.set_ylabel('Y (mm)')
# # ax.set_title('Z-Displacement (Uz)')
# # ax.set_aspect('equal')
# # ax.grid(True, alpha=0.3)

# # # Plot 4: |U| magnitude
# # ax = axes[1, 1]
# # scatter4 = ax.scatter(X, Y, c=U_mag, cmap='turbo', s=10, alpha=0.8)
# # cbar4 = plt.colorbar(scatter4, ax=ax)
# # cbar4.set_label('|U| (mm)')
# # ax.set_xlabel('X (mm)')
# # ax.set_ylabel('Y (mm)')
# # ax.set_title('Displacement Magnitude |U|')
# # ax.set_aspect('equal')
# # ax.grid(True, alpha=0.3)

# # plt.tight_layout()
# # plt.savefig('fig4_displacement_heatmaps.png', dpi=150, bbox_inches='tight')
# # plt.show()
# # print(" Saved: fig4_displacement_heatmaps.png")

# # # ==============================
# # # STATISTICS
# # # ==============================
# # print("\n=== DISPLACEMENT STATISTICS ===")
# # print(f"Total points: {len(X)}")
# # print(f"\nUx (X-displacement):")
# # print(f"  Range: [{dX.min():.6f}, {dX.max():.6f}] mm")
# # print(f"  Mean: {dX.mean():.6f} mm")
# # print(f"  Std: {dX.std():.6f} mm")

# # print(f"\nUy (Y-displacement):")
# # print(f"  Range: [{dY.min():.6f}, {dY.max():.6f}] mm")
# # print(f"  Mean: {dY.mean():.6f} mm")
# # print(f"  Std: {dY.std():.6f} mm")

# # print(f"\nUz (Z-displacement):")
# # print(f"  Range: [{dZ.min():.6f}, {dZ.max():.6f}] mm")
# # print(f"  Mean: {dZ.mean():.6f} mm")
# # print(f"  Std: {dZ.std():.6f} mm")

# # print(f"\n|U| (Magnitude):")
# # print(f"  Range: [{U_mag.min():.6f}, {U_mag.max():.6f}] mm")
# # print(f"  Mean: {U_mag.mean():.6f} mm")
# # print(f"  Std: {U_mag.std():.6f} mm")

# # print("\n" + "="*50)
# # print(" All plots generated!")
# # print("="*50)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ------------------------------
# # Global style for larger, clear text
# # ------------------------------
# plt.rcParams.update({
#     "font.size": 10,
#     "axes.titlesize": 10,     # we will avoid axes titles in PNGs to prevent clipping
#     "axes.labelsize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "legend.fontsize": 9,
# })

# # ==============================
# # LOAD DATA
# # ==============================
# DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
# df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# # Extract plane coordinates (MODEL_COORDINATES)
# X = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
# Y = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
# Z = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

# # Extract displacements (MODEL_DISPLACEMENT)
# dX = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
# dY = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
# dZ = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# # Calculate deformed positions
# X_def = X + dX
# Y_def = Y + dY
# Z_def = Z + dZ

# # Displacement magnitude
# U_mag = np.sqrt(dX**2 + dY**2 + dZ**2)

# # ==============================
# # HELPERS
# # ==============================
# def set_axes_equal(ax):
#     """Set 3D axes to equal scale."""
#     import numpy as np
#     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     centers = np.mean(limits, axis=1)
#     max_range = np.max(np.ptp(limits, axis=1))
#     ax.set_xlim3d([centers[0] - max_range/2, centers[0] + max_range/2])
#     ax.set_ylim3d([centers[1] - max_range/2, centers[1] + max_range/2])
#     ax.set_zlim3d([centers[2] - max_range/2, centers[2] + max_range/2])

# def save_tight(fig, path, dpi=300, pad=0.02):
#     """Tight, high-DPI save to reduce whitespace and prevent clipping."""
#     fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad)

# # ==============================
# # FIGURE 1: 3D Plane Coordinates (Initial)
# # ==============================
# fig1 = plt.figure(figsize=(7.5, 5.8))  # larger canvas
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.scatter(X, Y, Z, c='tab:blue', s=3, alpha=0.75)
# ax1.set_xlabel('X (mm)')
# ax1.set_ylabel('Y (mm)')
# ax1.set_zlabel('Z (mm)')
# ax1.view_init(elev=28, azim=-55)
# set_axes_equal(ax1)
# plt.tight_layout()
# save_tight(fig1, 'fig1_plane_coordinates_3d.png')
# plt.close(fig1)
# print("✓ Saved: fig1_plane_coordinates_3d.png")

# # ==============================
# # FIGURE 2: Deformed Surface Colored by |U|
# # ==============================
# fig2 = plt.figure(figsize=(7.5, 5.8))
# ax2 = fig2.add_subplot(111, projection='3d')
# sc2 = ax2.scatter(X_def, Y_def, Z_def, c=U_mag, cmap='turbo', s=3, alpha=0.85)
# cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.80, pad=0.08)
# cbar2.set_label('|U| (mm)')
# ax2.set_xlabel('X (mm)')
# ax2.set_ylabel('Y (mm)')
# ax2.set_zlabel('Z (mm)')
# ax2.view_init(elev=28, azim=-55)
# set_axes_equal(ax2)
# plt.tight_layout()
# save_tight(fig2, 'fig2_deformed_surface_magnitude.png')
# plt.close(fig2)
# print("✓ Saved: fig2_deformed_surface_magnitude.png")

# # ==============================
# # FIGURE 3: Displacement Vector Field (Arrows)
# # ==============================
# fig3 = plt.figure(figsize=(7.5, 5.8))
# ax3 = fig3.add_subplot(111, projection='3d')
# step = max(1, len(X)//400)  # subsample for clarity
# idx = np.arange(0, len(X), step)
# ax3.quiver(X[idx], Y[idx], Z[idx],
#            dX[idx], dY[idx], dZ[idx],
#            length=0.7, normalize=False, color='crimson',
#            linewidth=0.6, alpha=0.85)
# ax3.scatter(X[idx], Y[idx], Z[idx], s=6, color='k', alpha=0.6)
# ax3.set_xlabel('X (mm)')
# ax3.set_ylabel('Y (mm)')
# ax3.set_zlabel('Z (mm)')
# ax3.view_init(elev=28, azim=-55)
# set_axes_equal(ax3)
# plt.tight_layout()
# save_tight(fig3, 'fig3_displacement_vectors.png')
# plt.close(fig3)
# print("✓ Saved: fig3_displacement_vectors.png")

# # ==============================
# # Separate planar maps: Ux, Uy, Uz, |U|
# # ==============================
# def planar_map(x, y, c, cmap, label, fname):
#     fig = plt.figure(figsize=(6.8, 5.4))
#     ax = fig.add_subplot(111)
#     sc = ax.scatter(x, y, c=c, cmap=cmap, s=12, alpha=0.9, edgecolor='none')
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label(label)
#     ax.set_xlabel('X (mm)')
#     ax.set_ylabel('Y (mm)')
#     ax.set_aspect('equal')
#     ax.grid(True, alpha=0.25, linestyle=':')
#     plt.tight_layout()
#     save_tight(fig, fname)
#     plt.close(fig)
#     print(f"✓ Saved: {fname}")

# planar_map(X, Y, dX, cmap='RdBu_r', label='Ux (mm)', fname='fig4a_disp_Ux.png')
# planar_map(X, Y, dY, cmap='RdBu_r', label='Uy (mm)', fname='fig4b_disp_Uy.png')
# planar_map(X, Y, dZ, cmap='RdBu_r', label='Uz (mm)', fname='fig4c_disp_Uz.png')
# planar_map(X, Y, U_mag, cmap='turbo', label='|U| (mm)', fname='fig4d_disp_Umag.png')

# # ==============================
# # Terminal-only STATS (optional, keep minimal)
# # ==============================
# def _stats(name, arr):
#     arr = np.asarray(arr)
#     print(f"{name}: count={arr.size}, range=[{arr.min():.6f}, {arr.max():.6f}] mm, "
#           f"mean={arr.mean():.6f} mm, std={arr.std():.6f} mm")

# print("\n=== DISPLACEMENT STATS (terminal) ===")
# _stats("Ux", dX)
# _stats("Uy", dY)
# _stats("Uz", dZ)
# _stats("|U|", U_mag)
# print("="*50)

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

# ------------------------------
# Terminal-only summary (optional)
# ------------------------------
def _stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: count={arr.size}, range=[{arr.min():.6f}, {arr.max():.6f}] mm, "
          f"mean={arr.mean():.6f} mm, std={arr.std():.6f} mm")

print("\n=== DISPLACEMENT STATS (terminal) ===")
_stats("Ux", dX); _stats("Uy", dY); _stats("Uz", dZ); _stats("|U|", U_mag)
print("=" * 50)
