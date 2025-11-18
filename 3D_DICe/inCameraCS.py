# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# # # ==============================
# # # PATH SETTINGS
# # # ==============================
# # DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"


# # # ==============================
# # # TRANSFORMATION MATRIX (Specimen CS -> Camera CS)
# # # ==============================
# # R = np.array([
# #     [9.853007e-001, 4.568292e-003, -1.707676e-001],
# #     [4.191086e-003, -9.999879e-001, -2.569322e-003],
# #     [-1.707773e-001, 1.815853e-003, -9.853080e-001]
# # ])

# # t = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])


# # # ==============================
# # #  LOAD DATA
# # # ==============================
# # def load_dic_data(path):
# #     try:
# #         df = pd.read_csv(path, sep=r'\s+|,', engine='python')
# #     except Exception:
# #         raise RuntimeError("Could not read DICe data file")
# #     return df


# # df = load_dic_data(DATA_PATH)
# # print("Columns found:", list(df.columns))


# # # ==============================
# # #  EXTRACT COORDINATES AND DISPLACEMENTS
# # # ==============================
# # cx, cy, cz = "MODEL_COORDINATES_X", "MODEL_COORDINATES_Y", "MODEL_COORDINATES_Z"
# # ux, uy, uz = "MODEL_DISPLACEMENT_X", "MODEL_DISPLACEMENT_Y", "MODEL_DISPLACEMENT_Z"


# # X = df[cx].astype(float).to_numpy()
# # Y = df[cy].astype(float).to_numpy()
# # Z = df[cz].astype(float).to_numpy()
# # dX = df[ux].astype(float).to_numpy()
# # dY = df[uy].astype(float).to_numpy()
# # dZ = df[uz].astype(float).to_numpy()


# # # undeformed & deformed points in SPECIMEN CS
# # P0_specimen = np.vstack([X, Y, Z]).T
# # P1_specimen = np.vstack([X + dX, Y + dY, Z + dZ]).T
# # U = np.vstack([dX, dY, dZ]).T
# # Umag = np.linalg.norm(U, axis=1)


# # print(f"\n=== SPECIMEN CS ===")
# # print(f"Z range: {Z.min():.6f} to {Z.max():.6f}")
# # print(f"Displacement magnitude range: {Umag.min():.6f} to {Umag.max():.6f} mm")


# # # ==============================
# # # TRANSFORM TO CAMERA CS
# # # ==============================
# # # Transform: p_camera = R @ p_specimen + t
# # P0_camera = (R @ P0_specimen.T).T + t
# # P1_camera = (R @ P1_specimen.T).T + t

# # print(f"\n=== CAMERA CS ===")
# # print(f"X range: {P0_camera[:,0].min():.6f} to {P0_camera[:,0].max():.6f}")
# # print(f"Y range: {P0_camera[:,1].min():.6f} to {P0_camera[:,1].max():.6f}")
# # print(f"Z range: {P0_camera[:,2].min():.6f} to {P0_camera[:,2].max():.6f}")


# # # ==============================
# # #  VISUALIZATION HELPERS
# # # ==============================
# # def set_axes_equal(ax):
# #     """Fix 3D axis scaling to equal for better spatial accuracy."""
# #     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
# #     centers = np.mean(limits, axis=1)
# #     max_range = np.max(np.ptp(limits, axis=1))
# #     for ctr, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
# #         set_lim([ctr - max_range/2, ctr + max_range/2])


# # # ==============================
# # # PLOT 1: SPECIMEN CS - Initial (undeformed)
# # # ==============================
# # fig = plt.figure(figsize=(7,6))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(P0_specimen[:,0], P0_specimen[:,1], P0_specimen[:,2], s=4, color='gray')
# # ax.set_title("Specimen CS: Initial (undeformed) surface")
# # ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
# # ax.view_init(elev=35, azim=-60)
# # set_axes_equal(ax)
# # plt.tight_layout()
# # plt.show()


# # # ==============================
# # # PLOT 2: CAMERA CS - Initial (undeformed)
# # # ==============================
# # fig = plt.figure(figsize=(7,6))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(P0_camera[:,0], P0_camera[:,1], P0_camera[:,2], s=4, color='blue')
# # ax.set_title("Camera CS: Initial (undeformed) surface")
# # ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
# # ax.view_init(elev=35, azim=-60)
# # # ax.view_init(elev=0, azim=0)
# # set_axes_equal(ax)
# # plt.tight_layout()
# # plt.show()


# # # ==============================
# # # PLOT 3: CAMERA CS - Deformed colored by |U|
# # # ==============================
# # fig = plt.figure(figsize=(7,6))
# # ax = fig.add_subplot(111, projection='3d')
# # sc = ax.scatter(P1_camera[:,0], P1_camera[:,1], P1_camera[:,2], c=Umag, cmap='turbo', s=5)
# # plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label='|U| [mm]')
# # ax.set_title("Camera CS: Deformed surface colored by |U|")
# # ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
# # ax.view_init(elev=35, azim=-60)
# # set_axes_equal(ax)
# # plt.tight_layout()
# # plt.show()


# # # ==============================
# # # PLOT 4: CAMERA CS - Displacement vectors
# # # ==============================
# # # Calculate displacement in camera CS
# # U_camera = P1_camera - P0_camera

# # fig = plt.figure(figsize=(7,6))
# # ax = fig.add_subplot(111, projection='3d')

# # step = max(1, len(P0_camera)//400)
# # idx = np.arange(0, len(P0_camera), step)

# # ax.quiver(P0_camera[idx,0], P0_camera[idx,1], P0_camera[idx,2],
# #           U_camera[idx,0], U_camera[idx,1], U_camera[idx,2],
# #           length=0.5, normalize=False, color='red', linewidth=0.5)

# # ax.scatter(P0_camera[idx,0], P0_camera[idx,1], P0_camera[idx,2], s=5, color='k', alpha=0.6)
# # ax.set_title("Camera CS: Displacement vectors (undeformed → deformed)")
# # ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
# # ax.view_init(elev=35, azim=-60)
# # set_axes_equal(ax)
# # plt.tight_layout()
# # plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # ==============================
# # TRANSFORMATION & PLANE DATA
# # ==============================
# R = np.array([
#     [9.853007e-001, 4.568292e-003, -1.707676e-001],
#     [4.191086e-003, -9.999879e-001, -2.569322e-003],
#     [-1.707773e-001, 1.815853e-003, -9.853080e-001]
# ])

# t = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])

# # Best fit plane data
# origin_camera = np.array([-3.872996, -3.429066, 55.40866])
# x_axis_point_camera = np.array([2.836159, -3.397959, 54.24586])

# # Plane equation: -z = c[0]*x + c[1]*y + c[2]
# # Rearranged: z = -c[0]*x - c[1]*y - c[2]
# plane_coeffs = np.array([1.733238e-001, -1.842929e-003, -5.474370e+001])

# # ==============================
# # LOAD  DATA
# # ==============================
# DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
# df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python') # "," since data is comma seperated

# cx, cy, cz = "MODEL_COORDINATES_X", "MODEL_COORDINATES_Y", "MODEL_COORDINATES_Z"
# # converting coordinates to ararys
# X = df[cx].astype(float).to_numpy()
# Y = df[cy].astype(float).to_numpy()
# Z = df[cz].astype(float).to_numpy()

# P0_specimen = np.vstack([X, Y, Z]).T

# # vector transformation specimen CS <--> camera CS
# P0_camera = (R @ P0_specimen.T).T + t

# # ==============================
# # VISUALIZATION HELPER
# # ==============================
# def set_axes_equal(ax):
#     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     centers = np.mean(limits, axis=1)
#     max_range = np.max(np.ptp(limits, axis=1))
#     for ctr, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
#         set_lim([ctr - max_range/2, ctr + max_range/2])

# # ==============================
# # CREATE PLANE MESH
# # ==============================
# def create_plane_mesh(origin, coeffs, size=20, resolution=10):
#     """
#     Create a mesh grid for the best fit plane
#     Plane equation: z = -c[0]*x - c[1]*y - c[2]
#     """
#     c0, c1, c2 = coeffs
    
#     # Create grid centered around the origin
#     x_range = np.linspace(origin[0] - size, origin[0] + size, resolution)
#     y_range = np.linspace(origin[1] - size, origin[1] + size, resolution)
#     X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
#     # Calculate z from plane equation: z = -c[0]*x - c[1]*y - c[2]
#     Z_grid = -c0 * X_grid - c1 * Y_grid - c2
    
#     return X_grid, Y_grid, Z_grid

# # ==============================
# # PLOT: Camera CS with Best Fit Plane
# # ==============================
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 1. Plot the surface points
# ax.scatter(P0_camera[:,0], P0_camera[:,1], P0_camera[:,2], 
#            s=2, color='blue', alpha=0.5, label='Surface points')

# # 2. Plot the best fit plane origin (red point)
# ax.scatter(*origin_camera, s=100, color='red', marker='o', 
#            label='Best fit origin', edgecolors='black', linewidths=2)

# # 3. Plot the x-axis reference point (green point)
# ax.scatter(*x_axis_point_camera, s=100, color='green', marker='^',
#            label='X-axis point', edgecolors='black', linewidths=2)

# # 4. Draw axis line from origin to x-axis point
# ax.plot([origin_camera[0], x_axis_point_camera[0]],
#         [origin_camera[1], x_axis_point_camera[1]],
#         [origin_camera[2], x_axis_point_camera[2]],
#         'g--', linewidth=2, label='X-axis direction')

# # 5. Plot the best fit plane as a transparent surface
# X_grid, Y_grid, Z_grid = create_plane_mesh(origin_camera, plane_coeffs, size=15, resolution=20)
# ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='yellow', 
#                 edgecolor='orange', linewidth=0.5)

# # Labels and formatting
# ax.set_xlabel("X [mm]")
# ax.set_ylabel("Y [mm]")
# ax.set_zlabel("Z [mm]")
# ax.set_title("Camera CS: Surface with Best Fit Plane")
# ax.legend(loc='upper right')
# # ax.view_init(elev=25, azim=-70)
# ax.view_init(elev=0, azim=-70)
# set_axes_equal(ax)

# plt.tight_layout()
# plt.show()

# # ==============================
# # ALTERNATIVE: Side-by-side comparison
# # ==============================
# fig = plt.figure(figsize=(16, 6))

# # Left plot: Without plane
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(P0_camera[:,0], P0_camera[:,1], P0_camera[:,2], s=2, color='blue', alpha=0.6)
# ax1.scatter(*origin_camera, s=150, color='red', marker='o', edgecolors='black', linewidths=2)
# ax1.set_xlabel("X [mm]"); ax1.set_ylabel("Y [mm]"); ax1.set_zlabel("Z [mm]")
# ax1.set_title("Camera CS: Surface Points + Origin")
# ax1.view_init(elev=25, azim=-70)
# set_axes_equal(ax1)

# # Right plot: With plane
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(P0_camera[:,0], P0_camera[:,1], P0_camera[:,2], s=2, color='blue', alpha=0.4)
# ax2.scatter(*origin_camera, s=150, color='red', marker='o', edgecolors='black', linewidths=2)
# X_grid, Y_grid, Z_grid = create_plane_mesh(origin_camera, plane_coeffs, size=15, resolution=20)
# ax2.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.4, color='yellow', edgecolor='orange', linewidth=0.3)
# ax2.set_xlabel("X [mm]"); ax2.set_ylabel("Y [mm]"); ax2.set_zlabel("Z [mm]")
# ax2.set_title("Camera CS: Surface + Best Fit Plane")
# ax2.view_init(elev=25, azim=-70)
# set_axes_equal(ax2)

# plt.tight_layout()
# plt.show()

# print(f"\norigin point at: ({origin_camera[0]:.3f}, {origin_camera[1]:.3f}, {origin_camera[2]:.3f})")
# print(f"plane equation: z = -{plane_coeffs[0]:.4f}*x - {plane_coeffs[1]:.4f}*y - {plane_coeffs[2]:.4f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET

# ==============================
# 1. PARSE CAMERA CALIBRATION
# ==============================
def parse_camera_xml(xml_path):
    """Parse DICe camera calibration XML file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    cameras = {}
    
    for cam_list in root.findall('.//ParameterList[@name]'):
        cam_name = cam_list.get('name')
        if 'CAMERA' not in cam_name:
            continue
            
        cam_id = int(cam_name.split()[-1])
        cameras[cam_id] = {}
        
        for param in cam_list.findall('Parameter'):
            name = param.get('name')
            ptype = param.get('type')
            value = param.get('value')
            
            if ptype == 'double':
                cameras[cam_id][name] = float(value)
            elif name == 'IMAGE_HEIGHT_WIDTH':
                hw = value.strip('{}').split(',')
                cameras[cam_id]['HEIGHT'] = int(hw[0])
                cameras[cam_id]['WIDTH'] = int(hw[1])
        
        rot_list = cam_list.find('.//ParameterList[@name="rotation_3x3_matrix"]')
        if rot_list is not None:
            R = []
            for row_param in rot_list.findall('Parameter'):
                row_str = row_param.get('value')
                row_values = [float(x) for x in row_str.strip('{}').split(',')]
                R.append(row_values)
            cameras[cam_id]['R'] = np.array(R)
    
    return cameras

# ==============================
# 2. TRANSFORMATION FUNCTIONS
# ==============================
def plane_to_world_coords(points_plane, R_world2plane, t_world2plane):
    """Transform points from plane frame to world frame (Camera 0 CS)"""
    R_plane2world = R_world2plane.T
    t_plane2world = -R_world2plane.T @ t_world2plane
    
    return (R_plane2world @ points_plane.T).T + t_plane2world

def world_to_camera1_coords(points_world, cam1_params):
    """Transform points from world frame (Camera 0) to Camera 1 frame"""
    R1 = cam1_params['R']
    t1 = np.array([cam1_params['TX'], cam1_params['TY'], cam1_params['TZ']])
    
    return (R1 @ points_world.T).T + t1

# ==============================
# HELPER FUNCTION
# ==============================
def set_axes_equal(ax):
    """Fix 3D axis scaling to equal for better spatial accuracy."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    max_range = np.max(np.ptp(limits, axis=1))
    for ctr, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim([ctr - max_range/2, ctr + max_range/2])

# ==============================
# LOAD DATA
# ==============================
CAMERA_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"

# Transformation (plane → world)
R_world2plane = np.array([
    [9.853007e-001, 4.568292e-003, -1.707676e-001],
    [4.191086e-003, -9.999879e-001, -2.569322e-003],
    [-1.707773e-001, 1.815853e-003, -9.853080e-001]
])
t_world2plane = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])

# Load calibration
cameras = parse_camera_xml(CAMERA_XML)
cam0 = cameras[0]
cam1 = cameras[1]

# Load DICe data
df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# Extract coordinates (in plane frame)
X_plane = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
Y_plane = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
Z_plane = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()
points_plane = np.vstack([X_plane, Y_plane, Z_plane]).T

# Extract displacements (in plane frame)
dX_plane = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
dY_plane = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
dZ_plane = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()
disp_plane = np.vstack([dX_plane, dY_plane, dZ_plane]).T

print("=== COORDINATE TRANSFORMATIONS ===")
print(f"Loaded {len(points_plane)} points from plane frame")

# ==============================
# TRANSFORM: Plane → World (Camera 0)
# ==============================
print("\nTransforming plane → Camera 0 (world) coordinates...")
points_cam0 = plane_to_world_coords(points_plane, R_world2plane, t_world2plane)
disp_cam0 = (R_world2plane.T @ disp_plane.T).T  # Displacements also transform

print(f"✓ Transformed to Camera 0 coordinates")
print(f"  X range: [{points_cam0[:,0].min():.3f}, {points_cam0[:,0].max():.3f}]")
print(f"  Y range: [{points_cam0[:,1].min():.3f}, {points_cam0[:,1].max():.3f}]")
print(f"  Z range: [{points_cam0[:,2].min():.3f}, {points_cam0[:,2].max():.3f}]")

# ==============================
# TRANSFORM: World (Camera 0) → Camera 1
# ==============================
print("\nTransforming Camera 0 → Camera 1 coordinates...")
points_cam1 = world_to_camera1_coords(points_cam0, cam1)
disp_cam1 = (cam1['R'] @ disp_cam0.T).T  # Displacements also transform

print(f"✓ Transformed to Camera 1 coordinates")
print(f"  X range: [{points_cam1[:,0].min():.3f}, {points_cam1[:,0].max():.3f}]")
print(f"  Y range: [{points_cam1[:,1].min():.3f}, {points_cam1[:,1].max():.3f}]")
print(f"  Z range: [{points_cam1[:,2].min():.3f}, {points_cam1[:,2].max():.3f}]")

# Calculate displacement magnitudes
U_mag_plane = np.linalg.norm(disp_plane, axis=1)
U_mag_cam0 = np.linalg.norm(disp_cam0, axis=1)
U_mag_cam1 = np.linalg.norm(disp_cam1, axis=1)

# ==============================
# FIGURE 1: PLANE COORDINATES
# ==============================
print("\n=== GENERATING PLOTS ===")
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

scatter1 = ax1.scatter(X_plane, Y_plane, Z_plane, c=U_mag_plane, cmap='turbo', s=2, alpha=0.8)
cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.7, pad=0.1)
cbar1.set_label('|U| (mm)')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('Plane Coordinate System\n(Best-fit Plane Frame)')
ax1.view_init(elev=25, azim=-60)
set_axes_equal(ax1)

plt.tight_layout()
plt.savefig('plane_coordinates_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: plane_coordinates_3d.png")

# ==============================
# FIGURE 2: CAMERA 0 COORDINATES
# ==============================
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

scatter2 = ax2.scatter(points_cam0[:,0], points_cam0[:,1], points_cam0[:,2], 
                       c=U_mag_cam0, cmap='turbo', s=2, alpha=0.8)
cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7, pad=0.1)
cbar2.set_label('|U| (mm)')

ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')
ax2.set_title('Camera 0 Coordinate System\n(World Frame)')
ax2.view_init(elev=25, azim=-60)
set_axes_equal(ax2)

plt.tight_layout()
plt.savefig('camera0_coordinates_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: camera0_coordinates_3d.png")

# ==============================
# FIGURE 3: CAMERA 1 COORDINATES
# ==============================
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')

scatter3 = ax3.scatter(points_cam1[:,0], points_cam1[:,1], points_cam1[:,2], 
                       c=U_mag_cam1, cmap='turbo', s=2, alpha=0.8)
cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.7, pad=0.1)
cbar3.set_label('|U| (mm)')

ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_zlabel('Z (mm)')
ax3.set_title('Camera 1 Coordinate System\n(Relative to Camera 0)')
ax3.view_init(elev=25, azim=-60)
set_axes_equal(ax3)

plt.tight_layout()
plt.savefig('camera1_coordinates_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: camera1_coordinates_3d.png")

# ==============================
# FIGURE 4: SIDE-BY-SIDE COMPARISON
# ==============================
fig4 = plt.figure(figsize=(18, 6))

# Plane coordinates
ax4a = fig4.add_subplot(131, projection='3d')
scatter4a = ax4a.scatter(X_plane, Y_plane, Z_plane, c=U_mag_plane, cmap='turbo', s=2, alpha=0.8)
plt.colorbar(scatter4a, ax=ax4a, shrink=0.7, pad=0.1)
ax4a.set_xlabel('X (mm)')
ax4a.set_ylabel('Y (mm)')
ax4a.set_zlabel('Z (mm)')
ax4a.set_title('Plane Frame')
ax4a.view_init(elev=25, azim=-60)
set_axes_equal(ax4a)

# Camera 0 coordinates
ax4b = fig4.add_subplot(132, projection='3d')
scatter4b = ax4b.scatter(points_cam0[:,0], points_cam0[:,1], points_cam0[:,2], 
                         c=U_mag_cam0, cmap='turbo', s=2, alpha=0.8)
plt.colorbar(scatter4b, ax=ax4b, shrink=0.7, pad=0.1)
ax4b.set_xlabel('X (mm)')
ax4b.set_ylabel('Y (mm)')
ax4b.set_zlabel('Z (mm)')
ax4b.set_title('Camera 0 (World)')
ax4b.view_init(elev=25, azim=-60)
set_axes_equal(ax4b)

# Camera 1 coordinates
ax4c = fig4.add_subplot(133, projection='3d')
scatter4c = ax4c.scatter(points_cam1[:,0], points_cam1[:,1], points_cam1[:,2], 
                         c=U_mag_cam1, cmap='turbo', s=2, alpha=0.8)
plt.colorbar(scatter4c, ax=ax4c, shrink=0.7, pad=0.1)
ax4c.set_xlabel('X (mm)')
ax4c.set_ylabel('Y (mm)')
ax4c.set_zlabel('Z (mm)')
ax4c.set_title('Camera 1 (Stereo)')
ax4c.view_init(elev=25, azim=-60)
set_axes_equal(ax4c)

plt.tight_layout()
plt.savefig('all_coordinates_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: all_coordinates_comparison.png")

# ==============================
# STATISTICS
# ==============================
print("\n=== COORDINATE STATISTICS ===")

print("\nPlane Frame:")
print(f"  X: [{X_plane.min():.3f}, {X_plane.max():.3f}] mm")
print(f"  Y: [{Y_plane.min():.3f}, {Y_plane.max():.3f}] mm")
print(f"  Z: [{Z_plane.min():.3f}, {Z_plane.max():.3f}] mm")

print("\nCamera 0 Frame (World):")
print(f"  X: [{points_cam0[:,0].min():.3f}, {points_cam0[:,0].max():.3f}] mm")
print(f"  Y: [{points_cam0[:,1].min():.3f}, {points_cam0[:,1].max():.3f}] mm")
print(f"  Z: [{points_cam0[:,2].min():.3f}, {points_cam0[:,2].max():.3f}] mm")

print("\nCamera 1 Frame (Stereo):")
print(f"  X: [{points_cam1[:,0].min():.3f}, {points_cam1[:,0].max():.3f}] mm")
print(f"  Y: [{points_cam1[:,1].min():.3f}, {points_cam1[:,1].max():.3f}] mm")
print(f"  Z: [{points_cam1[:,2].min():.3f}, {points_cam1[:,2].max():.3f}] mm")

print("\n" + "="*50)
print("All transformation and visualization complete!")
print("="*50)
