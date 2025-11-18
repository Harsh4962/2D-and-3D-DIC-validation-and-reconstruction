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
