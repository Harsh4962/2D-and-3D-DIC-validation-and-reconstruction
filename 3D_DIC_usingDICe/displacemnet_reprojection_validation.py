import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# ==============================
# 1. PARSE CAMERA CALIBRATION FILE
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
        
        # Extract intrinsic parameters
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
        
        # Extract rotation matrix
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
# 2. PROJECTION FUNCTIONS
# ==============================
def project_point_to_image(point_3d, camera_params):
    """Project 3D point in camera coordinates to 2D image coordinates"""
    x, y, z = point_3d
    
    if z <= 0:
        return np.array([np.nan, np.nan])
    
    x_norm = x / z
    y_norm = y / z
    
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3
    
    K1 = camera_params.get('K1', 0.0)
    K2 = camera_params.get('K2', 0.0)
    K3 = camera_params.get('K3', 0.0)
    
    radial_distortion = 1 + K1*r2 + K2*r4 + K3*r6
    
    x_distorted = x_norm * radial_distortion
    y_distorted = y_norm * radial_distortion
    
    cx = camera_params['CX']
    cy = camera_params['CY']
    fx = camera_params['FX']
    fy = camera_params['FY']
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    return np.array([u, v])

def plane_to_world_coords(points_plane, R_world2plane, t_world2plane):
    """Transform points from best-fit plane frame to world frame (Camera 0 CS)"""
    R_plane2world = R_world2plane.T
    t_plane2world = -R_world2plane.T @ t_world2plane
    
    return (R_plane2world @ points_plane.T).T + t_plane2world

# ==============================
# 3. REPROJECTION ERROR CALCULATION
# ==============================
def compute_reprojection_errors(points_plane, R_world2plane, t_world2plane, 
                                 camera_params, measured_pixels=None):
    """Compute reprojection errors"""
    points_world = plane_to_world_coords(points_plane, R_world2plane, t_world2plane)
    
    reprojected_pixels = np.array([
        project_point_to_image(pt, camera_params) 
        for pt in points_world
    ])
    
    errors = None
    if measured_pixels is not None:
        valid_mask = ~np.isnan(reprojected_pixels).any(axis=1)
        if valid_mask.sum() > 0:
            errors = np.full(len(reprojected_pixels), np.nan)
            valid_reproj = reprojected_pixels[valid_mask]
            valid_measured = measured_pixels[valid_mask]
            errors[valid_mask] = np.linalg.norm(valid_reproj - valid_measured, axis=1)
    
    return reprojected_pixels, errors

# ==============================
# 4. MAIN ANALYSIS
# ==============================

# File paths
CAMERA_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"

# Best-fit plane to World CS transformation
R_world2plane = np.array([
    [9.853007e-001, 4.568292e-003, -1.707676e-001],
    [4.191086e-003, -9.999879e-001, -2.569322e-003],
    [-1.707773e-001, 1.815853e-003, -9.853080e-001]
])
t_world2plane = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])

print("=== DISPLACEMENT VALIDATION ===")
print("Validating deformed positions using displacement fields")

# Parse camera calibration
cameras = parse_camera_xml(CAMERA_XML)
cam0 = cameras[0]
cam1 = cameras[1]

print(f"\n=== CAMERA 0 PARAMETERS ===")
print(f"CX={cam0['CX']:.2f}, CY={cam0['CY']:.2f}")
print(f"FX={cam0['FX']:.2f}, FY={cam0['FY']:.2f}")

# Load DICe data
df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')

# ===== UNDEFORMED (REFERENCE) POSITIONS =====
x_ref = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
y_ref = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
z_ref = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

# ===== DISPLACEMENTS IN PLANE FRAME =====
dX = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
dY = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
dZ = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# ===== DEFORMED POSITIONS IN PLANE FRAME =====
X_def = x_ref + dX
Y_def = y_ref + dY
Z_def = z_ref + dZ
points_plane_deformed = np.vstack([X_def, Y_def, Z_def]).T

print(f"\n=== 3D DISPLACEMENT DATA ===")
print(f"Loaded {len(points_plane_deformed)} deformed 3D points")
print(f"Displacement magnitude range: [{np.linalg.norm([dX, dY, dZ], axis=0).min():.6f}, {np.linalg.norm([dX, dY, dZ], axis=0).max():.6f}] mm")

# Transform deformed points to camera 0 (world) and project
reprojected_cam0_deformed, _ = compute_reprojection_errors(
    points_plane_deformed, R_world2plane, t_world2plane, cam0
)
valid_reproj_deformed = reprojected_cam0_deformed[~np.isnan(reprojected_cam0_deformed).any(axis=1)]

# ===== MEASURED PIXEL COORDINATES (REFERENCE IMAGE) =====
measured_pixels_cam0_ref = None
if 'COORDINATE_X' in df.columns and 'COORDINATE_Y' in df.columns:
    measured_u_ref = df['COORDINATE_X'].astype(float).to_numpy()
    measured_v_ref = df['COORDINATE_Y'].astype(float).to_numpy()
    measured_pixels_cam0_ref = np.vstack([measured_u_ref, measured_v_ref]).T
    print(f"\nFound measured reference pixel coordinates (Camera 0)")

# ===== PIXEL DISPLACEMENTS =====
measured_disp_pixels_cam0 = None
if 'DISPLACEMENT_X' in df.columns and 'DISPLACEMENT_Y' in df.columns:
    measured_disp_u = df['DISPLACEMENT_X'].astype(float).to_numpy()
    measured_disp_v = df['DISPLACEMENT_Y'].astype(float).to_numpy()
    measured_disp_pixels_cam0 = np.vstack([measured_disp_u, measured_disp_v]).T
    print(f" Found measured pixel displacements (Camera 0)")
    print(f"  Pixel displacement range: [{np.linalg.norm(measured_disp_pixels_cam0, axis=1).min():.3f}, {np.linalg.norm(measured_disp_pixels_cam0, axis=1).max():.3f}] pixels")

# ===== DEFORMED PIXEL POSITIONS (MEASURED) =====
measured_pixels_cam0_deformed = None
if measured_disp_pixels_cam0 is not None and measured_pixels_cam0_ref is not None:
    measured_pixels_cam0_deformed = measured_pixels_cam0_ref + measured_disp_pixels_cam0
    print(f" Computed deformed pixel coordinates (reference + displacement)")

# ==============================
# 5. VISUALIZATION
# ==============================

# Figure 1: Reference positions (Camera 0)
fig1, ax1 = plt.subplots(figsize=(10, 8))
if measured_pixels_cam0_ref is not None:
    ax1.scatter(measured_pixels_cam0_ref[:,0], measured_pixels_cam0_ref[:,1], 
                s=5, c='gray', alpha=0.6, label='Reference positions')
ax1.set_xlim(0, cam0['WIDTH'])
ax1.set_ylim(cam0['HEIGHT'], 0)
ax1.add_patch(plt.Rectangle((0, 0), cam0['WIDTH'], cam0['HEIGHT'], 
                           fill=False, edgecolor='red', linewidth=2))
ax1.set_xlabel('u (pixels)')
ax1.set_ylabel('v (pixels)')
ax1.set_title('Camera 0: Reference (Undeformed) Positions')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.savefig('fig1_reference_positions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n Saved: fig1_reference_positions.png")

# Figure 2: Deformed positions - Measured vs Reprojected
if measured_pixels_cam0_deformed is not None:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Plot measured deformed positions (DICe result)
    ax2.scatter(measured_pixels_cam0_deformed[:,0], measured_pixels_cam0_deformed[:,1], 
                s=5, c='red', alpha=0.5, label='Measured deformed (DICe)', marker='o')
    
    # Plot reprojected deformed positions (from 3D displacement)
    if len(valid_reproj_deformed) > 0:
        ax2.scatter(valid_reproj_deformed[:,0], valid_reproj_deformed[:,1], 
                    s=5, c='blue', alpha=0.5, label='Reprojected deformed (3D)', marker='x')
    
    ax2.set_xlim(0, cam0['WIDTH'])
    ax2.set_ylim(cam0['HEIGHT'], 0)
    ax2.add_patch(plt.Rectangle((0, 0), cam0['WIDTH'], cam0['HEIGHT'], 
                               fill=False, edgecolor='black', linewidth=2))
    ax2.set_xlabel('u (pixels)')
    ax2.set_ylabel('v (pixels)')
    ax2.set_title('Camera 0: Deformed Positions (Measured vs Reprojected)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('fig2_deformed_overlay.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: fig2_deformed_overlay.png")

# Figure 3: Displacement vectors visualization
if measured_pixels_cam0_ref is not None and measured_pixels_cam0_deformed is not None:
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Sample points for cleaner visualization
    step = max(1, len(measured_pixels_cam0_ref) // 200)
    idx = np.arange(0, len(measured_pixels_cam0_ref), step)
    
    # Plot displacement arrows
    ax3.quiver(measured_pixels_cam0_ref[idx, 0], measured_pixels_cam0_ref[idx, 1],
               measured_disp_pixels_cam0[idx, 0], measured_disp_pixels_cam0[idx, 1],
               angles='xy', scale_units='xy', scale=1, color='blue', width=0.003, alpha=0.7)
    
    # Plot reference positions
    ax3.scatter(measured_pixels_cam0_ref[idx, 0], measured_pixels_cam0_ref[idx, 1], 
                s=10, c='red', alpha=0.7, label='Reference')
    
    # Plot deformed positions
    ax3.scatter(measured_pixels_cam0_deformed[idx, 0], measured_pixels_cam0_deformed[idx, 1], 
                s=10, c='green', alpha=0.7, label='Deformed')
    
    ax3.set_xlim(0, cam0['WIDTH'])
    ax3.set_ylim(cam0['HEIGHT'], 0)
    ax3.set_xlabel('u (pixels)')
    ax3.set_ylabel('v (pixels)')
    ax3.set_title('Camera 0: Pixel Displacement Vectors')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    plt.savefig('fig3_displacement_vectors.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: fig3_displacement_vectors.png")

# Figure 4: Zoomed comparison
if measured_pixels_cam0_deformed is not None and len(valid_reproj_deformed) > 0:
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    u_center = cam0['WIDTH'] / 2
    v_center = cam0['HEIGHT'] / 2
    zoom_width = cam0['WIDTH'] / 6
    zoom_height = cam0['HEIGHT'] / 6
    
    ax4.scatter(measured_pixels_cam0_deformed[:,0], measured_pixels_cam0_deformed[:,1], 
                s=20, c='red', alpha=0.6, label='Measured deformed', marker='o')
    ax4.scatter(valid_reproj_deformed[:,0], valid_reproj_deformed[:,1], 
                s=20, c='blue', alpha=0.6, label='Reprojected deformed', marker='x')
    ax4.set_xlim(u_center - zoom_width, u_center + zoom_width)
    ax4.set_ylim(v_center + zoom_height, v_center - zoom_height)
    ax4.set_xlabel('u (pixels)')
    ax4.set_ylabel('v (pixels)')
    ax4.set_title('Camera 0: ZOOMED Deformed Positions')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    plt.savefig('fig4_zoomed_deformed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: fig4_zoomed_deformed.png")

# Figure 5: Displacement validation error histogram
if measured_pixels_cam0_deformed is not None:
    _, errors_deformed = compute_reprojection_errors(
        points_plane_deformed, R_world2plane, t_world2plane, cam0, measured_pixels_cam0_deformed
    )
    
    valid_errors_deformed = errors_deformed[~np.isnan(errors_deformed)]
    
    if len(valid_errors_deformed) > 0:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.hist(valid_errors_deformed, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax5.axvline(valid_errors_deformed.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_errors_deformed.mean():.3f}px')
        ax5.axvline(np.median(valid_errors_deformed), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(valid_errors_deformed):.3f}px')
        ax5.set_xlabel('Displacement Validation Error (pixels)')
        ax5.set_ylabel('Count')
        ax5.set_title('Deformed Position Reprojection Error Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig5_displacement_error_histogram.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(" Saved: fig5_displacement_error_histogram.png")
        
        # Print statistics
        print(f"\n=== DISPLACEMENT VALIDATION ERROR STATISTICS ===")
        print(f"Valid error measurements: {len(valid_errors_deformed)}/{len(errors_deformed)}")
        print(f"Mean error: {valid_errors_deformed.mean():.4f} pixels")
        print(f"Median error: {np.median(valid_errors_deformed):.4f} pixels")
        print(f"Std dev: {valid_errors_deformed.std():.4f} pixels")
        print(f"Min error: {valid_errors_deformed.min():.4f} pixels")
        print(f"Max error: {valid_errors_deformed.max():.4f} pixels")
        print(f"RMS error: {np.sqrt((valid_errors_deformed**2).mean()):.4f} pixels")
        print(f"95th percentile: {np.percentile(valid_errors_deformed, 95):.4f} pixels")

print("\n" + "="*60)
print(" Displacement validation complete!")
print("="*60)