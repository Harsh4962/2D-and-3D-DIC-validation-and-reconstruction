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
                # Parse "{1200, 1920}" format
                hw = value.strip('{}').split(',')
                cameras[cam_id]['HEIGHT'] = int(hw[0])
                cameras[cam_id]['WIDTH'] = int(hw[1])
        
        # Extract rotation matrix
        rot_list = cam_list.find('.//ParameterList[@name="rotation_3x3_matrix"]')
        if rot_list is not None:
            R = []
            for row_param in rot_list.findall('Parameter'):
                row_str = row_param.get('value')
                # Parse "{ 1.0, 0.0, 0.0 }" format
                row_values = [float(x) for x in row_str.strip('{}').split(',')]
                R.append(row_values)
            cameras[cam_id]['R'] = np.array(R)

    print(cameras)
        
    return cameras

# ==============================
# 2. PROJECTION FUNCTIONS
# ==============================
def project_point_to_image(point_3d, camera_params):
    """
    Project 3D point in camera coordinates to 2D image coordinates
    
    Args:
        point_3d: (x, y, z) in camera coordinate system
        camera_params: dict with CX, CY, FX, FY, K1, K2, K3
    
    Returns:
        (u, v) in image pixel coordinates
    """
    x, y, z = point_3d
    
    # Check if z is positive (in front of camera)
    if z <= 0:
        return np.array([np.nan, np.nan])
    
    # Normalized coordinates
    x_norm = x / z
    y_norm = y / z
    
    # Radial distance
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2 # this is r^4
    r6 = r2**3 # this is r^6
    
    # Radial distortion (OpenCV model)
    K1 = camera_params.get('K1', 0.0)
    K2 = camera_params.get('K2', 0.0)
    K3 = camera_params.get('K3', 0.0)
    
    radial_distortion = 1 + K1*r2 + K2*r4 + K3*r6
    
    # Apply distortion
    x_distorted = x_norm * radial_distortion
    y_distorted = y_norm * radial_distortion
    
    # Project to image plane using intrinsics
    cx = camera_params['CX']
    cy = camera_params['CY']
    fx = camera_params['FX']
    fy = camera_params['FY']
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    return np.array([u, v])

def plane_to_world_coords(points_plane, R_world2plane, t_world2plane):
    """
    Transform points from best-fit plane frame to world frame (Camera 0 CS)
    
    CORRECTED: Uses inverse transformation
    """
    # Inverse transformation: p_world = R^T @ (p_plane - t),
    # P_world = tans(Rw->p)(P_plane - tw->p)
    R_plane2world = R_world2plane.T
    t_plane2world = -R_world2plane.T @ t_world2plane
    
    return (R_plane2world @ points_plane.T).T + t_plane2world

# ==============================
# 3. REPROJECTION ERROR CALCULATION
# ==============================
def compute_reprojection_errors(points_plane, R_world2plane, t_world2plane, 
                                 camera_params, measured_pixels=None):
    """
    Compute reprojection errors
    
    Args:
        points_plane: (N, 3) array of 3D points in best-fit plane frame
        R_world2plane: rotation matrix world -> plane (from DICe output)
        t_world2plane: translation vector world -> plane (from DICe output)
        camera_params: camera intrinsic parameters
        measured_pixels: (N, 2) array of measured pixel locations (optional)
    
    Returns:
        reprojected_pixels: (N, 2) array of reprojected pixel locations
        errors: (N,) array of reprojection errors (if measured_pixels provided)
    """
    # CORRECTED: Transform from plane frame to world frame (Camera 0 CS)
    points_world = plane_to_world_coords(points_plane, R_world2plane, t_world2plane)
    
    # Project to image
    reprojected_pixels = []
    for pt in points_world:
        pixel = project_point_to_image(pt, camera_params)
        reprojected_pixels.append(pixel)
    
    reprojected_pixels = np.array(reprojected_pixels)
    
    errors = None
    if measured_pixels is not None:
        # Only compute errors for valid projections (not NaN)
        valid_mask = ~np.isnan(reprojected_pixels).any(axis=1)
        if valid_mask.sum() > 0:
            errors = np.full(len(reprojected_pixels), np.nan)
            valid_reproj = reprojected_pixels[valid_mask]
            valid_measured = measured_pixels[valid_mask]
            # Euclidean distance between measured and reprojected
            errors[valid_mask] = np.linalg.norm(valid_reproj - valid_measured, axis=1)
    
    return reprojected_pixels, errors

# ==============================
# 4. MAIN ANALYSIS
# ==============================

# File paths
CAMERA_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"

# Best-fit plane to World CS transformation (from your DICe output)
R_world2plane = np.array([
    [9.853007e-001, 4.568292e-003, -1.707676e-001],
    [4.191086e-003, -9.999879e-001, -2.569322e-003],
    [-1.707773e-001, 1.815853e-003, -9.853080e-001]
])
t_world2plane = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])

print("=== TRANSFORMATION INFO ===")
print("DICe provides: World CS -> Best-fit plane CS transformation")
print("We need: Best-fit plane CS -> World CS (inverse)")

# Parse camera calibration
cameras = parse_camera_xml(CAMERA_XML)
cam0 = cameras[0]
cam1 = cameras[1]

print(f"\n=== CAMERA 0 PARAMETERS ===")
print(f"CX={cam0['CX']:.2f}, CY={cam0['CY']:.2f}")
print(f"FX={cam0['FX']:.2f}, FY={cam0['FY']:.2f}")
print(f"K1={cam0['K1']:.4f}, K2={cam0['K2']:.4f}, K3={cam0['K3']:.4f}")
print(f"Image size: {cam0['HEIGHT']}x{cam0['WIDTH']}")

# Load DICe data
df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')
# Loading best fit plane coordiates
cx, cy, cz = "MODEL_COORDINATES_X", "MODEL_COORDINATES_Y", "MODEL_COORDINATES_Z"
X = df[cx].astype(float).to_numpy()
Y = df[cy].astype(float).to_numpy()
Z = df[cz].astype(float).to_numpy()

# Points in best-fit plane frame
points_plane = np.vstack([X, Y, Z]).T

print(f"\n=== DATA INFO ===")
print(f"Loaded {len(points_plane)} 3D points from DICe results")

# Transform and project
reprojected_cam0, _ = compute_reprojection_errors(
    points_plane, R_world2plane, t_world2plane, cam0
)

# Remove NaN values
valid_reproj = reprojected_cam0[~np.isnan(reprojected_cam0).any(axis=1)]

# Load measured subset coordinates i.e actual pixel coordinates
measured_pixels_cam0 = None
if 'COORDINATE_X' in df.columns and 'COORDINATE_Y' in df.columns:
    measured_u = df['COORDINATE_X'].astype(float).to_numpy()
    measured_v = df['COORDINATE_Y'].astype(float).to_numpy()
    measured_pixels_cam0 = np.vstack([measured_u, measured_v]).T
    print(f"\n Found measured pixel coordinates for Camera 0")

# ==============================
# 5. SEPARATE FIGURES
# ==============================

# Figure 1: Camera 0 Reprojected Points
fig1, ax1 = plt.subplots(figsize=(8, 6))
if len(valid_reproj) > 0:
    ax1.scatter(valid_reproj[:,0], valid_reproj[:,1], s=1, c='blue', alpha=0.6)
    ax1.set_xlim(0, cam0['WIDTH'])
    ax1.set_ylim(cam0['HEIGHT'], 0)
    ax1.add_patch(plt.Rectangle((0, 0), cam0['WIDTH'], cam0['HEIGHT'], 
                               fill=False, edgecolor='red', linewidth=2))
ax1.set_xlabel('u (pixels)')
ax1.set_ylabel('v (pixels)')
ax1.set_title('Camera 0: Reprojected Points')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_camera0_reprojected.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: fig1_camera0_reprojected.png")

# Figure 2: Camera 1 Reprojected Points
fig2, ax2 = plt.subplots(figsize=(8, 6))
points_world = plane_to_world_coords(points_plane, R_world2plane, t_world2plane)
points_cam1 = (cam1['R'] @ points_world.T).T + np.array([cam1['TX'], cam1['TY'], cam1['TZ']])
reprojected_cam1 = np.array([project_point_to_image(pt, cam1) for pt in points_cam1])
valid_reproj_cam1 = reprojected_cam1[~np.isnan(reprojected_cam1).any(axis=1)]

if len(valid_reproj_cam1) > 0:
    ax2.scatter(valid_reproj_cam1[:,0], valid_reproj_cam1[:,1], s=1, c='green', alpha=0.6)
    ax2.set_xlim(0, cam1['WIDTH'])
    ax2.set_ylim(cam1['HEIGHT'], 0)
    ax2.add_patch(plt.Rectangle((0, 0), cam1['WIDTH'], cam1['HEIGHT'], 
                               fill=False, edgecolor='red', linewidth=2))
ax2.set_xlabel('u (pixels)')
ax2.set_ylabel('v (pixels)')
ax2.set_title('Camera 1: Reprojected Points')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_camera1_reprojected.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig2_camera1_reprojected.png")

# Figure 3: Overlay (Reprojected + Measured)
if measured_pixels_cam0 is not None:
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Plot measured first (underneath)
    ax3.scatter(measured_pixels_cam0[:,0], measured_pixels_cam0[:,1], 
                s=3, c='red', alpha=0.5, label='Measured (DICe)', marker='o')
    
    # Plot reprojected on top
    if len(valid_reproj) > 0:
        ax3.scatter(valid_reproj[:,0], valid_reproj[:,1], 
                    s=3, c='blue', alpha=0.5, label='Reprojected', marker='x')
    
    ax3.set_xlim(0, cam0['WIDTH'])
    ax3.set_ylim(cam0['HEIGHT'], 0)
    ax3.add_patch(plt.Rectangle((0, 0), cam0['WIDTH'], cam0['HEIGHT'], 
                               fill=False, edgecolor='black', linewidth=2))
    ax3.set_xlabel('u (pixels)')
    ax3.set_ylabel('v (pixels)')
    ax3.set_title('Camera 0: OVERLAY (Measured vs Reprojected)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig('fig3_overlay_measured_vs_reprojected.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: fig3_overlay_measured_vs_reprojected.png")

# Figure 4: Zoomed Overlay (Center Region)
if measured_pixels_cam0 is not None and len(valid_reproj) > 0:
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Calculate center region
    u_center = cam0['WIDTH'] / 2
    v_center = cam0['HEIGHT'] / 2
    # zoom_width = cam0['WIDTH'] / 100
    # zoom_height = cam0['HEIGHT'] / 100
    zoom_width = cam0['WIDTH'] / 6
    zoom_height = cam0['HEIGHT'] / 6
    
    ax4.scatter(measured_pixels_cam0[:,0], measured_pixels_cam0[:,1], 
                s=15, c='red', alpha=0.6, label='Measured', marker='o')
    ax4.scatter(valid_reproj[:,0], valid_reproj[:,1], 
                s=15, c='blue', alpha=0.6, label='Reprojected', marker='x')
    ax4.set_xlim(u_center - zoom_width, u_center + zoom_width)
    ax4.set_ylim(v_center + zoom_height, v_center - zoom_height)
    ax4.set_xlabel('u (pixels)')
    ax4.set_ylabel('v (pixels)')
    ax4.set_title('Camera 0: ZOOMED Center Region (Measured vs Reprojected)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig('fig4_zoomed_overlay.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: fig4_zoomed_overlay.png")

# Figure 5: Error Histogram
if measured_pixels_cam0 is not None:
    _, errors = compute_reprojection_errors(
        points_plane, R_world2plane, t_world2plane, cam0, measured_pixels_cam0
    )
    
    valid_errors = errors[~np.isnan(errors)]
    
    if len(valid_errors) > 0:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.hist(valid_errors, bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(valid_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_errors.mean():.3f}px')
        ax5.axvline(np.median(valid_errors), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(valid_errors):.3f}px')
        ax5.set_xlabel('Reprojection Error (pixels)')
        ax5.set_ylabel('Count')
        ax5.set_title('Corrected Reprojection Error Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig5_error_histogram.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(" Saved: fig5_error_histogram.png")
        
        # Print statistics
        print(f"\n=== REPROJECTION ERROR STATISTICS ===")
        print(f"Valid error measurements: {len(valid_errors)}/{len(errors)}")
        print(f"Mean error: {valid_errors.mean():.4f} pixels")
        print(f"Median error: {np.median(valid_errors):.4f} pixels")
        print(f"Std dev: {valid_errors.std():.4f} pixels")
        print(f"Min error: {valid_errors.min():.4f} pixels")
        print(f"Max error: {valid_errors.max():.4f} pixels")
        print(f"RMS error: {np.sqrt((valid_errors**2).mean()):.4f} pixels")
        print(f"95th percentile: {np.percentile(valid_errors, 95):.4f} pixels")

# print("\n All figures generated and saved separately!")
