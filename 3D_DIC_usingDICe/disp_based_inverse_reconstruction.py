import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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
        
        for param in cam_list.findall('Parameter'):
            name = param.get('name')
            ptype = param.get('type')
            value = param.get('value')
            
            if ptype == 'double':
                try:
                    cameras[cam_id][name] = float(value)
                except:
                    pass
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
    
    cx = camera_params.get('CX', 0.0)
    cy = camera_params.get('CY', 0.0)
    fx = camera_params.get('FX', 1.0)
    fy = camera_params.get('FY', 1.0)
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    return np.array([u, v])

def plane_to_world_coords(points_plane, R_world2plane, t_world2plane):
    """Transform points from plane to world frame"""
    R_plane2world = R_world2plane.T
    t_plane2world = -R_world2plane.T @ t_world2plane
    
    return (R_plane2world @ points_plane.T).T + t_plane2world

# ==============================
# 3. CREATE VALID MASK
# ==============================
def create_valid_mask_from_subsets(reference_pixels_cam0, 
                                   H, W,
                                   subset_size=31,
                                   step_size=15,
                                   method='overlap'):
    """
    Create valid region mask using subset information
    
    Args:
        reference_pixels_cam0: (N, 2) subset center positions
        H, W: Image dimensions
        subset_size: DICe subset size (31)
        step_size: DICe step size (15)
        method: 'subset', 'overlap', or 'conservative'
    """
    
    valid_region = np.zeros((H, W), dtype=bool)
    half_size = subset_size // 2  # 15 for 31×31
    
    if method == 'subset':
        # Each subset covers its 31×31 area
        for x, y in reference_pixels_cam0:
            x, y = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
            y_min, y_max = max(0, y-half_size), min(H, y+half_size+1)
            x_min, x_max = max(0, x-half_size), min(W, x+half_size+1)
            valid_region[y_min:y_max, x_min:x_max] = True
    
    elif method == 'overlap':
        # Account for overlapping subsets
        margin = step_size // 2  # 7 pixels
        for x, y in reference_pixels_cam0:
            x, y = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
            y_min = max(0, y - half_size - margin)
            y_max = min(H, y + half_size + margin + 1)
            x_min = max(0, x - half_size - margin)
            x_max = min(W, x + half_size + margin + 1)
            valid_region[y_min:y_max, x_min:x_max] = True
    
    elif method == 'conservative':
        # Only core measurement region
        for x, y in reference_pixels_cam0:
            x, y = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
            y_min, y_max = max(0, y-half_size), min(H, y+half_size+1)
            x_min, x_max = max(0, x-half_size), min(W, x+half_size+1)
            valid_region[y_min:y_max, x_min:x_max] = True
        
        # Erode to remove uncertain boundaries
        from scipy import ndimage
        kernel = ndimage.generate_binary_structure(2, 2)
        valid_region = ndimage.binary_erosion(valid_region, structure=kernel, iterations=5)
    
    return valid_region

# ==============================
# 4. FULL INVERSE RECONSTRUCTION
# ==============================
def full_inverse_reconstruction_stereo(
    reference_image_cam0,
    reference_image_cam1,
    reference_pixels_cam0,
    reference_pixels_cam1,
    measured_disp_pixels_cam0,
    measured_disp_pixels_cam1,
    points_plane_ref,
    dX_plane, dY_plane, dZ_plane,
    R_world2plane, t_world2plane,
    cam0, cam1,
    subset_size=31,
    step_size=15
):
    """
    Complete pipeline:
    1. Plane coords + displacements → Deformed 3D coords
    2. Transform to world → Project to both cameras
    3. Compute 2D displacements → Interpolate displacement fields
    4. Use Irec = Iref(x-u, y-v) → Reconstruct deformed images
    5. Compute errors
    """
    
    print("="*70)
    print("FULL INVERSE RECONSTRUCTION PIPELINE (MEMORY OPTIMIZED)")
    print("="*70)
    
    # ===== STEP 1: Compute deformed 3D positions =====
    print("\nStep 1: Computing deformed 3D positions in plane frame...")
    
    x_ref = points_plane_ref[:, 0]
    y_ref = points_plane_ref[:, 1]
    z_ref = points_plane_ref[:, 2]
    
    X_def = x_ref + dX_plane
    Y_def = y_ref + dY_plane
    Z_def = z_ref + dZ_plane
    points_plane_deformed = np.vstack([X_def, Y_def, Z_def]).T
    
    print(f" Generated {len(points_plane_deformed)} deformed points")
    
    # ===== STEP 2: Transform to world and project =====
    print("\nStep 2: Transforming to world frame and projecting to both cameras...")
    
    points_world_deformed = plane_to_world_coords(
        points_plane_deformed, R_world2plane, t_world2plane
    )
    
    # Project to Camera 0
    projected_cam0_deformed = np.array([
        project_point_to_image(pt, cam0) for pt in points_world_deformed
    ])
    valid_cam0 = ~np.isnan(projected_cam0_deformed).any(axis=1)
    
    # Project to Camera 1
    points_cam1_deformed = (cam1['R'] @ points_world_deformed.T).T + \
                          np.array([cam1['TX'], cam1['TY'], cam1['TZ']])
    projected_cam1_deformed = np.array([
        project_point_to_image(pt, cam1) for pt in points_cam1_deformed
    ])
    valid_cam1 = ~np.isnan(projected_cam1_deformed).any(axis=1)
    
    print(f" Camera 0: {valid_cam0.sum()}/{len(valid_cam0)} valid projections")
    print(f" Camera 1: {valid_cam1.sum()}/{len(valid_cam1)} valid projections")
    
    # ===== STEP 3: Compute pixel displacements (2D) =====
    print("\nStep 3: Computing pixel displacements in both image planes...")
    
    # Remove NaN
    disp_pixels_cam0 = projected_cam0_deformed[valid_cam0] - \
                       reference_pixels_cam0[valid_cam0]
    disp_pixels_cam1 = projected_cam1_deformed[valid_cam1] - \
                       reference_pixels_cam1[valid_cam1]
    
    valid_ref_pixels_cam0 = reference_pixels_cam0[valid_cam0]
    valid_ref_pixels_cam1 = reference_pixels_cam1[valid_cam1]
    
    print(f" Computed pixel displacements")
    
    # ===== STEP 4: Interpolate displacement fields (GRIDDATA) =====
    print("\nStep 4: Interpolating displacement fields using griddata...")
    
    H0, W0 = reference_image_cam0.shape[:2]
    H1, W1 = reference_image_cam1.shape[:2]
    
    # Create full-image grids
    x_grid0 = np.arange(W0, dtype=np.float32)
    y_grid0 = np.arange(H0, dtype=np.float32)
    xx0, yy0 = np.meshgrid(x_grid0, y_grid0)
    
    x_grid1 = np.arange(W1, dtype=np.float32)
    y_grid1 = np.arange(H1, dtype=np.float32)
    xx1, yy1 = np.meshgrid(x_grid1, y_grid1)
    
    # Interpolate using griddata
    print("  Camera 0: Interpolating displacement field...")
    disp_u0_full = griddata(valid_ref_pixels_cam0, disp_pixels_cam0[:,0], 
                            (xx0, yy0), method='cubic', fill_value=0)
    disp_v0_full = griddata(valid_ref_pixels_cam0, disp_pixels_cam0[:,1], 
                            (xx0, yy0), method='cubic', fill_value=0)
    
    print("  Camera 1: Interpolating displacement field...")
    disp_u1_full = griddata(valid_ref_pixels_cam1, disp_pixels_cam1[:,0], 
                            (xx1, yy1), method='cubic', fill_value=0)
    disp_v1_full = griddata(valid_ref_pixels_cam1, disp_pixels_cam1[:,1], 
                            (xx1, yy1), method='cubic', fill_value=0)
    
    print(f" Displacement fields interpolated")
    
    # ===== STEP 5: Reconstruct deformed images =====
    print("\nStep 5: Reconstructing deformed images using intensity conservation...")
    print("  Formula: Irec(x, y) = Iref(x - u(x,y), y - v(x,y))")
    
    # Camera 0 Reconstruction
    print("  Camera 0: Backward mapping...")
    xx0_src = xx0 - disp_u0_full
    yy0_src = yy0 - disp_v0_full
    
    rec_cam0_warped = cv2.remap(
        reference_image_cam0.astype(np.float32),
        xx0_src.astype(np.float32),
        yy0_src.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Create valid region mask
    valid_region_cam0 = create_valid_mask_from_subsets(
        valid_ref_pixels_cam0, H0, W0,
        subset_size=subset_size,
        step_size=step_size,
        method='overlap'
    )
    
    # Keep float (before clipping) for RRMSE calculations
    rec_cam0_float = rec_cam0_warped.copy()
    # Make uint8 copy for display/other metrics (unchanged behavior)
    rec_cam0 = np.clip(rec_cam0_float, 0, 255).astype(np.uint8)
    
    # Camera 1 Reconstruction
    print("  Camera 1: Backward mapping...")
    xx1_src = xx1 - disp_u1_full
    yy1_src = yy1 - disp_v1_full
    
    rec_cam1_warped = cv2.remap(
        reference_image_cam1.astype(np.float32),
        xx1_src.astype(np.float32),
        yy1_src.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Create valid region mask
    valid_region_cam1 = create_valid_mask_from_subsets(
        valid_ref_pixels_cam1, H1, W1,
        subset_size=subset_size,
        step_size=step_size,
        method='overlap'
    )
    
    rec_cam1_float = rec_cam1_warped.copy()
    rec_cam1 = np.clip(rec_cam1_float, 0, 255).astype(np.uint8)
    
    print(f" Reconstructed Camera 0 image")
    print(f" Reconstructed Camera 1 image")
    
    # Return float and u8 reconstructions and valid masks
    return {
        'rec_cam0': rec_cam0,
        'rec_cam1': rec_cam1,
        'rec_cam0_float': rec_cam0_float,
        'rec_cam1_float': rec_cam1_float,
        'valid_region_cam0': valid_region_cam0,
        'valid_region_cam1': valid_region_cam1
    }

# ==============================
# 5. COMPUTE RECONSTRUCTION ERRORS
# ==============================
def compute_reconstruction_errors(actual_def_cam0, actual_def_cam1,
                                  rec_cam0, rec_cam1,
                                  valid_region_cam0, valid_region_cam1):
    """Compute reconstruction error metrics"""
    
    print("\n" + "="*70)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("="*70)
    
    # Camera 0
    diff_cam0 = actual_def_cam0.astype(float) - rec_cam0.astype(float)
    diff_cam0_valid = diff_cam0[valid_region_cam0]
    
    rms_cam0 = np.sqrt(np.mean(diff_cam0_valid**2))
    mae_cam0 = np.mean(np.abs(diff_cam0_valid))
    ssim_cam0 = ssim(actual_def_cam0, rec_cam0, data_range=255)
    psnr_cam0 = psnr(actual_def_cam0, rec_cam0, data_range=255)
    
    print(f"\nCamera 0 Reconstruction Metrics:")
    print(f"  RMS Intensity Error: {rms_cam0:.2f} gray levels")
    print(f"  MAE Intensity Error: {mae_cam0:.2f} gray levels")
    print(f"  SSIM: {ssim_cam0:.4f} (1.0 = perfect match)")
    print(f"  PSNR: {psnr_cam0:.2f} dB (higher = better)")
    
    # Camera 1
    diff_cam1 = actual_def_cam1.astype(float) - rec_cam1.astype(float)
    diff_cam1_valid = diff_cam1[valid_region_cam1]
    
    rms_cam1 = np.sqrt(np.mean(diff_cam1_valid**2))
    mae_cam1 = np.mean(np.abs(diff_cam1_valid))
    ssim_cam1 = ssim(actual_def_cam1, rec_cam1, data_range=255)
    psnr_cam1 = psnr(actual_def_cam1, rec_cam1, data_range=255)
    
    print(f"\nCamera 1 Reconstruction Metrics:")
    print(f"  RMS Intensity Error: {rms_cam1:.2f} gray levels")
    print(f"  MAE Intensity Error: {mae_cam1:.2f} gray levels")
    print(f"  SSIM: {ssim_cam1:.4f} (1.0 = perfect match)")
    print(f"  PSNR: {psnr_cam1:.2f} dB (higher = better)")
    
    return {
        'cam0': {
            'rms': rms_cam0,
            'mae': mae_cam0,
            'ssim': ssim_cam0,
            'psnr': psnr_cam0,
            'diff': diff_cam0
        },
        'cam1': {
            'rms': rms_cam1,
            'mae': mae_cam1,
            'ssim': ssim_cam1,
            'psnr': psnr_cam1,
            'diff': diff_cam1
        }
    }

# ==============================
# RRMSE computation (DIC-style)
# ==============================
def compute_rrmse(actual_uint8, recon_float, valid_mask, threshold=5.0, eps=1e-6):
    """
    DIC-style Relative RMSE:
        RRMSE = sqrt( mean( (1 - Irec / Iactual)^2 ) )
    Uses recon_float (before uint8 clipping) and thresholds small actual intensities.
    Returns: (rrmse_value (dimensionless), rel_mask (boolean mask used), rel_map_pct (float array %))
    """
    actual = actual_uint8.astype(np.float32)
    recon = recon_float.astype(np.float32)
    rel_mask = valid_mask & (actual > threshold)
    if rel_mask.sum() == 0:
        return float('nan'), rel_mask, np.full_like(actual, np.nan, dtype=float)
    denom = actual[rel_mask] + eps
    rel = 1.0 - (recon[rel_mask] / denom)
    rrmse_scalar = float(np.sqrt(np.mean(rel**2)))
    # Absolute relative error percentage heatmap
    rel_map = np.full_like(actual, np.nan, dtype=float)
    rel_map[rel_mask] = 100.0 * np.abs(rel)
    return rrmse_scalar, rel_mask, rel_map

# ==============================
# 6. VISUALIZATION (CAMERA PAIRS TOGETHER, REST SEPARATE)
# ==============================
def visualize_full_reconstruction(reference_cam0, reference_cam1,
                                  actual_def_cam0, actual_def_cam1,
                                  rec_cam0, rec_cam1,
                                  errors,
                                  valid_region_cam0, valid_region_cam1,
                                  rel_thresh_display):
    """
    Visualization with camera pairs together:
    - Figures 1..7 unchanged
    - Figures 8..9: RRMSE heatmaps for camera 0 and camera 1
    """
    
    print("\nGenerating visualizations...")
    
    # ===== FIGURE 1: Camera 0 - Actual + Reconstructed =====
    print("  Generating Figure 1: Camera 0 - Actual + Reconstructed...")
    
    fig1, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].imshow(actual_def_cam0, cmap='gray')
    axes[0].set_title('Actual Deformed', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(rec_cam0, cmap='gray')
    axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f'Camera 0: Actual vs Reconstructed | SSIM: {errors["cam0"]["ssim"]:.4f} | PSNR: {errors["cam0"]["psnr"]:.2f} dB',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('01_camera0_actual_vs_reconstructed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved: 01_camera0_actual_vs_reconstructed.png")
    
    # ===== FIGURE 2: Camera 1 - Actual + Reconstructed =====
    print("  Generating Figure 2: Camera 1 - Actual + Reconstructed...")
    
    fig2, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].imshow(actual_def_cam1, cmap='gray')
    axes[0].set_title('Actual Deformed', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(rec_cam1, cmap='gray')
    axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f'Camera 1: Actual vs Reconstructed | SSIM: {errors["cam1"]["ssim"]:.4f} | PSNR: {errors["cam1"]["psnr"]:.2f} dB',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('02_camera1_actual_vs_reconstructed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved: 02_camera1_actual_vs_reconstructed.png")
    
    # ===== FIGURE 3: Error Heatmap - Camera 0 =====
    print("  Generating Figure 3: Error Heatmap Camera 0...")
    
    fig3, ax = plt.subplots(figsize=(12, 10))
    
    error_masked_cam0 = errors['cam0']['diff'].copy()
    error_masked_cam0[~valid_region_cam0] = 0
    
    im = ax.imshow(error_masked_cam0, cmap='RdBu_r', vmin=-50, vmax=50)
    ax.set_title(f"Camera 0: Reconstruction Error Heatmap\n" + 
                 f"RMS: {errors['cam0']['rms']:.2f} | MAE: {errors['cam0']['mae']:.2f} | " +
                 f"SSIM: {errors['cam0']['ssim']:.4f}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, label='Intensity Error (gray levels)')
    
    plt.tight_layout()
    plt.savefig('03_error_heatmap_cam0.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved: 03_error_heatmap_cam0.png")
    
    # ===== FIGURE 4: Error Heatmap - Camera 1 =====
    print("  Generating Figure 4: Error Heatmap Camera 1...")
    
    fig4, ax = plt.subplots(figsize=(12, 10))
    
    error_masked_cam1 = errors['cam1']['diff'].copy()
    error_masked_cam1[~valid_region_cam1] = 0
    
    im = ax.imshow(error_masked_cam1, cmap='RdBu_r', vmin=-50, vmax=50)
    ax.set_title(f"Camera 1: Reconstruction Error Heatmap\n" + 
                 f"RMS: {errors['cam1']['rms']:.2f} | MAE: {errors['cam1']['mae']:.2f} | " +
                 f"SSIM: {errors['cam1']['ssim']:.4f}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, label='Intensity Error (gray levels)')
    
    plt.tight_layout()
    plt.savefig('04_error_heatmap_cam1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: 04_error_heatmap_cam1.png")
    
    # ===== FIGURE 5: Error Histogram - Camera 0 =====
    print("Generating Figure 5: Error Histogram Camera 0")
    
    fig5, ax = plt.subplots(figsize=(12, 8))
    
    valid_errors_cam0 = errors['cam0']['diff'][valid_region_cam0]
    
    ax.hist(valid_errors_cam0, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(errors['cam0']['rms'], color='red', linestyle='--', linewidth=2.5,
              label=f"RMS: {errors['cam0']['rms']:.3f}")
    ax.axvline(errors['cam0']['mae'], color='green', linestyle='--', linewidth=2.5,
              label=f"MAE: {errors['cam0']['mae']:.3f}")
    ax.axvline(np.median(valid_errors_cam0), color='orange', linestyle='--', linewidth=2.5,
              label=f"Median: {np.median(valid_errors_cam0):.3f}")
    
    ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Camera 0: Error Distribution (Valid Region Only)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    stats_text = f"""Statistics (Valid Region):
    Count: {len(valid_errors_cam0)}
    Min: {valid_errors_cam0.min():.3f}
    Max: {valid_errors_cam0.max():.3f}
    Mean: {valid_errors_cam0.mean():.3f}
    Std: {valid_errors_cam0.std():.3f}
    95th %ile: {np.percentile(valid_errors_cam0, 95):.3f}"""
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('05_error_histogram_cam0.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 05_error_histogram_cam0.png")
    
    # ===== FIGURE 6: Error Histogram - Camera 1 =====
    print("  Generating Figure 6: Error Histogram Camera 1")
    
    fig6, ax = plt.subplots(figsize=(12, 8))
    
    valid_errors_cam1 = errors['cam1']['diff'][valid_region_cam1]
    
    ax.hist(valid_errors_cam1, bins=100, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(errors['cam1']['rms'], color='red', linestyle='--', linewidth=2.5,
              label=f"RMS: {errors['cam1']['rms']:.3f}")
    ax.axvline(errors['cam1']['mae'], color='green', linestyle='--', linewidth=2.5,
              label=f"MAE: {errors['cam1']['mae']:.3f}")
    ax.axvline(np.median(valid_errors_cam1), color='orange', linestyle='--', linewidth=2.5,
              label=f"Median: {np.median(valid_errors_cam1):.3f}")
    
    ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Camera 1: Error Distribution (Valid Region Only)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    stats_text = f"""Statistics (Valid Region):
    Count: {len(valid_errors_cam1)}
    Min: {valid_errors_cam1.min():.3f}
    Max: {valid_errors_cam1.max():.3f}
    Mean: {valid_errors_cam1.mean():.3f}
    Std: {valid_errors_cam1.std():.3f}
    95th %ile: {np.percentile(valid_errors_cam1, 95):.3f}"""
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('06_error_histogram_cam1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: 06_error_histogram_cam1.png")
    
    # ===== FIGURE 7: Error Comparison Histogram (Both Cameras) =====
    print("  Generating Figure 7: Error Comparison...")
    
    fig7, ax = plt.subplots(figsize=(14, 8))
    
    valid_errors_cam0 = errors['cam0']['diff'][valid_region_cam0]
    valid_errors_cam1 = errors['cam1']['diff'][valid_region_cam1]
    
    ax.hist(valid_errors_cam0, bins=80, alpha=0.6, label='Camera 0', color='steelblue', edgecolor='black')
    ax.hist(valid_errors_cam1, bins=80, alpha=0.6, label='Camera 1', color='coral', edgecolor='black')
    
    ax.axvline(errors['cam0']['rms'], color='steelblue', linestyle='--', linewidth=2, 
              label=f"Cam0 RMS: {errors['cam0']['rms']:.3f}")
    ax.axvline(errors['cam1']['rms'], color='coral', linestyle='--', linewidth=2,
              label=f"Cam1 RMS: {errors['cam1']['rms']:.3f}")
    
    ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reconstruction Error Comparison: Camera 0 vs Camera 1', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('07_error_comparison_cam0_vs_cam1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" Saved: 07_error_comparison_cam0_vs_cam1.png")
    
       # ===== FIGURE 8: RRMSE Heatmap Camera 0 =====
    print("  Generating Figure 8: RRMSE Heatmap Camera 0...")

    rrmap0 = errors['cam0'].get('rrmap', None)
    if rrmap0 is not None:
        fig8, ax = plt.subplots(figsize=(12, 10))

        rrmap0_masked = rrmap0.copy()
        rrmap0_masked[~valid_region_cam0] = np.nan

        vmax0 = np.nanpercentile(rrmap0_masked, 99)
        im8 = ax.imshow(rrmap0_masked, cmap='inferno', vmin=0, vmax=vmax0)

        ax.set_title(
            f"Camera 0: RRMSE Heatmap (threshold={rel_thresh_display})\n"
            f"Global RRMSE = {errors['cam0'].get('rrmse', np.nan):.4f}",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)

        plt.colorbar(im8, ax=ax, label='Relative Error (unitless)')
        plt.tight_layout()
        plt.savefig('08_rrmse_heatmap_cam0.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: 08_rrmse_heatmap_cam0.png")

    else:
        print("  No RRMSE map found for Camera 0")


    # ===== FIGURE 9: RRMSE Heatmap Camera 1 =====
    print("  Generating Figure 9: RRMSE Heatmap Camera 1...")

    rrmap1 = errors['cam1'].get('rrmap', None)
    if rrmap1 is not None:
        fig9, ax = plt.subplots(figsize=(12, 10))

        rrmap1_masked = rrmap1.copy()
        rrmap1_masked[~valid_region_cam1] = np.nan

        vmax1 = np.nanpercentile(rrmap1_masked, 99)
        im9 = ax.imshow(rrmap1_masked, cmap='inferno', vmin=0, vmax=vmax1)

        ax.set_title(
            f"Camera 1: RRMSE Heatmap (threshold={rel_thresh_display})\n"
            f"Global RRMSE = {errors['cam1'].get('rrmse', np.nan):.4f}",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)

        plt.colorbar(im9, ax=ax, label='Relative Error (unitless)')
        plt.tight_layout()
        plt.savefig('09_rrmse_heatmap_cam1.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: 09_rrmse_heatmap_cam1.png")

    else:
        print("No RRMSE map found for Camera 1")

# ==============================
# 7. MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    
    # ===== FILE PATHS =====
    CAMERA_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"
    DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
    REF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
    REF_CAM1 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_1.tif"
    DEF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_0.tif"
    DEF_CAM1 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_1.tif"

    
    # ===== TRANSFORMATION (from DICe output) =====
    R_world2plane = np.array([
        [9.853007e-001, 4.568292e-003, -1.707676e-001],
        [4.191086e-003, -9.999879e-001, -2.569322e-003],
        [-1.707773e-001, 1.815853e-003, -9.853080e-001]
    ])
    t_world2plane = np.array([1.329374e+001, -3.270429e+000, 5.393940e+001])
    
    print("="*70)
    print("FULL INVERSE RECONSTRUCTION: 3D DIC VALIDATION")
    print("="*70)
    
    # ===== LOAD CALIBRATION =====
    print("\nLoading camera calibration...")
    cameras = parse_camera_xml(CAMERA_XML)
    cam0 = cameras[0]
    cam1 = cameras[1]
    
    # ===== LOAD DATA =====
    print("Loading DICe results and images...")
    df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine='python')
    
    # Reference coordinates (plane frame)
    x_ref = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
    y_ref = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
    z_ref = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()
    points_plane_ref = np.vstack([x_ref, y_ref, z_ref]).T
    
    # Displacements (plane frame)
    dX = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
    dY = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
    dZ = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()
    
    # Reference pixels (Camera 0)
    measured_u_ref = df['COORDINATE_X'].astype(float).to_numpy()
    measured_v_ref = df['COORDINATE_Y'].astype(float).to_numpy()
    reference_pixels_cam0 = np.vstack([measured_u_ref, measured_v_ref]).T
    
    # Pixel displacements (Camera 0)
    measured_disp_u = df['DISPLACEMENT_X'].astype(float).to_numpy()
    measured_disp_v = df['DISPLACEMENT_Y'].astype(float).to_numpy()
    measured_disp_pixels_cam0 = np.vstack([measured_disp_u, measured_disp_v]).T
    
    # For Camera 1, compute projections
    print("Computing Camera 1 projections...")
    points_world_ref = plane_to_world_coords(points_plane_ref, R_world2plane, t_world2plane)
    points_cam1_ref = (cam1['R'] @ points_world_ref.T).T + \
                      np.array([cam1['TX'], cam1['TY'], cam1['TZ']])
    projected_cam1_ref = np.array([project_point_to_image(pt, cam1) for pt in points_cam1_ref])
    valid_cam1 = ~np.isnan(projected_cam1_ref).any(axis=1)
    reference_pixels_cam1 = projected_cam1_ref[valid_cam1]
    measured_disp_pixels_cam1 = measured_disp_pixels_cam0[valid_cam1]
    
    # Load images
    print("Loading reference and deformed images...")
    reference_image_cam0 = cv2.imread(REF_CAM0, cv2.IMREAD_GRAYSCALE)
    reference_image_cam1 = cv2.imread(REF_CAM1, cv2.IMREAD_GRAYSCALE)
    actual_def_cam0 = cv2.imread(DEF_CAM0, cv2.IMREAD_GRAYSCALE)
    actual_def_cam1 = cv2.imread(DEF_CAM1, cv2.IMREAD_GRAYSCALE)
    
    # ===== RUN FULL RECONSTRUCTION =====
    results = full_inverse_reconstruction_stereo(
        reference_image_cam0, reference_image_cam1,
        reference_pixels_cam0, reference_pixels_cam1,
        measured_disp_pixels_cam0, measured_disp_pixels_cam1,
        points_plane_ref, dX, dY, dZ,
        R_world2plane, t_world2plane,
        cam0, cam1,
        subset_size=31,
        step_size=15
    )
    
    # ===== COMPUTE ERRORS =====
    errors = compute_reconstruction_errors(
        actual_def_cam0, actual_def_cam1,
        results['rec_cam0'], results['rec_cam1'],
        results['valid_region_cam0'], results['valid_region_cam1']
    )
    
    # ===== COMPUTE RRMSE (thresholded, float recon) =====
    rel_thresh = 5.0   # T2 chosen by you
    eps = 1e-6
    
    rrmse_cam0, rel_mask_cam0, rel_map_cam0 = compute_rrmse(
        actual_def_cam0, results['rec_cam0_float'], results['valid_region_cam0'],
        threshold=rel_thresh, eps=eps
    )
    rrmse_cam1, rel_mask_cam1, rel_map_cam1 = compute_rrmse(
        actual_def_cam1, results['rec_cam1_float'], results['valid_region_cam1'],
        threshold=rel_thresh, eps=eps
    )
    
    errors['cam0']['rrmse'] = rrmse_cam0
    errors['cam1']['rrmse'] = rrmse_cam1
    errors['cam0']['rrmap'] = rel_map_cam0
    errors['cam1']['rrmap'] = rel_map_cam1
    
    print("\nRRMSE Results (DIC-style, thresholded):")
    print(f" Camera 0: RRMSE = {rrmse_cam0:.6f} (~{100*rrmse_cam0:.2f}%)   (threshold actual > {rel_thresh})")
    print(f" Camera 1: RRMSE = {rrmse_cam1:.6f} (~{100*rrmse_cam1:.2f}%)   (threshold actual > {rel_thresh})")
    
    # Save textual summary
    with open("rrmse_summary.txt", "w") as f:
        f.write("RRMSE summary (DIC-style, thresholded on actual intensity)\n")
        f.write("="*60 + "\n")
        f.write(f"rel_thresh = {rel_thresh}\n\n")
        f.write(f"Camera 0 RRMSE (dimensionless): {rrmse_cam0:.8f}\n")
        f.write(f"Camera 0 RRMSE (~%): {100*rrmse_cam0:.4f}\n\n")
        f.write(f"Camera 1 RRMSE (dimensionless): {rrmse_cam1:.8f}\n")
        f.write(f"Camera 1 RRMSE (~%): {100*rrmse_cam1:.4f}\n")
    print("Saved: rrmse_summary.txt")
    
    # ===== VISUALIZE (all plots, plus RRMSE heatmaps) =====
    visualize_full_reconstruction(
        reference_image_cam0, reference_image_cam1,
        actual_def_cam0, actual_def_cam1,
        results['rec_cam0'], results['rec_cam1'],
        errors,
        results['valid_region_cam0'], results['valid_region_cam1'],
        rel_thresh_display=rel_thresh
    )
    
    print("\n" + "="*70)
    print(" FULL INVERSE RECONSTRUCTION COMPLETE!")
    print("="*70)
    print("\nGenerated files (additional):")
    print("  08_rrmse_heatmap_cam0.png")
    print("  09_rrmse_heatmap_cam1.png")
    print("  rrmse_summary.txt")
