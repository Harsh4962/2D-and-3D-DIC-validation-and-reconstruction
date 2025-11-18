# # import numpy as np, cv2, pandas as pd, xml.etree.ElementTree as ET

# # # paths (adjust)
# # calib_xml = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\calibration_for_dice.xml"
# # excel_3d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC3D_to_Excel.xlsx"
# # frame = 10
# # sheet_pts = f"Points3D_f{frame:02d}"

# # # read 3D points
# # pts = pd.read_excel(excel_3d, sheet_name=sheet_pts)
# # X = pts[['X','Y','Z']].to_numpy(dtype=float)  # must be same units as T (likely mm)

# # # parse xml
# # root = ET.parse(calib_xml).getroot()
# # def read_cam(tag):
# #     node = root.find(tag)
# #     K = np.fromstring(node.find('K').text, sep=' ').reshape(3,3)
# #     dist = np.fromstring(node.find('dist').text, sep=' ')
# #     return K, dist
# # K1, dist1 = read_cam('left')
# # K2, dist2 = read_cam('right')

# # R = np.fromstring(root.find('extrinsics').find('R').text, sep=' ').reshape(3,3)
# # T = np.fromstring(root.find('extrinsics').find('T').text, sep=' ').reshape(3,1)

# # # project to left (assume left camera at identity)
# # rvec_left = np.zeros((3,1))
# # tvec_left = np.zeros((3,1))
# # proj1, _ = cv2.projectPoints(X, rvec_left, tvec_left, K1, dist1)
# # proj1 = proj1.reshape(-1,2)

# # # project to right using R,T as left->right (try this first)
# # rvec_right, _ = cv2.Rodrigues(R)
# # tvec_right = T
# # proj2, _ = cv2.projectPoints(X, rvec_right, tvec_right, K2, dist2)
# # proj2 = proj2.reshape(-1,2)

# # # load measured 2D points (from your DIC2D file)
# # cam1 = pd.read_excel(r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC2DpairResults.xlsx", sheet_name=f"cam1_f{frame:02d}")
# # cam2 = pd.read_excel(r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC2DpairResults.xlsx", sheet_name=f"cam2_f{frame:02d}")
# # u1 = cam1[['u','v']].to_numpy(dtype=float)
# # u2 = cam2[['u','v']].to_numpy(dtype=float)

# # # NOTE: you likely have different counts; match by PointID if available.
# # # For a quick check, compute nearest-neighbor matches from proj->measured (use KDTree)
# # from scipy.spatial import cKDTree
# # tree1 = cKDTree(u1); d1, idx1 = tree1.query(proj1, k=1)
# # tree2 = cKDTree(u2); d2, idx2 = tree2.query(proj2, k=1)

# # threshold = 10.0  # px
# # mask = (d1<threshold) & (d2<threshold)
# # print("Matched 3D points:", mask.sum(), "out of", len(X))
# # errs_left = np.linalg.norm(proj1[mask] - u1[idx1[mask]], axis=1)
# # errs_right = np.linalg.norm(proj2[mask] - u2[idx2[mask]], axis=1)
# # print("Left reproj mean/med (px):", errs_left.mean(), np.median(errs_left))
# # print("Right reproj mean/med (px):", errs_right.mean(), np.median(errs_right))

# #!/usr/bin/env python3
# """
# overlay_reproj.py
# Full reprojection overlay + histogram script.

# Edit the user settings below (file paths and frame number) and run:
# python overlay_reproj.py
# """

# import numpy as np
# import pandas as pd
# import xml.etree.ElementTree as ET
# import cv2
# from scipy.spatial import cKDTree
# import matplotlib.pyplot as plt
# from pathlib import Path

# # -------------------- USER SETTINGS --------------------
# calib_xml = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\calibration_for_dice.xml"         # your DICe xml (uploaded)
# excel_3d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC3D_to_Excel.xlsx"                # 3D results file you uploaded
# excel_2d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC2DpairResults.xlsx"              # 2D tracked points file you uploaded
# frame = 10                                      # frame index to inspect (integer)
# left_image_path = None                          # optional: set to explicit left image .tiff path
# right_image_path = None                         # optional: set to explicit right image .tiff path


# left_image_path  = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_01\Image_0019_0.tiff"
# right_image_path = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_02\Image_0019_0.tiff"

# # matching parameters
# threshold_px = 6.0      # accept match only if nearest measured 2D is within this many pixels
# require_both_cams = True # require both left & right measured points to be close (safer)

# # output files
# out_dir = Path("reproj_outputs")
# out_dir.mkdir(exist_ok=True)
# # -----------------------------------------------------

# def read_camera_from_dice_xml(root, tag):
#     node = root.find(tag)
#     if node is None:
#         raise ValueError(f"Camera tag '{tag}' not found in xml")
#     K = np.fromstring(node.find('K').text, sep=' ').reshape(3,3)
#     dist = np.fromstring(node.find('dist').text, sep=' ')
#     return K, dist

# def read_extrinsics_from_dice_xml(root):
#     node = root.find('extrinsics')
#     if node is None:
#         raise ValueError("extrinsics not found in xml")
#     R = np.fromstring(node.find('R').text, sep=' ').reshape(3,3)
#     T = np.fromstring(node.find('T').text, sep=' ')
#     return R, T

# def load_points3d(excel_3d, frame):
#     sheet = f"Points3D_f{frame:02d}"
#     df = pd.read_excel(excel_3d, sheet_name=sheet)
#     if not all(c in df.columns for c in ['X','Y','Z']):
#         raise ValueError(f"Sheet {sheet} must contain columns X,Y,Z")
#     X = df[['X','Y','Z']].to_numpy(dtype=float)
#     return X, df

# def load_2d_points(excel_2d, frame):
#     # expects sheets cam1_fXX and cam2_fXX with columns u,v
#     s1 = f"cam1_f{frame:02d}"
#     s2 = f"cam2_f{frame:02d}"
#     df1 = pd.read_excel(excel_2d, sheet_name=s1)
#     df2 = pd.read_excel(excel_2d, sheet_name=s2)
#     if not all(c in df1.columns for c in ['u','v']) or not all(c in df2.columns for c in ['u','v']):
#         raise ValueError("cam1/cam2 sheets must have 'u' and 'v' columns")
#     u1 = df1[['u','v']].to_numpy(dtype=float)
#     u2 = df2[['u','v']].to_numpy(dtype=float)
#     return u1, u2, df1, df2

# def parse_calib(calib_xml):
#     root = ET.parse(calib_xml).getroot()
#     K1, dist1 = read_camera_from_dice_xml(root, 'left')
#     K2, dist2 = read_camera_from_dice_xml(root, 'right')
#     R, T = read_extrinsics_from_dice_xml(root)
#     return K1, dist1, K2, dist2, R, T

# def project_points(X, K, dist, rvec, tvec):
#     # X: (N,3), rvec/tvec are OpenCV shape (3,1)
#     proj, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
#     return proj.reshape(-1,2)

# def find_matches(proj1, proj2, u1_meas, u2_meas, threshold_px, require_both):
#     # KD-tree NN
#     tree1 = cKDTree(u1_meas)
#     d1, idx1 = tree1.query(proj1, k=1)

#     tree2 = cKDTree(u2_meas)
#     d2, idx2 = tree2.query(proj2, k=1)

#     if require_both:
#         mask = (d1 <= threshold_px) & (d2 <= threshold_px)
#     else:
#         mask = (d1 <= threshold_px)
#     return mask, idx1, idx2, d1, d2

# def overlay_and_histogram(img_path, proj_pts, meas_pts, mask, idx_meas, title_prefix, outfile_prefix):
#     # Read image
#     img = plt.imread(img_path)
#     plt.figure(figsize=(9,7))
#     plt.imshow(img, cmap='gray')
#     # measured (for matched subset) - measured indices come from idx_meas
#     meas = meas_pts[idx_meas[mask]]
#     proj = proj_pts[mask]
#     plt.scatter(meas[:,0], meas[:,1], c='yellow', s=8, label='measured', edgecolors='k')
#     plt.scatter(proj[:,0], proj[:,1], c='red', s=6, label='projected', alpha=0.7)
#     plt.legend(loc='upper right')
#     plt.gca().invert_yaxis()
#     plt.title(f"{title_prefix}: measured (yellow) vs projected (red)")
#     plt.tight_layout()
#     out_img = out_dir / f"{outfile_prefix}_overlay.png"
#     plt.savefig(out_img, dpi=300, bbox_inches='tight')
#     plt.show()

#     # histogram of reprojection errors for matched set
#     errs = np.linalg.norm(proj_pts[mask] - meas_pts[idx_meas[mask]], axis=1)
#     plt.figure(figsize=(6,4))
#     plt.hist(errs, bins=40, color='C3', edgecolor='k')
#     plt.xlabel('Reprojection error (px)')
#     plt.ylabel('Count')
#     plt.title(f"{title_prefix} reprojection errors (matched count = {mask.sum()})")
#     out_hist = out_dir / f"{outfile_prefix}_hist.png"
#     plt.savefig(out_hist, dpi=300, bbox_inches='tight')
#     plt.show()
#     return errs

# def main():
#     global left_image_path, right_image_path

#     print("Loading calibration...")
#     K1, dist1, K2, dist2, R, T = parse_calib(calib_xml)
#     print("K1:\n", K1)
#     print("K2:\n", K2)
#     print("R:\n", R)
#     print("T:\n", T)

#     print(f"\nLoading 3D points (frame {frame})...")
#     X, df_pts = load_points3d(excel_3d, frame)
#     print("3D points:", X.shape[0])

#     print("Loading 2D measured points...")
#     u1_meas, u2_meas, df_cam1, df_cam2 = load_2d_points(excel_2d, frame)
#     print("Measured left 2D points:", u1_meas.shape[0], "Measured right 2D points:", u2_meas.shape[0])

#     # compute projections
#     print("\nProjecting 3D points into cameras (assume left is reference)...")
#     rvec_left = np.zeros((3,1), dtype=float)
#     tvec_left = np.zeros((3,1), dtype=float)
#     proj1 = project_points(X, K1, dist1, rvec_left, tvec_left)

#     rvec_right, _ = cv2.Rodrigues(R)   # R as given in xml (try R.T or -T if needed)
#     tvec_right = T.reshape(3,1)
#     proj2 = project_points(X, K2, dist2, rvec_right, tvec_right)

#     # find matches
#     print("\nFinding nearest-measured 2D points for projected points (threshold px = {})...".format(threshold_px))
#     mask, idx1, idx2, d1, d2 = find_matches(proj1, proj2, u1_meas, u2_meas, threshold_px, require_both_cams)
#     print(f"Matched 3D points: {mask.sum()} out of {X.shape[0]}")

#     # compute basic statistics
#     if mask.sum() > 0:
#         errs_left = np.linalg.norm(proj1[mask] - u1_meas[idx1[mask]], axis=1)
#         errs_right = np.linalg.norm(proj2[mask] - u2_meas[idx2[mask]], axis=1)
#         print("Left reproj mean/med (px):", errs_left.mean(), np.median(errs_left))
#         print("Right reproj mean/med (px):", errs_right.mean(), np.median(errs_right))
#     else:
#         print("No matches found, increase threshold_px or check alignment.")
#         return

#     # Attempt to figure out image paths if not given: read ImagePaths sheet from 2D excel (if exists)
#     if left_image_path is None or right_image_path is None:
#         try:
#             ip = pd.read_excel(excel_2d, sheet_name="ImagePaths")
#             # pick image path for the chosen frame (assumes sequential indexing)
#             # If the sheet has a column 'ImagePath', else user must set manually
#             if 'ImagePath' in ip.columns:
#                 # find a path that contains left/right suffix for this frame
#                 # simple heuristic: look for strings containing f{frame:04d}_0 or _1 etc
#                 # otherwise pick first two entries
#                 cand = ip['ImagePath'].astype(str).values
#                 # try simple matching by frame number substring
#                 fstr0 = f"Image_{frame:04d}_0"
#                 fstr1 = f"Image_{frame:04d}_1"
#                 left_match = next((p for p in cand if fstr0 in p), None)
#                 right_match = next((p for p in cand if fstr1 in p), None)
#                 if left_match and right_match:
#                     left_image_path = left_match
#                     right_image_path = right_match
#                 else:
#                     # fallback to first two paths if available
#                     left_image_path = cand[0] if cand.size>0 else None
#                     right_image_path = cand[1] if cand.size>1 else None
#         except Exception:
#             pass

#     if left_image_path is None or right_image_path is None:
#         print("\nWarning: Could not auto-detect image paths. Set left_image_path and right_image_path in script.")
#         # but still continue to produce histograms and stats; overlays require images

#     # produce overlays & histograms (left)
#     if left_image_path is not None:
#         print("\nProducing left overlay and histogram...")
#         errsL = overlay_and_histogram(left_image_path, proj1, u1_meas, mask, idx1,
#                                       title_prefix=f"Left frame {frame}", outfile_prefix=f"left_f{frame:02d}")
#     else:
#         print("Skipping left overlay (no left_image_path).")

#     # right
#     if right_image_path is not None:
#         print("\nProducing right overlay and histogram...")
#         errsR = overlay_and_histogram(right_image_path, proj2, u2_meas, mask, idx2,
#                                       title_prefix=f"Right frame {frame}", outfile_prefix=f"right_f{frame:02d}")
#     else:
#         print("Skipping right overlay (no right_image_path).")

#     print("\nSaved images (if overlays produced) to:", out_dir.resolve())

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
reproj_residuals.py
Compute and visualize reprojection residuals (mean vector + magnitude histogram).

Run:  python reproj_residuals.py
"""

import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- USER SETTINGS ----------
calib_xml = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\calibration_for_dice.xml"         # your DICe xml (uploaded)
excel_3d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC3D_to_Excel.xlsx"                # 3D results file you uploaded
excel_2d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC2DpairResults.xlsx"
frame     = 10
threshold_px = 6.0
require_both = True

# Optional overlay
left_image_path  = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_01\Image_0019_0.tiff"
right_image_path = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_02\Image_0019_1.tiff"

out_dir = Path("reproj_residuals_output"); out_dir.mkdir(exist_ok=True)
# -----------------------------------

def read_cam(root, tag):
    node = root.find(tag)
    K = np.fromstring(node.find('K').text, sep=' ').reshape(3,3)
    dist = np.fromstring(node.find('dist').text, sep=' ')
    return K, dist

# Load calibration
root = ET.parse(calib_xml).getroot()
K1, dist1 = read_cam(root, 'left')
K2, dist2 = read_cam(root, 'right')
R = np.fromstring(root.find('extrinsics').find('R').text, sep=' ').reshape(3,3)
T = np.fromstring(root.find('extrinsics').find('T').text, sep=' ')
rvec_right, _ = cv2.Rodrigues(R)
tvec_right = T.reshape(3,1)

# Load data
Pts3D = pd.read_excel(excel_3d, sheet_name=f"Points3D_f{frame:02d}")
X = Pts3D[['X','Y','Z']].to_numpy(float)
u1 = pd.read_excel(excel_2d, sheet_name=f"cam1_f{frame:02d}")[['u','v']].to_numpy(float)
u2 = pd.read_excel(excel_2d, sheet_name=f"cam2_f{frame:02d}")[['u','v']].to_numpy(float)

# Project 3D → 2D
proj1, _ = cv2.projectPoints(X, np.zeros((3,1)), np.zeros((3,1)), K1, dist1)
proj2, _ = cv2.projectPoints(X, rvec_right, tvec_right, K2, dist2)
proj1, proj2 = proj1.reshape(-1,2), proj2.reshape(-1,2)

# Match projected ↔ measured
tree1, tree2 = cKDTree(u1), cKDTree(u2)
d1, idx1 = tree1.query(proj1, k=1)
d2, idx2 = tree2.query(proj2, k=1)
mask = (d1 <= threshold_px) & (d2 <= threshold_px) if require_both else (d1 <= threshold_px)

# Residual analysis
residuals = proj1[mask] - u1[idx1[mask]]  # Nx2
magnitudes = np.linalg.norm(residuals, axis=1)
mean_vec = residuals.mean(axis=0)
median_mag = np.median(magnitudes)

print(f"Matched: {mask.sum()} / {len(mask)}")
print(f"Mean residual vector (px): {mean_vec}")
print(f"Median magnitude (px): {median_mag:.3f}")

# --- Visualization of residual arrows on left image ---
if left_image_path:
    img = plt.imread(left_image_path)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.imshow(img, cmap='gray')
    meas = u1[idx1[mask]]
    proj = proj1[mask]
    res = meas - proj
    ax.scatter(meas[:,0], meas[:,1], c='y', s=8, label='measured')
    ax.scatter(proj[:,0], proj[:,1], c='r', s=6, label='projected')
    # draw arrows every nth point to avoid clutter
    nth = max(1, int(len(res)/200))
    for i in range(0, len(res), nth):
        ax.arrow(proj[i,0], proj[i,1], res[i,0], res[i,1],
                 head_width=2.0, head_length=2.0, fc='cyan', ec='cyan', linewidth=0.5, alpha=0.8)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"Residual arrows (frame {frame})")
    fig.tight_layout()
    out_path = out_dir / f"residual_arrows_f{frame:02d}.png"
    fig.savefig(out_path, dpi=300)
    plt.show()

# --- Histogram of residual magnitudes ---
plt.figure(figsize=(6,4))
plt.hist(magnitudes, bins=40, color='steelblue', edgecolor='k')
plt.xlabel('Residual magnitude (px)')
plt.ylabel('Count')
plt.title(f'Reprojection residuals (frame {frame})')
plt.tight_layout()
plt.savefig(out_dir / f"residual_hist_f{frame:02d}.png", dpi=300)
plt.show()
