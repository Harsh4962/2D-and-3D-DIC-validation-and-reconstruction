#!/usr/bin/env python3
"""
viz_2d_matches.py
Visualize 2D stereo matching results from DIC2DpairResults.xlsx.

Edit USER SETTINGS below and run:
    python viz_2d_matches.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import cKDTree
from pathlib import Path

# ----------------- USER SETTINGS -----------------
excel_2d = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC2DpairResults.xlsx"   # path to your Excel
frame = 10                            # which frame to visualize (integer)
# Optional: paths to the left/right images for overlay. Set to None to skip.
left_image_path  = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_01\Image_0001_0.tiff"
right_image_path = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\sample_data\uniaxial_tension\cam_01\Image_0001_0.tiff"

out_dir = Path("viz_2d_outputs"); out_dir.mkdir(exist_ok=True)
# -------------------------------------------------

def load_sheet(excel_file, cam_tag, frame):
    sheet = f"{cam_tag}_f{frame:02d}"
    df = pd.read_excel(excel_file, sheet_name=sheet)
    return df

def find_uv_cols(df):
    # look for common names
    candidates = [('u','v'), ('U','V'), ('x','y'), ('X_px','Y_px'), ('x_px','y_px')]
    for a,b in candidates:
        if a in df.columns and b in df.columns:
            return a,b
    # fallback: take first two numeric columns
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    raise RuntimeError("Cannot find 2D u/v columns in sheet. Columns: " + ", ".join(df.columns))

def guess_quality_col(df):
    for cand in ['CorrCoeff','corr','Quality','FaceCorrComb','quality','corr_coeff']:
        if cand in df.columns:
            return cand
    return None

# Load data
cam1 = load_sheet(excel_2d, "cam1", frame)
cam2 = load_sheet(excel_2d, "cam2", frame)
u1_col, v1_col = find_uv_cols(cam1)
u2_col, v2_col = find_uv_cols(cam2)
u1 = cam1[[u1_col, v1_col]].to_numpy(dtype=float)
u2 = cam2[[u2_col, v2_col]].to_numpy(dtype=float)
print(f"Loaded cam1 {u1.shape[0]} points, cam2 {u2.shape[0]} points (frame {frame})")

# pick quality columns if present
q1_col = guess_quality_col(cam1)
q2_col = guess_quality_col(cam2)
q1 = cam1[q1_col].to_numpy(dtype=float) if q1_col else None
q2 = cam2[q2_col].to_numpy(dtype=float) if q2_col else None
if q1_col: print("Found quality column in cam1:", q1_col)
if q2_col: print("Found quality column in cam2:", q2_col)

# If rows correspond by index (common), we can directly pair them.
# If counts differ, do nearest-neighbor matching from cam1->cam2
pairs = None
if u1.shape[0] == u2.shape[0]:
    pairs = np.vstack([np.arange(u1.shape[0]), np.arange(u1.shape[0])]).T
else:
    # find nearest neighbor (use KDTree). This will give 1-to-many if dense.
    tree = cKDTree(u2)
    dists, idx2 = tree.query(u1, k=1)
    pairs = np.vstack([np.arange(u1.shape[0]), idx2]).T
    print("Note: matched cam1->cam2 by nearest neighbor (counts differ).")

# Optionally compute matching distance for diagnostics
dists = np.linalg.norm(u1[pairs[:,0]] - u2[pairs[:,1]], axis=1)

# ---- Plot 1: overlay points on left image ----
if left_image_path:
    imgL = plt.imread(left_image_path)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(imgL, cmap='gray')
    ax.scatter(u1[:,0], u1[:,1], c='yellow', s=10, edgecolors='k')
    ax.set_title(f"Left points (frame {frame})")
    ax.invert_yaxis(); ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
    fig.tight_layout(); fig.savefig(out_dir / f"left_points_f{frame:02d}.png", dpi=200)
    plt.show(); plt.close(fig)

# ---- Plot 2: overlay points on right image ----
if right_image_path:
    imgR = plt.imread(right_image_path)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(imgR, cmap='gray')
    ax.scatter(u2[:,0], u2[:,1], c='yellow', s=10, edgecolors='k')
    ax.set_title(f"Right points (frame {frame})")
    ax.invert_yaxis(); ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
    fig.tight_layout(); fig.savefig(out_dir / f"right_points_f{frame:02d}.png", dpi=200)
    plt.show(); plt.close(fig)

# ---- Plot 3: color-by-quality overlays if available ----
if q1 is not None and left_image_path:
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(imgL, cmap='gray')
    sc = ax.scatter(u1[:,0], u1[:,1], c=q1, cmap='plasma', s=20, edgecolors='k', linewidths=0.2)
    ax.invert_yaxis(); plt.colorbar(sc, ax=ax, label=q1_col)
    ax.set_title(f"Left points colored by {q1_col}")
    fig.tight_layout(); fig.savefig(out_dir / f"left_quality_f{frame:02d}.png", dpi=200)
    plt.show(); plt.close(fig)

if q2 is not None and right_image_path:
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(imgR, cmap='gray')
    sc = ax.scatter(u2[:,0], u2[:,1], c=q2, cmap='plasma', s=20, edgecolors='k', linewidths=0.2)
    ax.invert_yaxis(); plt.colorbar(sc, ax=ax, label=q2_col)
    ax.set_title(f"Right points colored by {q2_col}")
    fig.tight_layout(); fig.savefig(out_dir / f"right_quality_f{frame:02d}.png", dpi=200)
    plt.show(); plt.close(fig)

# ---- Plot 4: side-by-side correspondences with lines ----
fig, axs = plt.subplots(1,2, figsize=(14,8))
if left_image_path:
    axs[0].imshow(imgL, cmap='gray')
else:
    axs[0].set_facecolor('k')
axs[0].scatter(u1[:,0], u1[:,1], c='yellow', s=10, edgecolors='k')
axs[0].set_title('Left'); axs[0].invert_yaxis()

if right_image_path:
    axs[1].imshow(imgR, cmap='gray')
else:
    axs[1].set_facecolor('k')
axs[1].scatter(u2[:,0], u2[:,1], c='yellow', s=10, edgecolors='k')
axs[1].set_title('Right'); axs[1].invert_yaxis()

# draw connecting lines (a few, to avoid clutter)
n_show = min(300, pairs.shape[0])
idx_show = np.linspace(0, pairs.shape[0]-1, n_show, dtype=int)
for i in idx_show:
    i1, i2 = pairs[i]
    # convert coordinates into subplot coordinates by transforming x positions for right image
    p1 = u1[i1]
    p2 = u2[i2]
    # draw line from left subplot to right subplot using figure coordinates
    # get axis transforms
    ax1 = axs[0]; ax2 = axs[1]
    # transform data->display coords
    disp1 = ax1.transData.transform(p1)
    disp2 = ax2.transData.transform(p2)
    # create a line in figure coordinates by transforming to figure coordinates
    fig_coord1 = fig.transFigure.inverted().transform(disp1)
    fig_coord2 = fig.transFigure.inverted().transform(disp2)
    line = plt.Line2D((fig_coord1[0], fig_coord2[0]), (fig_coord1[1], fig_coord2[1]),
                      transform=fig.transFigure, color='cyan', alpha=0.25, linewidth=0.4)
    fig.add_artist(line)

fig.suptitle(f"Correspondences (showing {n_show} lines) frame {frame}")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(out_dir / f"correspondences_f{frame:02d}.png", dpi=200)
plt.show(); plt.close(fig)

# ---- Plot 5: histograms ----
plt.figure(figsize=(6,4))
plt.hist(dists, bins=60, color='C0', edgecolor='k')
plt.xlabel('u-v distance (px) between matched cams'); plt.ylabel('count')
plt.title('Matching distances (cam1->cam2)')
plt.tight_layout(); plt.savefig(out_dir / f"match_dists_hist_f{frame:02d}.png", dpi=200)
plt.show(); plt.close()

if q1 is not None:
    plt.figure(figsize=(6,4))
    plt.hist(q1, bins=60, color='C2', edgecolor='k')
    plt.title(f"{q1_col} distribution (left)")
    plt.tight_layout(); plt.savefig(out_dir / f"left_quality_hist_f{frame:02d}.png", dpi=200)
    plt.show(); plt.close()

print("Saved outputs to", out_dir.resolve())
