# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import os

# # # ==============================
# # # PATH SETTINGS
# # # ==============================
# # DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"

# # # ==============================
# # # LOAD DICe DATA
# # # ==============================
# # def load_dic_data(path):
# #     """Read DICe text output robustly (handles commas or whitespace)."""
# #     try:
# #         df = pd.read_csv(path, sep=r'\s+|,', engine='python', comment='#')
# #     except Exception as e:
# #         raise RuntimeError(f"Error reading file: {e}")
# #     return df

# # df = load_dic_data(DATA_PATH)
# # print("Columns found:", list(df.columns))

# # # ==============================
# # # EXTRACT IMAGE-PLANE COORDINATES
# # # ==============================
# # # In DICe, these are typically the left-camera subset centers in pixel coordinates
# # x_col, y_col = "COORDINATE_X", "COORDINATE_Y"

# # if x_col not in df.columns or y_col not in df.columns:
# #     raise ValueError(f"Could not find {x_col}, {y_col} in data columns.")

# # X_img = df[x_col].astype(float).to_numpy()
# # Y_img = df[y_col].astype(float).to_numpy()

# # # ==============================
# # # VISUALIZE IMAGE PLANE DISTRIBUTION
# # # ==============================
# # plt.figure(figsize=(7, 6))
# # plt.scatter(X_img, Y_img, s=8, color='royalblue', alpha=0.6, edgecolor='k', linewidth=0.2)
# # plt.gca().invert_yaxis()  # because image origin is top-left
# # plt.title("Subset Center Distribution in Left Camera Image Plane")
# # plt.xlabel("COORDINATE_X (pixels)")
# # plt.ylabel("COORDINATE_Y (pixels)")
# # plt.axis('equal')
# # plt.grid(alpha=0.3)
# # plt.tight_layout()
# # plt.show()

# # # ==============================
# # # PRINT ROI STATS
# # # ==============================
# # print(f"COORDINATE_X range: {X_img.min():.1f} – {X_img.max():.1f} px")
# # print(f"COORDINATE_Y range: {Y_img.min():.1f} – {Y_img.max():.1f} px")
# # print(f"ROI width:  {X_img.max() - X_img.min():.1f} px")
# # print(f"ROI height: {Y_img.max() - Y_img.min():.1f} px")
# # print(f"Number of subsets: {len(X_img)}")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # ---------------------------
# # paths (edit these two only)
# # ---------------------------
# LEFT_IMG_PATH = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
# DICE_TXT_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"  # <-- put your DICe text file here
# OUT_DIR       = r"C:\Users\harsh\OneDrive\Desktop\UGP\overlays"
# os.makedirs(OUT_DIR, exist_ok=True)

# # ---------------------------
# # load DICe text (robust to commas/whitespace)
# # ---------------------------
# def load_dice_table(path):
#     try:
#         df = pd.read_csv(path, sep=r'\s+|,', engine='python', comment='#')
#     except Exception as e:
#         raise RuntimeError(f"Could not read DICe file: {e}")
#     return df

# df = load_dice_table(DICE_TXT_PATH)
# cols = set(df.columns.str.strip())
# print("Columns found:", sorted(cols))

# # required columns
# x_col, y_col = "COORDINATE_X", "COORDINATE_Y"
# if x_col not in cols or y_col not in cols:
#     raise ValueError(f"Missing {x_col} / {y_col} in DICe file. Got: {sorted(cols)}")

# # optional: use STATUS_FLAG to mask bad subsets if present
# mask = np.ones(len(df), dtype=bool)
# if "STATUS_FLAG" in cols:
#     # in DICe, 0 is usually 'good'; keep only zeros
#     mask = (df["STATUS_FLAG"].astype(float).to_numpy() == 0)

# u = df[x_col].astype(float).to_numpy()[mask]
# v = df[y_col].astype(float).to_numpy()[mask]

# # ---------------------------
# # load the left image
# # ---------------------------
# img = plt.imread(LEFT_IMG_PATH)
# H, W = img.shape[0], img.shape[1]
# print(f"Image size: {W}x{H}px ; plotting {u.size} subsets")

# # ---------------------------
# # figure 1: subset centers overlay
# # ---------------------------
# plt.figure(figsize=(10,6))
# plt.imshow(img, cmap='gray')
# plt.scatter(u, v, s=8, c='yellow', edgecolors='k', linewidths=0.2, alpha=0.9, label='subset centers')
# plt.gca().invert_yaxis()          # image coordinates: y increases downward
# plt.axis('equal')
# plt.xlim(0, W); plt.ylim(H, 0)    # ensure full image frame
# plt.title("Left image with DICe subset centers")
# plt.xlabel("pixels"); plt.ylabel("pixels")
# plt.legend(loc='upper right', frameon=True)
# plt.tight_layout()
# save1 = os.path.join(OUT_DIR, "left_overlay_subsets.png")
# plt.savefig(save1, dpi=200)
# plt.show()
# print(f"Saved: {save1}")

# # ---------------------------
# # figure 2 (optional): 2D displacement arrows in image plane
# # ---------------------------
# if {"DISPLACEMENT_X", "DISPLACEMENT_Y"}.issubset(cols):
#     du_all = df["DISPLACEMENT_X"].astype(float).to_numpy()
#     dv_all = df["DISPLACEMENT_Y"].astype(float).to_numpy()
#     du = du_all[mask]
#     dv = dv_all[mask]

#     # sparsify for clarity (one arrow every ~N points)
#     step = max(1, len(u)//800)
#     idx = np.arange(0, len(u), step)

#     plt.figure(figsize=(10,6))
#     plt.imshow(img, cmap='gray')
#     plt.quiver(u[idx], v[idx], du[idx], dv[idx],
#                angles='xy', scale_units='xy', scale=1.0,
#                color='red', width=0.003, headwidth=3, headlength=4, headaxislength=3)
#     plt.scatter(u[idx], v[idx], s=6, c='yellow', edgecolors='k', linewidths=0.2, alpha=0.9)
#     plt.gca().invert_yaxis()
#     plt.axis('equal')
#     plt.xlim(0, W); plt.ylim(H, 0)
#     plt.title("Left image: subset centers + image-plane displacement vectors")
#     plt.xlabel("pixels"); plt.ylabel("pixels")
#     plt.tight_layout()
#     save2 = os.path.join(OUT_DIR, "left_overlay_displacements.png")
#     plt.savefig(save2, dpi=200)
#     plt.show()
#     print(f"Saved: {save2}")
# else:
#     print("DISPLACEMENT_X/Y not found—skipping arrows figure.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# paths
# ---------------------------
LEFT_IMG_PATH = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
DICE_TXT_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
OUT_DIR       = r"C:\Users\harsh\OneDrive\Desktop\UGP\overlays"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# load DICe data (robust)
# ---------------------------
def load_dice_table(path):
    try:
        df = pd.read_csv(path, sep=r'\s+|,', engine='python', comment='#')
    except Exception as e:
        raise RuntimeError(f"Could not read DICe file: {e}")
    # strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_dice_table(DICE_TXT_PATH)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Head:\n", df.head(3))

if "STATUS_FLAG" in df.columns:
    try:
        vc = df["STATUS_FLAG"].astype(float).value_counts(dropna=False).sort_index()
        print("\nSTATUS_FLAG counts:\n", vc)
    except Exception:
        print("\nSTATUS_FLAG present but not numeric.")

# ---------------------------
# pick coordinate columns (try common alternatives)
# ---------------------------
candidates_x = ["COORDINATE_X", "X", "u", "U", "IMAGE_COORDINATE_X"]
candidates_y = ["COORDINATE_Y", "Y", "v", "V", "IMAGE_COORDINATE_Y"]

def first_present(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

x_col = first_present(candidates_x)
y_col = first_present(candidates_y)

if x_col is None or y_col is None:
    raise ValueError("Could not find 2D image coordinate columns among: "
                     f"{candidates_x} / {candidates_y}")

print(f"\nUsing columns: {x_col} / {y_col}")

# keep only finite rows
u_all = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
v_all = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
finite_mask = np.isfinite(u_all) & np.isfinite(v_all)

u = u_all[finite_mask]
v = v_all[finite_mask]
print(f"Total rows: {len(df)} | finite (u,v): {u.size}")

# ---------------------------
# load image
# ---------------------------
img = plt.imread(LEFT_IMG_PATH)
H, W = img.shape[0], img.shape[1]
print(f"Image size: {W}x{H}px")

# how many points lie inside the image bounds?
in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
print(f"Points inside image bounds: {in_img.sum()} / {u.size}")

# ---------------------------
# overlay subsets
# ---------------------------
plt.figure(figsize=(10,6))
plt.imshow(img, cmap='gray')
plt.scatter(u[in_img], v[in_img], s=8, c='yellow', edgecolors='k', linewidths=0.2, alpha=0.9, label='subset centers')
plt.gca().invert_yaxis()
plt.axis('equal')
plt.xlim(0, W); plt.ylim(H, 0)
plt.title("Left image with DICe subset centers (no STATUS_FLAG filtering)")
plt.xlabel("pixels"); plt.ylabel("pixels")
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
save1 = os.path.join(OUT_DIR, "left_overlay_subsets.png")
plt.savefig(save1, dpi=200)
plt.show()
print(f"Saved: {save1}")

# ---------------------------
# optional: image-plane displacement arrows (if available)
# ---------------------------
has_disp = {"DISPLACEMENT_X", "DISPLACEMENT_Y"}.issubset(df.columns)
if has_disp:
    du_all = pd.to_numeric(df["DISPLACEMENT_X"], errors="coerce").to_numpy()
    dv_all = pd.to_numeric(df["DISPLACEMENT_Y"], errors="coerce").to_numpy()
    du = du_all[finite_mask][in_img]
    dv = dv_all[finite_mask][in_img]
    uu = u[in_img]
    vv = v[in_img]

    step = max(1, uu.size // 800)
    idx = np.arange(0, uu.size, step)

    plt.figure(figsize=(10,6))
    plt.imshow(img, cmap='gray')
    plt.quiver(uu[idx], vv[idx], du[idx], dv[idx],
               angles='xy', scale_units='xy', scale=1.0,
               color='red', width=0.003, headwidth=3, headlength=4, headaxislength=3)
    plt.scatter(uu[idx], vv[idx], s=6, c='yellow', edgecolors='k', linewidths=0.2, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xlim(0, W); plt.ylim(H, 0)
    plt.title("Left image: subset centers + image-plane displacement vectors")
    plt.xlabel("pixels"); plt.ylabel("pixels")
    plt.tight_layout()
    save2 = os.path.join(OUT_DIR, "left_overlay_displacements.png")
    plt.savefig(save2, dpi=200)
    plt.show()
    print(f"Saved: {save2}")
else:
    print("DISPLACEMENT_X/Y not found — skipping arrows.")
