import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# === Load reference image ===
ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"
ref_img = imageio.imread(ref_img_file).astype(float)
H, W = ref_img.shape

# === Load DICe subset data ===
df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt",
                 sep=",", comment="#")

x = df["COORDINATE_X"].values.astype(int)
y = df["COORDINATE_Y"].values.astype(int)

subset_size = 31  # from your DICe analysis
half = subset_size // 2

# === Build ROI mask from subset squares ===
roi_mask = np.zeros((H, W), dtype=bool)
for xi, yi in zip(x, y):
    x0, x1 = max(0, xi-half), min(W, xi+half+1)
    y0, y1 = max(0, yi-half), min(H, yi+half+1)
    roi_mask[y0:y1, x0:x1] = True

# === Auto-trim spill columns ===
xmin, xmax = x.min(), x.max()   # min/max subset centers
# enforce subset padding
xmin = max(0, xmin - half)
xmax = min(W, xmax + half + 1)

# clear everything outside this box
roi_mask[:, :xmin] = False
roi_mask[:, xmax:] = False

# === Overlay mask on reference image ===
plt.figure(figsize=(8,8))
plt.imshow(ref_img, cmap="gray")
plt.imshow(np.ma.masked_where(~roi_mask, roi_mask),
           cmap="Reds", alpha=0.4)
plt.title("Reference Image with Trimmed ROI Mask")
plt.axis("off")
plt.show()
