import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import convex_hull_image
from skimage.exposure import match_histograms

# === Inputs ===
ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed) image
def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed image
dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

subset_size = 31   # DICe subset size

# === Load reference and deformed images ===
ref_img = imageio.imread(ref_img_file).astype(float)
def_img = imageio.imread(def_img_file).astype(float)
H, W = ref_img.shape

# === Load DICe displacement data ===
df = pd.read_csv(dice_file, sep=",", comment="#")
x = df["COORDINATE_X"].values
y = df["COORDINATE_Y"].values
u_vals = df["DISPLACEMENT_X"].values
v_vals = df["DISPLACEMENT_Y"].values

# === Build ROI mask (convex hull of subset centers) ===
mask = np.zeros((H, W), dtype=bool)
xc = np.clip(x.astype(int), 0, W-1)
yc = np.clip(y.astype(int), 0, H-1)
mask[yc, xc] = True
mask = convex_hull_image(mask)

# === Interpolate displacements on full grid ===
grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
u = griddata((x, y), u_vals, (grid_x, grid_y), method="cubic", fill_value=0)
v = griddata((x, y), v_vals, (grid_x, grid_y), method="cubic", fill_value=0)

# === Warp reference image using subpixel interpolation ===
yy, xx = np.indices((H, W))
src_x = np.clip(xx - u, 0, W-1)
src_y = np.clip(yy - v, 0, H-1)

coords = np.array([src_y.ravel(), src_x.ravel()])
warped = map_coordinates(ref_img, coords, order=1, mode="reflect").reshape(H, W)

# Apply ROI mask (outside ROI = unchanged reference)
reconstructed = ref_img.copy()
reconstructed[mask] = warped[mask]

# Histogram match inside ROI
reconstructed_matched = reconstructed.copy()
reconstructed_matched[mask] = match_histograms(
    reconstructed[mask], def_img[mask]
)

# === Error analysis restricted to ROI ===
errors = (def_img[mask] - reconstructed_matched[mask]).ravel()
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
p_within_5  = np.mean(np.abs(errors) <= 5) * 100
p_within_10 = np.mean(np.abs(errors) <= 10) * 100
ssim_val = ssim(def_img[mask], reconstructed_matched[mask],
                data_range=def_img.max()-def_img.min())

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"SSIM: {ssim_val:.3f}")
print(f"% pixels within ±5 intensity error:  {p_within_5:.2f}%")
print(f"% pixels within ±10 intensity error: {p_within_10:.2f}%")

# === Error map ===
error_map = np.zeros_like(ref_img)
error_map[mask] = def_img[mask] - reconstructed_matched[mask]


# === Error Histogram ===
roi_errors = (def_img[mask] - reconstructed_matched[mask]).ravel()

plt.figure(figsize=(6,4))
plt.hist(roi_errors, bins=50, color="steelblue", edgecolor="k", alpha=0.7)
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.axvline(5, color="green", linestyle="--", linewidth=1, label="±5")
plt.axvline(-5, color="green", linestyle="--", linewidth=1)
plt.axvline(10, color="orange", linestyle="--", linewidth=1, label="±10")
plt.axvline(-10, color="orange", linestyle="--", linewidth=1)
plt.xlabel("Pixel Intensity Error (Actual - Reconstructed)")
plt.ylabel("Frequency")
plt.title("Distribution of Errors within ROI")
plt.legend()
plt.tight_layout()
plt.show()

#=== Visualization ===
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
plt.subplot(1,3,2); plt.imshow(reconstructed_matched, cmap="gray"); plt.title("Reconstructed (Subpixel, ROI)")
plt.subplot(1,3,3)
plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
plt.colorbar(label="Error (intensity)")
plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, MAE={mae:.2f}, SSIM={ssim_val:.3f}")
plt.tight_layout()
plt.show()
