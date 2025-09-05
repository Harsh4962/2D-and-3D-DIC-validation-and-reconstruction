import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- User settings ---
roi_path = r"C:\Users\harsh\Downloads\2dDIC_dataset\roi.tif"   # your binary ROI tif (white specimen, black background)
subset_radius = 15          # half subset size in pixels
step = 15                   # spacing between subset centers (pixels)
output_file = "subset_coords_exp1.txt"

# --- Load ROI mask ---
roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
roi = (roi > 127).astype(np.uint8)  # ensure binary (0/1)

h, w = roi.shape
coords = []

# --- Loop over grid points ---
for y in range(subset_radius, h - subset_radius, step):
    for x in range(subset_radius, w - subset_radius, step):
        # check if full subset fits inside ROI
        subset = roi[y - subset_radius:y + subset_radius + 1,
                     x - subset_radius:x + subset_radius + 1]
        if subset.min() == 1:  # all pixels inside ROI
            coords.append([x, y])

# --- Save coordinates ---
with open(output_file, "w") as f:
    f.write("x y\n")  # header line
    for c in coords:
        f.write(f"{c[0]} {c[1]}\n")

print(f"Saved {len(coords)} subset centers to {output_file}")

# --- Optional: visualize ---
plt.imshow(roi, cmap="gray")
ys, xs = zip(*coords)
plt.scatter(xs, ys, s=10, c='r')
plt.title("ROI with subset centers")
plt.gca().invert_yaxis()  # match image coordinates
plt.show()
