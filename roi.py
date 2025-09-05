# This script converts a binary ROI mask (TIF/PNG/JPG) into a DICe subset locations file.
# It uses pure NumPy + PIL so it should run anywhere.
#
# Usage example (after downloading this file):
#   python roi_to_subset_file.py --mask /path/to/roi.tif --subset-size 29 --step 10 --out subset_locations.txt
#
# The script ensures each subset window of size (subset_size x subset_size) lies fully inside the white (non-zero) ROI.
#
# I will save this as /mnt/data/roi_to_subset_file.py so you can download it.

from PIL import Image
import numpy as np
import argparse
import os
import textwrap

script_path = "/mnt/data/roi_to_subset_file.py"

code = r'''
from PIL import Image
import numpy as np
import argparse
import os

def integral_image(mask: np.ndarray) -> np.ndarray:
    """Compute summed-area table (integral image) with zero-padding at [0,:] and [:,0]."""
    # mask is 2D uint8 or bool; convert to uint32 for safe sums
    m = mask.astype(np.uint32)
    H, W = m.shape
    S = np.zeros((H + 1, W + 1), dtype=np.uint32)
    # cumulative sum over rows then columns
    S[1:, 1:] = np.cumsum(np.cumsum(m, axis=0), axis=1)
    return S

def window_sum(S: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    """Sum of mask[y0:y1, x0:x1] using integral image S."""
    return int(S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0])

def grid_centers(H, W, s, step):
    """Generate centers on a regular grid with margin s//2 from the border."""
    half = s // 2
    ys = np.arange(half, H - (s - half) + 1, step, dtype=int)  # ensure full window fits
    xs = np.arange(half, W - (s - half) + 1, step, dtype=int)
    # If image is small or step too large, ensure at least one point if possible
    if ys.size == 0 and H >= s:
        ys = np.array([H // 2], dtype=int)
    if xs.size == 0 and W >= s:
        xs = np.array([W // 2], dtype=int)
    return ys, xs

def centers_inside_roi(mask, s, step):
    """Return list of (x,y) centers where a full s x s window is inside white ROI."""
    H, W = mask.shape
    S = integral_image(mask)
    ys, xs = grid_centers(H, W, s, step)
    half = s // 2
    centers = []
    for y in ys:
        for x in xs:
            y0 = y - half
            x0 = x - half
            y1 = y0 + s
            x1 = x0 + s
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                continue
            # full square inside ROI?
            total = window_sum(S, y0, x0, y1, x1)
            if total == s * s:
                centers.append((x, y))
    return centers

def save_subset_file(centers, out_path):
    """Write centers to a DICe subset file with BEGIN/END SUBSET_COORDINATES block."""
    with open(out_path, 'w') as f:
        f.write("BEGIN SUBSET_COORDINATES\n")
        for (x, y) in centers:
            f.write(f"{x} {y}\n")
        f.write("END SUBSET_COORDINATES\n")

def main():
    ap = argparse.ArgumentParser(description="Convert ROI mask to DICe subset locations file.")
    ap.add_argument("--mask", required=True, help="Path to ROI mask image (white=inside, black=outside).")
    ap.add_argument("--subset-size", type=int, required=True, help="Subset size in pixels (odd number recommended).")
    ap.add_argument("--step", type=int, required=True, help="Grid spacing between subset centers in pixels.")
    ap.add_argument("--out", default="subset_locations.txt", help="Output subset file path.")
    args = ap.parse_args()

    # Load and binarize mask
    img = Image.open(args.mask).convert("L")
    mask = np.array(img)
    # Threshold: >0 considered inside (handles anti-aliased masks too)
    mask_bin = (mask > 0).astype(np.uint8)

    # Compute centers
    centers = centers_inside_roi(mask_bin, args.subset_size, args.step)

    if len(centers) == 0:
        raise SystemExit("No valid subset centers found. Try decreasing subset size or step, or check your mask.")

    # Save
    out_path = os.path.abspath(args.out)
    save_subset_file(centers, out_path)

    print(f"Wrote {len(centers)} subset centers to: {out_path}")
    print("Paste this into your DICe input.xml parameters:")
    print('  <Parameter name="subset_file" type="string" value="' + out_path.replace('\\','/') + '"/>')
    print("Remember to set 'subset_size' in your params.xml to match --subset-size.")

if __name__ == "__main__":
    main()
'''
# Write the script file
with open(script_path, "w") as f:
    f.write(code)

print(f"Saved converter script to {script_path}")

