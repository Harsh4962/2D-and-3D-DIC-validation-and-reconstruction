# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye
# from scipy.sparse.linalg import spsolve
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # === Input paths ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # undeformed
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_11.tif"   # actual deformed
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# # === Load reference & deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe strain data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")
# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# exx = df["VSG_STRAIN_XX"].values
# eyy = df["VSG_STRAIN_YY"].values
# exy = df["VSG_STRAIN_XY"].values

# # === Interpolate strain onto full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# exx_grid = griddata((x, y), exx, (grid_x, grid_y), method="linear", fill_value=0)
# eyy_grid = griddata((x, y), eyy, (grid_x, grid_y), method="linear", fill_value=0)
# exy_grid = griddata((x, y), exy, (grid_x, grid_y), method="linear", fill_value=0)

# # === Build Poisson system Ax = b for u and v ===
# N = H * W
# Ix = eye(W); Iy = eye(H)
# ex = diags([1, -2, 1], [-1, 0, 1], shape=(W, W))
# ey = diags([1, -2, 1], [-1, 0, 1], shape=(H, H))
# L = kron(Iy, ex) + kron(ey, Ix)  # Laplacian operator

# # Right-hand sides
# b_u = np.gradient(exx_grid, axis=1) + 0.5*np.gradient(exy_grid, axis=0)
# b_v = np.gradient(eyy_grid, axis=0) + 0.5*np.gradient(exy_grid, axis=1)
# b_u = b_u.ravel()
# b_v = b_v.ravel()

# # Solve for displacements
# u = spsolve(L, b_u).reshape(H, W)
# v = spsolve(L, b_v).reshape(H, W)

# # Normalize (fix rigid-body shift)
# u -= np.mean(u[0, :])
# v -= np.mean(v[:, 0])

# # === Warp reference image ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u, 0, W-1)
# src_y = np.clip(yy - v, 0, H-1)
# rec_img = ref_img[src_y.astype(int), src_x.astype(int)]

# # === Quantitative Comparison ===
# # Crop to common region
# h = min(def_img.shape[0], rec_img.shape[0])
# w = min(def_img.shape[1], rec_img.shape[1])
# def_crop = def_img[:h, :w]
# rec_crop = rec_img[:h, :w]

# err_img = np.abs(def_crop - rec_crop)
# rmse = np.sqrt(np.mean((def_crop - rec_crop)**2))
# ssim_index = ssim(def_crop, rec_crop, data_range=def_crop.max() - def_crop.min())

# # === Plot ===
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(def_crop, cmap="gray"); plt.title("Actual Deformed"); plt.axis("off")
# plt.subplot(1,3,2); plt.imshow(rec_crop, cmap="gray"); plt.title("Reconstructed (Poisson)"); plt.axis("off")
# plt.subplot(1,3,3); plt.imshow(err_img, cmap="hot"); plt.colorbar(label="Abs Error"); 
# plt.title(f"Error Heatmap\nRMSE={rmse:.3f}, SSIM={ssim_index:.3f}"); plt.axis("off")
# plt.tight_layout()
# plt.show()


# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference image
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed image
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# # === Load images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")
# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u = df["DISPLACEMENT_X"].values
# v = df["DISPLACEMENT_Y"].values

# # === Interpolate displacement onto full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u_grid = griddata((x, y), u, (grid_x, grid_y), method="linear", fill_value=0)
# v_grid = griddata((x, y), v, (grid_x, grid_y), method="linear", fill_value=0)

# # === Warp reference image ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u_grid, 0, W-1)
# src_y = np.clip(yy - v_grid, 0, H-1)
# reconstructed = ref_img[src_y.astype(int), src_x.astype(int)]

# # === Build ROI mask (where DICe had data) ===
# roi_mask = ~np.isnan(griddata((x, y), u, (grid_x, grid_y), method="nearest", fill_value=np.nan))

# # === Compute error metrics inside ROI ===
# error = np.abs(def_img - reconstructed)
# error_roi = error[roi_mask]

# rmse = np.sqrt(np.mean(error_roi**2))
# ssim_val = ssim(def_img, reconstructed, data_range=def_img.max()-def_img.min())

# # === Plot results ===
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed"); plt.axis("off")
# plt.subplot(1,3,2); plt.imshow(reconstructed, cmap="gray"); plt.title("Reconstructed (DICe U,V)"); plt.axis("off")
# plt.subplot(1,3,3); 
# plt.imshow(error, cmap="hot"); plt.colorbar(label="Abs Error")
# plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# plt.axis("off")
# plt.tight_layout()
# plt.show()




# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.ndimage import binary_dilation
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # === Load reference & deformed images ===
# ref_img = imageio.imread(r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif").astype(float)
# def_img = imageio.imread(r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif").astype(float)
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt",
#                  sep=",", comment="#")

# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u_pts = df["DISPLACEMENT_X"].values
# v_pts = df["DISPLACEMENT_Y"].values

# # --- Build ROI mask from subset centers ---
# roi_mask = np.zeros((H, W), dtype=bool)
# roi_mask[y.astype(int), x.astype(int)] = True
# roi_mask = binary_dilation(roi_mask, iterations=15)  # smooth/expand ROI to fill gaps

# # --- Interpolate displacement field onto full grid ---
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u_grid = griddata((x, y), u_pts, (grid_x, grid_y), method="linear", fill_value=0)
# v_grid = griddata((x, y), v_pts, (grid_x, grid_y), method="linear", fill_value=0)

# # --- Warp reference image ---
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u_grid, 0, W-1)
# src_y = np.clip(yy - v_grid, 0, H-1)
# warped = ref_img[src_y.astype(int), src_x.astype(int)]

# # === Compute error inside ROI only ===
# error = np.abs(def_img - warped)
# error_roi = np.where(roi_mask, error, 0)

# rmse = np.sqrt(np.mean((def_img[roi_mask] - warped[roi_mask])**2))
# ssim_val = ssim(def_img, warped, data_range=def_img.max()-def_img.min(), mask=roi_mask)

# # === Plot ===
# plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.imshow(def_img, cmap="gray")
# plt.title("Actual Deformed")

# plt.subplot(1,3,2)
# plt.imshow(warped, cmap="gray")
# plt.title("Reconstructed (DICe U,V)")

# plt.subplot(1,3,3)
# plt.imshow(error_roi, cmap="hot", vmin=0, vmax=np.percentile(error_roi[roi_mask], 99))
# plt.colorbar(label="Abs Error")
# plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")

# plt.tight_layout()
# plt.show()



# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # === Load images ===
# ref_img = imageio.imread(r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif").astype(float)
# def_img = imageio.imread(r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif").astype(float)  # final deformed
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(
#     r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt",
#     sep=",", comment="#"
# )
# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u = df["DISPLACEMENT_X"].values
# v = df["DISPLACEMENT_Y"].values

# # Build displacement grids
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u_grid = griddata((x, y), u, (grid_x, grid_y), method="cubic", fill_value=0)
# v_grid = griddata((x, y), v, (grid_x, grid_y), method="cubic", fill_value=0)

# # === Warp reference image using displacements ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u_grid, 0, W-1)
# src_y = np.clip(yy - v_grid, 0, H-1)
# warped = ref_img[src_y.astype(int), src_x.astype(int)]

# # === Construct ROI from subset centers automatically ===
# # bounding box of subset coordinates
# xmin, xmax = int(np.min(x)), int(np.max(x))
# ymin, ymax = int(np.min(y)), int(np.max(y))

# roiMask = np.zeros_like(ref_img, dtype=bool)
# roiMask[ymin:ymax+1, xmin:xmax+1] = True

# # === Compute error only inside ROI (not outside) ===
# error = np.abs(def_img - warped)
# error_masked = np.where(roiMask, error, np.nan)

# # Metrics (inside ROI only)
# rmse = np.sqrt(np.nanmean(error_masked**2))
# ssim_val = ssim(def_img, warped, data_range=def_img.max()-def_img.min(), win_size=7)

# # === Plot ===
# plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.imshow(def_img, cmap="gray")
# plt.title("Actual Deformed")

# plt.subplot(1,3,2)
# plt.imshow(warped, cmap="gray")
# plt.title("Reconstructed (DICe U,V)")

# plt.subplot(1,3,3)
# plt.imshow(error_masked, cmap="hot", vmin=0, vmax=np.nanpercentile(error_masked, 99))
# plt.colorbar(label="Abs Error")
# plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")

# plt.tight_layout()
# plt.show()



# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye
# from scipy.sparse.linalg import spsolve
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.morphology import square, dilation

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed) image
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed image
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size = 31   # DICe subset size

# # === Load reference and deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe strain data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")

# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# exx = df["VSG_STRAIN_XX"].values
# eyy = df["VSG_STRAIN_YY"].values
# exy = df["VSG_STRAIN_XY"].values

# # === Build ROI mask using subset centers + square dilation ===
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# mask = dilation(mask, square(subset_size))   # expand centers into square subsets

# # === Interpolate strains on full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# exx_grid = griddata((x, y), exx, (grid_x, grid_y), method="linear", fill_value=0)
# eyy_grid = griddata((x, y), eyy, (grid_x, grid_y), method="linear", fill_value=0)
# exy_grid = griddata((x, y), exy, (grid_x, grid_y), method="linear", fill_value=0)

# # === Solve Poisson equations for u,v (inverse problem) ===
# N = H * W
# Ix = eye(W); Iy = eye(H)
# ex = diags([1, -2, 1], [-1, 0, 1], shape=(W, W))
# ey = diags([1, -2, 1], [-1, 0, 1], shape=(H, H))
# L = kron(Iy, ex) + kron(ey, Ix)

# b_u = np.gradient(exx_grid, axis=1) + 0.5*np.gradient(exy_grid, axis=0)
# b_v = np.gradient(eyy_grid, axis=0) + 0.5*np.gradient(exy_grid, axis=1)

# u = spsolve(L, b_u.ravel()).reshape(H, W)
# v = spsolve(L, b_v.ravel()).reshape(H, W)

# # Normalize boundary mean
# u -= np.mean(u[0, :])
# v -= np.mean(v[:, 0])

# # === Warp reference image using computed displacements ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u, 0, W-1)
# src_y = np.clip(yy - v, 0, H-1)

# warped = ref_img[src_y.astype(int), src_x.astype(int)]

# # Apply ROI mask (outside ROI = reference image unchanged)
# reconstructed = ref_img.copy()
# reconstructed[mask] = warped[mask]

# # === Error map ===
# error_map = np.zeros_like(ref_img)
# error_map[mask] = def_img[mask] - reconstructed[mask]

# # === Visualization ===
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(ref_img, cmap="gray"); plt.title("Reference")
# plt.subplot(1,3,2); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
# plt.subplot(1,3,3); plt.imshow(reconstructed, cmap="gray"); plt.title("Reconstructed (ROI only)")
# plt.tight_layout()
# plt.show()

# # Error heatmap
# plt.figure(figsize=(6,5))
# plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
# plt.colorbar(label="Error (pixel intensity)")
# plt.title("Error map (Actual - Reconstructed) inside ROI")
# plt.show()


# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye
# from scipy.sparse.linalg import spsolve
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.morphology import square, dilation

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size = 31   # DICe subset size

# # === Load reference and deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe strain data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")

# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# exx = df["VSG_STRAIN_XX"].values
# eyy = df["VSG_STRAIN_YY"].values
# exy = df["VSG_STRAIN_XY"].values

# # === Build ROI mask ===
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# mask = dilation(mask, square(subset_size))

# # === Interpolate strains ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# exx_grid = griddata((x, y), exx, (grid_x, grid_y), method="linear", fill_value=0)
# eyy_grid = griddata((x, y), eyy, (grid_x, grid_y), method="linear", fill_value=0)
# exy_grid = griddata((x, y), exy, (grid_x, grid_y), method="linear", fill_value=0)

# # === Solve Poisson equations for u,v ===
# Ix = eye(W); Iy = eye(H)
# ex = diags([1, -2, 1], [-1, 0, 1], shape=(W, W))
# ey = diags([1, -2, 1], [-1, 0, 1], shape=(H, H))
# L = kron(Iy, ex) + kron(ey, Ix)

# b_u = np.gradient(exx_grid, axis=1) + 0.5*np.gradient(exy_grid, axis=0)
# b_v = np.gradient(eyy_grid, axis=0) + 0.5*np.gradient(exy_grid, axis=1)

# u = spsolve(L, b_u.ravel()).reshape(H, W)
# v = spsolve(L, b_v.ravel()).reshape(H, W)

# # Normalize boundary mean
# u -= np.mean(u[0, :])
# v -= np.mean(v[:, 0])

# # === Warp reference image ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u, 0, W-1)
# src_y = np.clip(yy - v, 0, H-1)
# warped = ref_img[src_y.astype(int), src_x.astype(int)]

# # Apply ROI mask only
# reconstructed = ref_img.copy()
# reconstructed[mask] = warped[mask]

# # === Error map ===
# error_map = np.zeros_like(ref_img)
# error_map[mask] = def_img[mask] - reconstructed[mask]

# rmse = np.sqrt(np.mean(error_map[mask]**2))

# # === Combined Visualization ===
# plt.figure(figsize=(18,6))

# plt.subplot(1,4,1); 
# plt.imshow(ref_img, cmap="gray"); plt.title("Reference"); plt.axis("off")

# plt.subplot(1,4,2); 
# plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed"); plt.axis("off")

# plt.subplot(1,4,3); 
# plt.imshow(reconstructed, cmap="gray"); plt.title("Reconstructed (ROI only)"); plt.axis("off")

# plt.subplot(1,4,4); 
# im = plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
# plt.colorbar(im, fraction=0.046, pad=0.04, label="Error (intensity)")
# plt.title(f"Error Heatmap\nRMSE={rmse:.2f}")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.morphology import square, dilation
# from skimage.metrics import structural_similarity as ssim

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size = 31   # DICe subset size

# # === Load reference and deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")

# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u = df["DISPLACEMENT_X"].values
# v = df["DISPLACEMENT_Y"].values

# # from scipy.ndimage import gaussian_filter
# # u = gaussian_filter(u, sigma=1)
# # v = gaussian_filter(v, sigma=1)


# # === Build ROI mask using subset centers ===
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# mask = dilation(mask, square(subset_size))   # expand each center into its square subset

# # === Interpolate displacements onto full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u_grid = griddata((x, y), u, (grid_x, grid_y), method="cubic", fill_value=0)
# v_grid = griddata((x, y), v, (grid_x, grid_y), method="cubic", fill_value=0)

# # === Warp reference image using displacement field ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u_grid, 0, W-1)
# src_y = np.clip(yy - v_grid, 0, H-1)

# warped = ref_img[src_y.astype(int), src_x.astype(int)]

# # Apply ROI mask (outside ROI = keep original reference)
# reconstructed = ref_img.copy()
# reconstructed[mask] = warped[mask]

# # === Error map inside ROI ===
# error_map = np.zeros_like(ref_img)
# error_map[mask] = def_img[mask] - reconstructed[mask]

# rmse = np.sqrt(np.mean(error_map[mask]**2))
# ssim_val = ssim(def_img, reconstructed, data_range=def_img.max() - def_img.min())

# # === Visualization ===
# plt.figure(figsize=(18,6))

# plt.subplot(1,4,1)
# plt.imshow(ref_img, cmap="gray")
# plt.title("Reference")
# plt.axis("off")

# plt.subplot(1,4,2)
# plt.imshow(def_img, cmap="gray")
# plt.title("Actual Deformed")
# plt.axis("off")

# plt.subplot(1,4,3)
# plt.imshow(reconstructed, cmap="gray")
# plt.title("Reconstructed (from Displacements)")
# plt.axis("off")

# plt.subplot(1,4,4)
# im = plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
# plt.colorbar(im, fraction=0.046, pad=0.04, label="Error (intensity)")
# plt.title(f"Error Heatmap\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# plt.axis("off")

# plt.tight_layout()
# plt.show()



############################# BEST SO FAR ####################################################
# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.ndimage import map_coordinates
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.morphology import square, dilation

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed) image
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed image
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size = 31   # DICe subset size

# # === Load reference and deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")
# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u_vals = df["DISPLACEMENT_X"].values
# v_vals = df["DISPLACEMENT_Y"].values

# # === Build ROI mask using subset centers + square dilation ===
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# mask = dilation(mask, square(subset_size))   # expand centers into square subsets

# # === Interpolate displacements on full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u = griddata((x, y), u_vals, (grid_x, grid_y), method="cubic", fill_value=0)
# v = griddata((x, y), v_vals, (grid_x, grid_y), method="cubic", fill_value=0)

# # === Warp reference image using subpixel interpolation ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u, 0, W-1)
# src_y = np.clip(yy - v, 0, H-1)

# coords = np.array([src_y.ravel(), src_x.ravel()])
# warped = map_coordinates(ref_img, coords, order=1, mode="reflect").reshape(H, W)

# # Apply ROI mask (outside ROI = keep original reference image)
# reconstructed = ref_img.copy()
# reconstructed[mask] = warped[mask]

# # === Error analysis ===
# error_map = np.zeros_like(ref_img)
# error_map[mask] = def_img[mask] - reconstructed[mask]

# rmse = np.sqrt(np.mean((def_img[mask] - reconstructed[mask])**2))
# ssim_val = ssim(def_img, reconstructed, data_range=def_img.max()-def_img.min())

# # === Visualization ===
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
# plt.subplot(1,3,2); plt.imshow(reconstructed, cmap="gray"); plt.title("Reconstructed (Subpixel)")
# plt.subplot(1,3,3); 
# plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50); 
# plt.colorbar(label="Error (intensity)")
# plt.title(f"Error Heatmap\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# plt.tight_layout()
# plt.show()







# ###################### BETTER ####################################################################
# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.ndimage import map_coordinates
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.morphology import square, dilation

# from skimage.exposure import match_histograms

# # === Inputs ===
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed) image
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed image
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size = 31   # DICe subset size

# # === Load reference and deformed images ===
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # === Load DICe displacement data ===
# df = pd.read_csv(dice_file, sep=",", comment="#")
# x = df["COORDINATE_X"].values
# y = df["COORDINATE_Y"].values
# u_vals = df["DISPLACEMENT_X"].values
# v_vals = df["DISPLACEMENT_Y"].values

# # === Build ROI mask using subset centers + square dilation ===
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# # mask = dilation(mask, square(subset_size))   # expand centers into square subsets
# # restrict mask strictly to convex hull of subset centers
# from skimage.morphology import convex_hull_image
# mask = convex_hull_image(mask)


# # === Interpolate displacements on full grid ===
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# u = griddata((x, y), u_vals, (grid_x, grid_y), method="cubic", fill_value=0)
# v = griddata((x, y), v_vals, (grid_x, grid_y), method="cubic", fill_value=0)

# # === Warp reference image using subpixel interpolation ===
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u, 0, W-1)
# src_y = np.clip(yy - v, 0, H-1)

# coords = np.array([src_y.ravel(), src_x.ravel()])
# warped = map_coordinates(ref_img, coords, order=1, mode="reflect").reshape(H, W)

# # Apply ROI mask (outside ROI = unchanged reference)
# reconstructed = ref_img.copy()
# reconstructed[mask] = warped[mask]


# reconstructed_matched = reconstructed.copy()
# reconstructed_matched[mask] = match_histograms(
#     reconstructed[mask], def_img[mask]
# )


# # === Error analysis restricted to ROI ===
# # error_map = np.zeros_like(ref_img)
# # error_map[mask] = def_img[mask] - reconstructed[mask]

# # rmse = np.sqrt(np.mean((def_img[mask] - reconstructed[mask])**2))
# # ssim_val = ssim(def_img[mask], reconstructed[mask], data_range=def_img.max()-def_img.min())


# # === Error analysis restricted to ROI (use histogram matched version) ===
# error_map = np.zeros_like(ref_img)
# error_map[mask] = def_img[mask] - reconstructed_matched[mask]

# rmse = np.sqrt(np.mean((def_img[mask] - reconstructed_matched[mask])**2))
# ssim_val = ssim(def_img[mask], reconstructed_matched[mask], data_range=def_img.max()-def_img.min())



# # === Visualization ===
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
# plt.subplot(1,3,2); plt.imshow(reconstructed, cmap="gray"); plt.title("Reconstructed (Subpixel, ROI)")
# plt.subplot(1,3,3)
# plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
# plt.colorbar(label="Error (intensity)")
# plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# plt.tight_layout()
# plt.show()




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

# === Visualization ===
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
plt.subplot(1,3,2); plt.imshow(reconstructed_matched, cmap="gray"); plt.title("Reconstructed (Subpixel, ROI)")
plt.subplot(1,3,3)
plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50)
plt.colorbar(label="Error (intensity)")
plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, MAE={mae:.2f}, SSIM={ssim_val:.3f}")
plt.tight_layout()
plt.show()
