# # import numpy as np
# # import pandas as pd
# # from scipy.interpolate import griddata
# # from scipy.sparse import diags, kron, eye
# # from scipy.sparse.linalg import spsolve
# # from scipy.ndimage import gaussian_filter, map_coordinates
# # import imageio.v2 as imageio
# # import matplotlib.pyplot as plt
# # from skimage.metrics import structural_similarity as ssim
# # from skimage.morphology import convex_hull_image
# # from skimage.exposure import match_histograms

# # # === Inputs ===
# # ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"
# # def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"
# # dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# # # === Load reference and deformed images ===
# # ref_img = imageio.imread(ref_img_file).astype(float)
# # def_img = imageio.imread(def_img_file).astype(float)
# # H, W = ref_img.shape

# # # === Load DICe strain + displacement data ===
# # df = pd.read_csv(dice_file, sep=",", comment="#")
# # x, y = df["COORDINATE_X"].values, df["COORDINATE_Y"].values
# # exx, eyy, exy = df["VSG_STRAIN_XX"].values, df["VSG_STRAIN_YY"].values, df["VSG_STRAIN_XY"].values
# # u_bc, v_bc = df["DISPLACEMENT_X"].values, df["DISPLACEMENT_Y"].values

# # # === ROI mask from convex hull ===
# # mask = np.zeros((H, W), dtype=bool)
# # mask[np.clip(y.astype(int), 0, H-1), np.clip(x.astype(int), 0, W-1)] = True
# # mask = convex_hull_image(mask)

# # # === Interpolate strains to grid ===
# # grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
# # exx_grid = griddata((x, y), exx, (grid_x, grid_y), method="cubic", fill_value=0)
# # eyy_grid = griddata((x, y), eyy, (grid_x, grid_y), method="cubic", fill_value=0)
# # exy_grid = griddata((x, y), exy, (grid_x, grid_y), method="cubic", fill_value=0)

# # # --- Smooth strain fields to reduce noise ---
# # exx_grid = gaussian_filter(exx_grid, sigma=3)
# # eyy_grid = gaussian_filter(eyy_grid, sigma=3)
# # exy_grid = gaussian_filter(exy_grid, sigma=3)

# # # === Poisson solver matrices ===
# # Ix = eye(W); Iy = eye(H)
# # ex_mat = diags([1, -2, 1], [-1, 0, 1], shape=(W, W))
# # ey_mat = diags([1, -2, 1], [-1, 0, 1], shape=(H, H))
# # L = kron(Iy, ex_mat) + kron(ey_mat, Ix)

# # # Right-hand sides
# # b_u = np.gradient(exx_grid, axis=1) + 0.5*np.gradient(exy_grid, axis=0)
# # b_v = np.gradient(eyy_grid, axis=0) + 0.5*np.gradient(exy_grid, axis=1)

# # # Flatten
# # b_u, b_v = b_u.ravel(), b_v.ravel()

# # # === Solve Poisson equations ===
# # u = spsolve(L, b_u).reshape(H, W)
# # v = spsolve(L, b_v).reshape(H, W)

# # # --- Boundary condition correction using DICe displacement ---
# # u -= np.mean(u[0, :]) - np.mean(u_bc)  # align with mean displacement
# # v -= np.mean(v[:, 0]) - np.mean(v_bc)

# # # === Warp reference image ===
# # yy, xx = np.indices((H, W))
# # src_x = np.clip(xx - u, 0, W-1)
# # src_y = np.clip(yy - v, 0, H-1)
# # coords = np.array([src_y.ravel(), src_x.ravel()])
# # warped = map_coordinates(ref_img, coords, order=1, mode="reflect").reshape(H, W)

# # # Apply ROI mask
# # reconstructed = ref_img.copy()
# # reconstructed[mask] = warped[mask]

# # # Histogram matching for illumination normalization
# # reconstructed_matched = reconstructed.copy()
# # reconstructed_matched[mask] = match_histograms(reconstructed[mask], def_img[mask])

# # # === Error analysis ===
# # error_map = np.zeros_like(ref_img)
# # error_map[mask] = def_img[mask] - reconstructed_matched[mask]

# # rmse = np.sqrt(np.mean((def_img[mask] - reconstructed_matched[mask])**2))
# # ssim_val = ssim(def_img[mask], reconstructed_matched[mask], data_range=def_img.max()-def_img.min())

# # # === Visualization ===
# # plt.figure(figsize=(15,5))
# # plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
# # plt.subplot(1,3,2); plt.imshow(reconstructed_matched, cmap="gray"); plt.title("Reconstructed (Strain+Poisson)")
# # plt.subplot(1,3,3); plt.imshow(error_map, cmap="bwr", vmin=-50, vmax=50); plt.colorbar(label="Error")
# # plt.title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# # plt.tight_layout()
# # plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio

# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye, csr_matrix
# from scipy.sparse.linalg import spsolve
# from scipy.ndimage import map_coordinates, gaussian_filter
# from skimage.metrics import structural_similarity as ssim
# from skimage.morphology import convex_hull_image, binary_erosion, square

# # --------------------
# # Inputs
# # --------------------
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size     = 31         # used only to understand spacing (not critical here)
# smooth_sigma_px = 1.0        # Gaussian smoothing of strain maps (set 0 to disable)
# lambda_reg      = 1e-2       # Tikhonov regularization ( λ * I ) for Poisson
# erode_px        = 1          # erode ROI inwards to define interior unknowns

# # --------------------
# # Load images
# # --------------------
# ref_img = imageio.imread(ref_img_file).astype(float)
# def_img = imageio.imread(def_img_file).astype(float)
# H, W = ref_img.shape

# # --------------------
# # Load DICe data
# # --------------------
# df = pd.read_csv(dice_file, sep=",", comment="#")

# x   = df["COORDINATE_X"].values
# y   = df["COORDINATE_Y"].values
# exx = df["VSG_STRAIN_XX"].values
# eyy = df["VSG_STRAIN_YY"].values
# exy = df["VSG_STRAIN_XY"].values
# u_d = df["DISPLACEMENT_X"].values
# v_d = df["DISPLACEMENT_Y"].values

# # --------------------
# # Build ROI mask from subset centers (convex hull)
# # --------------------
# mask_pts = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask_pts[yc, xc] = True
# roi_mask = convex_hull_image(mask_pts)  # generous but contained

# # Split ROI into boundary (Dirichlet) and interior (unknowns)
# if erode_px > 0:
#     interior_mask = binary_erosion(roi_mask, footprint=square(1+2*erode_px))
# else:
#     interior_mask = binary_erosion(roi_mask, footprint=square(1))

# boundary_mask = roi_mask & (~interior_mask)

# # --------------------
# # Interpolate DICe (u,v) for boundary BC and (ε) to full grid
# # --------------------
# grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

# # Displacements for BCs
# u_bc_full = griddata((x, y), u_d, (grid_x, grid_y), method="linear", fill_value=0.0)
# v_bc_full = griddata((x, y), v_d, (grid_x, grid_y), method="linear", fill_value=0.0)

# # Strains for RHS
# exx_grid = griddata((x, y), exx, (grid_x, grid_y), method="linear", fill_value=0.0)
# eyy_grid = griddata((x, y), eyy, (grid_x, grid_y), method="linear", fill_value=0.0)
# exy_grid = griddata((x, y), exy, (grid_x, grid_y), method="linear", fill_value=0.0)

# # Optional denoising of strain (helps a lot)
# if smooth_sigma_px and smooth_sigma_px > 0:
#     exx_grid = gaussian_filter(exx_grid, smooth_sigma_px)
#     eyy_grid = gaussian_filter(eyy_grid, smooth_sigma_px)
#     exy_grid = gaussian_filter(exy_grid, smooth_sigma_px)

# # --------------------
# # Build Poisson RHS:  ∇²u = ∂ε_xx/∂x + 0.5 ∂ε_xy/∂y,  ∇²v = ∂ε_yy/∂y + 0.5 ∂ε_xy/∂x
# # --------------------
# b_u = np.gradient(exx_grid, axis=1) + 0.5 * np.gradient(exy_grid, axis=0)
# b_v = np.gradient(eyy_grid, axis=0) + 0.5 * np.gradient(exy_grid, axis=1)

# # --------------------
# # Discrete Laplacian on the full image grid
# # --------------------
# Ix = eye(W, format="csr")
# Iy = eye(H, format="csr")
# ex1d = diags([1, -2, 1], [-1, 0, 1], shape=(W, W), format="csr")
# ey1d = diags([1, -2, 1], [-1, 0, 1], shape=(H, H), format="csr")
# L = kron(Iy, ex1d) + kron(ey1d, Ix)        # N×N
# L = csr_matrix(L)

# # --------------------
# # Reduce the system to unknowns = interior pixels; boundary = Dirichlet from DICe
# # --------------------
# flat_idx = np.arange(H*W).reshape(H, W)
# ii = flat_idx[interior_mask].ravel()   # unknowns
# ib = flat_idx[boundary_mask].ravel()   # Dirichlet boundary nodes

# # Prepare vectors
# b_u_flat = b_u.ravel()
# b_v_flat = b_v.ravel()
# u_bc_vec = np.zeros(H*W); u_bc_vec[ib] = u_bc_full.ravel()[ib]
# v_bc_vec = np.zeros(H*W); v_bc_vec[ib] = v_bc_full.ravel()[ib]

# # Build reduced systems with Dirichlet BC and regularization
# A_ii = L[ii, :][:, ii] + lambda_reg * eye(ii.size, format="csr")
# B_ui = b_u_flat[ii] - L[ii, :][:, ib].dot(u_bc_vec[ib])
# B_vi = b_v_flat[ii] - L[ii, :][:, ib].dot(v_bc_vec[ib])

# # Solve
# u_sol = spsolve(A_ii, B_ui)
# v_sol = spsolve(A_ii, B_vi)

# # Reassemble full fields (only ROI is meaningful)
# u_field = np.zeros((H, W), dtype=float)
# v_field = np.zeros((H, W), dtype=float)
# u_field[interior_mask]  = u_sol
# v_field[interior_mask]  = v_sol
# u_field[boundary_mask]  = u_bc_full[boundary_mask]
# v_field[boundary_mask]  = v_bc_full[boundary_mask]
# # Outside ROI remains zeros; we won't use it for warping.

# # Optional zero-mean anchoring along top/left edges of ROI to remove constant offsets
# top = np.where(roi_mask[0, :])[0]
# left = np.where(roi_mask[:, 0])[0]
# if top.size > 0:  u_field[:, top] -= np.mean(u_field[:, top])
# if left.size > 0: v_field[left, :] -= np.mean(v_field[left, :])

# # --------------------
# # Warp the reference image using recovered (u,v) — only inside ROI
# # --------------------
# yy, xx = np.indices((H, W))
# src_x = np.clip(xx - u_field, 0, W-1)
# src_y = np.clip(yy - v_field, 0, H-1)
# coords = np.array([src_y.ravel(), src_x.ravel()])

# warped = map_coordinates(ref_img, coords, order=1, mode="reflect").reshape(H, W)

# recon = ref_img.copy()
# recon[roi_mask] = warped[roi_mask]   # leave outside ROI as reference

# # --------------------
# # Error analysis (ROI only)
# # --------------------
# err = np.zeros_like(ref_img)
# err[roi_mask] = def_img[roi_mask] - recon[roi_mask]

# rmse = np.sqrt(np.mean((def_img[roi_mask] - recon[roi_mask])**2))
# mae  = np.mean(np.abs(def_img[roi_mask] - recon[roi_mask]))
# ssim_val = ssim(def_img[roi_mask], recon[roi_mask],
#                 data_range=def_img.max()-def_img.min())

# within5  = np.mean(np.abs(err[roi_mask]) <= 5)  * 100.0
# within10 = np.mean(np.abs(err[roi_mask]) <= 10) * 100.0

# print(f"RMSE: {rmse:.3f}")
# print(f"MAE : {mae:.3f}")
# print(f"SSIM: {ssim_val:.3f}")
# print(f"% pixels within ±5:  {within5:.2f}%")
# print(f"% pixels within ±10: {within10:.2f}%")

# # --------------------
# # Plots
# # --------------------
# fig, axs = plt.subplots(1, 3, figsize=(14, 5))
# axs[0].imshow(def_img, cmap="gray"); axs[0].set_title("Actual Deformed"); axs[0].axis("off")
# axs[1].imshow(recon,   cmap="gray"); axs[1].set_title("Reconstructed (Strain + BCs)"); axs[1].axis("off")

# im = axs[2].imshow(err, cmap="bwr", vmin=-50, vmax=50)
# axs[2].set_title(f"Error Heatmap (ROI)\nRMSE={rmse:.2f}, SSIM={ssim_val:.3f}")
# axs[2].axis("off")
# cbar = plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
# cbar.set_label("Error (intensity)")
# plt.tight_layout(); plt.show()

# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye, vstack
# from scipy.sparse.linalg import spsolve
# from scipy.ndimage import map_coordinates
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.morphology import convex_hull_image
# from skimage.exposure import match_histograms

# # =========================
# # User inputs
# # =========================
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed)
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size  = 31          # from DICe
# lambda_grid  = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # candidates
# do_hist_match = True       # illumination normalization inside ROI
# report_hist   = True       # also show histogram of errors for the best lambda

# # =========================
# # Load images
# # =========================
# ref_img = imageio.imread(ref_img_file).astype(np.float32)
# def_img = imageio.imread(def_img_file).astype(np.float32)
# H, W = ref_img.shape

# # =========================
# # Load DICe data
# # =========================
# df = pd.read_csv(dice_file, sep=",", comment="#")
# x   = df["COORDINATE_X"].values
# y   = df["COORDINATE_Y"].values
# exx = df["VSG_STRAIN_XX"].values
# eyy = df["VSG_STRAIN_YY"].values
# exy = df["VSG_STRAIN_XY"].values
# u_bc = df["DISPLACEMENT_X"].values
# v_bc = df["DISPLACEMENT_Y"].values

# # =========================
# # Build ROI mask via convex hull of subset centers
# # =========================
# mask = np.zeros((H, W), dtype=bool)
# xc = np.clip(x.astype(int), 0, W-1)
# yc = np.clip(y.astype(int), 0, H-1)
# mask[yc, xc] = True
# mask = convex_hull_image(mask)

# # Crop to ROI bounding box to reduce problem size
# rows = np.where(mask.any(axis=1))[0]
# cols = np.where(mask.any(axis=0))[0]
# r0, r1 = rows.min(), rows.max() + 1
# c0, c1 = cols.min(), cols.max() + 1

# ref_c  = ref_img[r0:r1, c0:c1]
# def_c  = def_img[r0:r1, c0:c1]
# mask_c = mask[r0:r1, c0:c1]
# Hc, Wc = ref_c.shape
# Nc = Hc * Wc

# # Build target grid for the crop (in ABSOLUTE image coordinates for interpolation)
# gx_abs, gy_abs = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))

# # Interpolate strains (absolute coordinates -> crop grid)
# exx_c = griddata((x, y), exx, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
# eyy_c = griddata((x, y), eyy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
# exy_c = griddata((x, y), exy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)

# # Interpolate BC displacements onto crop grid
# u_bc_c = griddata((x, y), u_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()
# v_bc_c = griddata((x, y), v_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()

# # Build Laplacian on the crop
# Ix = eye(Wc, format="csr")
# Iy = eye(Hc, format="csr")
# ex = diags([1, -2, 1], [-1, 0, 1], shape=(Wc, Wc), format="csr")
# ey = diags([1, -2, 1], [-1, 0, 1], shape=(Hc, Hc), format="csr")
# L  = kron(Iy, ex, format="csr") + kron(ey, Ix, format="csr")  # Nc x Nc

# # RHS from compatibility
# b_u = (np.gradient(exx_c, axis=1) + 0.5*np.gradient(exy_c, axis=0)).astype(np.float32).ravel()
# b_v = (np.gradient(eyy_c, axis=0) + 0.5*np.gradient(exy_c, axis=1)).astype(np.float32).ravel()

# # Boundary selection (inside ROI)
# bc_idx = np.where(mask_c.ravel())[0]
# M_diag = np.zeros(Nc, dtype=np.float32)
# M_diag[bc_idx] = 1.0
# M = diags(M_diag, 0, shape=(Nc, Nc), format="csr")

# # Precompute L^T L and L^T b to avoid recomputing them for every lambda
# LtL    = (L.T @ L).tocsr()         # Nc x Nc
# LTb_u  = (L.T @ b_u).astype(np.float32)
# LTb_v  = (L.T @ b_v).astype(np.float32)

# def solve_for_lambda(lmbda):
#     """Solve (L^T L + λ M) u = L^T b + λ M u_bc and same for v; return u,v (crop)."""
#     A = (LtL + lmbda * M).tocsr()
#     bu = (LTb_u + lmbda * (M @ u_bc_c)).astype(np.float32)
#     bv = (LTb_v + lmbda * (M @ v_bc_c)).astype(np.float32)

#     u = spsolve(A, bu).reshape(Hc, Wc).astype(np.float32)
#     v = spsolve(A, bv).reshape(Hc, Wc).astype(np.float32)
#     return u, v

# def reconstruct_and_score(u, v):
#     """Warp reference crop with (u,v), apply ROI mask, optional histogram match, compute metrics."""
#     yy, xx = np.indices((Hc, Wc), dtype=np.float32)
#     src_x = np.clip(xx - u, 0, Wc - 1)
#     src_y = np.clip(yy - v, 0, Hc - 1)
#     coords = np.array([src_y.ravel(), src_x.ravel()])
#     warped_c = map_coordinates(ref_c, coords, order=1, mode="reflect").reshape(Hc, Wc).astype(np.float32)

#     recon_c = ref_c.copy()
#     recon_c[mask_c] = warped_c[mask_c]

#     # Histogram match inside ROI (optional)
#     if do_hist_match:
#         recon_c2 = recon_c.copy()
#         recon_c2[mask_c] = match_histograms(recon_c[mask_c], def_c[mask_c]).astype(np.float32)
#     else:
#         recon_c2 = recon_c

#     err = np.zeros_like(ref_c, dtype=np.float32)
#     err[mask_c] = def_c[mask_c] - recon_c2[mask_c]

#     rmse = float(np.sqrt(np.mean((def_c[mask_c] - recon_c2[mask_c])**2)))
#     mae  = float(np.mean(np.abs(def_c[mask_c] - recon_c2[mask_c])))
#     # SSIM on ROI pixels flattened (skimage supports 1D arrays too)
#     ssim_val = float(ssim(def_c[mask_c], recon_c2[mask_c], data_range=float(def_c.max()-def_c.min())))
#     within5  = float(np.mean(np.abs(err[mask_c]) <= 5) * 100.0)
#     within10 = float(np.mean(np.abs(err[mask_c]) <= 10) * 100.0)
#     return recon_c2, err, dict(rmse=rmse, mae=mae, ssim=ssim_val, within5=within5, within10=within10)

# # =========================
# # Lambda sweep
# # =========================
# best = None
# print("Tuning regularization λ ...")
# for lam in lambda_grid:
#     try:
#         u_c, v_c = solve_for_lambda(lam)
#         _, _, metrics = reconstruct_and_score(u_c, v_c)
#         print(f"λ={lam:g} -> SSIM={metrics['ssim']:.4f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")
#         if (best is None) or (metrics["ssim"] > best["metrics"]["ssim"]):
#             best = dict(lam=lam, u=u_c, v=v_c, metrics=metrics)
#     except Exception as e:
#         print(f"λ={lam:g} failed: {e}")

# if best is None:
#     raise RuntimeError("All λ candidates failed to solve. Consider shrinking ROI or trying larger λ values.")

# print("\nSelected λ = {:.4g}".format(best["lam"]))
# print("Best metrics: ",
#       f"SSIM={best['metrics']['ssim']:.4f}, RMSE={best['metrics']['rmse']:.3f}, "
#       f"MAE={best['metrics']['mae']:.3f}, <=5: {best['metrics']['within5']:.2f}%, <=10: {best['metrics']['within10']:.2f}%")

# # =========================
# # Final reconstruction (best λ)
# # =========================
# recon_c_best, err_c_best, metrics_best = reconstruct_and_score(best["u"], best["v"])

# # Stitch the crop back to full-size images for display
# recon_full = ref_img.copy()
# recon_full[r0:r1, c0:c1][mask_c] = recon_c_best[mask_c]
# err_full = np.zeros_like(ref_img, dtype=np.float32)
# err_full[r0:r1, c0:c1][mask_c] = err_c_best[mask_c]

# # =========================
# # Plots
# # =========================
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
# plt.subplot(1,3,2); plt.imshow(recon_full, cmap="gray"); plt.title(f"Reconstructed (Strain+λ*, ROI)\nλ*={best['lam']:.3g}")
# plt.subplot(1,3,3)
# plt.imshow(err_full, cmap="bwr", vmin=-50, vmax=50); plt.colorbar(label="Error (intensity)")
# plt.title(f"Error Heatmap (ROI)\nSSIM={metrics_best['ssim']:.3f}, RMSE={metrics_best['rmse']:.2f}")
# plt.tight_layout(); plt.show()

# if report_hist:
#     plt.figure(figsize=(6,4))
#     plt.hist(np.abs(err_c_best[mask_c]).ravel(), bins=50, color="gray", edgecolor="black")
#     plt.xlabel("Absolute Intensity Error (ROI)"); plt.ylabel("Frequency")
#     plt.title("Histogram of Reconstruction Errors (best λ)")
#     plt.tight_layout(); plt.show()




import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import convex_hull_image, binary_erosion, disk
from skimage.exposure import match_histograms

# =========================
# User inputs
# =========================
ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed)
def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed
dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

subset_size   = 31          # from DICe
lambda_grid   = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # candidates
do_hist_match = True        # illumination normalization inside ROI
report_hist   = True        # also show histogram of errors for the best lambda
border_px     = 10          # pixels to exclude from ROI boundary

# =========================
# Load images
# =========================
ref_img = imageio.imread(ref_img_file).astype(np.float32)
def_img = imageio.imread(def_img_file).astype(np.float32)
H, W = ref_img.shape

# =========================
# Load DICe data
# =========================
df = pd.read_csv(dice_file, sep=",", comment="#")
x   = df["COORDINATE_X"].values
y   = df["COORDINATE_Y"].values
exx = df["VSG_STRAIN_XX"].values
eyy = df["VSG_STRAIN_YY"].values
exy = df["VSG_STRAIN_XY"].values
u_bc = df["DISPLACEMENT_X"].values
v_bc = df["DISPLACEMENT_Y"].values

# =========================
# Build ROI mask via convex hull of subset centers
# =========================
mask = np.zeros((H, W), dtype=bool)
xc = np.clip(x.astype(int), 0, W-1)
yc = np.clip(y.astype(int), 0, H-1)
mask[yc, xc] = True
mask = convex_hull_image(mask)

# Shrink ROI inward to avoid border artifacts
mask_interior = binary_erosion(mask, footprint=disk(border_px))

# Crop to ROI bounding box
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
r0, r1 = rows.min(), rows.max() + 1
c0, c1 = cols.min(), cols.max() + 1

ref_c  = ref_img[r0:r1, c0:c1]
def_c  = def_img[r0:r1, c0:c1]
mask_c = mask[r0:r1, c0:c1]
mask_int_c = mask_interior[r0:r1, c0:c1]
Hc, Wc = ref_c.shape
Nc = Hc * Wc

# Build target grid (absolute coords for interpolation)
gx_abs, gy_abs = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))

# Interpolate strains and boundary displacements
exx_c = griddata((x, y), exx, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
eyy_c = griddata((x, y), eyy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
exy_c = griddata((x, y), exy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
u_bc_c = griddata((x, y), u_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()
v_bc_c = griddata((x, y), v_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()

# Build Laplacian
Ix = eye(Wc, format="csr")
Iy = eye(Hc, format="csr")
ex = diags([1, -2, 1], [-1, 0, 1], shape=(Wc, Wc), format="csr")
ey = diags([1, -2, 1], [-1, 0, 1], shape=(Hc, Hc), format="csr")
L  = kron(Iy, ex, format="csr") + kron(ey, Ix, format="csr")

# RHS from compatibility
b_u = (np.gradient(exx_c, axis=1) + 0.5*np.gradient(exy_c, axis=0)).astype(np.float32).ravel()
b_v = (np.gradient(eyy_c, axis=0) + 0.5*np.gradient(exy_c, axis=1)).astype(np.float32).ravel()

# Boundary condition matrix
bc_idx = np.where(mask_c.ravel())[0]
M_diag = np.zeros(Nc, dtype=np.float32)
M_diag[bc_idx] = 1.0
M = diags(M_diag, 0, shape=(Nc, Nc), format="csr")

# Precompute
LtL   = (L.T @ L).tocsr()
LTb_u = (L.T @ b_u).astype(np.float32)
LTb_v = (L.T @ b_v).astype(np.float32)

def solve_for_lambda(lmbda):
    A = (LtL + lmbda * M).tocsr()
    bu = (LTb_u + lmbda * (M @ u_bc_c)).astype(np.float32)
    bv = (LTb_v + lmbda * (M @ v_bc_c)).astype(np.float32)
    u = spsolve(A, bu).reshape(Hc, Wc).astype(np.float32)
    v = spsolve(A, bv).reshape(Hc, Wc).astype(np.float32)
    return u, v

def reconstruct_and_score(u, v):
    yy, xx = np.indices((Hc, Wc), dtype=np.float32)
    src_x = np.clip(xx - u, 0, Wc - 1)
    src_y = np.clip(yy - v, 0, Hc - 1)
    coords = np.array([src_y.ravel(), src_x.ravel()])
    warped_c = map_coordinates(ref_c, coords, order=1, mode="reflect").reshape(Hc, Wc).astype(np.float32)

    recon_c = ref_c.copy()
    recon_c[mask_c] = warped_c[mask_c]

    if do_hist_match:
        recon_c2 = recon_c.copy()
        recon_c2[mask_c] = match_histograms(recon_c[mask_c], def_c[mask_c]).astype(np.float32)
    else:
        recon_c2 = recon_c

    err = np.zeros_like(ref_c, dtype=np.float32)
    err[mask_c] = def_c[mask_c] - recon_c2[mask_c]

    # Metrics on *interior* mask
    rmse = float(np.sqrt(np.mean((def_c[mask_int_c] - recon_c2[mask_int_c])**2)))
    mae  = float(np.mean(np.abs(def_c[mask_int_c] - recon_c2[mask_int_c])))
    ssim_val = float(ssim(def_c[mask_int_c], recon_c2[mask_int_c], 
                          data_range=float(def_c.max()-def_c.min())))
    within5  = float(np.mean(np.abs(err[mask_int_c]) <= 5) * 100.0)
    within10 = float(np.mean(np.abs(err[mask_int_c]) <= 10) * 100.0)

    return recon_c2, err, dict(rmse=rmse, mae=mae, ssim=ssim_val, within5=within5, within10=within10)

# =========================
# Lambda sweep
# =========================
best = None
print("Tuning regularization λ ...")
for lam in lambda_grid:
    try:
        u_c, v_c = solve_for_lambda(lam)
        _, _, metrics = reconstruct_and_score(u_c, v_c)
        print(f"λ={lam:g} -> SSIM={metrics['ssim']:.4f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")
        if (best is None) or (metrics["ssim"] > best["metrics"]["ssim"]):
            best = dict(lam=lam, u=u_c, v=v_c, metrics=metrics)
    except Exception as e:
        print(f"λ={lam:g} failed: {e}")

if best is None:
    raise RuntimeError("All λ candidates failed.")

print("\nSelected λ = {:.4g}".format(best["lam"]))
print("Best metrics: ",
      f"SSIM={best['metrics']['ssim']:.4f}, RMSE={best['metrics']['rmse']:.3f}, "
      f"MAE={best['metrics']['mae']:.3f}, <=5: {best['metrics']['within5']:.2f}%, <=10: {best['metrics']['within10']:.2f}%")

# =========================
# Final reconstruction
# =========================
recon_c_best, err_c_best, metrics_best = reconstruct_and_score(best["u"], best["v"])

recon_full = ref_img.copy()
recon_full[r0:r1, c0:c1][mask_c] = recon_c_best[mask_c]
err_full = np.zeros_like(ref_img, dtype=np.float32)
err_full[r0:r1, c0:c1][mask_c] = err_c_best[mask_c]

# =========================
# Plots
# =========================
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed")
plt.subplot(1,3,2); plt.imshow(recon_full, cmap="gray"); plt.title(f"Reconstructed (Strain+λ*, ROI)\nλ*={best['lam']:.3g}")
plt.subplot(1,3,3)
plt.imshow(err_full, cmap="bwr", vmin=-50, vmax=50); plt.colorbar(label="Error (intensity)")
plt.title(f"Error Heatmap (ROI, interior)\nSSIM={metrics_best['ssim']:.3f}, RMSE={metrics_best['rmse']:.2f}")
plt.tight_layout(); plt.show()

if report_hist:
    # Use signed error (not absolute)
    err_vals = err_c_best[mask_int_c].ravel()

    plt.figure(figsize=(10,5))
    plt.hist(err_vals, bins=100, color="skyblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="-", linewidth=1, label="Zero error")
    plt.axvline(5, color="green", linestyle="--", label="±5")
    plt.axvline(-5, color="green", linestyle="--")
    plt.axvline(10, color="orange", linestyle="--", label="±10")
    plt.axvline(-10, color="orange", linestyle="--")

    plt.xlabel("Pixel Intensity Error (Actual - Reconstructed, ROI interior)")
    plt.ylabel("Pixel Count")
    plt.title("Distribution of Reconstruction Errors (best λ, signed)")
    plt.legend()

    # Annotate percentages on plot
    # within5  = metrics_best["within5"]
    # within10 = metrics_best["within10"]
    # plt.text(0.65, 0.85, f"Within ±5:  {within5:.2f}%", 
    #          transform=plt.gca().transAxes, fontsize=11, color="green")
    # plt.text(0.65, 0.78, f"Within ±10: {within10:.2f}%", 
    #          transform=plt.gca().transAxes, fontsize=11, color="orange")

    plt.tight_layout()
    plt.show()