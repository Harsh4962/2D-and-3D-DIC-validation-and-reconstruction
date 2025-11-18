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




# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.sparse import diags, kron, eye
# from scipy.sparse.linalg import spsolve
# from scipy.ndimage import map_coordinates
# import imageio.v2 as imageio
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.morphology import convex_hull_image, binary_erosion, disk
# from skimage.exposure import match_histograms

# # =========================
# # User inputs
# # =========================
# ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed)
# def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed
# dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# subset_size   = 31          # from DICe
# lambda_grid   = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # candidates
# do_hist_match = True        # illumination normalization inside ROI
# report_hist   = True        # also show histogram of errors for the best lambda
# border_px     = 10          # pixels to exclude from ROI boundary

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

# # Shrink ROI inward to avoid border artifacts
# mask_interior = binary_erosion(mask, footprint=disk(border_px))

# # Crop to ROI bounding box
# rows = np.where(mask.any(axis=1))[0]
# cols = np.where(mask.any(axis=0))[0]
# r0, r1 = rows.min(), rows.max() + 1
# c0, c1 = cols.min(), cols.max() + 1

# ref_c  = ref_img[r0:r1, c0:c1]
# def_c  = def_img[r0:r1, c0:c1]
# mask_c = mask[r0:r1, c0:c1]
# mask_int_c = mask_interior[r0:r1, c0:c1]
# Hc, Wc = ref_c.shape
# Nc = Hc * Wc

# # Build target grid (absolute coords for interpolation)
# gx_abs, gy_abs = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))

# # Interpolate strains and boundary displacements
# exx_c = griddata((x, y), exx, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
# eyy_c = griddata((x, y), eyy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
# exy_c = griddata((x, y), exy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
# u_bc_c = griddata((x, y), u_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()
# v_bc_c = griddata((x, y), v_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()

# # Build Laplacian
# Ix = eye(Wc, format="csr")
# Iy = eye(Hc, format="csr")
# ex = diags([1, -2, 1], [-1, 0, 1], shape=(Wc, Wc), format="csr")
# ey = diags([1, -2, 1], [-1, 0, 1], shape=(Hc, Hc), format="csr")
# L  = kron(Iy, ex, format="csr") + kron(ey, Ix, format="csr")

# # RHS from compatibility
# b_u = (np.gradient(exx_c, axis=1) + 0.5*np.gradient(exy_c, axis=0)).astype(np.float32).ravel()
# b_v = (np.gradient(eyy_c, axis=0) + 0.5*np.gradient(exy_c, axis=1)).astype(np.float32).ravel()

# # Boundary condition matrix
# bc_idx = np.where(mask_c.ravel())[0]
# M_diag = np.zeros(Nc, dtype=np.float32)
# M_diag[bc_idx] = 1.0
# M = diags(M_diag, 0, shape=(Nc, Nc), format="csr")

# # Precompute
# LtL   = (L.T @ L).tocsr()
# LTb_u = (L.T @ b_u).astype(np.float32)
# LTb_v = (L.T @ b_v).astype(np.float32)

# def solve_for_lambda(lmbda):
#     A = (LtL + lmbda * M).tocsr()
#     bu = (LTb_u + lmbda * (M @ u_bc_c)).astype(np.float32)
#     bv = (LTb_v + lmbda * (M @ v_bc_c)).astype(np.float32)
#     u = spsolve(A, bu).reshape(Hc, Wc).astype(np.float32)
#     v = spsolve(A, bv).reshape(Hc, Wc).astype(np.float32)
#     return u, v

# def reconstruct_and_score(u, v):
#     yy, xx = np.indices((Hc, Wc), dtype=np.float32)
#     src_x = np.clip(xx - u, 0, Wc - 1)
#     src_y = np.clip(yy - v, 0, Hc - 1)
#     coords = np.array([src_y.ravel(), src_x.ravel()])
#     warped_c = map_coordinates(ref_c, coords, order=1, mode="reflect").reshape(Hc, Wc).astype(np.float32)

#     recon_c = ref_c.copy()
#     recon_c[mask_c] = warped_c[mask_c]

#     if do_hist_match:
#         recon_c2 = recon_c.copy()
#         recon_c2[mask_c] = match_histograms(recon_c[mask_c], def_c[mask_c]).astype(np.float32)
#     else:
#         recon_c2 = recon_c

#     err = np.zeros_like(ref_c, dtype=np.float32)
#     err[mask_c] = def_c[mask_c] - recon_c2[mask_c]

#     # Metrics on *interior* mask
#     rmse = float(np.sqrt(np.mean((def_c[mask_int_c] - recon_c2[mask_int_c])**2)))
#     mae  = float(np.mean(np.abs(def_c[mask_int_c] - recon_c2[mask_int_c])))
#     ssim_val = float(ssim(def_c[mask_int_c], recon_c2[mask_int_c], 
#                           data_range=float(def_c.max()-def_c.min())))
#     within5  = float(np.mean(np.abs(err[mask_int_c]) <= 5) * 100.0)
#     within10 = float(np.mean(np.abs(err[mask_int_c]) <= 10) * 100.0)

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
#     raise RuntimeError("All λ candidates failed.")

# print("\nSelected λ = {:.4g}".format(best["lam"]))
# print("Best metrics: ",
#       f"SSIM={best['metrics']['ssim']:.4f}, RMSE={best['metrics']['rmse']:.3f}, "
#       f"MAE={best['metrics']['mae']:.3f}, <=5: {best['metrics']['within5']:.2f}%, <=10: {best['metrics']['within10']:.2f}%")

# # =========================
# # Final reconstruction
# # =========================
# recon_c_best, err_c_best, metrics_best = reconstruct_and_score(best["u"], best["v"])

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
# plt.title(f"Error Heatmap (ROI, interior)\nSSIM={metrics_best['ssim']:.3f}, RMSE={metrics_best['rmse']:.2f}")
# plt.tight_layout(); plt.show()

# if report_hist:
#     # Use signed error (not absolute)
#     err_vals = err_c_best[mask_int_c].ravel()

#     plt.figure(figsize=(10,5))
#     plt.hist(err_vals, bins=100, color="skyblue", edgecolor="black")
#     plt.axvline(0, color="red", linestyle="-", linewidth=1, label="Zero error")
#     plt.axvline(5, color="green", linestyle="--", label="±5")
#     plt.axvline(-5, color="green", linestyle="--")
#     plt.axvline(10, color="orange", linestyle="--", label="±10")
#     plt.axvline(-10, color="orange", linestyle="--")

#     plt.xlabel("Pixel Intensity Error (Actual - Reconstructed, ROI interior)")
#     plt.ylabel("Pixel Count")
#     plt.title("Distribution of Reconstruction Errors (best λ, signed)")
#     plt.legend()

#     # Annotate percentages on plot
#     # within5  = metrics_best["within5"]
#     # within10 = metrics_best["within10"]
#     # plt.text(0.65, 0.85, f"Within ±5:  {within5:.2f}%", 
#     #          transform=plt.gca().transAxes, fontsize=11, color="green")
#     # plt.text(0.65, 0.78, f"Within ±10: {within10:.2f}%", 
#     #          transform=plt.gca().transAxes, fontsize=11, color="orange")

#     plt.tight_layout()
#     plt.show()




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
from skimage.restoration import denoise_tv_chambolle

# =========================
# User inputs
# =========================
ref_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_00.tif"   # Reference (undeformed)
def_img_file = r"C:\Users\harsh\Downloads\2dDIC_dataset\ohtcfrp_10.tif"   # Actual deformed
dice_file    = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

subset_size   = 31   # for context (not directly used below)
lambda_grid   = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # candidates
do_hist_match = True        # illumination normalization inside ROI
border_px     = 10          # pixels to exclude from ROI boundary (interior mask)
use_first_order = False     # False: zeroth-order (L^T L); True: first-order (G^T G)
do_tv_postprocess = False   # apply TV to (u,v) after solving (edge-preserving polish)
tv_weight          = 0.05   # TV weight if enabled
noise_frac_for_study = 0.05 # 5% noise level for robustness study
plot_error_vlim = 50        # colorbar clamp for error heatmaps

# =========================
# Utils
# =========================
def safe_get(df, name_candidates):
    for name in name_candidates:
        if name in df.columns:
            return df[name].values
    raise KeyError(f"None of {name_candidates} found in CSV columns: {list(df.columns)}")

def build_roi_from_points(H, W, x, y, border_px):
    mask = np.zeros((H, W), dtype=bool)
    xc = np.clip(np.round(x).astype(int), 0, W-1)
    yc = np.clip(np.round(y).astype(int), 0, H-1)
    mask[yc, xc] = True
    mask = convex_hull_image(mask)
    mask_interior = binary_erosion(mask, footprint=disk(border_px))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()+1
    c0, c1 = cols.min(), cols.max()+1
    return mask, mask_interior, (r0, r1, c0, c1)

def build_operators(Hc, Wc):
    Ix = eye(Wc, format="csr")
    Iy = eye(Hc, format="csr")
    ex = diags([1, -2, 1], [-1, 0, 1], shape=(Wc, Wc), format="csr")
    ey = diags([1, -2, 1], [-1, 0, 1], shape=(Hc, Hc), format="csr")
    L  = kron(Iy, ex, format="csr") + kron(ey, Ix, format="csr")  # 2D Laplacian

    # First-order forward-difference operators (Neumann-like interior)
    Ex = diags([-1, 1], [0, 1], shape=(Wc-1, Wc), format="csr")
    Ey = diags([-1, 1], [0, 1], shape=(Hc-1, Hc), format="csr")
    Dx = kron(Iy, Ex, format="csr")  # (Hc*(Wc-1)) x (Hc*Wc)
    Dy = kron(Ey, Ix, format="csr")  # ((Hc-1)*Wc) x (Hc*Wc)
    GtG = (Dx.T @ Dx + Dy.T @ Dy).tocsr()
    return L, GtG

def objective_terms(L, M, u, v, b_u, b_v, lam, u_bc_c, v_bc_c):
    du = (L @ u.ravel()) - b_u
    dv = (L @ v.ravel()) - b_v
    data_term = float(du.dot(du) + dv.dot(dv))
    bu_term = (M @ (u.ravel() - u_bc_c))
    bv_term = (M @ (v.ravel() - v_bc_c))
    bc_term = float(bu_term.dot(bu_term) + bv_term.dot(bv_term))
    total = data_term + lam * bc_term
    return data_term, bc_term, total

def reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u, v, do_hist_match=True):
    Hc, Wc = ref_c.shape
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

    # Metrics on interior mask
    rmse = float(np.sqrt(np.mean((def_c[mask_int_c] - recon_c2[mask_int_c])**2)))
    mae  = float(np.mean(np.abs(def_c[mask_int_c] - recon_c2[mask_int_c])))
    # data_range: use actual interior range for stability
    dr = float(def_c[mask_int_c].max() - def_c[mask_int_c].min() + 1e-6)
    ssim_val = float(ssim(def_c[mask_int_c], recon_c2[mask_int_c], data_range=dr))
    within5  = float(np.mean(np.abs(err[mask_int_c]) <= 5) * 100.0)
    within10 = float(np.mean(np.abs(err[mask_int_c]) <= 10) * 100.0)

    return recon_c2, err, dict(rmse=rmse, mae=mae, ssim=ssim_val, within5=within5, within10=within10)

def add_noise(arr, frac, valid_mask):
    s = np.std(arr[valid_mask])
    return (arr + np.random.normal(0, frac * s, size=arr.shape)).astype(np.float32)

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

x   = safe_get(df, ["COORDINATE_X", "X", "x"])
y   = safe_get(df, ["COORDINATE_Y", "Y", "y"])
exx = safe_get(df, ["VSG_STRAIN_XX", "STRAIN_XX", "EXX"])
eyy = safe_get(df, ["VSG_STRAIN_YY", "STRAIN_YY", "EYY"])
exy = safe_get(df, ["VSG_STRAIN_XY", "STRAIN_XY", "EXY"])
u_bc = safe_get(df, ["DISPLACEMENT_X", "UX", "U"])
v_bc = safe_get(df, ["DISPLACEMENT_Y", "UY", "V"])

# =========================
# ROI from convex hull of subset centers
# =========================
mask, mask_interior, (r0, r1, c0, c1) = build_roi_from_points(H, W, x, y, border_px)
ref_c  = ref_img[r0:r1, c0:c1]
def_c  = def_img[r0:r1, c0:c1]
mask_c = mask[r0:r1, c0:c1]
mask_int_c = mask_interior[r0:r1, c0:c1]
Hc, Wc = ref_c.shape
Nc = Hc * Wc

# Target grid (absolute coords for interpolation)
gx_abs, gy_abs = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))

# Interpolate strains and boundary displacements onto cropped grid
exx_c = griddata((x, y), exx, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
eyy_c = griddata((x, y), eyy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
exy_c = griddata((x, y), exy, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32)
u_bc_c = griddata((x, y), u_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()
v_bc_c = griddata((x, y), v_bc, (gx_abs, gy_abs), method="linear", fill_value=0).astype(np.float32).ravel()

# =========================
# Operators and RHS
# =========================
L, GtG = build_operators(Hc, Wc)

# Compatibility RHS (Poisson-type)
b_u = (np.gradient(exx_c, axis=1) + 0.5*np.gradient(exy_c, axis=0)).astype(np.float32).ravel()
b_v = (np.gradient(eyy_c, axis=0) + 0.5*np.gradient(exy_c, axis=1)).astype(np.float32).ravel()

# Boundary condition matrix (enforce inside ROI only)
M_diag = np.zeros(Nc, dtype=np.float32)
M_diag[mask_c.ravel()] = 1.0
M = diags(M_diag, 0, shape=(Nc, Nc), format="csr")

# Precompute adjoint-projected RHS for data term (for zeroth-order) — keep for both to be consistent
LTb_u = (L.T @ b_u).astype(np.float32)
LTb_v = (L.T @ b_v).astype(np.float32)

# Choose regularizer matrix
RegMat = GtG if use_first_order else (L.T @ L)

def solve_for_lambda(lmbda):
    A = (RegMat + lmbda * M).tocsr()
    # project b via L^T for both cases — treats data term consistently
    bu = (LTb_u + lmbda * (M @ u_bc_c)).astype(np.float32)
    bv = (LTb_v + lmbda * (M @ v_bc_c)).astype(np.float32)
    u = spsolve(A, bu).reshape(Hc, Wc).astype(np.float32)
    v = spsolve(A, bv).reshape(Hc, Wc).astype(np.float32)
    return u, v

# =========================
# Lambda sweep + objective tracking
# =========================
curve = []
best = None
print("Tuning regularization λ ...")
for lam in lambda_grid:
    try:
        u_c, v_c = solve_for_lambda(lam)
        _, _, metrics = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_c, v_c, do_hist_match)
        data_term, bc_term, total = objective_terms(L, M, u_c, v_c, b_u, b_v, lam, u_bc_c, v_bc_c)
        curve.append(dict(lam=lam, metrics=metrics, data=data_term, bc=bc_term, total=total))
        print(f"λ={lam:g} -> SSIM={metrics['ssim']:.4f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
              f"||Lu-b||^2={data_term:.2e}  ||M(..)||^2={bc_term:.2e}")
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
# L-curve and Metrics vs λ plots
# =========================
lam_vals = np.array([d["lam"] for d in curve], dtype=float)
data_vals = np.array([d["data"] for d in curve], dtype=float)
bc_vals   = np.array([d["bc"]   for d in curve], dtype=float)
ssim_vals = np.array([d["metrics"]["ssim"] for d in curve], dtype=float)
rmse_vals = np.array([d["metrics"]["rmse"] for d in curve], dtype=float)

best_idx = np.argmax(ssim_vals)

plt.figure(figsize=(6,5))
plt.loglog(data_vals, bc_vals, marker='o')
plt.scatter([data_vals[best_idx]], [bc_vals[best_idx]], s=80)
plt.title("L-curve: data term vs boundary term")
plt.xlabel(r"$\|Lu-b\|_2^2 + \|Lv-b\|_2^2$")
plt.ylabel(r"$\|M(u-u_{bc})\|_2^2 + \|M(v-v_{bc})\|_2^2$")
plt.grid(True, which="both")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.semilogx(lam_vals, ssim_vals, marker='o')
plt.scatter([lam_vals[best_idx]], [ssim_vals[best_idx]], s=80)
plt.xlabel("λ"); plt.ylabel("SSIM"); plt.title("SSIM vs λ"); plt.grid(True, which="both")
plt.subplot(1,2,2)
plt.semilogx(lam_vals, rmse_vals, marker='o')
plt.scatter([lam_vals[best_idx]], [rmse_vals[best_idx]], s=80)
plt.xlabel("λ"); plt.ylabel("RMSE"); plt.title("RMSE vs λ"); plt.grid(True, which="both")
plt.tight_layout(); plt.show()

# =========================
# Final reconstruction (best λ)
# =========================
recon_c_best, err_c_best, metrics_best = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, best["u"], best["v"], do_hist_match)

if do_tv_postprocess:
    u_tv = denoise_tv_chambolle(best["u"], weight=tv_weight)
    v_tv = denoise_tv_chambolle(best["v"], weight=tv_weight)
    recon_c_tv, err_c_tv, metrics_tv = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_tv.astype(np.float32), v_tv.astype(np.float32), do_hist_match)
else:
    recon_c_tv, err_c_tv, metrics_tv = None, None, None

# Stitch back to full image (optional, for visuals)
recon_full = ref_img.copy()
err_full = np.zeros_like(ref_img, dtype=np.float32)
recon_full[r0:r1, c0:c1][mask_c] = recon_c_best[mask_c]
err_full[r0:r1, c0:c1][mask_c]   = err_c_best[mask_c]

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(def_img, cmap="gray"); plt.title("Actual Deformed"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(recon_full, cmap="gray"); plt.title(f"Reconstructed (best λ={best['lam']:.3g})"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(err_full, cmap="bwr", vmin=-plot_error_vlim, vmax=plot_error_vlim); 
plt.colorbar(label="Error (intensity)")
plt.title(f"Error Heatmap\nSSIM={metrics_best['ssim']:.3f}, RMSE={metrics_best['rmse']:.2f}")
plt.axis('off'); plt.tight_layout(); plt.show()

# =========================
# Small λ vs Best λ vs Large λ (triptych)
# =========================
lam_small = lam_vals[0]
lam_large = lam_vals[-1]
u_s, v_s = solve_for_lambda(lam_small)
u_b, v_b = best["u"], best["v"]
u_l, v_l = solve_for_lambda(lam_large)

recon_s, err_s, met_s = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_s, v_s, do_hist_match)
recon_b, err_b, met_b = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_b, v_b, do_hist_match)
recon_l, err_l, met_l = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_l, v_l, do_hist_match)

def panel_gray(imgs, titles, suptitle):
    plt.figure(figsize=(15,5))
    for i,(im,ti) in enumerate(zip(imgs,titles),1):
        plt.subplot(1,3,i); plt.imshow(im, cmap="gray"); plt.title(ti); plt.axis('off')
    plt.suptitle(suptitle); plt.tight_layout(); plt.show()

panel_gray(
  [recon_s, recon_b, recon_l],
  [f"λ={lam_small:g}\nSSIM={met_s['ssim']:.3f}",
   f"λ*={best['lam']:.3g}\nSSIM={met_b['ssim']:.3f}",
   f"λ={lam_large:g}\nSSIM={met_l['ssim']:.3f}"],
  "Strain-based reconstruction: small λ vs best λ vs large λ"
)

# Optional: show error heatmaps for triptych
plt.figure(figsize=(15,5))
for i,(err,ti) in enumerate([(err_s, f"λ={lam_small:g}"),
                             (err_b, f"λ*={best['lam']:.3g}"),
                             (err_l, f"λ={lam_large:g}")], 1):
    plt.subplot(1,3,i)
    plt.imshow(err, cmap="bwr", vmin=-plot_error_vlim, vmax=plot_error_vlim)
    plt.title(f"Error heatmap ({ti})"); plt.axis('off')
plt.tight_layout(); plt.show()

# =========================
# Noise robustness: SSIM vs λ (clean vs noisy strains)
# =========================
exx_n = add_noise(exx_c, noise_frac_for_study, mask_c)
eyy_n = add_noise(eyy_c, noise_frac_for_study, mask_c)
exy_n = add_noise(exy_c, noise_frac_for_study, mask_c)

b_u_n = (np.gradient(exx_n, axis=1) + 0.5*np.gradient(exy_n, axis=0)).astype(np.float32).ravel()
b_v_n = (np.gradient(eyy_n, axis=0) + 0.5*np.gradient(exy_n, axis=1)).astype(np.float32).ravel()
LTb_u_n = (L.T @ b_u_n).astype(np.float32)
LTb_v_n = (L.T @ b_v_n).astype(np.float32)

def solve_for_lambda_noisy(lmbda):
    A = (RegMat + lmbda * M).tocsr()
    bu = (LTb_u_n + lmbda * (M @ u_bc_c)).astype(np.float32)
    bv = (LTb_v_n + lmbda * (M @ v_bc_c)).astype(np.float32)
    u = spsolve(A, bu).reshape(Hc, Wc).astype(np.float32)
    v = spsolve(A, bv).reshape(Hc, Wc).astype(np.float32)
    return u, v

ssim_noisy = []
for lam in lambda_grid:
    try:
        u_n, v_n = solve_for_lambda_noisy(lam)
        _, _, met_n = reconstruct_and_score(ref_c, def_c, mask_c, mask_int_c, u_n, v_n, do_hist_match)
        ssim_noisy.append(met_n["ssim"])
    except Exception:
        ssim_noisy.append(np.nan)
ssim_noisy = np.array(ssim_noisy, dtype=float)

plt.figure(figsize=(6,4))
plt.semilogx(lam_vals, ssim_vals, marker='o', label="clean")
plt.semilogx(lam_vals, ssim_noisy, marker='o', label=f"noisy ({int(100*noise_frac_for_study)}%)")
plt.xlabel("λ"); plt.ylabel("SSIM"); plt.title("Noise robustness: SSIM vs λ")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()

# =========================
# Optional: TV post-processing result panel
# =========================
if do_tv_postprocess:
    panel_gray(
        [recon_b, recon_c_tv, def_c],
        [f"Before TV (λ*={best['lam']:.3g})\nSSIM={metrics_best['ssim']:.3f}",
         f"After TV (w={tv_weight})\nSSIM={metrics_tv['ssim']:.3f}",
         "Actual deformed (crop)"],
        "Effect of TV post-processing on displacement fields"
    )
