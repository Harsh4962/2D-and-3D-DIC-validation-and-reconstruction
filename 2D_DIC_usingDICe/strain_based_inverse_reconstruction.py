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
# Small λ vs Best λ vs Large λ
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
