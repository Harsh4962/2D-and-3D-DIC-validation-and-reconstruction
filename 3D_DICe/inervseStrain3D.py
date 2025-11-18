# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from scipy.interpolate import griddata
# # from scipy.sparse import csr_matrix, lil_matrix
# # from scipy.sparse.linalg import spsolve
# # import cv2
# # from skimage.metrics import structural_similarity as ssim
# # from skimage.metrics import peak_signal_noise_ratio as psnr

# # # ==============================
# # # 1) PATHS
# # # ==============================
# # DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
# # REF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
# # DEF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_0.tif"

# # print("=" * 100)
# # print(" INVERSE STRAIN INTEGRATION — CORRECTED ")
# # print("=" * 100)

# # # ==============================
# # # 2) LOAD DICe DATA
# # # ==============================
# # df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine="python")

# # # Plane coordinates
# # X = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
# # Y = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
# # Z = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

# # # Strains
# # exx = df["VSG_STRAIN_XX"].astype(float).to_numpy()
# # eyy = df["VSG_STRAIN_YY"].astype(float).to_numpy()
# # exy = df["VSG_STRAIN_XY"].astype(float).to_numpy()

# # # DIC displacements (used as boundary conditions)
# # u_meas = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
# # v_meas = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
# # w_meas = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# # # Image coordinates
# # u_img = df["COORDINATE_X"].astype(float).to_numpy()
# # v_img = df["COORDINATE_Y"].astype(float).to_numpy()

# # reference_image = cv2.imread(REF_CAM0, cv2.IMREAD_GRAYSCALE)
# # actual_deformed = cv2.imread(DEF_CAM0, cv2.IMREAD_GRAYSCALE)

# # H, W = reference_image.shape
# # N = len(X)

# # print(f"Loaded {N} points")
# # print(f"Image size: {W} × {H}")

# # # ============================================================
# # # 3) GRID IN SPECIMEN PLANE
# # # ============================================================
# # sx = 0.12  # mm
# # sy = 0.12

# # x_min, x_max = X.min(), X.max()
# # y_min, y_max = Y.min(), Y.max()

# # nx = int((x_max - x_min) / sx) + 1
# # ny = int((y_max - y_min) / sy) + 1

# # xg = np.linspace(x_min, x_max, nx)
# # yg = np.linspace(y_min, y_max, ny)
# # XX, YY = np.meshgrid(xg, yg)

# # pts_spec = np.vstack([X, Y]).T

# # # ============================================================
# # # 4) INTERPOLATE Z and strains to plane grid   <<< FIXED
# # # ============================================================
# # Z_plane = griddata(pts_spec, Z, (XX, YY), method='linear', fill_value=np.nan)
# # exx_g   = griddata(pts_spec, exx, (XX, YY), method='cubic', fill_value=0)
# # eyy_g   = griddata(pts_spec, eyy, (XX, YY), method='cubic', fill_value=0)
# # exy_g   = griddata(pts_spec, exy, (XX, YY), method='cubic', fill_value=0)

# # # Clean NaN in Z
# # Z_plane = np.nan_to_num(Z_plane)

# # print(f"Grid created: {nx} × {ny}")
# # print(f"Z plane min,max = {Z_plane.min():.4f}, {Z_plane.max():.4f}")

# # # ============================================================
# # # 5) BUILD COMPATIBILITY SYSTEM   (same as before)
# # # ============================================================
# # dx = xg[1] - xg[0]
# # dy = yg[1] - yg[0]
# # n_nodes = nx * ny

# # def build_poisson(nx, ny, dx, dy, bc_mask, bc_vals):
# #     n = nx * ny
# #     A = lil_matrix((n, n))
# #     b = np.zeros(n)

# #     for j in range(ny):
# #         for i in range(nx):
# #             idx = j * nx + i
# #             if bc_mask[j, i]:
# #                 A[idx, idx] = 1
# #                 b[idx] = bc_vals[j, i]
# #             else:
# #                 A[idx, idx] = -4
# #                 if i > 0:     A[idx, idx - 1] = 1
# #                 if i < nx-1: A[idx, idx + 1] = 1
# #                 if j > 0:     A[idx, idx - nx] = 1
# #                 if j < ny-1: A[idx, idx + nx] = 1

# #     return A.tocsr(), b

# # # Create BC mask
# # bc_mask = np.zeros((ny, nx), bool)
# # bc_u = np.zeros((ny, nx))
# # bc_v = np.zeros((ny, nx))
# # bc_w = np.zeros((ny, nx))

# # for k in range(N):
# #     i = np.argmin(np.abs(xg - X[k]))
# #     j = np.argmin(np.abs(yg - Y[k]))
# #     bc_mask[j, i] = True
# #     bc_u[j, i] = u_meas[k]
# #     bc_v[j, i] = v_meas[k]
# #     bc_w[j, i] = w_meas[k]

# # A_u, b_u = build_poisson(nx, ny, dx, dy, bc_mask, bc_u)
# # A_v, b_v = build_poisson(nx, ny, dx, dy, bc_mask, bc_v)
# # A_w, b_w = build_poisson(nx, ny, dx, dy, bc_mask, bc_w)

# # print("Systems built:", A_u.shape)

# # # ============================================================
# # # 6) SOLVE FOR U, V, W
# # # ============================================================
# # print("Solving Poisson...")
# # u_vec = spsolve(A_u, b_u)
# # v_vec = spsolve(A_v, b_v)
# # w_vec = spsolve(A_w, b_w)

# # u_rec = u_vec.reshape(ny, nx)
# # v_rec = v_vec.reshape(ny, nx)
# # w_rec = w_vec.reshape(ny, nx)

# # print("Solutions recovered")

# # # ============================================================
# # # 7) FORM 3D POINT CLOUD <<< FIXED block
# # # ============================================================
# # P0_plane = np.stack([XX, YY, Z_plane], axis=-1).reshape(-1, 3)
# # P1_plane = np.stack([XX + u_rec, YY + v_rec, Z_plane + w_rec], axis=-1).reshape(-1, 3)

# # P0_plane = np.nan_to_num(P0_plane)
# # P1_plane = np.nan_to_num(P1_plane)

# # print("3D cloud created")


# # print("\n Finished up to P1 construction (no broadcast errors)")
# # print("Next: apply camera projection → warp → compare images")

# # # ==============================
# # # 8) PROJECTION + REMAP + METRICS
# # # ==============================
# # import re, xml.etree.ElementTree as ET

# # # --- set these two paths ---
# # BEST_PLANE = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\best_fit_plane_out.dat"
# # CAL_XML    = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"

# # def parse_best_plane_transform(path):
# #     """
# #     Reads the 'Transform from current world coords to best fit plane origin and bases'
# #     3x4 block from best_fit_plane_out.dat
# #     plane = R * cam + t   =>   cam = R^T * (plane - t)
# #     """
# #     txt = open(path, 'r').read()
# #     nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
# #     vals = list(map(float, nums))
# #     if len(vals) < 12:
# #         raise RuntimeError("best_fit_plane_out.dat: 3x4 transform not found.")
# #     block = vals[-12:]
# #     R = np.array([[block[0], block[1], block[2]],
# #                   [block[4], block[5], block[6]],
# #                   [block[8], block[9], block[10]]], dtype=float)
# #     t = np.array([block[3], block[7], block[11]], dtype=float)
# #     return R, t

# # def parse_left_cam_from_calxml(path):
# #     tree = ET.parse(path); root = tree.getroot()
# #     cam0 = None
# #     for pl in root.iter('ParameterList'):
# #         if pl.get('name','') == 'CAMERA 0':
# #             cam0 = pl; break
# #     if cam0 is None:
# #         raise RuntimeError("CAMERA 0 not found in cal.xml")

# #     def getp(name, default=0.0):
# #         for p in cam0.iter('Parameter'):
# #             if p.get('name') == name:
# #                 try: return float(p.get('value'))
# #                 except: return default
# #         return default

# #     fx = getp('FX'); fy = getp('FY'); cx = getp('CX'); cy = getp('CY')
# #     k1 = getp('K1'); k2 = getp('K2'); k3 = getp('K3')
# #     p1 = getp('P1', 0.0); p2 = getp('P2', 0.0)  # may be missing
# #     return dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)

# # def distort_opencv(xn, yn, k1, k2, k3, p1, p2):
# #     r2 = xn*xn + yn*yn
# #     radial = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
# #     x_t = 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
# #     y_t = p1*(r2 + 2*yn*yn) + 2*p2*xn*yn
# #     xd = xn*radial + x_t
# #     yd = yn*radial + y_t
# #     return xd, yd

# # def project_points_cam0(P_cam, K):
# #     X, Y, Z = P_cam[:,0], P_cam[:,1], P_cam[:,2]
# #     # guard against Z<=0
# #     Z = np.where(Z<=1e-6, 1e-6, Z)
# #     xn, yn = X/Z, Y/Z
# #     xd, yd = distort_opencv(xn, yn, K['k1'], K['k2'], K['k3'], K['p1'], K['p2'])
# #     u = K['fx'] * xd + K['cx']
# #     v = K['fy'] * yd + K['cy']
# #     return np.vstack([u, v]).T

# # # --- parse calibration + transform ---
# # R_plane_cam, t_plane_cam = parse_best_plane_transform(BEST_PLANE)  # plane = R*cam + t
# # K = parse_left_cam_from_calxml(CAL_XML)

# # # inverse: plane -> camera
# # # cam = R^T * (plane - t)
# # P0_cam = (R_plane_cam.T @ (P0_plane - t_plane_cam).T).T
# # P1_cam = (R_plane_cam.T @ (P1_plane - t_plane_cam).T).T

# # # project both
# # uv0 = project_points_cam0(P0_cam, K)   # (N,2)
# # uv1 = project_points_cam0(P1_cam, K)

# # # pixel flow on the projected specimen grid
# # du_pix = (uv1[:,0] - uv0[:,0]).reshape(ny, nx)
# # dv_pix = (uv1[:,1] - uv0[:,1]).reshape(ny, nx)

# # # interpolate flow to full image grid
# # x_img = np.arange(W, dtype=np.float32)
# # y_img = np.arange(H, dtype=np.float32)
# # XX_img, YY_img = np.meshgrid(x_img, y_img)

# # # Keep only points whose base projection lies inside the image
# # in_bounds = (uv0[:,0] >= 0) & (uv0[:,0] < W) & (uv0[:,1] >= 0) & (uv0[:,1] < H)
# # uv0_in   = uv0[in_bounds]
# # du_in    = du_pix.ravel()[in_bounds]
# # dv_in    = dv_pix.ravel()[in_bounds]

# # # If griddata complains near edges, you can change 'cubic' -> 'linear'
# # du_im = griddata(uv0_in, du_in, (XX_img, YY_img), method='cubic', fill_value=0.0)
# # dv_im = griddata(uv0_in, dv_in, (XX_img, YY_img), method='cubic', fill_value=0.0)

# # # build a validity mask around measured subset centers for fair metrics
# # valid = np.zeros((H, W), dtype=bool)
# # uu = np.clip(u_img.astype(int), 0, W-1)
# # vv = np.clip(v_img.astype(int), 0, H-1)
# # patch = 10  # px half-size around each subset center
# # for u0, v0 in zip(uu, vv):
# #     valid[max(0, v0-patch):min(H, v0+patch+1),
# #           max(0, u0-patch):min(W, u0+patch+1)] = True

# # # remap the reference image using the flow
# # map_x = (XX_img - du_im).astype(np.float32)
# # map_y = (YY_img - dv_im).astype(np.float32)
# # recon = cv2.remap(reference_image.astype(np.float32), map_x, map_y,
# #                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
# # recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)

# # # metrics (on valid mask)
# # diff = actual_deformed.astype(float) - recon_u8.astype(float)
# # diff_valid = diff[valid]
# # RMS  = float(np.sqrt(np.mean(diff_valid**2)))
# # MAE  = float(np.mean(np.abs(diff_valid)))
# # SSIM = float(ssim(actual_deformed, recon_u8, data_range=255))
# # PSNR = float(psnr(actual_deformed, recon_u8, data_range=255))

# # print("\n--- Projection & Remap metrics (left camera) ---")
# # print(f"RMS  : {RMS:.2f} gray")
# # print(f"MAE  : {MAE:.2f} gray")
# # print(f"SSIM : {SSIM:.4f}")
# # print(f"PSNR : {PSNR:.2f} dB")

# # # quick visuals
# # plt.figure(figsize=(13,5))
# # plt.subplot(1,2,1); plt.imshow(actual_deformed, cmap='gray'); plt.title('Actual deformed'); plt.axis('off')
# # plt.subplot(1,2,2); plt.imshow(recon_u8, cmap='gray'); plt.title('Reconstructed (projection-based)'); plt.axis('off')
# # plt.tight_layout(); plt.show()

# # plt.figure(figsize=(12,5))
# # plt.subplot(1,2,1); plt.imshow(du_im, cmap='coolwarm'); plt.title('Δu (px)'); plt.colorbar()
# # plt.subplot(1,2,2); plt.imshow(dv_im, cmap='coolwarm'); plt.title('Δv (px)'); plt.colorbar()
# # plt.tight_layout(); plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from scipy.sparse import csr_matrix, lil_matrix
# from scipy.sparse.linalg import spsolve
# import cv2
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# import re
# import xml.etree.ElementTree as ET

# # ==============================
# # PATHS
# # ==============================
# DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
# REF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
# DEF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_0.tif"
# BEST_PLANE = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\best_fit_plane_out.dat"
# CAL_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"

# print("=" * 100)
# print(" STRAIN-BASED INVERSE RECONSTRUCTION (CORRECTED) ")
# print("=" * 100)

# # ==============================
# # LOAD DICe DATA
# # ==============================
# df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine="python")

# # Plane coordinates
# X = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
# Y = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
# Z = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

# # Strains (PRIMARY INPUT for strain-based approach!)
# exx = df["VSG_STRAIN_XX"].astype(float).to_numpy()
# eyy = df["VSG_STRAIN_YY"].astype(float).to_numpy()
# exy = df["VSG_STRAIN_XY"].astype(float).to_numpy()

# # Measured displacements (used ONLY as boundary conditions)
# u_meas = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
# v_meas = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
# w_meas = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

# # Image coordinates
# u_img = df["COORDINATE_X"].astype(float).to_numpy()
# v_img = df["COORDINATE_Y"].astype(float).to_numpy()

# reference_image = cv2.imread(REF_CAM0, cv2.IMREAD_GRAYSCALE)
# actual_deformed = cv2.imread(DEF_CAM0, cv2.IMREAD_GRAYSCALE)

# H, W = reference_image.shape
# N = len(X)

# print(f"\nLoaded {N} measurement points")
# print(f"Image size: {W} × {H} pixels")
# print(f"\nStrain statistics:")
# print(f"  εxx: [{exx.min():.6f}, {exx.max():.6f}], mean={exx.mean():.6f}")
# print(f"  εyy: [{eyy.min():.6f}, {eyy.max():.6f}], mean={eyy.mean():.6f}")
# print(f"  εxy: [{exy.min():.6f}, {exy.max():.6f}], mean={exy.mean():.6f}")

# # ==============================
# # CREATE PLANE GRID
# # ==============================
# sx = 0.12  # mm
# sy = 0.12

# x_min, x_max = X.min(), X.max()
# y_min, y_max = Y.min(), Y.max()

# nx = int((x_max - x_min) / sx) + 1
# ny = int((y_max - y_min) / sy) + 1

# xg = np.linspace(x_min, x_max, nx)
# yg = np.linspace(y_min, y_max, ny)
# XX, YY = np.meshgrid(xg, yg)

# dx = xg[1] - xg[0]
# dy = yg[1] - yg[0]

# pts_spec = np.vstack([X, Y]).T

# print(f"\nPlane grid created: {nx} × {ny} = {nx*ny} points")
# print(f"Grid spacing: dx={dx:.4f} mm, dy={dy:.4f} mm")

# # ==============================
# # INTERPOLATE Z AND STRAINS TO GRID
# # ==============================
# print("\nInterpolating geometry and strain fields to grid...")

# Z_plane = griddata(pts_spec, Z, (XX, YY), method='linear', fill_value=np.nan)
# Z_plane = np.nan_to_num(Z_plane)

# exx_g = griddata(pts_spec, exx, (XX, YY), method='cubic', fill_value=0)
# eyy_g = griddata(pts_spec, eyy, (XX, YY), method='cubic', fill_value=0)
# exy_g = griddata(pts_spec, exy, (XX, YY), method='cubic', fill_value=0)

# print("✓ Interpolation complete")
# print(f"  Z range: [{Z_plane.min():.4f}, {Z_plane.max():.4f}] mm")

# # ==============================
# # COMPUTE STRAIN DERIVATIVES (RHS)
# # ==============================
# print("\nComputing strain compatibility RHS...")

# # Strain derivatives for u-displacement
# deps_xx_dx = np.gradient(exx_g, dx, axis=1)  # ∂εxx/∂x
# deps_xy_dy = np.gradient(exy_g, dy, axis=0)  # ∂εxy/∂y
# rhs_u = deps_xx_dx + deps_xy_dy

# # Strain derivatives for v-displacement
# deps_xy_dx = np.gradient(exy_g, dx, axis=1)  # ∂εxy/∂x
# deps_yy_dy = np.gradient(eyy_g, dy, axis=0)  # ∂εyy/∂y
# rhs_v = deps_xy_dx + deps_yy_dy

# # For w: assume small out-of-plane strain, use Laplace smoothing
# rhs_w = np.zeros_like(rhs_u)

# print(f"✓ RHS computed:")
# print(f"  RHS_u range: [{rhs_u.min():.6f}, {rhs_u.max():.6f}]")
# print(f"  RHS_v range: [{rhs_v.min():.6f}, {rhs_v.max():.6f}]")

# # ==============================
# # BUILD STRAIN COMPATIBILITY POISSON SYSTEM
# # ==============================
# print("\nBuilding Poisson system with strain compatibility...")

# n_nodes = nx * ny

# def build_strain_poisson(nx, ny, dx, dy, rhs_field, bc_mask, bc_vals):
#     """
#     Build Poisson system: ∇²u = rhs_field
#     with Dirichlet BCs at bc_mask locations
#     """
#     n = nx * ny
#     A = lil_matrix((n, n))
#     b = np.zeros(n)
    
#     # Laplacian coefficients (proper scaling)
#     coeff_x = 1.0 / (dx * dx)
#     coeff_y = 1.0 / (dy * dy)
#     coeff_center = -2 * (coeff_x + coeff_y)
    
#     for j in range(ny):
#         for i in range(nx):
#             idx = j * nx + i
            
#             if bc_mask[j, i]:
#                 # Dirichlet boundary condition
#                 A[idx, idx] = 1.0
#                 b[idx] = bc_vals[j, i]
#             else:
#                 # Interior: Laplacian with strain RHS
#                 A[idx, idx] = coeff_center
                
#                 # Neighbors
#                 if i > 0:
#                     A[idx, idx - 1] = coeff_x
#                 else:
#                     A[idx, idx] -= coeff_x  # Neumann BC at boundary
                    
#                 if i < nx - 1:
#                     A[idx, idx + 1] = coeff_x
#                 else:
#                     A[idx, idx] -= coeff_x
                    
#                 if j > 0:
#                     A[idx, idx - nx] = coeff_y
#                 else:
#                     A[idx, idx] -= coeff_y
                    
#                 if j < ny - 1:
#                     A[idx, idx + nx] = coeff_y
#                 else:
#                     A[idx, idx] -= coeff_y
                
#                 # RHS from strain compatibility!
#                 b[idx] = rhs_field[j, i]
    
#     return A.tocsr(), b

# # Create BC mask: enforce measured displacements at measurement points
# bc_mask = np.zeros((ny, nx), dtype=bool)
# bc_u = np.zeros((ny, nx))
# bc_v = np.zeros((ny, nx))
# bc_w = np.zeros((ny, nx))

# for k in range(N):
#     i = np.argmin(np.abs(xg - X[k]))
#     j = np.argmin(np.abs(yg - Y[k]))
#     bc_mask[j, i] = True
#     bc_u[j, i] = u_meas[k]
#     bc_v[j, i] = v_meas[k]
#     bc_w[j, i] = w_meas[k]

# print(f"✓ Dirichlet BCs applied at {bc_mask.sum()} grid nodes")

# # Build systems
# print("\nBuilding system matrices...")
# A_u, b_u = build_strain_poisson(nx, ny, dx, dy, rhs_u, bc_mask, bc_u)
# A_v, b_v = build_strain_poisson(nx, ny, dx, dy, rhs_v, bc_mask, bc_v)
# A_w, b_w = build_strain_poisson(nx, ny, dx, dy, rhs_w, bc_mask, bc_w)

# print(f"✓ System size: {A_u.shape[0]} × {A_u.shape[1]}")
# print(f"  Non-zeros: {A_u.nnz}")

# # ==============================
# # SOLVE FOR DISPLACEMENTS FROM STRAIN INTEGRATION
# # ==============================
# print("\nSolving Poisson systems (integrating strains)...")

# u_vec = spsolve(A_u, b_u)
# v_vec = spsolve(A_v, b_v)
# w_vec = spsolve(A_w, b_w)

# u_rec = u_vec.reshape(ny, nx)
# v_rec = v_vec.reshape(ny, nx)
# w_rec = w_vec.reshape(ny, nx)

# print("✓ Solutions obtained:")
# print(f"  u range: [{u_rec.min():.6f}, {u_rec.max():.6f}] mm")
# print(f"  v range: [{v_rec.min():.6f}, {v_rec.max():.6f}] mm")
# print(f"  w range: [{w_rec.min():.6f}, {w_rec.max():.6f}] mm")

# # ==============================
# # FORM 3D POINT CLOUDS
# # ==============================
# print("\nConstructing 3D surfaces...")

# P0_plane = np.stack([XX, YY, Z_plane], axis=-1).reshape(-1, 3)
# P1_plane = np.stack([XX + u_rec, YY + v_rec, Z_plane + w_rec], axis=-1).reshape(-1, 3)

# P0_plane = np.nan_to_num(P0_plane)
# P1_plane = np.nan_to_num(P1_plane)

# print(f"✓ Point clouds created: {P0_plane.shape[0]} points")

# # ==============================
# # CAMERA PROJECTION FUNCTIONS
# # ==============================
# def parse_best_plane_transform(path):
#     """Reads 3×4 transform from best_fit_plane_out.dat"""
#     txt = open(path, 'r').read()
#     nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
#     vals = list(map(float, nums))
#     if len(vals) < 12:
#         raise RuntimeError("best_fit_plane_out.dat: 3×4 transform not found.")
#     block = vals[-12:]
#     R = np.array([[block[0], block[1], block[2]],
#                   [block[4], block[5], block[6]],
#                   [block[8], block[9], block[10]]], dtype=float)
#     t = np.array([block[3], block[7], block[11]], dtype=float)
#     return R, t

# def parse_left_cam_from_calxml(path):
#     """Parse camera 0 intrinsics from cal.xml"""
#     tree = ET.parse(path)
#     root = tree.getroot()
#     cam0 = None
#     for pl in root.iter('ParameterList'):
#         if pl.get('name', '') == 'CAMERA 0':
#             cam0 = pl
#             break
#     if cam0 is None:
#         raise RuntimeError("CAMERA 0 not found in cal.xml")
    
#     def getp(name, default=0.0):
#         for p in cam0.iter('Parameter'):
#             if p.get('name') == name:
#                 try:
#                     return float(p.get('value'))
#                 except:
#                     return default
#         return default
    
#     fx = getp('FX')
#     fy = getp('FY')
#     cx = getp('CX')
#     cy = getp('CY')
#     k1 = getp('K1')
#     k2 = getp('K2')
#     k3 = getp('K3')
#     p1 = getp('P1', 0.0)
#     p2 = getp('P2', 0.0)
    
#     return dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)

# def distort_opencv(xn, yn, k1, k2, k3, p1, p2):
#     """Apply OpenCV distortion model"""
#     r2 = xn*xn + yn*yn
#     radial = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
#     x_t = 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
#     y_t = p1*(r2 + 2*yn*yn) + 2*p2*xn*yn
#     xd = xn*radial + x_t
#     yd = yn*radial + y_t
#     return xd, yd

# def project_points_cam0(P_cam, K):
#     """Project 3D points to image plane"""
#     X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
#     Z = np.where(Z <= 1e-6, 1e-6, Z)  # Guard against Z <= 0
#     xn, yn = X/Z, Y/Z
#     xd, yd = distort_opencv(xn, yn, K['k1'], K['k2'], K['k3'], K['p1'], K['p2'])
#     u = K['fx'] * xd + K['cx']
#     v = K['fy'] * yd + K['cy']
#     return np.vstack([u, v]).T

# # ==============================
# # TRANSFORM AND PROJECT
# # ==============================
# print("\nLoading calibration and transforming to camera frame...")

# R_plane_cam, t_plane_cam = parse_best_plane_transform(BEST_PLANE)
# K = parse_left_cam_from_calxml(CAL_XML)

# print(f"✓ Calibration loaded:")
# print(f"  Intrinsics: fx={K['fx']:.2f}, fy={K['fy']:.2f}, cx={K['cx']:.2f}, cy={K['cy']:.2f}")
# print(f"  Distortion: k1={K['k1']:.4e}, k2={K['k2']:.4e}, k3={K['k3']:.4e}")

# # Transform: plane → camera
# # cam = R^T * (plane - t)
# P0_cam = (R_plane_cam.T @ (P0_plane - t_plane_cam).T).T
# P1_cam = (R_plane_cam.T @ (P1_plane - t_plane_cam).T).T

# # Project both surfaces
# print("\nProjecting 3D surfaces to image plane...")
# uv0 = project_points_cam0(P0_cam, K)
# uv1 = project_points_cam0(P1_cam, K)

# # Compute optical flow on plane grid
# du_pix = (uv1[:, 0] - uv0[:, 0]).reshape(ny, nx)
# dv_pix = (uv1[:, 1] - uv0[:, 1]).reshape(ny, nx)

# print(f"✓ Projection complete:")
# print(f"  Flow range: du=[{du_pix.min():.2f}, {du_pix.max():.2f}] px")
# print(f"              dv=[{dv_pix.min():.2f}, {dv_pix.max():.2f}] px")

# # ==============================
# # INTERPOLATE FLOW TO FULL IMAGE
# # ==============================
# print("\nInterpolating optical flow to full image resolution...")

# x_img = np.arange(W, dtype=np.float32)
# y_img = np.arange(H, dtype=np.float32)
# XX_img, YY_img = np.meshgrid(x_img, y_img)

# # Filter valid projections
# in_bounds = (uv0[:, 0] >= 0) & (uv0[:, 0] < W) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H)
# uv0_in = uv0[in_bounds]
# du_in = du_pix.ravel()[in_bounds]
# dv_in = dv_pix.ravel()[in_bounds]

# # Interpolate to image grid
# du_im = griddata(uv0_in, du_in, (XX_img, YY_img), method='cubic', fill_value=0.0)
# dv_im = griddata(uv0_in, dv_in, (XX_img, YY_img), method='cubic', fill_value=0.0)

# print("✓ Flow interpolated to full image")

# # ==============================
# # CREATE VALIDITY MASK
# # ==============================
# valid = np.zeros((H, W), dtype=bool)
# uu = np.clip(u_img.astype(int), 0, W-1)
# vv = np.clip(v_img.astype(int), 0, H-1)
# patch = 15  # pixels

# for u0, v0 in zip(uu, vv):
#     valid[max(0, v0-patch):min(H, v0+patch+1),
#           max(0, u0-patch):min(W, u0+patch+1)] = True

# print(f"✓ Valid region: {valid.sum()} pixels ({100*valid.sum()/(H*W):.1f}% of image)")

# # ==============================
# # RECONSTRUCT DEFORMED IMAGE
# # ==============================
# print("\nReconstructing deformed image...")

# map_x = (XX_img - du_im).astype(np.float32)
# map_y = (YY_img - dv_im).astype(np.float32)

# recon = cv2.remap(reference_image.astype(np.float32), map_x, map_y,
#                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
# recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)

# print("✓ Reconstruction complete")

# # ==============================
# # COMPUTE METRICS
# # ==============================
# print("\n" + "="*100)
# print("VALIDATION METRICS (Strain-Based Reconstruction)")
# print("="*100)

# diff = actual_deformed.astype(float) - recon_u8.astype(float)
# diff_valid = diff[valid]

# RMS = float(np.sqrt(np.mean(diff_valid**2)))
# MAE = float(np.mean(np.abs(diff_valid)))
# SSIM = float(ssim(actual_deformed, recon_u8, data_range=255))
# PSNR = float(psnr(actual_deformed, recon_u8, data_range=255))

# print(f"\nReconstruction Error:")
# print(f"  RMS:  {RMS:.2f} gray levels")
# print(f"  MAE:  {MAE:.2f} gray levels")
# print(f"  SSIM: {SSIM:.4f}")
# print(f"  PSNR: {PSNR:.2f} dB")

# # ==============================
# # VISUALIZATIONS
# # ==============================
# print("\nGenerating visualizations...")

# # Image comparison
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# axes[0].imshow(reference_image, cmap='gray')
# axes[0].set_title('Reference Image', fontsize=14, fontweight='bold')
# axes[0].axis('off')

# axes[1].imshow(actual_deformed, cmap='gray')
# axes[1].set_title('Actual Deformed', fontsize=14, fontweight='bold')
# axes[1].axis('off')

# axes[2].imshow(recon_u8, cmap='gray')
# axes[2].set_title(f'Strain-Based Reconstruction\nRMS={RMS:.2f}, SSIM={SSIM:.4f}', 
#                  fontsize=14, fontweight='bold')
# axes[2].axis('off')

# plt.tight_layout()
# plt.show()

# # Optical flow
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# im1 = axes[0].imshow(du_im, cmap='coolwarm', vmin=-np.abs(du_im).max(), vmax=np.abs(du_im).max())
# axes[0].set_title('Horizontal Flow Δu (pixels)', fontsize=13, fontweight='bold')
# plt.colorbar(im1, ax=axes[0])

# im2 = axes[1].imshow(dv_im, cmap='coolwarm', vmin=-np.abs(dv_im).max(), vmax=np.abs(dv_im).max())
# axes[1].set_title('Vertical Flow Δv (pixels)', fontsize=13, fontweight='bold')
# plt.colorbar(im2, ax=axes[1])

# plt.tight_layout()
# plt.show()

# # Error heatmap
# fig, ax = plt.subplots(figsize=(12, 9))

# error_masked = diff.copy()
# error_masked[~valid] = 0

# im = ax.imshow(error_masked, cmap='RdBu_r', vmin=-50, vmax=50)
# ax.set_title(f'Reconstruction Error\nRMS={RMS:.2f}, MAE={MAE:.2f}', 
#              fontsize=14, fontweight='bold')
# ax.set_xlabel('X (pixels)')
# ax.set_ylabel('Y (pixels)')
# cbar = plt.colorbar(im, ax=ax, label='Error (gray levels)')

# plt.tight_layout()
# plt.show()

# # Error histogram
# fig, ax = plt.subplots(figsize=(12, 7))

# ax.hist(diff_valid, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
# ax.axvline(RMS, color='red', linestyle='--', linewidth=2.5, label=f'RMS: {RMS:.2f}')
# ax.axvline(MAE, color='green', linestyle='--', linewidth=2.5, label=f'MAE: {MAE:.2f}')
# ax.axvline(np.median(diff_valid), color='orange', linestyle='--', linewidth=2.5,
#           label=f'Median: {np.median(diff_valid):.2f}')

# ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12, fontweight='bold')
# ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
# ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
# ax.legend(fontsize=11)
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# print("\n" + "="*100)
# print(" STRAIN-BASED INVERSE RECONSTRUCTION COMPLETE!")
# print("="*100)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import re
import xml.etree.ElementTree as ET

# ==============================
# PATHS
# ==============================
DATA_PATH = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt"
REF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_0.tif"
DEF_CAM0 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_0.tif"
REF_CAM1 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0110_1.tif"
DEF_CAM1 = r"C:\Users\harsh\Downloads\DICe_examples\DICe_examples\stereo_d_sample\images\GT4-0129_1.tif"
BEST_PLANE = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\best_fit_plane_out.dat"
CAL_XML = r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\cal.xml"

print("=" * 100)
print(" STRAIN-BASED INVERSE RECONSTRUCTION - STEREO VALIDATION ")
print("=" * 100)

# ==============================
# LOAD DICe DATA
# ==============================
df = pd.read_csv(DATA_PATH, sep=r'\s+|,', engine="python")

X = df["MODEL_COORDINATES_X"].astype(float).to_numpy()
Y = df["MODEL_COORDINATES_Y"].astype(float).to_numpy()
Z = df["MODEL_COORDINATES_Z"].astype(float).to_numpy()

exx = df["VSG_STRAIN_XX"].astype(float).to_numpy()
eyy = df["VSG_STRAIN_YY"].astype(float).to_numpy()
exy = df["VSG_STRAIN_XY"].astype(float).to_numpy()

u_meas = df["MODEL_DISPLACEMENT_X"].astype(float).to_numpy()
v_meas = df["MODEL_DISPLACEMENT_Y"].astype(float).to_numpy()
w_meas = df["MODEL_DISPLACEMENT_Z"].astype(float).to_numpy()

u_img = df["COORDINATE_X"].astype(float).to_numpy()
v_img = df["COORDINATE_Y"].astype(float).to_numpy()

# Load images for both cameras
reference_cam0 = cv2.imread(REF_CAM0, cv2.IMREAD_GRAYSCALE)
actual_def_cam0 = cv2.imread(DEF_CAM0, cv2.IMREAD_GRAYSCALE)
reference_cam1 = cv2.imread(REF_CAM1, cv2.IMREAD_GRAYSCALE)
actual_def_cam1 = cv2.imread(DEF_CAM1, cv2.IMREAD_GRAYSCALE)

H, W = reference_cam0.shape
N = len(X)

print(f"\nLoaded {N} measurement points")
print(f"Image size: {W} × {H} pixels")
print(f"\nStrain statistics:")
print(f"  εxx: [{exx.min():.6f}, {exx.max():.6f}], mean={exx.mean():.6f}")
print(f"  εyy: [{eyy.min():.6f}, {eyy.max():.6f}], mean={eyy.mean():.6f}")
print(f"  εxy: [{exy.min():.6f}, {exy.max():.6f}], mean={exy.mean():.6f}")

# ==============================
# CREATE PLANE GRID
# ==============================
sx, sy = 0.12, 0.12
x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()

nx = int((x_max - x_min) / sx) + 1
ny = int((y_max - y_min) / sy) + 1

xg = np.linspace(x_min, x_max, nx)
yg = np.linspace(y_min, y_max, ny)
XX, YY = np.meshgrid(xg, yg)

dx = xg[1] - xg[0]
dy = yg[1] - yg[0]
pts_spec = np.vstack([X, Y]).T

print(f"\nPlane grid: {nx} × {ny} = {nx*ny} points")
print(f"Spacing: dx={dx:.4f} mm, dy={dy:.4f} mm")

# ==============================
# INTERPOLATE Z AND STRAINS
# ==============================
print("\nInterpolating geometry and strains...")

Z_plane = griddata(pts_spec, Z, (XX, YY), method='linear', fill_value=np.nan)
Z_plane = np.nan_to_num(Z_plane)

exx_g = griddata(pts_spec, exx, (XX, YY), method='cubic', fill_value=0)
eyy_g = griddata(pts_spec, eyy, (XX, YY), method='cubic', fill_value=0)
exy_g = griddata(pts_spec, exy, (XX, YY), method='cubic', fill_value=0)

print("✓ Interpolation complete")

# ==============================
# COMPUTE STRAIN DERIVATIVES
# ==============================
print("\nComputing strain compatibility RHS...")

deps_xx_dx = np.gradient(exx_g, dx, axis=1)
deps_xy_dy = np.gradient(exy_g, dy, axis=0)
rhs_u = deps_xx_dx + deps_xy_dy

deps_xy_dx = np.gradient(exy_g, dx, axis=1)
deps_yy_dy = np.gradient(eyy_g, dy, axis=0)
rhs_v = deps_xy_dx + deps_yy_dy

rhs_w = np.zeros_like(rhs_u)

print(f"RHS computed:")
print(f"  RHS_u: [{rhs_u.min():.6f}, {rhs_u.max():.6f}]")
print(f"  RHS_v: [{rhs_v.min():.6f}, {rhs_v.max():.6f}]")

# ==============================
# BUILD POISSON SYSTEM
# ==============================
print("\nBuilding Poisson system...")

n_nodes = nx * ny

def build_strain_poisson(nx, ny, dx, dy, rhs_field, bc_mask, bc_vals):
    n = nx * ny
    A = lil_matrix((n, n))
    b = np.zeros(n)
    
    coeff_x = 1.0 / (dx * dx)
    coeff_y = 1.0 / (dy * dy)
    coeff_center = -2 * (coeff_x + coeff_y)
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            
            if bc_mask[j, i]:
                A[idx, idx] = 1.0
                b[idx] = bc_vals[j, i]
            else:
                A[idx, idx] = coeff_center
                
                if i > 0:
                    A[idx, idx - 1] = coeff_x
                else:
                    A[idx, idx] -= coeff_x
                    
                if i < nx - 1:
                    A[idx, idx + 1] = coeff_x
                else:
                    A[idx, idx] -= coeff_x
                    
                if j > 0:
                    A[idx, idx - nx] = coeff_y
                else:
                    A[idx, idx] -= coeff_y
                    
                if j < ny - 1:
                    A[idx, idx + nx] = coeff_y
                else:
                    A[idx, idx] -= coeff_y
                
                b[idx] = rhs_field[j, i]
    
    return A.tocsr(), b

bc_mask = np.zeros((ny, nx), dtype=bool)
bc_u = np.zeros((ny, nx))
bc_v = np.zeros((ny, nx))
bc_w = np.zeros((ny, nx))

for k in range(N):
    i = np.argmin(np.abs(xg - X[k]))
    j = np.argmin(np.abs(yg - Y[k]))
    bc_mask[j, i] = True
    bc_u[j, i] = u_meas[k]
    bc_v[j, i] = v_meas[k]
    bc_w[j, i] = w_meas[k]

print(f" BCs applied at {bc_mask.sum()} nodes")

A_u, b_u = build_strain_poisson(nx, ny, dx, dy, rhs_u, bc_mask, bc_u)
A_v, b_v = build_strain_poisson(nx, ny, dx, dy, rhs_v, bc_mask, bc_v)
A_w, b_w = build_strain_poisson(nx, ny, dx, dy, rhs_w, bc_mask, bc_w)

print(f" System: {A_u.shape[0]} × {A_u.shape[1]}, {A_u.nnz} non-zeros")

# ==============================
# SOLVE FOR DISPLACEMENTS
# ==============================
print("\nSolving Poisson systems...")

u_vec = spsolve(A_u, b_u)
v_vec = spsolve(A_v, b_v)
w_vec = spsolve(A_w, b_w)

u_rec = u_vec.reshape(ny, nx)
v_rec = v_vec.reshape(ny, nx)
w_rec = w_vec.reshape(ny, nx)

print(" Solutions obtained:")
print(f"  u: [{u_rec.min():.6f}, {u_rec.max():.6f}] mm")
print(f"  v: [{v_rec.min():.6f}, {v_rec.max():.6f}] mm")
print(f"  w: [{w_rec.min():.6f}, {w_rec.max():.6f}] mm")

# ==============================
# FORM 3D SURFACES
# ==============================
print("\nConstructing 3D surfaces...")

P0_plane = np.stack([XX, YY, Z_plane], axis=-1).reshape(-1, 3)
P1_plane = np.stack([XX + u_rec, YY + v_rec, Z_plane + w_rec], axis=-1).reshape(-1, 3)

P0_plane = np.nan_to_num(P0_plane)
P1_plane = np.nan_to_num(P1_plane)

print(f"✓ Point clouds: {P0_plane.shape[0]} points")

# ==============================
# CAMERA CALIBRATION FUNCTIONS
# ==============================
def parse_best_plane_transform(path):
    txt = open(path, 'r').read()
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
    vals = list(map(float, nums))
    if len(vals) < 12:
        raise RuntimeError("Transform not found")
    block = vals[-12:]
    R = np.array([[block[0], block[1], block[2]],
                  [block[4], block[5], block[6]],
                  [block[8], block[9], block[10]]], dtype=float)
    t = np.array([block[3], block[7], block[11]], dtype=float)
    return R, t

def parse_cam_from_calxml(path, cam_id):
    """Parse camera intrinsics from cal.xml"""
    tree = ET.parse(path)
    root = tree.getroot()
    cam = None
    for pl in root.iter('ParameterList'):
        if pl.get('name', '') == f'CAMERA {cam_id}':
            cam = pl
            break
    if cam is None:
        raise RuntimeError(f"CAMERA {cam_id} not found")
    
    def getp(name, default=0.0):
        for p in cam.iter('Parameter'):
            if p.get('name') == name:
                try:
                    return float(p.get('value'))
                except:
                    return default
        return default
    
    return dict(
        fx=getp('FX'), fy=getp('FY'), cx=getp('CX'), cy=getp('CY'),
        k1=getp('K1'), k2=getp('K2'), k3=getp('K3'),
        p1=getp('P1', 0.0), p2=getp('P2', 0.0)
    )

def get_stereo_transform(path):
    """Get rotation and translation from cam0 to cam1"""
    tree = ET.parse(path)
    root = tree.getroot()
    
    # Find transform parameters
    tx = ty = tz = 0.0
    rx = ry = rz = 0.0
    
    for pl in root.iter('ParameterList'):
        if 'CAMERA_SYSTEM' in pl.get('name', ''):
            for p in pl.iter('Parameter'):
                name = p.get('name', '')
                try:
                    val = float(p.get('value', 0))
                    if name == 'TX': tx = val
                    elif name == 'TY': ty = val
                    elif name == 'TZ': tz = val
                    elif name == 'RX': rx = val
                    elif name == 'RY': ry = val
                    elif name == 'RZ': rz = val
                except:
                    pass
    
    # Build rotation matrix from Euler angles
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    t = np.array([tx, ty, tz])
    
    return R, t

def distort_opencv(xn, yn, k1, k2, k3, p1, p2):
    r2 = xn*xn + yn*yn
    radial = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
    x_t = 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
    y_t = p1*(r2 + 2*yn*yn) + 2*p2*xn*yn
    xd = xn*radial + x_t
    yd = yn*radial + y_t
    return xd, yd

def project_points(P_cam, K):
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    Z = np.where(Z <= 1e-6, 1e-6, Z)
    xn, yn = X/Z, Y/Z
    xd, yd = distort_opencv(xn, yn, K['k1'], K['k2'], K['k3'], K['p1'], K['p2'])
    u = K['fx'] * xd + K['cx']
    v = K['fy'] * yd + K['cy']
    return np.vstack([u, v]).T

# ==============================
# LOAD CALIBRATION
# ==============================
print("\nLoading calibration...")

R_plane_cam, t_plane_cam = parse_best_plane_transform(BEST_PLANE)
K0 = parse_cam_from_calxml(CAL_XML, 0)
K1 = parse_cam_from_calxml(CAL_XML, 1)
R_01, t_01 = get_stereo_transform(CAL_XML)

print(f"✓ Camera 0: fx={K0['fx']:.2f}, fy={K0['fy']:.2f}")
print(f"✓ Camera 1: fx={K1['fx']:.2f}, fy={K1['fy']:.2f}")

# ==============================
# PROCESS BOTH CAMERAS
# ==============================
def process_camera(cam_id, P0_plane, P1_plane, R_plane_cam, t_plane_cam, K, 
                   R_stereo, t_stereo, ref_img, def_img, u_img, v_img):
    """Process reconstruction for one camera"""
    
    print(f"\n{'='*100}")
    print(f"PROCESSING CAMERA {cam_id}")
    print(f"{'='*100}")
    
    # Transform to camera frame
    if cam_id == 0:
        # Plane → Camera 0
        P0_cam = (R_plane_cam.T @ (P0_plane - t_plane_cam).T).T
        P1_cam = (R_plane_cam.T @ (P1_plane - t_plane_cam).T).T
    else:
        # Plane → Camera 0 → Camera 1
        P0_cam0 = (R_plane_cam.T @ (P0_plane - t_plane_cam).T).T
        P1_cam0 = (R_plane_cam.T @ (P1_plane - t_plane_cam).T).T
        P0_cam = (R_stereo @ P0_cam0.T + t_stereo[:, None]).T
        P1_cam = (R_stereo @ P1_cam0.T + t_stereo[:, None]).T
    
    # Project
    uv0 = project_points(P0_cam, K)
    uv1 = project_points(P1_cam, K)
    
    du_pix = (uv1[:, 0] - uv0[:, 0]).reshape(ny, nx)
    dv_pix = (uv1[:, 1] - uv0[:, 1]).reshape(ny, nx)
    
    print(f"Flow: du=[{du_pix.min():.2f}, {du_pix.max():.2f}] px")
    print(f"      dv=[{dv_pix.min():.2f}, {dv_pix.max():.2f}] px")
    
    # Interpolate to image
    x_img = np.arange(W, dtype=np.float32)
    y_img = np.arange(H, dtype=np.float32)
    XX_img, YY_img = np.meshgrid(x_img, y_img)
    
    in_bounds = (uv0[:, 0] >= 0) & (uv0[:, 0] < W) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H)
    uv0_in = uv0[in_bounds]
    du_in = du_pix.ravel()[in_bounds]
    dv_in = dv_pix.ravel()[in_bounds]
    
    du_im = griddata(uv0_in, du_in, (XX_img, YY_img), method='cubic', fill_value=0.0)
    dv_im = griddata(uv0_in, dv_in, (XX_img, YY_img), method='cubic', fill_value=0.0)
    
    # Valid mask
    valid = np.zeros((H, W), dtype=bool)
    uu = np.clip(u_img.astype(int), 0, W-1)
    vv = np.clip(v_img.astype(int), 0, H-1)
    patch = 15
    for u0, v0 in zip(uu, vv):
        valid[max(0, v0-patch):min(H, v0+patch+1),
              max(0, u0-patch):min(W, u0+patch+1)] = True
    
    # Reconstruct
    map_x = (XX_img - du_im).astype(np.float32)
    map_y = (YY_img - dv_im).astype(np.float32)
    
    recon = cv2.remap(ref_img.astype(np.float32), map_x, map_y,
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
    
    # Metrics
    diff = def_img.astype(float) - recon_u8.astype(float)
    diff_valid = diff[valid]
    
    RMS = float(np.sqrt(np.mean(diff_valid**2)))
    MAE = float(np.mean(np.abs(diff_valid)))
    SSIM = float(ssim(def_img, recon_u8, data_range=255))
    PSNR = float(psnr(def_img, recon_u8, data_range=255))
    
    print(f"\nMetrics:")
    print(f"  RMS:  {RMS:.2f} gray levels")
    print(f"  MAE:  {MAE:.2f} gray levels")
    print(f"  SSIM: {SSIM:.4f}")
    print(f"  PSNR: {PSNR:.2f} dB")
    
    return {
        'recon': recon_u8,
        'diff': diff,
        'valid': valid,
        'du_im': du_im,
        'dv_im': dv_im,
        'RMS': RMS,
        'MAE': MAE,
        'SSIM': SSIM,
        'PSNR': PSNR
    }

# Process both cameras
results_cam0 = process_camera(0, P0_plane, P1_plane, R_plane_cam, t_plane_cam, K0,
                              None, None, reference_cam0, actual_def_cam0, u_img, v_img)

results_cam1 = process_camera(1, P0_plane, P1_plane, R_plane_cam, t_plane_cam, K1,
                              R_01, t_01, reference_cam1, actual_def_cam1, u_img, v_img)

# # ==============================
# # VISUALIZATIONS
# # ==============================
# print("\n" + "="*100)
# print("GENERATING VISUALIZATIONS")
# print("="*100)

# # Combined comparison
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # Camera 0
# axes[0, 0].imshow(reference_cam0, cmap='gray')
# axes[0, 0].set_title('Cam0: Reference', fontsize=13, fontweight='bold')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(actual_def_cam0, cmap='gray')
# axes[0, 1].set_title('Cam0: Actual Deformed', fontsize=13, fontweight='bold')
# axes[0, 1].axis('off')

# axes[0, 2].imshow(results_cam0['recon'], cmap='gray')
# axes[0, 2].set_title(f'Cam0: Reconstructed\nRMS={results_cam0["RMS"]:.2f}, SSIM={results_cam0["SSIM"]:.4f}',
#                      fontsize=13, fontweight='bold')
# axes[0, 2].axis('off')

# # Camera 1
# axes[1, 0].imshow(reference_cam1, cmap='gray')
# axes[1, 0].set_title('Cam1: Reference', fontsize=13, fontweight='bold')
# axes[1, 0].axis('off')

# axes[1, 1].imshow(actual_def_cam1, cmap='gray')
# axes[1, 1].set_title('Cam1: Actual Deformed', fontsize=13, fontweight='bold')
# axes[1, 1].axis('off')

# axes[1, 2].imshow(results_cam1['recon'], cmap='gray')
# axes[1, 2].set_title(f'Cam1: Reconstructed\nRMS={results_cam1["RMS"]:.2f}, SSIM={results_cam1["SSIM"]:.4f}',
#                      fontsize=13, fontweight='bold')
# axes[1, 2].axis('off')

# plt.tight_layout()
# plt.show()

# # Error comparison
# fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# err0 = results_cam0['diff'].copy()
# err0[~results_cam0['valid']] = 0
# im0 = axes[0, 0].imshow(err0, cmap='RdBu_r', vmin=-50, vmax=50)
# axes[0, 0].set_title(f'Cam0 Error (RMS={results_cam0["RMS"]:.2f})', fontsize=13, fontweight='bold')
# plt.colorbar(im0, ax=axes[0, 0])

# axes[0, 1].hist(results_cam0['diff'][results_cam0['valid']], bins=100, 
#                 alpha=0.7, color='steelblue', edgecolor='black')
# axes[0, 1].axvline(results_cam0['RMS'], color='red', linestyle='--', linewidth=2)
# axes[0, 1].set_title('Cam0 Error Distribution', fontsize=13, fontweight='bold')
# axes[0, 1].set_xlabel('Error (gray levels)')
# axes[0, 1].grid(True, alpha=0.3)

# err1 = results_cam1['diff'].copy()
# err1[~results_cam1['valid']] = 0
# im1 = axes[1, 0].imshow(err1, cmap='RdBu_r', vmin=-50, vmax=50)
# axes[1, 0].set_title(f'Cam1 Error (RMS={results_cam1["RMS"]:.2f})', fontsize=13, fontweight='bold')
# plt.colorbar(im1, ax=axes[1, 0])

# axes[1, 1].hist(results_cam1['diff'][results_cam1['valid']], bins=100, 
#                 alpha=0.7, color='coral', edgecolor='black')
# axes[1, 1].axvline(results_cam1['RMS'], color='red', linestyle='--', linewidth=2)
# axes[1, 1].set_title('Cam1 Error Distribution', fontsize=13, fontweight='bold')
# axes[1, 1].set_xlabel('Error (gray levels)')
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # Summary
# print("\n" + "="*100)
# print("STEREO VALIDATION SUMMARY")
# print("="*100)
# print(f"\nCamera 0:")
# print(f"  RMS:  {results_cam0['RMS']:.2f} | SSIM: {results_cam0['SSIM']:.4f}")
# print(f"\nCamera 1:")
# print(f"  RMS:  {results_cam1['RMS']:.2f} | SSIM: {results_cam1['SSIM']:.4f}")
# print(f"\nAverage:")
# print(f"  RMS:  {(results_cam0['RMS'] + results_cam1['RMS'])/2:.2f}")
# print(f"  SSIM: {(results_cam0['SSIM'] + results_cam1['SSIM'])/2:.4f}")

# print("\n" + "="*100)
# print("STEREO STRAIN-BASED RECONSTRUCTION COMPLETE!")
# print("="*100)

# ==============================
# VISUALIZATIONS (SEPARATE PLOTS)
# ==============================
print("\n" + "="*100)
print("GENERATING VISUALIZATIONS")
print("="*100)

# ===== FIGURE 1: Camera 0 - Actual + Reconstructed =====
print("  Generating Figure 1: Camera 0 - Actual + Reconstructed...")

fig1, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(actual_def_cam0, cmap='gray')
axes[0].set_title('Actual Deformed', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(results_cam0['recon'], cmap='gray')
axes[1].set_title('Reconstructed (Strain-Based)', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle(f'Camera 0: Actual vs Reconstructed | SSIM: {results_cam0["SSIM"]:.4f} | PSNR: {results_cam0["PSNR"]:.2f} dB',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('strain_01_camera0_actual_vs_reconstructed.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_01_camera0_actual_vs_reconstructed.png")

# ===== FIGURE 2: Camera 1 - Actual + Reconstructed =====
print("  Generating Figure 2: Camera 1 - Actual + Reconstructed...")

fig2, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(actual_def_cam1, cmap='gray')
axes[0].set_title('Actual Deformed', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(results_cam1['recon'], cmap='gray')
axes[1].set_title('Reconstructed (Strain-Based)', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle(f'Camera 1: Actual vs Reconstructed | SSIM: {results_cam1["SSIM"]:.4f} | PSNR: {results_cam1["PSNR"]:.2f} dB',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('strain_02_camera1_actual_vs_reconstructed.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_02_camera1_actual_vs_reconstructed.png")

# ===== FIGURE 3: Error Heatmap - Camera 0 =====
print("  Generating Figure 3: Error Heatmap Camera 0...")

fig3, ax = plt.subplots(figsize=(12, 10))

err0 = results_cam0['diff'].copy()
err0[~results_cam0['valid']] = 0
im0 = ax.imshow(err0, cmap='RdBu_r', vmin=-50, vmax=50)
ax.set_title(f'Camera 0: Reconstruction Error Heatmap\nRMS={results_cam0["RMS"]:.2f} | MAE={results_cam0["MAE"]:.2f} | SSIM={results_cam0["SSIM"]:.4f}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
plt.colorbar(im0, ax=ax, label='Intensity Error (gray levels)')

plt.tight_layout()
plt.savefig('strain_03_error_heatmap_cam0.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_03_error_heatmap_cam0.png")

# ===== FIGURE 4: Error Heatmap - Camera 1 =====
print("  Generating Figure 4: Error Heatmap Camera 1...")

fig4, ax = plt.subplots(figsize=(12, 10))

err1 = results_cam1['diff'].copy()
err1[~results_cam1['valid']] = 0
im1 = ax.imshow(err1, cmap='RdBu_r', vmin=-50, vmax=50)
ax.set_title(f'Camera 1: Reconstruction Error Heatmap\nRMS={results_cam1["RMS"]:.2f} | MAE={results_cam1["MAE"]:.2f} | SSIM={results_cam1["SSIM"]:.4f}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
plt.colorbar(im1, ax=ax, label='Intensity Error (gray levels)')

plt.tight_layout()
plt.savefig('strain_04_error_heatmap_cam1.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_04_error_heatmap_cam1.png")

# ===== FIGURE 5: Error Histogram - Camera 0 =====
print("  Generating Figure 5: Error Histogram Camera 0...")

fig5, ax = plt.subplots(figsize=(12, 8))

valid_errors_cam0 = results_cam0['diff'][results_cam0['valid']]

ax.hist(valid_errors_cam0, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(results_cam0['RMS'], color='red', linestyle='--', linewidth=2.5,
          label=f"RMS: {results_cam0['RMS']:.3f}")
ax.axvline(results_cam0['MAE'], color='green', linestyle='--', linewidth=2.5,
          label=f"MAE: {results_cam0['MAE']:.3f}")
ax.axvline(np.median(valid_errors_cam0), color='orange', linestyle='--', linewidth=2.5,
          label=f"Median: {np.median(valid_errors_cam0):.3f}")

ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Camera 0: Error Distribution (Valid Region Only)', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

stats_text = f"""Statistics (Valid Region):
Count: {len(valid_errors_cam0)}
Min: {valid_errors_cam0.min():.3f}
Max: {valid_errors_cam0.max():.3f}
Mean: {valid_errors_cam0.mean():.3f}
Std: {valid_errors_cam0.std():.3f}
95th %ile: {np.percentile(valid_errors_cam0, 95):.3f}"""

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('strain_05_error_histogram_cam0.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_05_error_histogram_cam0.png")

# ===== FIGURE 6: Error Histogram - Camera 1 =====
print("  Generating Figure 6: Error Histogram Camera 1...")

fig6, ax = plt.subplots(figsize=(12, 8))

valid_errors_cam1 = results_cam1['diff'][results_cam1['valid']]

ax.hist(valid_errors_cam1, bins=100, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(results_cam1['RMS'], color='red', linestyle='--', linewidth=2.5,
          label=f"RMS: {results_cam1['RMS']:.3f}")
ax.axvline(results_cam1['MAE'], color='green', linestyle='--', linewidth=2.5,
          label=f"MAE: {results_cam1['MAE']:.3f}")
ax.axvline(np.median(valid_errors_cam1), color='orange', linestyle='--', linewidth=2.5,
          label=f"Median: {np.median(valid_errors_cam1):.3f}")

ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Camera 1: Error Distribution (Valid Region Only)', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

stats_text = f"""Statistics (Valid Region):
Count: {len(valid_errors_cam1)}
Min: {valid_errors_cam1.min():.3f}
Max: {valid_errors_cam1.max():.3f}
Mean: {valid_errors_cam1.mean():.3f}
Std: {valid_errors_cam1.std():.3f}
95th %ile: {np.percentile(valid_errors_cam1, 95):.3f}"""

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('strain_06_error_histogram_cam1.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_06_error_histogram_cam1.png")

# ===== FIGURE 7: Error Comparison (Both Cameras) =====
print("  Generating Figure 7: Error Comparison...")

fig7, ax = plt.subplots(figsize=(14, 8))

valid_errors_cam0 = results_cam0['diff'][results_cam0['valid']]
valid_errors_cam1 = results_cam1['diff'][results_cam1['valid']]

ax.hist(valid_errors_cam0, bins=80, alpha=0.6, label='Camera 0', color='steelblue', edgecolor='black')
ax.hist(valid_errors_cam1, bins=80, alpha=0.6, label='Camera 1', color='coral', edgecolor='black')

ax.axvline(results_cam0['RMS'], color='steelblue', linestyle='--', linewidth=2,
          label=f"Cam0 RMS: {results_cam0['RMS']:.3f}")
ax.axvline(results_cam1['RMS'], color='coral', linestyle='--', linewidth=2,
          label=f"Cam1 RMS: {results_cam1['RMS']:.3f}")

ax.set_xlabel('Reconstruction Error (gray levels)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Reconstruction Error Comparison: Camera 0 vs Camera 1', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strain_07_error_comparison_cam0_vs_cam1.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: strain_07_error_comparison_cam0_vs_cam1.png")

# Summary
print("\n" + "="*100)
print("STEREO VALIDATION SUMMARY")
print("="*100)
print(f"\nCamera 0:")
print(f"  RMS:  {results_cam0['RMS']:.2f} | SSIM: {results_cam0['SSIM']:.4f}")
print(f"\nCamera 1:")
print(f"  RMS:  {results_cam1['RMS']:.2f} | SSIM: {results_cam1['SSIM']:.4f}")
print(f"\nAverage:")
print(f"  RMS:  {(results_cam0['RMS'] + results_cam1['RMS'])/2:.2f}")
print(f"  SSIM: {(results_cam0['SSIM'] + results_cam1['SSIM'])/2:.4f}")

print("\n" + "="*100)
print("GENERATED FILES:")
print("="*100)
print("  strain_01_camera0_actual_vs_reconstructed.png")
print("  strain_02_camera1_actual_vs_reconstructed.png")
print("  strain_03_error_heatmap_cam0.png")
print("  strain_04_error_heatmap_cam1.png")
print("  strain_05_error_histogram_cam0.png")
print("  strain_06_error_histogram_cam1.png")
print("  strain_07_error_comparison_cam0_vs_cam1.png")

print("\n" + "="*100)
print("STEREO STRAIN-BASED RECONSTRUCTION COMPLETE!")
print("="*100)
