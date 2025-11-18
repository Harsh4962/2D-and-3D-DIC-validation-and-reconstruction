# import pandas as pd, numpy as np
# from scipy.spatial import cKDTree

# # ===== Paths to files =====
# p_nc = r"C:\Users\harsh\OneDrive\Desktop\NCorr_final\ncorr_inc_011.csv"
# p_di = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"

# # ===== Load NCorr results (x,y,u,v,...) =====
# nc = pd.read_csv(p_nc)

# # ===== Load DICe results =====
# di = pd.read_csv(p_di, sep=",", comment="#")

# # ===== Interpolate DICe results onto NCorr grid =====
# tree = cKDTree(np.c_[di.COORDINATE_X.values, di.COORDINATE_Y.values])
# dist, idx = tree.query(np.c_[nc["x"].values, nc["y"].values], k=1)

# u_dice_interp = di.DISPLACEMENT_X.values[idx]
# v_dice_interp = di.DISPLACEMENT_Y.values[idx]

# # ===== Differences =====
# u_diff = nc["u"].values - u_dice_interp
# v_diff = nc["v"].values - v_dice_interp

# # ===== Statistics function =====
# def stats(d):
#     return dict(
#         mean=float(np.nanmean(d)),   # Average signed difference
#         mae=float(np.nanmean(np.abs(d))),  # Mean Absolute Error
#         rmse=float(np.sqrt(np.nanmean(d**2))),  # Root Mean Squared Error
#         p_within_0p05=float(np.mean(np.abs(d) <= 0.05))*100,  # % within ±0.05 px
#         p_within_0p10=float(np.mean(np.abs(d) <= 0.10))*100,  # % within ±0.10 px
#     )

# u_stats = stats(u_diff)
# v_stats = stats(v_diff)

# # ===== Detailed Report =====
# print("\n===== Cross-Validation Report: NCorr vs. DICe =====\n")

# print("➡ Displacement U (x-direction):")
# print(f"   • Mean difference: {u_stats['mean']:.5f} px "
#       "(bias: NCorr - DICe, ideally ~0)")
# print(f"   • Mean Absolute Error (MAE): {u_stats['mae']:.5f} px "
#       "(average unsigned difference)")
# print(f"   • Root Mean Squared Error (RMSE): {u_stats['rmse']:.5f} px "
#       "(sensitive to larger deviations)")
# print(f"   • % of points within ±0.05 px: {u_stats['p_within_0p05']:.2f}%")
# print(f"   • % of points within ±0.10 px: {u_stats['p_within_0p10']:.2f}%\n")

# print("➡ Displacement V (y-direction):")
# print(f"   • Mean difference: {v_stats['mean']:.5f} px")
# print(f"   • Mean Absolute Error (MAE): {v_stats['mae']:.5f} px")
# print(f"   • Root Mean Squared Error (RMSE): {v_stats['rmse']:.5f} px")
# print(f"   • % of points within ±0.05 px: {v_stats['p_within_0p05']:.2f}%")
# print(f"   • % of points within ±0.10 px: {v_stats['p_within_0p10']:.2f}%\n")

# print("=============================================")
# print("Interpretation:")
# print("• The mean differences are nearly zero, showing no systematic bias.")
# print("• MAE and RMSE are much less than 1 px, confirming subpixel agreement.")
# print("• Over 97–99% of points lie within ±0.05–0.10 px, which demonstrates "
#       "that NCorr and DICe give consistent displacement results.")
# print("=============================================\n")


# import pandas as pd
# import numpy as np
# from scipy.spatial import cKDTree

# # ===== Paths to files =====
# p_zn = r"C:\Users\harsh\Downloads\ncorr_2D_matlab-master_ZNSSD\ncorr_2D_matlab-master\ZNSSDexports_clean_final\ncorr_inc_006.csv"
# p_sd = r"C:\Users\harsh\Downloads\ncorr_2D_matlab-master-clean1\ncorr_2D_matlab-master\SSDexports_clean_final\ncorr_inc_006.csv"

# # ===== Load NCorr ZNSSD results =====
# znssd = pd.read_csv(p_zn)

# # ===== Load NCorr NCC results =====
# ssd = pd.read_csv(p_sd)

# # ===== Interpolate NCC results onto ZNSSD grid =====
# tree = cKDTree(np.c_[ssd["x"].values, ssd["y"].values])
# dist, idx = tree.query(np.c_[znssd["x"].values, znssd["y"].values], k=1)

# u_ssd_interp = ssd["u"].values[idx]
# v_ssd_interp = ssd["v"].values[idx]

# # ===== Differences =====
# u_diff = znssd["u"].values - u_ssd_interp
# v_diff = znssd["v"].values - v_ssd_interp

# # ===== Statistics function =====
# def stats(d):
#     return dict(
#         mean=float(np.nanmean(d)),        # Average signed difference
#         mae=float(np.nanmean(np.abs(d))), # Mean Absolute Error
#         rmse=float(np.sqrt(np.nanmean(d**2))), # Root Mean Squared Error
#         p_within_0p05=float(np.mean(np.abs(d) <= 0.05))*100, # % within ±0.05 px
#         p_within_0p10=float(np.mean(np.abs(d) <= 0.10))*100, # % within ±0.10 px
#     )

# u_stats = stats(u_diff)
# v_stats = stats(v_diff)

# # ===== Detailed Report =====
# print("\n===== Cross-Validation Report: NCorr ZNSSD vs NCorr SSD =====\n")

# print("➡ Displacement U (x-direction):")
# print(f"   • Mean difference: {u_stats['mean']:.5f} px "
#       "(bias: ZNSSD - SSD, ideally ~0)")
# print(f"   • Mean Absolute Error (MAE): {u_stats['mae']:.5f} px")
# print(f"   • Root Mean Squared Error (RMSE): {u_stats['rmse']:.5f} px")
# print(f"   • % of points within ±0.05 px: {u_stats['p_within_0p05']:.2f}%")
# print(f"   • % of points within ±0.10 px: {u_stats['p_within_0p10']:.2f}%\n")

# print("➡ Displacement V (y-direction):")
# print(f"   • Mean difference: {v_stats['mean']:.5f} px")
# print(f"   • Mean Absolute Error (MAE): {v_stats['mae']:.5f} px")
# print(f"   • Root Mean Squared Error (RMSE): {v_stats['rmse']:.5f} px")
# print(f"   • % of points within ±0.05 px: {v_stats['p_within_0p05']:.2f}%")
# print(f"   • % of points within ±0.10 px: {v_stats['p_within_0p10']:.2f}%\n")


import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ===== Paths to files =====
p_zn = r"C:\Users\harsh\Downloads\ncorr_2D_matlab-master_ZNSSD\ncorr_2D_matlab-master\ZNSSDexports_clean_final\ncorr_inc_006.csv"
p_nc = r"C:\Users\harsh\Downloads\ncorr_2D_matlab-master_NCC@\ncorr_2D_matlab-master\NCCFinalOuts\ncorr_inc_006.csv"

# ===== Load NCorr ZNSSD results =====
znssd = pd.read_csv(p_zn)

# ===== Load NCorr NCC results =====
ncc = pd.read_csv(p_nc)

# ===== Interpolate NCC results onto ZNSSD grid =====
tree = cKDTree(np.c_[ncc["x"].values, ncc["y"].values])
dist, idx = tree.query(np.c_[znssd["x"].values, znssd["y"].values], k=1)

u_ncc_interp = ncc["u"].values[idx]
v_ncc_interp = ncc["v"].values[idx]

# ===== Differences =====
u_diff = znssd["u"].values - u_ncc_interp
v_diff = znssd["v"].values - v_ncc_interp

# ===== Statistics function =====
def stats(d):
    return dict(
        mean=float(np.nanmean(d)),        # Average signed difference
        mae=float(np.nanmean(np.abs(d))), # Mean Absolute Error
        rmse=float(np.sqrt(np.nanmean(d**2))), # Root Mean Squared Error
        p_within_0p05=float(np.mean(np.abs(d) <= 0.05))*100, # % within ±0.05 px
        p_within_0p10=float(np.mean(np.abs(d) <= 0.10))*100, # % within ±0.10 px
    )

u_stats = stats(u_diff)
v_stats = stats(v_diff)

# ===== Detailed Report =====
print("\n===== Cross-Validation Report: NCorr ZNSSD vs NCorr NCC =====\n")

print("➡ Displacement U (x-direction):")
print(f"   • Mean difference: {u_stats['mean']:.5f} px "
      "(bias: ZNSSD - NCC, ideally ~0)")
print(f"   • Mean Absolute Error (MAE): {u_stats['mae']:.5f} px")
print(f"   • Root Mean Squared Error (RMSE): {u_stats['rmse']:.5f} px")
print(f"   • % of points within ±0.05 px: {u_stats['p_within_0p05']:.2f}%")
print(f"   • % of points within ±0.10 px: {u_stats['p_within_0p10']:.2f}%\n")

print("➡ Displacement V (y-direction):")
print(f"   • Mean difference: {v_stats['mean']:.5f} px")
print(f"   • Mean Absolute Error (MAE): {v_stats['mae']:.5f} px")
print(f"   • Root Mean Squared Error (RMSE): {v_stats['rmse']:.5f} px")
print(f"   • % of points within ±0.05 px: {v_stats['p_within_0p05']:.2f}%")
print(f"   • % of points within ±0.10 px: {v_stats['p_within_0p10']:.2f}%\n")
