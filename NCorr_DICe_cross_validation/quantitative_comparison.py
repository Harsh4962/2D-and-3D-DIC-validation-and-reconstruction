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
