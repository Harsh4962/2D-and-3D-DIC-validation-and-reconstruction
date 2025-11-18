import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load NCorr ZNSSD results ===
znssd_file = r"C:\Users\harsh\OneDrive\Desktop\NCorr_final\ncorr_inc_011.csv"
znssd_df = pd.read_csv(znssd_file)

# === Load NCorr NCC results ===
ncc_file = r"C:\Users\harsh\Downloads\ncorr_2D_matlab-master (1)\ncorr_2D_matlab-master\NCCexports\ncorr_inc_011.csv"
ncc_df = pd.read_csv(ncc_file)

# === Function to grid data ===
def make_grid(x, y, field):
    xi = np.unique(x)
    yi = np.unique(y)
    X, Y = np.meshgrid(xi, yi)
    Z = np.full_like(X, np.nan, dtype=float)
    for i in range(len(x)):
        xi_idx = np.where(xi == x[i])[0][0]
        yi_idx = np.where(yi == y[i])[0][0]
        Z[yi_idx, xi_idx] = field[i]
    return X, Y, Z

# === Create NCorr grids ===
X_zn, Y_zn, U_zn = make_grid(znssd_df["x"], znssd_df["y"], znssd_df["u"])
_, _, V_zn = make_grid(znssd_df["x"], znssd_df["y"], znssd_df["v"])

X_nc, Y_nc, U_nc = make_grid(ncc_df["x"], ncc_df["y"], ncc_df["u"])
_, _, V_nc = make_grid(ncc_df["x"], ncc_df["y"], ncc_df["v"])

# === Plot side by side for U and V ===
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# U displacement
im1 = axs[0,0].pcolormesh(X_zn, Y_zn, U_zn, shading="auto", cmap="jet")
axs[0,0].set_title("NCorr ZNSSD U")
axs[0,0].set_aspect("equal"); axs[0,0].invert_yaxis()
fig.colorbar(im1, ax=axs[0,0])

im2 = axs[0,1].pcolormesh(X_nc, Y_nc, U_nc, shading="auto", cmap="jet")
axs[0,1].set_title("NCorr NCC U")
axs[0,1].set_aspect("equal"); axs[0,1].invert_yaxis()
fig.colorbar(im2, ax=axs[0,1])

# V displacement
im3 = axs[1,0].pcolormesh(X_zn, Y_zn, V_zn, shading="auto", cmap="jet")
axs[1,0].set_title("NCorr ZNSSD V")
axs[1,0].set_aspect("equal"); axs[1,0].invert_yaxis()
fig.colorbar(im3, ax=axs[1,0])

im4 = axs[1,1].pcolormesh(X_nc, Y_nc, V_nc, shading="auto", cmap="jet")
axs[1,1].set_title("NCorr NCC V")
axs[1,1].set_aspect("equal"); axs[1,1].invert_yaxis()
fig.colorbar(im4, ax=axs[1,1])

plt.tight_layout()
plt.show()
