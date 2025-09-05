import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load NCorr results ===
ncorr_file = r"C:\Users\harsh\OneDrive\Desktop\NCorr_final\ncorr_inc_011.csv"
ncorr_df = pd.read_csv(ncorr_file)

# === Load DICe results ===
dice_file = r"C:\Users\harsh\OneDrive\Desktop\dice_working_dir\results\DICe_solution_10.txt"
dice_df = pd.read_csv(dice_file, sep=",", comment='#')

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

# === NCorr grids ===
X_nc, Y_nc, U_nc = make_grid(ncorr_df["x"], ncorr_df["y"], ncorr_df["u"])
_, _, V_nc = make_grid(ncorr_df["x"], ncorr_df["y"], ncorr_df["v"])

# === DICe grids ===
X_di, Y_di, U_di = make_grid(dice_df["COORDINATE_X"], dice_df["COORDINATE_Y"], dice_df["DISPLACEMENT_X"])
_, _, V_di = make_grid(dice_df["COORDINATE_X"], dice_df["COORDINATE_Y"], dice_df["DISPLACEMENT_Y"])

# === Plot side by side for U and V ===
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# U displacement
im1 = axs[0,0].pcolormesh(X_nc, Y_nc, U_nc, shading="auto", cmap="jet")
axs[0,0].set_title("NCorr Displacement U")
axs[0,0].set_aspect("equal"); axs[0,0].invert_yaxis()
fig.colorbar(im1, ax=axs[0,0])

im2 = axs[0,1].pcolormesh(X_di, Y_di, U_di, shading="auto", cmap="jet")
axs[0,1].set_title("DICe Displacement U")
axs[0,1].set_aspect("equal"); axs[0,1].invert_yaxis()
fig.colorbar(im2, ax=axs[0,1])

# V displacement
im3 = axs[1,0].pcolormesh(X_nc, Y_nc, V_nc, shading="auto", cmap="jet")
axs[1,0].set_title("NCorr Displacement V")
axs[1,0].set_aspect("equal"); axs[1,0].invert_yaxis()
fig.colorbar(im3, ax=axs[1,0])

im4 = axs[1,1].pcolormesh(X_di, Y_di, V_di, shading="auto", cmap="jet")
axs[1,1].set_title("DICe Displacement V")
axs[1,1].set_aspect("equal"); axs[1,1].invert_yaxis()
fig.colorbar(im4, ax=axs[1,1])

plt.tight_layout()
plt.show()
