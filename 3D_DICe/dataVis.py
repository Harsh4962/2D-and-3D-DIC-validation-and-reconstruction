import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load DICe result file
df = pd.read_csv(r"C:\Users\harsh\dice_working_dir\3D_anal\.dice\results\DICe_solution_0.txt")

# Extract coordinates & displacements
x = df["MODEL_COORDINATES_X"]
y = df["MODEL_COORDINATES_Y"]
z = df["MODEL_COORDINATES_Z"]
ux = df["MODEL_DISPLACEMENT_X"]
uy = df["MODEL_DISPLACEMENT_Y"]
uz = df["MODEL_DISPLACEMENT_Z"]

# Compute displacement magnitude
disp_mag = np.sqrt(ux**2 + uy**2 + uz**2)

# Scatter plot (2D projection)
plt.figure(figsize=(8,6))
sc = plt.scatter(x, y, c=disp_mag, cmap="viridis", s=20)
plt.colorbar(sc, label="Displacement Magnitude")
plt.xlabel("Model X (mm)")
plt.ylabel("Model Y (mm)")
plt.title("3D DIC Displacement Field (Top View)")
plt.show()

# Optional: strain visualization
strain = df["VSG_STRAIN_XX"]
plt.figure(figsize=(8,6))
sc = plt.scatter(x, y, c=strain, cmap="coolwarm", s=20)
plt.colorbar(sc, label="Strain XX")
plt.xlabel("Model X (mm)")
plt.ylabel("Model Y (mm)")
plt.title("Local Strain Field (εxx)")
plt.show()
