# Digital Image Correlation (DIC) Analysis and Inverse Reconstruction

This repository contains a complete pipeline for performing 2D and 3D Digital Image Correlation (DIC) analysis, displacement and strain-based inverse reconstructions, and cross-validation between DICe and Ncorr. The scripts are organized to support both quantitative and qualitative evaluations of DIC accuracy, reconstruction fidelity, and stereo reprojection performance.

The repository is structured into three major modules:

* **2D DIC using DICe**
* **3D Stereo DIC using DICe**
* **Ncorr–DICe Cross-Validation**

Each module contains focused scripts for reconstruction, visualization, and validation.

---

## Repository Structure

### 1. 2D_DIC_usingDICe

Scripts for performing inverse reconstruction and visualization using 2D DIC (single camera).

* `disp_based_inverse_reconstruction.py`
  Displacement-based inverse reconstruction from measured pixel displacements using intensity conservation.

* `strain_based_inverse_reconstruction.py`
  Strain-based inverse reconstruction using Poisson equation with DICe strain fields as input.

---

### 2. 3D_DIC_usingDICe

Scripts supporting 3D stereo DIC analysis, stereo reprojection, and reconstruction.

* `disp_based_inverse_reconstruction.py`
  Full 3D inverse reconstruction using measured 3D displacements projected into both cameras.

* `displacement_reprojection_validation.py`
  Reprojection of reconstructed 3D points into camera coordinates to validate extrinsic and intrinsic calibration.

* `in_camera_coord_system_visualization.py`
  Visualization of reconstructed surfaces, displacements, and strains in the camera coordinate system.

* `in_plane_coord_system_visualization.py`
  Visualization of DIC results in the specimen (plane) coordinate system.

* `strain_based_inverse_reconstruction.py`
  Poisson-based strain-driven 3D surface reconstruction using strain compatibility equations.

* `strain_visualization_in_plane_coord_system.py`
  Visual exploration of strain components in the specimen coordinate system.

* `subset_center_reprojection_validation.py`
  Reprojection of DIC subset centers to validate calibration accuracy and grid alignment.

---

### 3. NCorr_DICe_cross_validation

Scripts for comparing Ncorr and DICe output fields.

* `qualitative_comparison.py`
  Side-by-side visualization of displacement and strain fields from both solvers.

* `quantitative_comparison.py`
  Statistical comparisons, error maps, and global metrics for evaluating solver consistency.

---

## Features

The repository provides:

### Inverse Reconstruction (2D and 3D)

* Displacement-based image reconstruction
* Strain-based reconstruction via Poisson integration
* Full stereo back-projection and forward-projection validation
* Robust image warping and intensity-mapped reconstruction

### Validation Metrics

Computed metrics include:

* RMS intensity error
* Mean absolute error (MAE)
* Structural Similarity Index (SSIM)
* Peak Signal-to-Noise Ratio (PSNR)
* Relative Root Mean Square Error (RRMSE), with thresholding
* Full-field error heatmaps and relative error maps
* Histograms and spatial statistics

### Visualization Utilities

* Specimen-frame and camera-frame visualization
* 3D point-cloud surface construction
* Strain component plotting: εxx, εyy, εxy
* Reprojection of subset centers and 3D reconstructed points

---

## Dependencies

Typical dependencies include:

* Python 3.8+
* NumPy
* SciPy
* OpenCV
* Matplotlib
* scikit-image
* pandas

```bash
pip install numpy scipy opencv-python matplotlib scikit-image pandas
```

---

## Input Requirements

Most scripts require:

* DICe output file (`DICe_solution_0.txt`)
* Calibration XML (`cal.xml`)
* Reference and deformed images (single or stereo)
* Best-fit plane file for 3D DIC (`best_fit_plane_out.dat`)

Paths can be configured directly in each script.

---

## Usage

Each script is standalone and can be executed directly:

```bash
python disp_based_inverse_reconstruction.py
```

Results are saved automatically as PNG figures and printed in the console.

---

## Applications

This repository is suitable for:

* Correlation algorithm validation
* Optical metrology research
* Calibration verification
* 2D and 3D surface reconstruction
* Comparison of DIC solvers (DICe vs. Ncorr)
* Laboratory experiment analysis

---

## Notes

* RRMSE is provided for completeness but is not the preferred metric for DIC reconstruction quality.
  Due to the division by local pixel intensity, speckle patterns with many low-intensity regions can produce inflated relative errors despite high SSIM/PSNR and visually correct reconstructions.
