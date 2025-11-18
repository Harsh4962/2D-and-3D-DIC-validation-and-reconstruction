import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------- settings --------
excel_path = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC3D_to_Excel.xlsx"   # your file
frame = 15                             # which frame index to visualize (01 -> sheet suffix f01)
points_sheet = f"Points3D_f{frame:02d}"
disp_sheet   = f"Disp_f{frame:02d}"
faces_sheet  = "Faces"

# -------- load data --------
Pts = pd.read_excel(excel_path, sheet_name=points_sheet)
Disp = pd.read_excel(excel_path, sheet_name=disp_sheet)
Faces = pd.read_excel(excel_path, sheet_name=faces_sheet, header=None).values

# Ensure 0-based indices for Faces if they are 1-based
if Faces.min() == 1:
    Faces = Faces - 1

# Extract arrays
V = Pts[['X','Y','Z']].to_numpy(dtype=float)          # N x 3 vertices
U = Disp[['Ux','Uy','Uz']].to_numpy(dtype=float)      # N x 3 displacement
Umag = Disp['Umag'].to_numpy(dtype=float)             # N

# -------- helper to draw a colored mesh --------
def plot_colormap(vertices, faces, scalars, title, cmap='viridis'):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Create polygons for each face
    polys = [vertices[tri] for tri in faces]
    coll = Poly3DCollection(polys, linewidths=0.2, edgecolors='k', alpha=1.0)
    # Face color = average scalar over the triangle
    face_vals = np.mean(scalars[faces], axis=1)
    coll.set_array(face_vals)
    coll.set_cmap(cmap)
    coll.set_clim(vmin=np.nanmin(scalars), vmax=np.nanmax(scalars))
    ax.add_collection3d(coll)

    # Axes limits
    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(scalars)
    plt.colorbar(m, ax=ax, shrink=0.7, pad=0.1, label=title)
    plt.tight_layout()
    plt.show()

# -------- heat maps --------
# 1) Total displacement magnitude
plot_colormap(V, Faces.astype(int), Umag, title='|U| (displacement magnitude)')

# 2) Component maps (Ux, Uy, Uz)
plot_colormap(V, Faces.astype(int), U[:,0], title='Ux (X-component)')
plot_colormap(V, Faces.astype(int), U[:,1], title='Uy (Y-component)')
plot_colormap(V, Faces.astype(int), U[:,2], title='Uz (Z-component)')

# -------- optional: deformed shape colored by |U| --------
V_def = V + U
plot_colormap(V_def, Faces.astype(int), Umag, title='Deformed shape colored by |U|')


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib import cm
# from matplotlib.colors import Normalize
# from pathlib import Path

# # ----------------- user settings -----------------
# excel_path = r"C:\Users\harsh\Downloads\DuoDIC-main\DuoDIC-main\DIC3D_to_Excel.xlsx"
# frame = 10   # frame index to visualize (will look for f10 sheets)
# points_sheet = f"Points3D_f{frame:02d}"
# disp_sheet   = f"Disp_f{frame:02d}"
# faces_sheet  = "Faces"
# strain_sheet = f"Strain_E_f{frame:02d}"   # expected sheet name for strain per face

# out_folder = Path(".") / "DIC_plots"      # where PNGs will be saved
# out_folder.mkdir(parents=True, exist_ok=True)
# # -------------------------------------------------

# # -------- load data --------
# Pts = pd.read_excel(excel_path, sheet_name=points_sheet)
# Disp = pd.read_excel(excel_path, sheet_name=disp_sheet)
# Faces = pd.read_excel(excel_path, sheet_name=faces_sheet, header=None).values

# # ensure 0-based face indices
# Faces = Faces.astype(int)
# if Faces.min() == 1:
#     Faces = Faces - 1

# # vertex arrays
# V = Pts[['X','Y','Z']].to_numpy(dtype=float)          # N x 3 vertices
# U = Disp[['Ux','Uy','Uz']].to_numpy(dtype=float)      # N x 3 displacement
# Umag = Disp['Umag'].to_numpy(dtype=float)             # N

# # load strain (per-face) if available
# strain_available = False
# try:
#     S = pd.read_excel(excel_path, sheet_name=strain_sheet)
#     # expected columns: FaceID, epc1, epc2, eeq  (many DIC exports use these names)
#     # convert to zero-based face index if needed
#     if 'FaceID' in S.columns:
#         face_ids = S['FaceID'].to_numpy(dtype=int)
#         min_id = face_ids.min()
#         if min_id == 1:
#             face_index = face_ids - 1
#         else:
#             face_index = face_ids
#         # create array of eeq/epc1/epc2 ordered by face index
#         num_faces = Faces.shape[0]
#         eeq_arr = np.full((num_faces,), np.nan)
#         epc1_arr = np.full((num_faces,), np.nan)
#         epc2_arr = np.full((num_faces,), np.nan)
#         for i, idx in enumerate(face_index):
#             if idx >=0 and idx < num_faces:
#                 if 'eeq' in S.columns:
#                     eeq_arr[idx] = S.iloc[i]['eeq']
#                 elif 'Eqv' in S.columns:
#                     eeq_arr[idx] = S.iloc[i]['Eqv']  # some exports use different name
#                 if 'epc1' in S.columns:
#                     epc1_arr[idx] = S.iloc[i]['epc1']
#                 if 'epc2' in S.columns:
#                     epc2_arr[idx] = S.iloc[i]['epc2']
#         strain_available = True
#     else:
#         # If the sheet has no FaceID but is already ordered per-face just extract columns
#         if 'eeq' in S.columns:
#             eeq_arr = S['eeq'].to_numpy(dtype=float)
#             epc1_arr = S['epc1'].to_numpy(dtype=float) if 'epc1' in S.columns else np.full_like(eeq_arr, np.nan)
#             epc2_arr = S['epc2'].to_numpy(dtype=float) if 'epc2' in S.columns else np.full_like(eeq_arr, np.nan)
#             strain_available = True
# except Exception as e:
#     print("Could not load strain sheet:", e)
#     strain_available = False

# # -------- helper to draw a colored face mesh with nicer shading --------
# def face_colormap_plot(vertices, faces, face_scalars, title, cmap_name='viridis',
#                        elev=30, azim=-60, edge_on=False, alpha=1.0, save_name=None):
#     """
#     vertices: (N,3)
#     faces: (M,3) indices into vertices
#     face_scalars: (M,) scalar value per face (not per vertex)
#     """
#     # normalize and colormap
#     cmap = cm.get_cmap(cmap_name)
#     norm = Normalize(vmin=np.nanmin(face_scalars), vmax=np.nanmax(face_scalars))
#     face_colors = cmap(norm(face_scalars))  # RGBA per face

#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')
#     polys = [vertices[tri] for tri in faces]

#     coll = Poly3DCollection(polys, linewidths=0.05 if edge_on else 0.0,
#                             edgecolors='k' if edge_on else None)
#     coll.set_facecolor(face_colors)
#     coll.set_alpha(alpha)
#     ax.add_collection3d(coll)

#     # scale axes equally
#     xyz_min = vertices.min(axis=0)
#     xyz_max = vertices.max(axis=0)
#     max_range = (xyz_max - xyz_min).max()
#     mid = (xyz_max + xyz_min) / 2.0
#     ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
#     ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
#     ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_title(title)

#     # Colorbar (create a ScalarMappable)
#     m = cm.ScalarMappable(norm=norm, cmap=cmap)
#     m.set_array(face_scalars)
#     cbar = plt.colorbar(m, ax=ax, shrink=0.6, pad=0.08)
#     cbar.set_label(title)

#     # tweak lighting-ish effect: slightly darken faces facing away using normals (visual improvement)
#     # compute facet normals to modulate brightness (optional)
#     try:
#         normals = np.cross(vertices[faces][:,1] - vertices[faces][:,0],
#                            vertices[faces][:,2] - vertices[faces][:,0])
#         norms = np.linalg.norm(normals, axis=1)
#         norms[norms==0] = 1.0
#         normals = normals / norms[:,None]
#         # viewer direction (approx)
#         view_dir = np.array([0.0, 0.0, 1.0])
#         shade = 0.6 + 0.4 * np.clip(np.dot(normals, view_dir), 0, 1)
#         # apply shading to face_colors RGB channels (leave alpha)
#         shaded = (face_colors[:,:3].T * shade).T
#         face_colors_shaded = np.concatenate([shaded, face_colors[:,3:4]], axis=1)
#         coll.set_facecolor(face_colors_shaded)
#     except Exception:
#         pass

#     plt.tight_layout()
#     if save_name is not None:
#         plt.savefig(str(out_folder / save_name), dpi=300, bbox_inches='tight')
#     plt.show()

# # -------- plot displacement magnitude (per-face) by averaging vertex Umag over each triangle --------
# # compute per-face scalar from vertex scalar by averaging
# num_faces = Faces.shape[0]
# face_Umag = np.mean(Umag[Faces], axis=1)

# # Better viewing arrangement: show three subplots with different view angles
# face_colormap_plot(V, Faces, face_Umag, title=f'|U| (frame {frame})', cmap_name='inferno',
#                    elev=25, azim=-60, edge_on=False, alpha=1.0, save_name=f'U_mag_f{frame:02d}.png')

# # -------- strain maps (if available) --------
# if strain_available:
#     # epc1, epc2, eeq arrays are per-face already
#     # For visualization reasons, clip extreme outliers to 1st/99th percentile
#     def clipped(arr, qlo=1, qhi=99):
#         lo = np.nanpercentile(arr, qlo)
#         hi = np.nanpercentile(arr, qhi)
#         return np.clip(arr, lo, hi)

#     # epc1
#     if not np.all(np.isnan(epc1_arr)):
#         e1 = clipped(epc1_arr)
#         face_colormap_plot(V, Faces, e1, title=f'epc1 (principal strain 1) f{frame:02d}',
#                            cmap_name='RdBu_r', elev=45, azim=30, edge_on=False, alpha=1.0,
#                            save_name=f'epc1_f{frame:02d}.png')

#     # epc2
#     if not np.all(np.isnan(epc2_arr)):
#         e2 = clipped(epc2_arr)
#         face_colormap_plot(V, Faces, e2, title=f'epc2 (principal strain 2) f{frame:02d}',
#                            cmap_name='RdBu_r', elev=60, azim=120, edge_on=False, alpha=1.0,
#                            save_name=f'epc2_f{frame:02d}.png')

#     # eeq (equivalent)
#     if not np.all(np.isnan(eeq_arr)):
#         eq = clipped(eeq_arr)
#         face_colormap_plot(V, Faces, eq, title=f'eeq (equivalent strain) f{frame:02d}',
#                            cmap_name='magma', elev=30, azim=140, edge_on=False, alpha=1.0,
#                            save_name=f'eeq_f{frame:02d}.png')
# else:
#     print("Strain sheet not found or could not be parsed. Skipping strain maps.")

# # -------- optional: deformed surface colored by eeq (if available) --------
# if strain_available and not np.all(np.isnan(eeq_arr)):
#     V_def = V + U   # deformed vertex positions
#     face_colormap_plot(V_def, Faces, eeq_arr, title=f'Deformed surface colored by eeq f{frame:02d}',
#                        cmap_name='magma', elev=20, azim=-40, edge_on=False, alpha=1.0,
#                        save_name=f'deformed_eeq_f{frame:02d}.png')

# print("Finished plotting. PNG files are saved to:", out_folder.resolve())
