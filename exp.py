# # pip install pandas matplotlib openpyxl

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ========= USER SETTINGS =========
# INPUT_PATH = Path(r"C:\Users\harsh\Downloads\TensileTest_Aug_MS.xls (1) (1)(1) (1).xlsx")   # folder OR a single .xlsx file
# OUTPUT_DIR = Path(r"./plots_out_2")

# GAUGE_LENGTH_MM = 84.48
# WIDTH_MM = 12.71
# THICKNESS_MM = 3.30
# # =================================

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AREA_MM2 = WIDTH_MM * THICKNESS_MM  # mm^2
# # 1 kN = 1000 N, and 1 N/mm^2 = 1 MPa → stress will be in MPa directly

# def ensure_iter_of_files(path: Path):
#     if path.is_dir():
#         files = sorted([p for p in path.iterdir() if p.suffix.lower() in [".xlsx", ".xls"]])
#     else:
#         files = [path]
#     if not files:
#         raise FileNotFoundError(f"No Excel files found in {path}")
#     return files

# def clean_df(df: pd.DataFrame) -> pd.DataFrame:
#     # Normalize expected column names
#     rename_map = {}
#     for c in df.columns:
#         cl = str(c).strip().lower()
#         if cl in ["time_s", "time (s)", "time[s]", "time", "t", "t_s"]:
#             rename_map[c] = "Time_s"
#         elif cl in ["displacement_mm", "disp_mm", "displacement (mm)", "extension_mm", "ext_mm", "displacement"]:
#             rename_map[c] = "Displacement_mm"
#         elif cl in ["load_kn", "load (kn)", "load", "force_kn", "force (kn)"]:
#             rename_map[c] = "Load_kN"
#     if rename_map:
#         df = df.rename(columns=rename_map)

#     # Keep only the needed columns if they exist
#     needed = ["Time_s", "Displacement_mm", "Load_kN"]
#     missing = [c for c in needed if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

#     # Make numeric & drop rows with no data
#     for c in needed:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df.dropna(subset=needed).copy()

#     # Sort by time if present
#     df = df.sort_values("Time_s").reset_index(drop=True)
#     return df

# def compute_engineering_stress_strain(df: pd.DataFrame) -> pd.DataFrame:
#     # Strain (mm/mm), Stress (MPa)
#     df["Strain"] = df["Displacement_mm"] / GAUGE_LENGTH_MM
#     df["Stress_MPa"] = (df["Load_kN"] * 1000.0) / AREA_MM2
#     return df

# def plot_xy(x, y, xlabel, ylabel, title, outpath):
#     plt.figure()
#     plt.plot(x, y)  # no explicit colors or styles
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(True, linestyle="--", alpha=0.4)
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()

# def process_excel(file_path: Path):
#     # Read first sheet by default
#     df = pd.read_excel(file_path)
#     df = clean_df(df)
#     df = compute_engineering_stress_strain(df)

#     # Save computed table for reference
#     out_csv = OUTPUT_DIR / f"{file_path.stem}_with_stress_strain.csv"
#     df.to_csv(out_csv, index=False)

#     # Make plots
#     base = OUTPUT_DIR / file_path.stem

#     plot_xy(
#         df["Strain"], df["Stress_MPa"],
#         xlabel="Strain (mm/mm)",
#         ylabel="Stress (MPa)",
#         title=f"Stress vs Strain — {file_path.name}",
#         outpath=str(base) + "_stress_vs_strain.png"
#     )

#     plot_xy(
#         df["Time_s"], df["Displacement_mm"],
#         xlabel="Time (s)",
#         ylabel="Displacement (mm)",
#         title=f"Time vs Displacement — {file_path.name}",
#         outpath=str(base) + "_time_vs_displacement.png"
#     )

#     plot_xy(
#         df["Time_s"], df["Load_kN"],
#         xlabel="Time (s)",
#         ylabel="Load (kN)",
#         title=f"Time vs Load — {file_path.name}",
#         outpath=str(base) + "_time_vs_load.png"
#     )

#     print(f"Done: {file_path.name}")
#     print(f"  -> {out_csv}")
#     print(f"  -> {base}_stress_vs_strain.png")
#     print(f"  -> {base}_time_vs_displacement.png")
#     print(f"  -> {base}_time_vs_load.png")

# def main():
#     files = ensure_iter_of_files(INPUT_PATH)
#     for f in files:
#         try:
#             process_excel(f)
#         except Exception as e:
#             print(f"[WARN] Skipped {f.name}: {e}")

# if __name__ == "__main__":
#     main()


# pip install pandas matplotlib openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= USER SETTINGS =========
INPUT_PATH = Path(r"C:\Users\harsh\Downloads\TensileTest_Aug_MS.xls (1) (1)(1) (1).xlsx")   # folder OR a single .xlsx file
OUTPUT_DIR = Path(r"./plots_out_2")

GAUGE_LENGTH_MM = 84.48
WIDTH_MM = 12.71
THICKNESS_MM = 3.30
YIELD_OFFSET = 0.002  # 0.2% offset
# =================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AREA_MM2 = WIDTH_MM * THICKNESS_MM  # mm^2  (1 N/mm^2 = 1 MPa)

def ensure_iter_of_files(path: Path):
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in [".xlsx", ".xls"]])
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No Excel files found in {path}")
    return files

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize expected column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["time_s", "time (s)", "time[s]", "time", "t", "t_s"]:
            rename_map[c] = "Time_s"
        elif cl in ["displacement_mm", "disp_mm", "displacement (mm)", "extension_mm", "ext_mm", "displacement", "extension"]:
            rename_map[c] = "Displacement_mm"
        elif cl in ["load_kn", "load (kn)", "load", "force_kn", "force (kn)", "force"]:
            rename_map[c] = "Load_kN"
    if rename_map:
        df = df.rename(columns=rename_map)

    needed = ["Time_s", "Displacement_mm", "Load_kN"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed).copy()

    # If early displacement is negative, flip sign (common in some logs)
    if df["Displacement_mm"].head(20).median() < 0:
        df["Displacement_mm"] = -df["Displacement_mm"]

    df = df.sort_values("Time_s").reset_index(drop=True)
    return df

def compute_engineering_stress_strain(df: pd.DataFrame) -> pd.DataFrame:
    df["Strain"] = df["Displacement_mm"] / GAUGE_LENGTH_MM           # mm/mm
    df["Stress_MPa"] = (df["Load_kN"] * 1000.0) / AREA_MM2           # MPa
    # keep nonnegative strain and sort by strain for integration
    df = df[df["Strain"] >= 0].sort_values("Strain").drop_duplicates(subset="Strain").reset_index(drop=True)
    return df

# ---------- Property helpers ----------
def estimate_modulus(df: pd.DataFrame) -> float:
    """Linear fit in low-strain/low-stress region."""
    strain = df["Strain"].to_numpy()
    stress = df["Stress_MPa"].to_numpy()
    uts = stress.max() if len(stress) else np.nan
    mask = (strain <= 0.0025) | (stress <= 0.35 * uts)
    sub = df[mask]
    if len(sub) < 5:
        sub = df.iloc[:min(20, len(df))]
    x = sub["Strain"].to_numpy()
    y = sub["Stress_MPa"].to_numpy()
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        return float(m)
    return np.nan

def yield_02_offset(df: pd.DataFrame, E: float, offset=YIELD_OFFSET):
    """Return (yield_strain, yield_stress, crossed_flag) by searching for intersection with 0.2% offset line."""
    strain = df["Strain"].to_numpy()
    stress = df["Stress_MPa"].to_numpy()

    # dense grid interpolation for robust crossing search
    grid = np.linspace(strain.min(), strain.max(), 5000)
    sig_grid = np.interp(grid, strain, stress)
    f = sig_grid - E * (grid - offset)

    sign = np.sign(f)
    sign[sign == 0] = 1
    idx = np.where((sign[1:] >= 0) & (sign[:-1] < 0))[0]

    if idx.size > 0:
        i = int(idx[0])
        x0, x1 = grid[i], grid[i+1]
        y0, y1 = f[i], f[i+1]
        x_star = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        y_star = float(np.interp(x_star, strain, stress))
        return float(x_star), y_star, True
    # fallback: closest approach
    j = int(np.argmin(np.abs(f)))
    return float(grid[j]), float(sig_grid[j]), False

def trapz_area(x, y):
    """Numerical integral using trapezoidal rule."""
    return float(np.trapz(y, x))  # MPa * (dimensionless) => MJ/m^3

def plot_xy(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.plot(x, y)  # default style
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def process_excel(file_path: Path):
    df = pd.read_excel(file_path, sheet_name=0)
    df = clean_df(df)
    df = compute_engineering_stress_strain(df)

    # ---- properties ----
    E = estimate_modulus(df)                           # MPa
    ys, ysig, crossed = yield_02_offset(df, E)         # yield strain & stress
    uts = float(df["Stress_MPa"].max())
    uts_strain = float(df.loc[df["Stress_MPa"].idxmax(), "Strain"])
    frac_stress = float(df["Stress_MPa"].iloc[-1])
    frac_strain = float(df["Strain"].iloc[-1])

    # Areas
    total_area = trapz_area(df["Strain"].to_numpy(), df["Stress_MPa"].to_numpy())          # toughness (MJ/m^3)
    # modulus of resilience = area up to yield (0.2% offset yield)
    elastic_mask = df["Strain"] <= ys
    if elastic_mask.any():
        res_area = trapz_area(df.loc[elastic_mask, "Strain"].to_numpy(),
                              df.loc[elastic_mask, "Stress_MPa"].to_numpy())
    else:
        # fallback: use linear elastic estimate 0.5*σy*εy
        res_area = 0.5 * ysig * ys

    # Save computed table
    out_csv = OUTPUT_DIR / f"{file_path.stem}_with_stress_strain.csv"
    df.to_csv(out_csv, index=False)

    # Save properties CSV
    props = pd.DataFrame([{
        "file": file_path.name,
        "Area_mm2": AREA_MM2,
        "GaugeLength_mm": GAUGE_LENGTH_MM,
        "E_MPa": E,
        "Yield_strain": ys,
        "Yield_stress_MPa": ysig,
        "Yield_crossed_exact": crossed,
        "UTS_MPa": uts,
        "UTS_strain": uts_strain,
        "Fracture_stress_MPa": frac_stress,
        "Fracture_strain": frac_strain,
        "Toughness_MJ_per_m3": total_area,       # numeric value equals MPa
        "Modulus_of_Resilience_MJ_per_m3": res_area
    }])
    props_csv = OUTPUT_DIR / f"{file_path.stem}_properties.csv"
    props.to_csv(props_csv, index=False)

    # ---- Plots ----
    base = OUTPUT_DIR / file_path.stem

    # Stress–Strain (with shaded areas + offset line + markers)
    plt.figure()
    plt.plot(df["Strain"], df["Stress_MPa"], label="Stress–Strain")
    # 0.2% offset line (draw across same strain range)
    plt.plot(df["Strain"], E * (df["Strain"] - YIELD_OFFSET), "--", label="0.2% Offset")
    # Shade resilience (up to yield)
    mask = df["Strain"] <= ys
    if mask.any():
        plt.fill_between(df.loc[mask, "Strain"], 0, df.loc[mask, "Stress_MPa"], alpha=0.2, label="Resilience area")
    # Shade total area (optional light overlay)
    plt.fill_between(df["Strain"], 0, df["Stress_MPa"], alpha=0.08, label="Total area (Toughness)")

    # Key points
    plt.scatter([ys, uts_strain, frac_strain], [ysig, uts, frac_stress], label="Yield / UTS / Fracture")

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.title(
        f"Stress–Strain — {file_path.name}\n"
        f"E={E:.1f} MPa | σy={ysig:.1f} MPa | UTS={uts:.1f} MPa\n"
        f"Toughness={total_area:.2f} MJ/m³ | Resilience={res_area:.2f} MJ/m³"
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(base) + "_stress_vs_strain.png", dpi=220)
    plt.close()

    # Time–Displacement
    plot_xy(
        df["Time_s"], df["Displacement_mm"],
        xlabel="Time (s)", ylabel="Displacement (mm)",
        title=f"Time vs Displacement — {file_path.name}",
        outpath=str(base) + "_time_vs_displacement.png"
    )

    # Time–Load
    plot_xy(
        df["Time_s"], df["Load_kN"],
        xlabel="Time (s)", ylabel="Load (kN)",
        title=f"Time vs Load — {file_path.name}",
        outpath=str(base) + "_time_vs_load.png"
    )

    # Console summary
    print(f"[OK] {file_path.name}")
    print(f"  E (MPa): {E:.2f}")
    print(f"  Yield  : {ysig:.2f} MPa at strain {ys:.6f} (exact crossing: {crossed})")
    print(f"  UTS    : {uts:.2f} MPa at strain {uts_strain:.6f}")
    print(f"  Fract. : {frac_stress:.2f} MPa at strain {frac_strain:.6f}")
    print(f"  Toughness (area under full curve): {total_area:.3f} MJ/m^3")
    print(f"  Modulus of Resilience            : {res_area:.3f} MJ/m^3")
    print(f"  -> {props_csv}")
    print(f"  -> {out_csv}")

def main():
    files = ensure_iter_of_files(INPUT_PATH)
    for f in files:
        try:
            process_excel(f)
        except Exception as e:
            print(f"[WARN] Skipped {f.name}: {e}")

if __name__ == "__main__":
    main()
