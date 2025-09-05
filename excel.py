import pandas as pd

# Try reading the uploaded file to inspect its structure
file_path = r"C:\Users\harsh\Downloads\B1_S75_F165"

# Attempt to read as text first (since no extension is given)
with open(file_path, "r", errors="ignore") as f:
    sample = f.read(500)

sample[:500]
df = pd.read_csv(file_path, sep="\t", header=None, names=["Fx", "Fy", "Fz"])

# Show first few rows
df.head()
excel_path = r"C:\Users\harsh\Downloads\forces6.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Force Data", index=False)
    