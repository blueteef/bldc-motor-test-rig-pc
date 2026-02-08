from pathlib import Path
from bldc_loader import load_folder

logs = Path(r"C:\path\to\your\logs")  # <-- change this
rs = load_folder(logs)

print(f"Folder: {rs.folder}")
print(f"Runs loaded: {len(rs.runs)}")
print(f"Issues: {rs.issues}")

if rs.summary is not None:
    print("RUN_SUMMARY rows:", len(rs.summary))
if rs.summary_short is not None:
    print("RUN_SUMMARY_SHORT rows:", len(rs.summary_short))

# print a couple runs
for r in rs.runs[:3]:
    print(r.name, r.df.shape, r.issues, "meta_keys:", list(r.meta.keys())[:6])
