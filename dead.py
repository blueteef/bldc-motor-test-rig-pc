from pathlib import Path
import sys

# Ensure this folder (containing bldc_loader/) is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from bldc_loader import load_folder

LOGS = Path(r"C:\Users\snyder\Documents\Motor Logs\SD_dump")  # change if logs are elsewhere

rs = load_folder(LOGS)

print("Folder:", rs.folder)
print("Runs:", len(rs.runs))
print("Folder issues:", rs.issues)

r = rs.runs[0]

print("First run:", r.name)
print("Columns:", r.df.columns.tolist())
print("Rows:", len(r.df))
print("Detected fs:", r.meta.get("derived", {}).get("fs_hz"))
print("Issues:", r.issues[:5])


for r in rs.runs[:5]:
    fs = r.meta.get("derived", {}).get("fs_hz")
    print(r.name, fs, r.issues)
