import os
from pathlib import Path

print("ALL_ENV_KEY_NAMES:", sorted(os.environ.keys()))

root = Path("/marimo/runs")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(f"{p} ({p.stat().st_size} bytes)")
    elif p.is_dir() and not any(p.iterdir()):
        print(f"{p}/ (EMPTY DIR)")

print("\n--- /marimo/alloy_scripts ---")
for p in sorted(Path("/marimo/alloy_scripts").rglob("*")):
    if p.is_file():
        print(f"{p} ({p.stat().st_size} bytes)")
