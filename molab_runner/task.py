from pathlib import Path
txt = Path("/marimo/notebook.py").read_text()
for idx, line in enumerate(txt.splitlines(), 1):
    if any(k in line for k in ("swe_pool", "chemistry_pool", "SWE-rebench", "build_phase1_raw_pools", "start_alloy_job")):
        print(f"{idx}: {line}")
