from pathlib import Path

for fn in ["setup_and_pools_v2.py", "phase1_distill.py", "phase1_r2_distill.py"]:
    p = Path("/marimo/alloy_scripts") / fn
    print(f"================ {fn} ================")
    print(p.read_text()[:18000])
