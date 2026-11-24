import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("POOL_PREP:", json.dumps(g["get_alloy_job_status"]("phase1_pool_prep", tail_lines=20), indent=2))
print("=== PHASE1_DISTILL.PY ===")
print(Path("/marimo/alloy_scripts/phase1_distill.py").read_text())
