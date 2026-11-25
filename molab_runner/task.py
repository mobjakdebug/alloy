import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_pools", tail_lines=25), indent=2))
ve = Path("/marimo/alloy_scripts/vllm_eval.py")
if ve.exists():
    print("=== VLLM_EVAL.PY ===")
    print(ve.read_text())
