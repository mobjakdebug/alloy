import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_BASELINE_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_baseline", tail_lines=35), indent=2))
bp = Path("/marimo/alloy_gate/baseline_public_vllm.json")
if bp.exists():
    d = json.loads(bp.read_text(encoding="utf-8"))
    print("BASELINE_METRICS:", json.dumps({k: v for k, v in d.items() if k != "detailed"}, indent=2, ensure_ascii=False))
