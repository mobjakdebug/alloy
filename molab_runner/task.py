import json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("JOB_STATUS:", json.dumps(g["get_alloy_job_status"]("r2_baseline_and_phase1", tail_lines=40), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())

bp = Path("/marimo/alloy_gate/baseline_public_vllm.json")
if bp.exists():
    d = json.loads(bp.read_text(encoding="utf-8"))
    print("BASELINE_PUBLIC:", json.dumps({k: v for k, v in d.items() if k not in ("detailed", "identity_probes", "tool_probes")}, indent=2, ensure_ascii=False))

r2s = Path("/marimo/alloy_data/phase1_r2_verified/round2_stats.json")
if r2s.exists():
    print("ROUND2_STATS:", r2s.read_text(encoding="utf-8"))
