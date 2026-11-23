import json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("BASELINE:", json.dumps(g["get_alloy_job_status"]("phase0_vllm_baseline", tail_lines=30), indent=2, ensure_ascii=False))
print("POOL_PREP:", json.dumps(g["get_alloy_job_status"]("phase1_pool_prep", tail_lines=30), indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())
res_p = Path("/marimo/alloy_gate/baseline_ornith_vllm.json")
if res_p.exists():
    d = json.loads(res_p.read_text(encoding="utf-8"))
    print("FULL_METRICS:", json.dumps(d["metrics"], indent=2, ensure_ascii=False))
    print("IDENTITY:", json.dumps(d["identity_probes"], indent=2, ensure_ascii=False))
    print("TOOLS:", json.dumps(d["tool_probes"], indent=2, ensure_ascii=False))
# check2
# check3
