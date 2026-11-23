import json, subprocess
import marimo._code_mode as cm
from datasets import load_dataset

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_vllm_baseline", tail_lines=40), indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

# Quick check of SWE-rebench-V2 columns & sample row (streaming 3 rows)
ds = load_dataset("nebius/SWE-rebench-V2", split="train", streaming=True)
for i, r in enumerate(ds):
    if i == 0:
        print("SWE_COLS:", list(r.keys()))
        print("SWE_SAMPLE0:", {k: (str(v)[:140] if v is not None else None) for k, v in r.items()})
    if i >= 2:
        break
