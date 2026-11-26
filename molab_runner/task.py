import json
import subprocess
from pathlib import Path

if "alloy_job_status" in globals():
    print("GRPO_SMOKE_STATUS:", json.dumps(alloy_job_status("grpo_smoke"), indent=2))
log_p = Path("/marimo/alloy_logs/grpo_smoke.log")
if log_p.exists():
    print("LOG_TAIL:\n" + "\n".join(log_p.read_text(encoding="utf-8").splitlines()[-60:]))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())
