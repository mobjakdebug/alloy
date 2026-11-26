import json
import subprocess

print("GRPO_SMOKE_STATUS:", json.dumps(alloy_job_status("grpo_smoke"), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())
