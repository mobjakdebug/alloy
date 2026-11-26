import json
import subprocess

print("GRPO_SMOKE_STATUS:", json.dumps(get_alloy_job_status("grpo_smoke", tail_lines=80), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())
