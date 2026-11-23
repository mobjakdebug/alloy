import json, subprocess
import marimo._code_mode as cm

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_vllm_baseline", tail_lines=40), indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())
