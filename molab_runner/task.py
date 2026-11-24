import json
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_JOB:", json.dumps(g["get_alloy_job_status"]("setup_vllm_and_cache", tail_lines=25), indent=2))
