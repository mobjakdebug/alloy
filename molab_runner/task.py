import json
import marimo._code_mode as cm
g = cm.get_context().globals
print(json.dumps(g["get_alloy_job_status"]("setup_vllm_env", tail_lines=40), indent=2, ensure_ascii=False))
