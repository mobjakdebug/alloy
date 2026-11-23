import json
import marimo._code_mode as cm

ctx = cm.get_context()
st = ctx.globals["get_alloy_job_status"]("deep_dataset_audit", tail_lines=40)
print("DEEP_AUDIT_STATUS:", json.dumps(st, indent=2, ensure_ascii=False))
