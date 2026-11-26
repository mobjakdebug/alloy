import json
from pathlib import Path
import marimo._code_mode as cm

if "get_alloy_job_status" not in cm.get_context().globals:
    async with cm.get_context() as _ctx:
        _by_name = {c.name: c.id for c in _ctx.cells}
        for _nm in ["imports", "header", "guard_helpers", "job_manager"]:
            if _nm in _by_name:
                _ctx.run_cell(_by_name[_nm])

g = cm.get_context().globals
print("JOB_STATUS:", json.dumps(g["get_alloy_job_status"]("alloy_r2_rebuild_and_push"), indent=2, default=str))
log_p = Path("/marimo/alloy_logs/alloy_r2_rebuild_and_push.log")
if log_p.exists():
    lines = log_p.read_text(errors="replace").splitlines()
    print("--- LOG TAIL ---")
    print("\n".join(lines[-80:]))
