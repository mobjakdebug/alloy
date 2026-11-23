import json
import marimo as mo
import marimo._code_mode as cm

ctx = cm.get_context()
st = ctx.globals["get_alloy_job_status"]("phase0_gate_train", tail_lines=40)
print("GATE_TRAIN_STATUS:", json.dumps(st, indent=2, ensure_ascii=False))
mo.status.toast(
    title=f"Phase 0 Gate+Train: {st.get('status')}",
    description="Check status output for peak VRAM, train tok/s, and Konkur 1404 counts.",
)
