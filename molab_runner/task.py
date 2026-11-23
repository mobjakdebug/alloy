import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_konkur_baseline", tail_lines=30), indent=2, ensure_ascii=False))
print("UI_STATE:", {
    "base_model": g["base_model_ui"].value,
    "hf_org_set": bool(g["hf_org_ui"].value.strip()),
    "hf_org_val": g["hf_org_ui"].value.strip(),
    "hf_token_set": bool(g["get_hf_token"]()),
})
gate_p = Path("/marimo/alloy_gate/frozen_gate.json")
if gate_p.exists():
    gate = json.loads(gate_p.read_text(encoding="utf-8"))
    from collections import Counter
    print("GATE_EXAMS_250:", dict(Counter(x["exam_name"] for x in gate["konkur_gate"])))
    print("GATE_EXAMS_60:", dict(Counter(x["exam_name"] for x in gate["konkur_gate"][:60])))
