import json, subprocess
from pathlib import Path
import marimo as mo
import marimo._code_mode as cm

g = cm.get_context().globals
st = g["get_alloy_job_status"]("phase1_distill", tail_lines=40)
print("PHASE1_DISTILL:", json.dumps(st, indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

out_dir = Path("/marimo/alloy_data/phase1_verified")
if out_dir.exists():
    print("OUT_FILES:", [(p.name, round(p.stat().st_size / (1024**2), 2)) for p in sorted(out_dir.iterdir())])

if st.get("status") == "completed" and (out_dir / "alloy_round1_train.jsonl").exists():
    r1_st = g["get_alloy_job_status"]("phase2_round1_train", tail_lines=10)
    if r1_st.get("status") in ("not_found", "failed"):
        msg = g["start_alloy_job"](
            "phase2_round1_train",
            g["run_phase2_round1_train"],
            base_model=g["base_model_ui"].value,
            hf_org=g["hf_org_ui"].value.strip(),
            hf_token=g["get_hf_token"](),
            max_hours=4.2,
            lr=4e-5,
            max_length=6144,
        )
        print("AUTO_STARTED_TRAIN:", msg)
        mo.status.toast(title="Round 1 Training Started!", description="Phase 1 dataset ready -> running anti-forgetting rsLoRA SFT with Hub sync.")
