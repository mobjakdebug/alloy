import json
import subprocess
from pathlib import Path

print("STAGE2_STATUS:", json.dumps(get_alloy_job_status("alloy_stage2", tail_lines=45), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())

for label, p in [
    ("GRPO_SUMMARY", Path("/marimo/runs/alloy_round3_grpo/grpo_train_summary.json")),
    ("GRPO_EVAL", Path("/marimo/runs/alloy_evals/grpo_alloy_public_vllm.json")),
    ("PHASE4_BASELINE", Path("/marimo/runs/alloy_evals/phase4_baseline_ralph.json")),
    ("PHASE4_ALLOY", Path("/marimo/runs/alloy_evals/phase4_alloy_ralph.json")),
]:
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.pop("detailed", None)
            data.pop("tasks", None)
        print(f"{label}:", json.dumps(data, indent=2, ensure_ascii=False))
