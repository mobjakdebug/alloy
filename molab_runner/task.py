import json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("R2_SFT_EVAL_STATUS:", json.dumps(g["get_alloy_job_status"]("r2_sft_and_eval", tail_lines=35), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())

for label, p in [
    ("ROUND2_DATASET_STATS", "/marimo/alloy_data/phase1_r2_verified/round2_stats.json"),
    ("ROUND2_TRAIN_SUMMARY", "/marimo/runs/alloy_round2/round2_train_summary.json"),
    ("ROUND2_EVAL", "/marimo/alloy_gate/round2_alloy_public_vllm.json"),
]:
    fp = Path(p)
    if fp.exists():
        d = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "detailed" in d:
            d = {k: v for k, v in d.items() if k != "detailed"}
        print(f"{label}:", json.dumps(d, indent=2, ensure_ascii=False))
