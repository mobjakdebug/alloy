import json
import subprocess
from pathlib import Path

print("PHASE4_V2_STATUS:", json.dumps(get_alloy_job_status("alloy_phase4_v2", tail_lines=35), indent=2))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip())

for label, p in [
    ("PHASE4_V2_BASELINE", Path("/marimo/runs/alloy_evals/phase4_v2_baseline_ralph.json")),
    ("PHASE4_V2_ALLOY_R2", Path("/marimo/runs/alloy_evals/phase4_v2_r2_ralph.json")),
]:
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        tasks = d.pop("tasks", [])
        print(f"{label}:", json.dumps(d, indent=2))
        for t in tasks:
            print(f"  {label} task={t['id']} solved={t['solved']} iters={t['iterations_used']} attempts={[(a['applied'], a['passed'], a['err'][:80]) for a in t['attempts']]}")
