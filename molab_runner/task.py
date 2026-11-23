import json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import marimo._code_mode as cm

print("NOW_UTC:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
g = cm.get_context().globals
st = g["get_alloy_job_status"]("phase1_distill", tail_lines=40)
print("PHASE1_DISTILL:", json.dumps(st, indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

out_dir = Path("/marimo/alloy_data/phase1_verified")
if out_dir.exists():
    print("OUT_FILES:", [(p.name, round(p.stat().st_size / (1024**2), 2)) for p in sorted(out_dir.iterdir())])
    ap = out_dir / "on_policy_anchors.jsonl"
    if ap.exists():
        with open(ap, "r", encoding="utf-8") as f:
            samples = [json.loads(next(f)) for _ in range(3)]
        for i, s in enumerate(samples):
            print(f"ANCHOR_SAMPLE_{i} ({s['source']}):", s["messages"][0]["content"][:120], "-->", s["messages"][1]["content"][:180], "...", s["messages"][1]["content"][-120:])
