import datetime
import json
import os
import subprocess
from pathlib import Path
import marimo._code_mode as cm

print("UTC_NOW:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader"],
        text=True,
    ).strip()
    print("GPU:", gpu)
except Exception as e:
    print("GPU_ERR:", e)

async with cm.get_context() as ctx:
    print("CELLS:", [(c.id, c.name) for c in ctx.cells])
    print("GLOBALS_SAMPLE:", sorted(k for k in ctx.globals.keys() if not k.startswith("_"))[:60])

marimo_dir = Path("/marimo")
if marimo_dir.exists():
    print("MARIMO_SUBDIRS:", sorted(p.name for p in marimo_dir.iterdir()))
for sub in ["/marimo/alloy_data/phase1_verified", "/marimo/alloy_data/phase1_pool", "/marimo/alloy_gate", "/marimo/alloy_logs", "/marimo/runs/alloy_round1"]:
    p = Path(sub)
    if p.exists():
        print(f"DIR {sub}:", {x.name: round(x.stat().st_size / (1024 * 1024), 2) for x in sorted(p.iterdir())})

for logp in ["/marimo/alloy_logs/phase1_distill.log", "/marimo/alloy_logs/phase2_auto_launcher.log", "/marimo/alloy_logs/phase2_round1_train.log"]:
    lp = Path(logp)
    if lp.exists():
        lines = lp.read_text(errors="replace").splitlines()
        print(f"LOG_TAIL {logp} (total {len(lines)} lines):")
        for line in lines[-25:]:
            print(" ", line)

print("HF_TOKEN_ENV:", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")))
