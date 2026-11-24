import datetime, json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("UTC_NOW:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
print("GPU:", subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader"],
    text=True,
).strip())

vdir = Path("/marimo/alloy_data/phase1_verified")
if vdir.exists():
    print("PHASE1_VERIFIED_MB:", {p.name: round(p.stat().st_size / (1024 * 1024), 2) for p in sorted(vdir.iterdir())})

rdir = Path("/marimo/runs/alloy_round1")
if rdir.exists():
    print("RUNS_R1:", [p.name for p in sorted(rdir.iterdir())])

for jn in ["phase1_distill", "phase2_round1_train"]:
    st = g["get_alloy_job_status"](jn, tail_lines=25)
    print(f"=== {jn} ({st['status']}) started={st.get('started_at')} finished={st.get('finished_at')} ===")
    if st.get("error"):
        print("ERROR:", st["error"])
    if st.get("result"):
        print("RESULT:", json.dumps(st["result"], indent=2)[:1500])
    for line in st.get("log_tail", []):
        print(" ", line)
