import datetime, json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("UTC_NOW:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
print("GPU:", subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader"],
    text=True,
).strip())

st = g["get_alloy_job_status"]("phase2_round1_train", tail_lines=35)
print(f"=== phase2_round1_train ({st['status']}) started={st.get('started_at')} finished={st.get('finished_at')} ===")
if st.get("error"):
    print("ERROR:", st["error"])
for line in st.get("log_tail", []):
    print(" ", line)

for fn in ["baseline_ornith_vllm.json", "round1_alloy_vllm.json"]:
    p = Path("/marimo/alloy_gate") / fn
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        summary = {k: v for k, v in d.items() if k not in ("predictions", "samples", "details", "rows")}
        print(f"=== {fn} ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])
    else:
        print(f"MISSING {fn}")
