import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
st = g["get_alloy_job_status"]("setup_and_pools", tail_lines=40)
print("SETUP_STATUS:", json.dumps(st, indent=2))

gate_p = Path("/marimo/alloy_gate/public_gate_659.json")
if gate_p.exists():
    gate = json.loads(gate_p.read_text(encoding="utf-8"))
    print("PUBLIC_GATE_KEYS:", {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in gate.items()})
    for k in ("science_bench", "konkur_gate", "physlogic_gate"):
        if k in gate and gate[k]:
            print(f"SAMPLE_{k}:", json.dumps(gate[k][0], ensure_ascii=False)[:400])

pool_dir = Path("/marimo/alloy_data/phase1_pool")
if pool_dir.exists():
    counts = {}
    for p in sorted(pool_dir.glob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            counts[p.name] = sum(1 for l in f if l.strip())
    print("POOL_COUNTS:", json.dumps(counts, indent=2))
