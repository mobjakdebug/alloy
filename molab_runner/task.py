import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_V2_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_pools_v2", tail_lines=40), indent=2))

gate_p = Path("/marimo/alloy_gate/public_gate_659.json")
if gate_p.exists():
    gate = json.loads(gate_p.read_text(encoding="utf-8"))
    print("PUBLIC_GATE_KEYS:", {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in gate.items()})

pool_dir = Path("/marimo/alloy_data/phase1_pool")
if pool_dir.exists():
    counts = {}
    for p in sorted(pool_dir.glob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            counts[p.name] = sum(1 for l in f if l.strip())
    print("POOL_COUNTS:", json.dumps(counts, indent=2))
