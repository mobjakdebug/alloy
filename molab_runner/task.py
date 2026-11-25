import json
from pathlib import Path

d = json.loads(Path("/marimo/alloy_gate/baseline_public_vllm.json").read_text(encoding="utf-8"))
sci = [x for x in d["detailed"] if x["kind"].startswith("science_")][:25]
for x in sci:
    print(f"{x['id']} | ok={x['ok']} | gold={x['gold']!r} | pred={x['pred']!r}")
print("=== TOOL PROBES ===")
for t in d["tool_probes"]:
    print(t["prompt"], "-->", repr(t["response"][:200]))
