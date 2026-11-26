import json
from pathlib import Path

for name in ["phase4_baseline_ralph.json", "phase4_alloy_ralph.json"]:
    p = Path("/marimo/runs/alloy_evals") / name
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"=== {name} ===")
        for t in d.get("tasks", []):
            print(t["id"], "solved=", t["solved"], "iters=", t["iterations_used"], "calls=", t["calls"], "attempts=", [(a["applied"], a["passed"], a["err"][:120]) for a in t["attempts"]])

# Also verify Round 2 SFT vs Baseline under updated evaluate_public_regression_gate
import sys
sys.path.insert(0, "/marimo/alloy_repo")
from alloy.verifiers import evaluate_public_regression_gate

base = json.loads(Path("/marimo/runs/alloy_evals/baseline_public_vllm.json").read_text(encoding="utf-8"))
r2 = json.loads(Path("/marimo/runs/alloy_evals/round2_alloy_public_vllm.json").read_text(encoding="utf-8"))
grpo = json.loads(Path("/marimo/runs/alloy_evals/grpo_alloy_public_vllm.json").read_text(encoding="utf-8"))
print("R2_GATE:", json.dumps(evaluate_public_regression_gate(r2, base), indent=2))
print("GRPO_GATE:", json.dumps(evaluate_public_regression_gate(grpo, base), indent=2))
