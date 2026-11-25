import json, os, subprocess, sys
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals

# Kill any lingering setup_and_pools_v2 process stuck at interpreter teardown
subprocess.run(["pkill", "-9", "-f", "setup_and_pools_v2.py"], check=False)

pool_dir = Path("/marimo/alloy_data/phase1_pool")
counts = {}
for p in sorted(pool_dir.glob("*.jsonl")):
    with open(p, "r", encoding="utf-8") as f:
        counts[p.name] = sum(1 for l in f if l.strip())
print("POOL_COUNTS:", json.dumps(counts, indent=2))

gate = json.loads(Path("/marimo/alloy_gate/public_gate_659.json").read_text(encoding="utf-8"))
print("PUBLIC_GATE_COUNTS:", {k: len(v) for k, v in gate.items()})

repo_dir = Path("/marimo/alloy_repo")
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT

scripts_dir = Path("/marimo/alloy_scripts")
scripts_dir.mkdir(parents=True, exist_ok=True)
eval_script = scripts_dir / "public_vllm_eval.py"
eval_script.write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
p1_r2_script = scripts_dir / "phase1_r2_distill.py"
p1_r2_script.write_text(PHASE1_R2_SCRIPT, encoding="utf-8")

def run_baseline_and_phase1_r2(log):
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"

    # 1. Baseline eval on public_gate_659.json
    base_out = Path("/marimo/alloy_gate/baseline_public_vllm.json")
    if not base_out.exists():
        log("=== Step 1: Baseline vLLM Eval on ornith-ai/Ornith-1.5-9B (public_gate_659.json) ===")
        ep = subprocess.Popen(
            [
                "/marimo/vllm_venv/bin/python", str(eval_script),
                "--model", "ornith-ai/Ornith-1.5-9B",
                "--gate-path", "/marimo/alloy_gate/public_gate_659.json",
                "--out-path", str(base_out),
                "--max-tokens", "6144",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        for line in ep.stdout:
            s = line.rstrip()
            if s and any(k in s for k in ["Running vLLM", "PUBLIC_EVAL_SUMMARY", "elapsed_s", "total_tokens", "tok_per_sec", "identity_ok", "tool_ok", "konkur", "physlogic", "science_bench", "science_chem", "science_phys", "by_kind", "accuracy_pct", "parsed_pct", "truncated", "Error", "Traceback", "Processed prompts"]):
                log(s)
        rc = ep.wait()
        if rc != 0:
            raise RuntimeError(f"baseline public_vllm_eval exited rc={rc}")
    else:
        log("Baseline public_vllm.json already exists; skipping.")

    # 2. Phase 1 Round 2: Pass 1 (On-Policy Anchors + pass@4 Probes), Pass 2 (Qwen3.8-27B-FP8), Assemble
    out_dir = Path("/marimo/alloy_data/phase1_r2_verified")
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in ("pass1", "pass2", "assemble"):
        log(f"=== Step 2.{step}: Phase 1 Round 2 ({step}) ===")
        p = subprocess.Popen(
            [
                "/marimo/vllm_venv/bin/python", str(p1_r2_script),
                "--step", step,
                "--base-model", "ornith-ai/Ornith-1.5-9B",
                "--teacher-model", "Qwen/Qwen3.8-27B-FP8",
                "--pool-dir", "/marimo/alloy_data/phase1_pool",
                "--out-dir", str(out_dir),
                "--gate-dir", "/marimo/alloy_gate",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        for line in p.stdout:
            s = line.rstrip()
            if s and any(k in s for k in ["[Pass", "[Pool]", "FINAL_ROUND2_DATASET", "total_examples", "total_tokens", "avg_tokens", "on_policy_pct", "source_counts", "Error", "Traceback", "Processed prompts"]):
                log(s)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"phase1_r2_distill step={step} exited rc={rc}")

    stats = json.loads((out_dir / "round2_stats.json").read_text(encoding="utf-8"))
    base_metrics = json.loads(base_out.read_text(encoding="utf-8"))["metrics"]
    return {"baseline_metrics": base_metrics, "round2_dataset": stats}

print("STARTED r2_baseline_and_phase1:", g["start_alloy_job"]("r2_baseline_and_phase1", run_baseline_and_phase1_r2))
