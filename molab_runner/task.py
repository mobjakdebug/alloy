import json, os, subprocess, sys
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals

repo_dir = Path("/marimo/alloy_repo")
if not (repo_dir / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", "arena/2b4e287c-alloy", "https://github.com/mobjakdebug/alloy.git", str(repo_dir)], check=True)
else:
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)

if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT
from alloy.phase2_and_3 import PHASE2_R2_SFT_SCRIPT, PHASE3_GRPO_SCRIPT
from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT
from alloy.verifiers import answers_match, evaluate_public_regression_gate

# Re-score baseline_public_vllm.json science_bench with upgraded verifier
bp = Path("/marimo/alloy_gate/baseline_public_vllm.json")
if bp.exists():
    bdata = json.loads(bp.read_text(encoding="utf-8"))
    chem_ok = chem_tot = chem_p = 0
    phys_ok = phys_tot = phys_p = 0
    for row in bdata["detailed"]:
        if row["kind"] == "science_chem":
            chem_tot += 1
            chem_p += int(row["pred"] is not None)
            row["ok"] = bool(row["pred"] is not None and answers_match(row["pred"], row["gold"]))
            chem_ok += int(row["ok"])
        elif row["kind"] == "science_phys":
            phys_tot += 1
            phys_p += int(row["pred"] is not None)
            row["ok"] = bool(row["pred"] is not None and answers_match(row["pred"], row["gold"]))
            phys_ok += int(row["ok"])
    tot = max(1, chem_tot + phys_tot)
    bdata["metrics"]["science_bench"]["correct"] = chem_ok + phys_ok
    bdata["metrics"]["science_bench"]["accuracy_pct"] = round(100.0 * (chem_ok + phys_ok) / tot, 2)
    bdata["metrics"]["science_bench"]["by_kind"] = {
        "science_chem": {"total": chem_tot, "correct": chem_ok, "parsed": chem_p},
        "science_phys": {"total": phys_tot, "correct": phys_ok, "parsed": phys_p},
    }
    bp.write_text(json.dumps(bdata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESCORED_BASELINE_METRICS:", json.dumps(bdata["metrics"], indent=2))

scripts_dir = Path("/marimo/alloy_scripts")
scripts_dir.mkdir(parents=True, exist_ok=True)
(scripts_dir / "public_vllm_eval.py").write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
(scripts_dir / "phase1_r2_distill.py").write_text(PHASE1_R2_SCRIPT, encoding="utf-8")
(scripts_dir / "phase2_r2_sft.py").write_text(PHASE2_R2_SFT_SCRIPT, encoding="utf-8")
(scripts_dir / "phase3_grpo.py").write_text(PHASE3_GRPO_SCRIPT, encoding="utf-8")
(scripts_dir / "phase4_ralph_eval.py").write_text(PHASE4_RALPH_EVAL_SCRIPT, encoding="utf-8")

def run_r2_sft_and_eval(log):
    # Ensure antlr4-python3-runtime is available for SymPy parse_latex in vllm_venv
    subprocess.run(["uv", "pip", "install", "--python", "/marimo/vllm_venv/bin/python", "-q", "antlr4-python3-runtime==4.13.2"], check=False)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"

    out_dir = Path("/marimo/alloy_data/phase1_r2_verified")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phase 1 Round 2
    for step in ("pass1", "pass2", "assemble"):
        log(f"=== Phase 1 Round 2 ({step}) ===")
        p = subprocess.Popen(
            [
                "/marimo/vllm_venv/bin/python", str(scripts_dir / "phase1_r2_distill.py"),
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
            if s and any(k in s for k in ["[Pass", "[Pool]", "FINAL_ROUND2_DATASET", "total_examples", "total_tokens", "avg_tokens", "on_policy_pct", "source_counts", "dropped", "Error", "Traceback", "Processed prompts"]):
                log(s)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"phase1_r2_distill step={step} exited rc={rc}")

    # 2. Phase 2 Round 2 SFT (fresh Ornith-1.5-9B, use_rslora=False, r=64, alpha=64, lr=1.5e-5)
    log("=== Phase 2 Round 2 SFT ===")
    p2 = subprocess.Popen(
        [
            "/tmp/uv-venv/bin/python", str(scripts_dir / "phase2_r2_sft.py"),
            "--data-path", str(out_dir / "alloy_round2_train.jsonl"),
            "--out-dir", "/marimo/runs/alloy_round2",
            "--max-hours", "3.5",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    for line in p2.stdout:
        s = line.rstrip()
        if s and any(k in s for k in ["[Phase2", "PHASE2_R2_DONE", "Error", "Traceback"]):
            log(s)
    rc2 = p2.wait()
    if rc2 != 0:
        raise RuntimeError(f"phase2_r2_sft exited rc={rc2}")

    # 3. Evaluate Round 2 Adapter on public_gate_659.json
    r2_eval_path = Path("/marimo/alloy_gate/round2_alloy_public_vllm.json")
    log("=== Evaluating Round 2 Adapter on public_gate_659.json ===")
    ev = subprocess.Popen(
        [
            "/marimo/vllm_venv/bin/python", str(scripts_dir / "public_vllm_eval.py"),
            "--model", "ornith-ai/Ornith-1.5-9B",
            "--lora-path", "/marimo/runs/alloy_round2/final_adapter",
            "--gate-path", "/marimo/alloy_gate/public_gate_659.json",
            "--out-path", str(r2_eval_path),
            "--max-tokens", "6144",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    for line in ev.stdout:
        s = line.rstrip()
        if s and any(k in s for k in ["Running vLLM", "PUBLIC_EVAL_SUMMARY", "elapsed_s", "total_tokens", "tok_per_sec", "identity_ok", "tool_ok", "konkur", "physlogic", "science_bench", "science_chem", "science_phys", "by_kind", "accuracy_pct", "parsed_pct", "truncated", "Error", "Traceback", "Processed prompts"]):
            log(s)
    rcev = ev.wait()
    if rcev != 0:
        raise RuntimeError(f"Round 2 eval exited rc={rcev}")

    r2_payload = json.loads(r2_eval_path.read_text(encoding="utf-8"))
    base_payload = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
    passed, rep = evaluate_public_regression_gate(r2_payload, base_payload)
    log(f"ROUND2_GATE passed={passed}: {json.dumps(rep)}")
    return {"round2_gate_passed": passed, "round2_report": rep}

print("STARTED r2_sft_and_eval:", g["start_alloy_job"]("r2_sft_and_eval", run_r2_sft_and_eval))
