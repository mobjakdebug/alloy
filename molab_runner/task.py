import json, os, subprocess, sys
from collections import deque
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals

repo_dir = Path("/marimo/alloy_repo")
subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)

if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

# Clear any cached alloy modules in persistent marimo kernel
for _m in list(sys.modules.keys()):
    if _m == "alloy" or _m.startswith("alloy."):
        del sys.modules[_m]

from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT
from alloy.phase2_and_3 import PHASE2_R2_SFT_SCRIPT, PHASE3_GRPO_SCRIPT
from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT
from alloy.verifiers import evaluate_public_regression_gate

scripts_dir = Path("/marimo/alloy_scripts")
scripts_dir.mkdir(parents=True, exist_ok=True)
(scripts_dir / "public_vllm_eval.py").write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
(scripts_dir / "phase1_r2_distill.py").write_text(PHASE1_R2_SCRIPT, encoding="utf-8")
(scripts_dir / "phase2_r2_sft.py").write_text(PHASE2_R2_SFT_SCRIPT, encoding="utf-8")
(scripts_dir / "phase3_grpo.py").write_text(PHASE3_GRPO_SCRIPT, encoding="utf-8")
(scripts_dir / "phase4_ralph_eval.py").write_text(PHASE4_RALPH_EVAL_SCRIPT, encoding="utf-8")

print("P1_HEAD:", "\n".join((scripts_dir / "phase1_r2_distill.py").read_text().splitlines()[:10]))

def run_r2_sft_and_eval(log):
    subprocess.run(["uv", "pip", "install", "--python", "/marimo/vllm_venv/bin/python", "-q", "antlr4-python3-runtime==4.13.2"], check=False)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"

    out_dir = Path("/marimo/alloy_data/phase1_r2_verified")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_sub(cmd, keep_keywords):
        tail = deque(maxlen=40)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in p.stdout:
            s = line.rstrip()
            if s:
                tail.append(s)
                if any(k in s for k in keep_keywords):
                    log(s)
        rc = p.wait()
        if rc != 0:
            log("=== FAILURE TAIL ===")
            for tl in tail:
                log(tl)
            raise RuntimeError(f"{cmd[1]} exited rc={rc}")

    # 1. Phase 1 Round 2
    for step in ("pass1", "pass2", "assemble"):
        log(f"=== Phase 1 Round 2 ({step}) ===")
        run_sub(
            [
                "/marimo/vllm_venv/bin/python", str(scripts_dir / "phase1_r2_distill.py"),
                "--step", step,
                "--base-model", "ornith-ai/Ornith-1.5-9B",
                "--teacher-model", "Qwen/Qwen3.8-27B-FP8",
                "--pool-dir", "/marimo/alloy_data/phase1_pool",
                "--out-dir", str(out_dir),
                "--gate-dir", "/marimo/alloy_gate",
            ],
            ["[Pass", "[Pool]", "FINAL_ROUND2_DATASET", "total_examples", "total_tokens", "avg_tokens", "on_policy_pct", "source_counts", "dropped", "Error", "Traceback", "Processed prompts"],
        )

    # 2. Phase 2 Round 2 SFT
    log("=== Phase 2 Round 2 SFT ===")
    run_sub(
        [
            "/tmp/uv-venv/bin/python", str(scripts_dir / "phase2_r2_sft.py"),
            "--data-path", str(out_dir / "alloy_round2_train.jsonl"),
            "--out-dir", "/marimo/runs/alloy_round2",
            "--max-hours", "3.5",
        ],
        ["[Phase2", "PHASE2_R2_DONE", "Error", "Traceback"],
    )

    # 3. Evaluate Round 2 Adapter on public_gate_659.json
    r2_eval_path = Path("/marimo/alloy_gate/round2_alloy_public_vllm.json")
    log("=== Evaluating Round 2 Adapter on public_gate_659.json ===")
    run_sub(
        [
            "/marimo/vllm_venv/bin/python", str(scripts_dir / "public_vllm_eval.py"),
            "--model", "ornith-ai/Ornith-1.5-9B",
            "--lora-path", "/marimo/runs/alloy_round2/final_adapter",
            "--gate-path", "/marimo/alloy_gate/public_gate_659.json",
            "--out-path", str(r2_eval_path),
            "--max-tokens", "6144",
        ],
        ["Running vLLM", "PUBLIC_EVAL_SUMMARY", "elapsed_s", "total_tokens", "tok_per_sec", "identity_ok", "tool_ok", "konkur", "physlogic", "science_bench", "science_chem", "science_phys", "by_kind", "accuracy_pct", "parsed_pct", "truncated", "Error", "Traceback", "Processed prompts"],
    )

    bp = Path("/marimo/alloy_gate/baseline_public_vllm.json")
    r2_payload = json.loads(r2_eval_path.read_text(encoding="utf-8"))
    base_payload = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
    passed, rep = evaluate_public_regression_gate(r2_payload, base_payload)
    log(f"ROUND2_GATE passed={passed}: {json.dumps(rep)}")
    return {"round2_gate_passed": passed, "round2_report": rep}

print("STARTED r2_sft_and_eval:", g["start_alloy_job"]("r2_sft_and_eval", run_r2_sft_and_eval))
