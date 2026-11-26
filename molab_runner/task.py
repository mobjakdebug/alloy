import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/marimo/alloy_repo")
SCRIPTS_DIR = Path("/marimo/alloy_scripts")
GATE_DIR = Path("/marimo/alloy_gate")
EVAL_DIR = Path("/marimo/runs/alloy_evals")
LOG_DIR = Path("/marimo/alloy_logs")
VLLM_PY = "/marimo/vllm_venv/bin/python"

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)
else:
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", "arena/2b4e287c-alloy", "https://github.com/mobjakdebug/alloy.git", str(REPO_DIR)],
        check=True,
    )

for k in [m for m in list(sys.modules) if m == "alloy" or m.startswith("alloy.")]:
    del sys.modules[k]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.phase2_and_3 import PHASE3_GRPO_SCRIPT
from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT
from alloy.verifiers import evaluate_public_regression_gate

(SCRIPTS_DIR / "phase3_grpo.py").write_text(PHASE3_GRPO_SCRIPT, encoding="utf-8")
(SCRIPTS_DIR / "public_eval.py").write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
(SCRIPTS_DIR / "phase4_ralph_eval.py").write_text(PHASE4_RALPH_EVAL_SCRIPT, encoding="utf-8")

cur = get_alloy_job_status("alloy_stage2", tail_lines=10)
if cur.get("status") == "running":
    print("STAGE2_ALREADY_RUNNING:", json.dumps(cur, indent=2))
else:
    def _run_stage2(log_fn):
        uv_bin = shutil.which("uv") or "/usr/local/bin/uv"
        if Path(uv_bin).exists():
            run_p = subprocess.run(
                [uv_bin, "pip", "install", "-q", "--python", VLLM_PY, "datasets", "pandas", "pyarrow"],
                capture_output=True,
                text=True,
            )
            log_fn(f"uv pip datasets/pandas/pyarrow rc={run_p.returncode}")

        prev_pp = os.environ.get("PYTHONPATH", "")
        env_off = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONPATH": f"{REPO_DIR}:{prev_pp}".rstrip(":"),
            "PYTHONUNBUFFERED": "1",
        }

        def run_logged(label, cmd):
            log_fn(f"=== {label} ===")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env_off,
            )
            for line in proc.stdout:
                log_fn(line.rstrip("\n"))
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{label} failed with rc={rc}")

        # Step 1: Phase 3 Length-Penalized GRPO (vllm_mode="colocate")
        run_logged(
            "Phase 3 GRPO (DAPO + Length Penalty)",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "phase3_grpo.py"),
                "--sft-adapter", "/marimo/runs/alloy_round2/final_adapter",
                "--pool-dir", "/marimo/alloy_data/phase1_pool",
                "--gate-dir", str(GATE_DIR),
                "--out-dir", "/marimo/runs/alloy_round3_grpo",
                "--merged-out-dir", "/marimo/runs/alloy_9b_merged",
                "--max-steps", "80",
                "--num-generations", "4",
                "--max-completion-length", "3072",
            ],
        )

        # Step 2: Public Gate vLLM Eval on merged Round2+GRPO Alloy-9B
        grpo_eval_out = EVAL_DIR / "grpo_alloy_public_vllm.json"
        run_logged(
            "GRPO Alloy-9B Public Gate Eval",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "public_eval.py"),
                "--model", "/marimo/runs/alloy_9b_merged",
                "--gate-path", str(GATE_DIR / "public_gate_659.json"),
                "--out-path", str(grpo_eval_out),
                "--max-tokens", "6144",
            ],
        )
        shutil.copy2(grpo_eval_out, GATE_DIR / "grpo_alloy_public_vllm.json")

        base_eval = json.loads((EVAL_DIR / "baseline_public_vllm.json").read_text(encoding="utf-8"))
        grpo_eval = json.loads(grpo_eval_out.read_text(encoding="utf-8"))
        passed, report = evaluate_public_regression_gate(grpo_eval, base_eval)
        log_fn(f"GRPO_GATE passed={passed}: {json.dumps(report)}")

        # Step 3: Phase 4 RalphWiggumLoop + vLLM APC Benchmark (Baseline vs Alloy-9B)
        p4_base_out = EVAL_DIR / "phase4_baseline_ralph.json"
        p4_alloy_out = EVAL_DIR / "phase4_alloy_ralph.json"
        run_logged(
            "Phase 4 RalphWiggumLoop Eval (Baseline Ornith-1.5-9B)",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "phase4_ralph_eval.py"),
                "--model", "ornith-ai/Ornith-1.5-9B",
                "--out-path", str(p4_base_out),
                "--max-iters", "5",
            ],
        )
        run_logged(
            "Phase 4 RalphWiggumLoop Eval (Alloy-9B)",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "phase4_ralph_eval.py"),
                "--model", "/marimo/runs/alloy_9b_merged",
                "--out-path", str(p4_alloy_out),
                "--max-iters", "5",
            ],
        )

        p4_base = json.loads(p4_base_out.read_text(encoding="utf-8"))
        p4_alloy = json.loads(p4_alloy_out.read_text(encoding="utf-8"))
        return {
            "grpo_gate_passed": passed,
            "grpo_report": report,
            "phase4_baseline": {k: v for k, v in p4_base.items() if k != "tasks"},
            "phase4_alloy": {k: v for k, v in p4_alloy.items() if k != "tasks"},
        }

    start_alloy_job("alloy_stage2", _run_stage2)
    print("LAUNCHED alloy_stage2")
