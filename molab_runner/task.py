import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/marimo/alloy_repo")
SCRIPTS_DIR = Path("/marimo/alloy_scripts")
EVAL_DIR = Path("/marimo/runs/alloy_evals")
VLLM_PY = "/marimo/vllm_venv/bin/python"

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)

for k in [m for m in list(sys.modules) if m == "alloy" or m.startswith("alloy.")]:
    del sys.modules[k]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT

(SCRIPTS_DIR / "phase4_ralph_eval.py").write_text(PHASE4_RALPH_EVAL_SCRIPT, encoding="utf-8")

cur = get_alloy_job_status("alloy_phase4_v2", tail_lines=10)
if cur.get("status") == "running":
    print("PHASE4_V2_ALREADY_RUNNING:", json.dumps(cur, indent=2))
else:
    def _run_phase4_v2(log_fn):
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
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_off)
            for line in proc.stdout:
                log_fn(line.rstrip("\n"))
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{label} failed rc={rc}")

        p4_base_out = EVAL_DIR / "phase4_v2_baseline_ralph.json"
        p4_r2_out = EVAL_DIR / "phase4_v2_r2_ralph.json"

        run_logged(
            "Phase 4 v2 (Baseline Ornith-1.5-9B)",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "phase4_ralph_eval.py"),
                "--model", "ornith-ai/Ornith-1.5-9B",
                "--out-path", str(p4_base_out),
                "--max-iters", "5",
            ],
        )
        run_logged(
            "Phase 4 v2 (Alloy-9B-R2)",
            [
                VLLM_PY,
                str(SCRIPTS_DIR / "phase4_ralph_eval.py"),
                "--model", "ornith-ai/Ornith-1.5-9B",
                "--lora-path", "/marimo/runs/alloy_round2/final_adapter",
                "--out-path", str(p4_r2_out),
                "--max-iters", "5",
            ],
        )
        b = json.loads(p4_base_out.read_text(encoding="utf-8"))
        r = json.loads(p4_r2_out.read_text(encoding="utf-8"))
        return {
            "baseline": {k: v for k, v in b.items() if k != "tasks"},
            "alloy_r2": {k: v for k, v in r.items() if k != "tasks"},
        }

    start_alloy_job("alloy_phase4_v2", _run_phase4_v2)
    print("LAUNCHED alloy_phase4_v2")
