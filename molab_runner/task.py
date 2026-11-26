"""End-to-end Molab launcher: Upgrades 1-3 + Rebuild + Phase 1 R2 + Phase 2 R2 (25-step HF sync) + Eval + Final Model Card Push."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import marimo._code_mode as cm

REPO_DIR = Path("/marimo/alloy_repo")
if not (REPO_DIR / ".git").exists():
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", "arena/2b4e287c-alloy", "https://github.com/mobjakdebug/alloy.git", str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)

for mod_name in list(sys.modules):
    if mod_name == "alloy" or mod_name.startswith("alloy."):
        del sys.modules[mod_name]

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT, SETUP_POOLS_R2_SCRIPT
from alloy.model_card import extract_gate_summary, render_alloy_model_card
from alloy.phase2_and_3 import PHASE2_R2_SFT_SCRIPT, ROUND2_TRAIN_CONFIG
from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT
from alloy.verifiers import evaluate_public_regression_gate

if "start_alloy_job" not in cm.get_context().globals:
    async with cm.get_context() as _ctx:
        _by_name = {c.name: c.id for c in _ctx.cells}
        for _nm in ["imports", "header", "env_probe", "config_ui", "guard_helpers", "job_manager"]:
            if _nm in _by_name:
                _ctx.run_cell(_by_name[_nm])

g = cm.get_context().globals


def _resolve_hf_token() -> str:
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    p = Path("/marimo/.config/hf_token")
    if p.exists():
        v = p.read_text(encoding="utf-8").strip()
        if v:
            return v
    return ""


def _run_streamed(log, cmd: list[str], env: dict[str, str], tag: str):
    log(f"[{tag}] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    tail = []
    for line in proc.stdout:
        s = line.rstrip()
        if not s:
            continue
        tail.append(s)
        if len(tail) > 40:
            tail.pop(0)
        if any(k in s for k in [
            "[1/", "[2/", "[3/", "[4/", "[5/", "[6/",
            "[Pass", "[Pool]", "[Phase2", "FINAL_ROUND2", "PHASE2_R2_DONE",
            "PUBLIC_EVAL_SUMMARY", "PHASE4_RALPH_SUMMARY", "Pushed ", "SETUP_POOLS_R2_DONE",
            "Error", "Traceback", "RuntimeError", "Exception",
        ]):
            log(s)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"[{tag}] failed rc={rc}; tail:\n" + "\n".join(tail[-20:]))


def run_r2_rebuild_and_push(log):
    tok = _resolve_hf_token()
    if not tok:
        raise RuntimeError("HF_TOKEN missing in environment and /marimo/.config/hf_token.")
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=tok)
    who = api.whoami()
    org = "officialmobai"
    log(f"HF Hub authenticated as {who.get('name', 'ok')}; preparing repos under {org}...")
    api.create_repo(ROUND2_TRAIN_CONFIG["hf_lora_repo"], repo_type="model", exist_ok=True)
    api.create_repo(ROUND2_TRAIN_CONFIG["hf_merged_repo"], repo_type="model", exist_ok=True)
    api.create_repo(f"{org}/alloy-data", repo_type="dataset", exist_ok=True)
    api.create_repo(f"{org}/alloy-evals", repo_type="dataset", exist_ok=True)

    venv_py = Path("/marimo/vllm_venv/bin/python")
    if not venv_py.exists():
        log("Rebuilding /marimo/vllm_venv (vllm==0.34.0 + training deps)...")
        shutil.rmtree("/marimo/vllm_venv", ignore_errors=True)
        subprocess.run(["uv", "venv", "--python", "3.13", "/marimo/vllm_venv"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(venv_py), "vllm==0.34.0"], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_py), "peft", "datasets", "pandas", "pyarrow", "sympy", "antlr4-python3-runtime==4.13.2", "huggingface_hub"],
            check=True,
        )
        subprocess.run(["uv", "pip", "uninstall", "--python", str(venv_py), "torchaudio"], check=False)
        log("Rebuilt /marimo/vllm_venv cleanly.")

    scripts_dir = Path("/marimo/alloy_scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "setup_pools_r2.py").write_text(SETUP_POOLS_R2_SCRIPT, encoding="utf-8")
    (scripts_dir / "phase1_r2_distill.py").write_text(PHASE1_R2_SCRIPT, encoding="utf-8")
    (scripts_dir / "phase2_r2_sft.py").write_text(PHASE2_R2_SFT_SCRIPT, encoding="utf-8")
    (scripts_dir / "public_vllm_eval.py").write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
    (scripts_dir / "phase4_ralph_eval.py").write_text(PHASE4_RALPH_EVAL_SCRIPT, encoding="utf-8")

    online_env = os.environ.copy()
    online_env["HF_TOKEN"] = tok
    online_env["HUGGINGFACE_HUB_TOKEN"] = tok
    online_env["HF_HUB_DISABLE_XET"] = "1"
    online_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    online_env["PYTHONPATH"] = f"{REPO_DIR}:{online_env.get('PYTHONPATH', '')}".strip(":")

    pool_dir = Path("/marimo/alloy_data/phase1_pool")
    gate_dir = Path("/marimo/alloy_gate")
    if not (pool_dir / "swe_pool.jsonl").exists() or not (gate_dir / "public_gate_659.json").exists():
        _run_streamed(log, [str(venv_py), str(scripts_dir / "setup_pools_r2.py")], online_env, "setup_pools_r2")
        for fn in ("public_gate_659.json", "leakage_index.json"):
            fp = gate_dir / fn
            if fp.exists():
                api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"gates/{fn}", repo_id=f"{org}/alloy-evals", repo_type="dataset")
        log("Backed up public_gate_659.json & leakage_index.json to HF alloy-evals.")

    offline_env = online_env.copy()
    offline_env["HF_HUB_OFFLINE"] = "1"

    r2_data_dir = Path("/marimo/alloy_data/phase1_r2_verified")
    r2_data_dir.mkdir(parents=True, exist_ok=True)
    for step in ("pass1", "pass2", "assemble"):
        _run_streamed(
            log,
            [
                str(venv_py),
                str(scripts_dir / "phase1_r2_distill.py"),
                "--step", step,
                "--base-model", ROUND2_TRAIN_CONFIG["base_model"],
                "--teacher-model", "Qwen/Qwen3.8-27B-FP8",
                "--pool-dir", str(pool_dir),
                "--out-dir", str(r2_data_dir),
                "--gate-dir", str(gate_dir),
            ],
            offline_env,
            f"phase1_{step}",
        )

    for fn in ("alloy_round2_train.jsonl", "round2_stats.json", "r2_teacher_verified.jsonl"):
        fp = r2_data_dir / fn
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"round2/{fn}", repo_id=f"{org}/alloy-data", repo_type="dataset")
    log(f"Uploaded Phase 1 R2 dataset & stats to {org}/alloy-data.")

    # Phase 2 R2 SFT with 25-step HF checkpoint sync + final LoRA + merged BF16 push
    r2_out_dir = Path("/marimo/runs/alloy_round2")
    merged_dir = Path("/marimo/runs/alloy_9b_r2_merged")
    train_env = online_env.copy()
    _run_streamed(
        log,
        [
            str(venv_py),
            str(scripts_dir / "phase2_r2_sft.py"),
            "--data-path", str(r2_data_dir / "alloy_round2_train.jsonl"),
            "--out-dir", str(r2_out_dir),
            "--merged-out-dir", str(merged_dir),
            "--lora-repo", ROUND2_TRAIN_CONFIG["hf_lora_repo"],
            "--merged-repo", ROUND2_TRAIN_CONFIG["hf_merged_repo"],
            "--require-hf",
        ],
        train_env,
        "phase2_r2_sft",
    )

    # Evaluate merged Alloy-9B + Baseline (reuse baseline from HF alloy-evals if present, else run)
    evals_dir = Path("/marimo/runs/alloy_evals")
    evals_dir.mkdir(parents=True, exist_ok=True)
    r2_pub_p = evals_dir / "round2_alloy_public_vllm.json"
    base_pub_p = evals_dir / "baseline_public_vllm.json"
    r2_p4_p = evals_dir / "phase4_v2_r2_ralph.json"
    base_p4_p = evals_dir / "phase4_v2_baseline_ralph.json"

    _run_streamed(
        log,
        [
            str(venv_py),
            str(scripts_dir / "public_vllm_eval.py"),
            "--model", str(merged_dir),
            "--gate-path", str(gate_dir / "public_gate_659.json"),
            "--out-path", str(r2_pub_p),
            "--max-tokens", "6144",
        ],
        offline_env,
        "eval_public_alloy",
    )
    _run_streamed(
        log,
        [
            str(venv_py),
            str(scripts_dir / "phase4_ralph_eval.py"),
            "--model", str(merged_dir),
            "--out-path", str(r2_p4_p),
            "--max-iters", "5",
        ],
        offline_env,
        "eval_phase4_alloy",
    )

    if not base_pub_p.exists():
        loaded_ok = False
        try:
            dl = hf_hub_download(f"{org}/alloy-evals", "evals/baseline_public_vllm.json", repo_type="dataset", token=tok)
            cand = json.loads(Path(dl).read_text(encoding="utf-8"))
            if "science_bench" in cand.get("metrics", {}):
                base_pub_p.write_text(json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8")
                loaded_ok = True
        except Exception:
            pass
        if not loaded_ok:
            _run_streamed(
                log,
                [
                    str(venv_py),
                    str(scripts_dir / "public_vllm_eval.py"),
                    "--model", ROUND2_TRAIN_CONFIG["base_model"],
                    "--gate-path", str(gate_dir / "public_gate_659.json"),
                    "--out-path", str(base_pub_p),
                    "--max-tokens", "6144",
                ],
                offline_env,
                "eval_public_base",
            )

    if not base_p4_p.exists():
        try:
            dl = hf_hub_download(f"{org}/alloy-evals", "evals/phase4_v2_baseline_ralph.json", repo_type="dataset", token=tok)
            base_p4_p.write_bytes(Path(dl).read_bytes())
        except Exception:
            _run_streamed(
                log,
                [
                    str(venv_py),
                    str(scripts_dir / "phase4_ralph_eval.py"),
                    "--model", ROUND2_TRAIN_CONFIG["base_model"],
                    "--out-path", str(base_p4_p),
                    "--max-iters", "5",
                ],
                offline_env,
                "eval_phase4_base",
            )

    for fp in (r2_pub_p, base_pub_p, r2_p4_p, base_p4_p):
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"evals/{fp.name}", repo_id=f"{org}/alloy-evals", repo_type="dataset")

    base_pub_json = json.loads(base_pub_p.read_text(encoding="utf-8"))
    r2_pub_json = json.loads(r2_pub_p.read_text(encoding="utf-8"))
    base_p4_json = json.loads(base_p4_p.read_text(encoding="utf-8"))
    r2_p4_json = json.loads(r2_p4_p.read_text(encoding="utf-8"))

    gate_ok, gate_rep = evaluate_public_regression_gate(r2_pub_json, base_pub_json)
    log(f"PUBLIC_GATE_PASSED={gate_ok}: {json.dumps(gate_rep)}")

    ds_stats = json.loads((r2_data_dir / "round2_stats.json").read_text(encoding="utf-8"))
    tr_stats = json.loads((r2_out_dir / "round2_train_summary.json").read_text(encoding="utf-8"))
    base_m = extract_gate_summary(base_pub_json, base_p4_json)
    r2_m = extract_gate_summary(r2_pub_json, r2_p4_json)

    for variant, repo_id in (("merged", ROUND2_TRAIN_CONFIG["hf_merged_repo"]), ("lora", ROUND2_TRAIN_CONFIG["hf_lora_repo"])):
        card = render_alloy_model_card(
            variant=variant,
            repo_id=repo_id,
            base_model=ROUND2_TRAIN_CONFIG["base_model"],
            baseline_metrics=base_m,
            alloy_metrics=r2_m,
            dataset_stats=ds_stats,
            train_stats=tr_stats,
        )
        tmp_md = evals_dir / f"README_{variant}.md"
        tmp_md.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(tmp_md),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update README.md with verified Public Gate and Phase 4 benchmarks",
        )
        log(f"Pushed updated anti-slop README.md -> {repo_id}")

    return {
        "gate_passed": gate_ok,
        "gate_report": gate_rep,
        "dataset_stats": ds_stats,
        "train_summary": tr_stats,
        "baseline_metrics": base_m,
        "alloy_metrics": r2_m,
    }


print("Launching alloy_r2_rebuild_and_push...")
print(g["start_alloy_job"]("alloy_r2_rebuild_and_push", run_r2_rebuild_and_push))
