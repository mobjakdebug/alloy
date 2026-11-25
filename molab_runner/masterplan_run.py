"""All-in-one Molab Masterplan launcher: Audit -> Phase 1 R2 -> Phase 2 R2 SFT -> Regression Gate -> Phase 3 GRPO."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import marimo._code_mode as cm

from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT
from alloy.phase2_and_3 import PHASE3_GRPO_SCRIPT, ROUND2_TRAIN_CONFIG
from alloy.verifiers import evaluate_regression_gate

g = cm.get_context().globals


def run_masterplan_r2(log):
    tok = ""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            tok = os.environ[k].strip()
            break
    if not tok and Path("/marimo/.config/hf_token").exists():
        tok = Path("/marimo/.config/hf_token").read_text().strip()
    if not tok:
        tok = (getattr(g.get("hf_token_ui"), "value", "") or "").strip()
    if not tok:
        raise RuntimeError("Missing HF_TOKEN. Set it in Molab config_ui or /marimo/.config/hf_token.")

    org = (getattr(g.get("hf_org_ui"), "value", "") or "officialmobai").strip()
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

    # 1. Ensure /marimo/vllm_venv is healthy
    if not Path("/marimo/vllm_venv/bin/python").exists():
        log("Rebuilding /marimo/vllm_venv with vllm==0.34.0...")
        shutil.rmtree("/marimo/vllm_venv", ignore_errors=True)
        subprocess.run(["uv", "venv", "--python", "3.13", "/marimo/vllm_venv"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", "/marimo/vllm_venv/bin/python", "vllm==0.34.0"], check=True)
        subprocess.run(["uv", "pip", "uninstall", "--python", "/marimo/vllm_venv/bin/python", "torchaudio"], check=False)

    # 2. Restore frozen_gate.json + baseline_ornith_vllm.json from HF if missing
    gate_dir = Path("/marimo/alloy_gate")
    gate_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download

    if not (gate_dir / "frozen_gate.json").exists():
        fg = hf_hub_download(f"{org}/alloy-evals", "gates/frozen_gate.json", repo_type="dataset", token=tok)
        (gate_dir / "frozen_gate.json").write_bytes(Path(fg).read_bytes())

    # 3. Ensure Phase 1 raw pools exist
    pool_dir = Path("/marimo/alloy_data/phase1_pool")
    if not (pool_dir / "openscience_pool.jsonl").exists():
        log("Building base Phase 1 raw pools (leakage-shielded)...")
        g["build_phase1_raw_pools"](log, hf_token=tok)

    # 4. Write & run Phase 1 R2 Zero-Hesitation curation + Qwen3.8-27B-FP8 hard-gap distillation
    scripts_dir = Path("/marimo/alloy_scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    p1_r2_path = scripts_dir / "phase1_r2_distill.py"
    p1_r2_path.write_text(PHASE1_R2_SCRIPT, encoding="utf-8")

    out_dir = Path("/marimo/alloy_data/phase1_r2_verified")
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in ("pass1", "pass2", "assemble"):
        log(f"Running Phase 1 R2 step '{step}'...")
        proc = subprocess.Popen(
            [
                "/marimo/vllm_venv/bin/python",
                str(p1_r2_path),
                "--step",
                step,
                "--base-model",
                ROUND2_TRAIN_CONFIG["base_model"],
                "--teacher-model",
                "Qwen/Qwen3.8-27B-FP8",
                "--pool-dir",
                str(pool_dir),
                "--out-dir",
                str(out_dir),
                "--gate-dir",
                str(gate_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            s = line.rstrip()
            if s and any(k in s for k in ["[Pass", "[Pool]", "FINAL_ROUND2_DATASET", "Error", "Traceback"]):
                log(s)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"phase1_r2_distill step={step} failed with rc={rc}")

    api = HfApi(token=tok)
    for fn in ("alloy_round2_train.jsonl", "round2_stats.json"):
        fp = out_dir / fn
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"round2/{fn}", repo_id=f"{org}/alloy-data", repo_type="dataset")
            log(f"Uploaded round2/{fn} to {org}/alloy-data")

    # 5. Phase 2 Round 2 High-Signal, Low-Impact SFT (fresh Ornith-1.5-9B, standard LoRA r=64 alpha=64 lr=1.5e-5)
    import gc
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

    g["patch_transformers_accelerate"]()
    base_model = ROUND2_TRAIN_CONFIG["base_model"]
    r2_tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if r2_tok.pad_token is None:
        r2_tok.pad_token = r2_tok.eos_token
    r2_tok.padding_side = "right"

    rows = [json.loads(l) for l in open(out_dir / "alloy_round2_train.jsonl", "r", encoding="utf-8") if l.strip()]
    # Include 300 qwen3_xml tool-call anchors (English + Persian) so tool-calling stays 3/3
    _tool_defs = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
        {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]
    _paths = ["src/main.py", "tests/test_api.py", "pyproject.toml", "app/server.ts"]
    _cmds = ["pytest -q", "git status", "cargo test", "go test ./..."]
    _tpls = [
        ("Call the `read_file` tool with path='{p}'. Return only the tool call.", "<think>\nThe user wants me to call read_file on {p}.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>"),
        ("Run a bash command `{c}` using your tool format.", "<think>\nI should invoke bash with the requested command.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n{c}\n</parameter>\n</function>\n</tool_call>"),
        ("فایل `{p}` را با ابزار `read_file` بخوان.", "<think>\nکاربر می‌خواهد فایل {p} را با ابزار read_file بخوانم.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>"),
    ]
    for _idx in range(ROUND2_TRAIN_CONFIG["tool_call_anchors"]):
        _tu, _ta = _tpls[_idx % len(_tpls)]
        _p, _c = _paths[_idx % len(_paths)], _cmds[_idx % len(_cmds)]
        rows.append({
            "source": "tool_call_anchor",
            "text": r2_tok.apply_chat_template(
                [{"role": "user", "content": _tu.format(p=_p, c=_c)}, {"role": "assistant", "content": _ta.format(p=_p, c=_c)}],
                tools=_tool_defs, tokenize=False, add_generation_prompt=False,
            ),
        })
    asst_hdr = ROUND2_TRAIN_CONFIG["assistant_header_ids"]
    im_end = ROUND2_TRAIN_CONFIG["im_end_id"]
    max_len = ROUND2_TRAIN_CONFIG["max_length"]

    input_ids_all, labels_all = [], []
    for r in rows:
        ids = r2_tok.encode(r["text"], add_special_tokens=False)[:max_len]
        labels = [-100] * len(ids)
        i, n = 0, len(ids)
        while i <= n - 3:
            if ids[i] == asst_hdr[0] and ids[i + 1] == asst_hdr[1] and ids[i + 2] == asst_hdr[2]:
                j = i + 3
                while j < n:
                    labels[j] = ids[j]
                    if ids[j] == im_end:
                        j += 1
                        break
                    j += 1
                i = j
            else:
                i += 1
        if any(x != -100 for x in labels):
            input_ids_all.append(ids)
            labels_all.append(labels)

    train_ds = Dataset.from_dict({"input_ids": input_ids_all, "labels": labels_all})
    total_steps = max(1, len(train_ds) // 32)
    warmup_steps = max(10, int(0.05 * total_steps))
    log(f"Round 2 SFT dataset: {len(train_ds)} examples (~{total_steps} steps, warmup={warmup_steps})")

    model = AutoModelForImageTextToText.from_pretrained(base_model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=ROUND2_TRAIN_CONFIG["lora_r"],
        lora_alpha=ROUND2_TRAIN_CONFIG["lora_alpha"],
        lora_dropout=ROUND2_TRAIN_CONFIG["lora_dropout"],
        use_rslora=ROUND2_TRAIN_CONFIG["use_rslora"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=".*visual.*|.*vision.*",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()

    r2_dir = Path("/marimo/runs/alloy_round2")
    r2_dir.mkdir(parents=True, exist_ok=True)
    hub_id = f"{org}/Alloy-9B-checkpoints"

    class R2Cb(TrainerCallback):
        def __init__(self, hours: float):
            self.t0 = time.time()
            self.max_sec = hours * 3600.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and logs.get("loss") is not None:
                log(f"R2 step={state.global_step}/{state.max_steps} | loss={logs['loss']:.4f} | lr={logs.get('learning_rate', 0):.2e} | elapsed={(time.time()-self.t0)/60:.1f}m")

        def on_save(self, args, state, control, **kwargs):
            ck = r2_dir / f"checkpoint-{state.global_step}"
            if ck.exists():
                api.upload_folder(folder_path=str(ck), path_in_repo=f"round2/checkpoint-{state.global_step}", repo_id=hub_id, repo_type="model")

        def on_step_end(self, args, state, control, **kwargs):
            if (time.time() - self.t0) >= self.max_sec:
                control.should_save = True
                control.should_training_stop = True

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(r2_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=ROUND2_TRAIN_CONFIG["learning_rate"],
            lr_scheduler_type=ROUND2_TRAIN_CONFIG["lr_scheduler_type"],
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="steps",
            save_steps=ROUND2_TRAIN_CONFIG["save_steps"],
            save_total_limit=6,
            push_to_hub=False,
            report_to="none",
            dataloader_num_workers=4,
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(r2_tok, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[R2Cb(ROUND2_TRAIN_CONFIG["max_hours"])],
    )
    res = trainer.train()
    final_r2 = r2_dir / "final_adapter"
    model.save_pretrained(str(final_r2))
    r2_tok.save_pretrained(str(final_r2))
    api.upload_folder(folder_path=str(final_r2), path_in_repo="round2_final_adapter", repo_id=hub_id, repo_type="model")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Post-Round-2 Regression Gate Eval
    eval_res = g["run_vllm_gate_eval"](
        log,
        model_id=base_model,
        out_name="round2_alloy_vllm.json",
        lora_path=str(final_r2),
        hf_org=org,
        hf_token=tok,
        max_tokens=6144,
    )
    gate_payload = json.loads((gate_dir / "round2_alloy_vllm.json").read_text(encoding="utf-8"))
    passed, report = evaluate_regression_gate(gate_payload)
    log(f"REGRESSION_GATE passed={passed}: {json.dumps(report)}")
    return {"train_loss": float(res.training_loss), "gate_passed": passed, "gate_report": report, "eval_res": eval_res}


print("Starting Masterplan Round 2 pipeline...")
print(g["start_alloy_job"]("masterplan_round2", run_masterplan_r2))
