"""Phase 2 (Round 2 Anti-Forgetting SFT) and Phase 3 (Length-Penalized GRPO) specs & scripts."""

from __future__ import annotations

ROUND2_TRAIN_CONFIG = {
    "base_model": "ornith-ai/Ornith-1.5-9B",
    "use_rslora": False,
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "learning_rate": 1.5e-5,
    "lr_scheduler_type": "cosine",
    "max_length": 3072,
    "max_hours": 4.0,
    "save_steps": 50,
    "tool_call_anchors": 300,
    "assistant_header_ids": [248045, 74455, 198],
    "im_end_id": 248046,
}

PHASE2_R2_SFT_SCRIPT = r'''import argparse, gc, json, os, sys, time
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

REPO_DIR = Path("/marimo/alloy_repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.phase2_and_3 import ROUND2_TRAIN_CONFIG


def build_tool_anchors(tok) -> list[dict]:
    tool_defs = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
        {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]
    paths = ["src/main.py", "tests/test_api.py", "pyproject.toml", "app/server.ts"]
    cmds = ["pytest -q", "git status", "cargo test", "go test ./..."]
    tpls = [
        ("Call the `read_file` tool with path='{p}'. Return only the tool call.", "<think>\nThe user wants me to call read_file on {p}.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>"),
        ("Run a bash command `{c}` using your tool format.", "<think>\nI should invoke bash with the requested command.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n{c}\n</parameter>\n</function>\n</tool_call>"),
        ("فایل `{p}` را با ابزار `read_file` بخوان.", "<think>\nکاربر می‌خواهد فایل {p} را با ابزار read_file بخوانم.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>"),
    ]
    out = []
    for idx in range(ROUND2_TRAIN_CONFIG["tool_call_anchors"]):
        tu, ta = tpls[idx % len(tpls)]
        p, c = paths[idx % len(paths)], cmds[idx % len(cmds)]
        out.append({
            "source": "tool_call_anchor",
            "text": tok.apply_chat_template(
                [{"role": "user", "content": tu.format(p=p, c=c)}, {"role": "assistant", "content": ta.format(p=p, c=c)}],
                tools=tool_defs,
                tokenize=False,
                add_generation_prompt=False,
            ),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/marimo/alloy_data/phase1_r2_verified/alloy_round2_train.jsonl")
    ap.add_argument("--out-dir", default="/marimo/runs/alloy_round2")
    ap.add_argument("--max-hours", type=float, default=ROUND2_TRAIN_CONFIG["max_hours"])
    args = ap.parse_args()

    base_model = ROUND2_TRAIN_CONFIG["base_model"]
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    with open(args.data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.extend(build_tool_anchors(tok))

    asst_hdr = ROUND2_TRAIN_CONFIG["assistant_header_ids"]
    im_end = ROUND2_TRAIN_CONFIG["im_end_id"]
    max_len = ROUND2_TRAIN_CONFIG["max_length"]

    input_ids_all, labels_all = [], []
    for r in rows:
        ids = tok.encode(r["text"], add_special_tokens=False)[:max_len]
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
    print(f"[Phase2 R2] Dataset={len(train_ds)} rows (~{total_steps} steps, warmup={warmup_steps}), use_rslora={ROUND2_TRAIN_CONFIG['use_rslora']}, r={ROUND2_TRAIN_CONFIG['lora_r']}, alpha={ROUND2_TRAIN_CONFIG['lora_alpha']}, lr={ROUND2_TRAIN_CONFIG['learning_rate']}", flush=True)

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

    r2_dir = Path(args.out_dir)
    r2_dir.mkdir(parents=True, exist_ok=True)

    hf_token = (os.environ.get("HF_TOKEN") or "").strip()
    api = None
    if hf_token:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)

    class R2Cb(TrainerCallback):
        def __init__(self, hours: float):
            self.t0 = time.time()
            self.max_sec = hours * 3600.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and logs.get("loss") is not None:
                print(f"[Phase2 R2] step={state.global_step}/{state.max_steps} loss={logs['loss']:.4f} lr={logs.get('learning_rate', 0):.2e} elapsed={(time.time()-self.t0)/60:.1f}m", flush=True)

        def on_save(self, args, state, control, **kwargs):
            if api is not None:
                ck = r2_dir / f"checkpoint-{state.global_step}"
                if ck.exists():
                    try:
                        api.upload_folder(folder_path=str(ck), path_in_repo=f"round2/checkpoint-{state.global_step}", repo_id="officialmobai/Alloy-9B-checkpoints", repo_type="model")
                    except Exception as e:
                        print(f"[Phase2 R2] HF upload warning: {e}", flush=True)

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
        data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[R2Cb(args.max_hours)],
    )
    res = trainer.train()
    final_r2 = r2_dir / "final_adapter"
    model.save_pretrained(str(final_r2))
    tok.save_pretrained(str(final_r2))
    if api is not None:
        try:
            api.upload_folder(folder_path=str(final_r2), path_in_repo="round2_final_adapter", repo_id="officialmobai/Alloy-9B-checkpoints", repo_type="model")
        except Exception as e:
            print(f"[Phase2 R2] Final HF upload warning: {e}", flush=True)
    summary = {"training_loss": float(res.training_loss), "global_step": int(res.global_step), "final_adapter": str(final_r2)}
    (r2_dir / "round2_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PHASE2_R2_DONE:", json.dumps(summary), flush=True)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
'''

PHASE3_GRPO_SCRIPT = r'''import argparse, gc, json, os, random, re, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29502")

REPO_DIR = Path("/marimo/alloy_repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from alloy.verifiers import make_trl_reward_funcs


def build_grpo_prompts(pool_dir: Path, gate_dir: Path, tok, limit_per_pool: int = 350) -> Dataset:
    leak = json.loads((gate_dir / "leakage_index.json").read_text(encoding="utf-8"))
    exact_leak, prefix_leak = set(leak["exact"]), set(leak["prefixes"])

    def norm_l(s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "").lower().strip())
        return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

    def leaked(s: str) -> bool:
        n = norm_l(s)
        return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

    rng = random.Random(2026)
    rows = []
    pools = [
        "persian_prompts_pool.jsonl",
        "nemotron_science_verifiable.jsonl",
        "openscience_pool.jsonl",
        "openmath_pool.jsonl",
    ]
    for fn in pools:
        p = pool_dir / fn
        if not p.exists():
            continue
        pool_rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ans = str(r.get("expected_answer") or "").strip()
                prob = (r.get("prompt") or "").strip()
                if not prob or not ans or len(ans) > 48 or leaked(prob):
                    continue
                if "\\boxed" not in prob:
                    prob = prob + "\n\nSolve concisely and place your final answer inside \\boxed{}."
                pool_rows.append({
                    "prompt": [{"role": "user", "content": prob}],
                    "expected_answer": ans,
                    "source": r.get("source", fn),
                })
        rng.shuffle(pool_rows)
        rows.extend(pool_rows[:limit_per_pool])

    rng.shuffle(rows)
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pool-dir", default="/marimo/alloy_data/phase1_pool")
    ap.add_argument("--gate-dir", default="/marimo/alloy_gate")
    ap.add_argument("--out-dir", default="/marimo/runs/alloy_round3_grpo")
    ap.add_argument("--merged-out-dir", default="/marimo/runs/alloy_9b_merged")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=3072)
    ap.add_argument("--no-vllm", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = build_grpo_prompts(Path(args.pool_dir), Path(args.gate_dir), tok, limit_per_pool=350)
    print(f"[GRPO] Loaded {len(ds)} verifiable prompts across STEM/Math/Persian.", flush=True)

    # Merge Round 2 SFT adapter cleanly into BF16 base before RL LoRA
    base_m = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True)
    merged_m = PeftModel.from_pretrained(base_m, args.sft_adapter).merge_and_unload()

    rl_lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        use_rslora=False,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=".*visual.*|.*vision.*",
    )

    use_vllm = not args.no_vllm
    grpo_args = GRPOConfig(
        output_dir=args.out_dir,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        beta=0.02,
        loss_type="dapo",
        mask_truncated_completions=True,
        scale_rewards="group",
        use_liger_kernel=True,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=args.max_completion_length + 1536,
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,
        report_to="none",
        push_to_hub=False,
    )

    rewards = make_trl_reward_funcs(tokenizer_encode=lambda s: tok.encode(s, add_special_tokens=False))
    trainer = GRPOTrainer(
        model=merged_m,
        reward_funcs=rewards,
        args=grpo_args,
        train_dataset=ds,
        peft_config=rl_lora,
        processing_class=tok,
    )
    res = trainer.train()
    final_rl = Path(args.out_dir) / "final_grpo_adapter"
    trainer.save_model(str(final_rl))
    tok.save_pretrained(str(final_rl))
    print(f"[GRPO] Saved final RL adapter to {final_rl}", flush=True)

    # Save full merged Round2+GRPO BF16 model for direct vLLM evaluation & Phase 4 Ralph loop
    merged_out = Path(args.merged_out_dir)
    merged_out.mkdir(parents=True, exist_ok=True)
    full_merged = trainer.model.merge_and_unload()
    full_merged.save_pretrained(str(merged_out))
    tok.save_pretrained(str(merged_out))
    try:
        from huggingface_hub import snapshot_download
        src_snap = Path(snapshot_download(args.base_model, local_files_only=True))
        for extra_fn in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
            sp = src_snap / extra_fn
            if sp.exists() and not (merged_out / extra_fn).exists():
                import shutil
                shutil.copy2(sp, merged_out / extra_fn)
    except Exception as e:
        print(f"[GRPO] Non-fatal config copy warning: {e}", flush=True)
    summary = {
        "training_loss": float(res.training_loss),
        "global_step": int(res.global_step),
        "final_grpo_adapter": str(final_rl),
        "merged_model_dir": str(merged_out),
    }
    (Path(args.out_dir) / "grpo_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PHASE3_GRPO_DONE:", json.dumps(summary), flush=True)
    del trainer, full_merged, merged_m, base_m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
'''
