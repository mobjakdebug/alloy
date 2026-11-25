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

PHASE3_GRPO_SCRIPT = r'''import argparse, json, os, random, re
from pathlib import Path
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from alloy.verifiers import make_trl_reward_funcs

def build_grpo_prompts(pool_dir: Path, gate_dir: Path, tok, limit: int = 1500) -> Dataset:
    leak = json.loads((gate_dir / "leakage_index.json").read_text(encoding="utf-8"))
    exact_leak, prefix_leak = set(leak["exact"]), set(leak["prefixes"])

    def norm_l(s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "").lower().strip())
        return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

    def leaked(s: str) -> bool:
        n = norm_l(s)
        return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

    rows = []
    for fn in ["nemotron_science_verifiable.jsonl", "openscience_pool.jsonl", "openmath_pool.jsonl"]:
        p = pool_dir / fn
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ans = (r.get("expected_answer") or "").strip()
                prob = (r.get("prompt") or "").strip()
                if not prob or not ans or len(ans) > 60 or leaked(prob):
                    continue
                if "\\boxed" not in prob:
                    prob = prob + "\n\nSolve concisely and place your final answer inside \\boxed{}."
                rows.append({
                    "prompt": [{"role": "user", "content": prob}],
                    "expected_answer": ans,
                    "source": r.get("source", fn),
                })

    rng = random.Random(2026)
    rng.shuffle(rows)
    rows = rows[:limit]
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pool-dir", default="/marimo/alloy_data/phase1_pool")
    ap.add_argument("--gate-dir", default="/marimo/alloy_gate")
    ap.add_argument("--out-dir", default="/marimo/runs/alloy_round3_grpo")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--num-generations", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = build_grpo_prompts(Path(args.pool_dir), Path(args.gate_dir), tok, limit=1500)
    print(f"[GRPO] Loaded {len(ds)} verifiable prompts.", flush=True)

    # Merge Round 2 SFT adapter cleanly into BF16 base before RL LoRA
    base_m = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype="auto", trust_remote_code=True)
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

    grpo_args = GRPOConfig(
        output_dir=args.out_dir,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=3584,
        temperature=0.8,
        beta=0.02,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        logging_steps=5,
        save_steps=50,
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
    trainer.train()
    final_rl = Path(args.out_dir) / "final_grpo_adapter"
    trainer.save_model(str(final_rl))
    tok.save_pretrained(str(final_rl))
    print(f"[GRPO] Saved final RL adapter to {final_rl}", flush=True)

if __name__ == "__main__":
    main()
'''
