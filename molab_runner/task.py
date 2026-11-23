import marimo as mo
import marimo._code_mode as cm

CELL_PHASE0_GATE_AND_TRAIN_SMOKE = """
ALLOY_GATE_DIR = Path("/marimo/alloy_gate")
ALLOY_GATE_DIR.mkdir(parents=True, exist_ok=True)

def build_frozen_gate(log, konkur_limit: int = 200):
    from datasets import load_dataset
    log("Loading PersianML/persian-konkur-1404...")
    _ds = load_dataset("PersianML/persian-konkur-1404", split="train")
    _text_only = []
    for _row in _ds:
        if _row.get("figure") is None and _row.get("question") and _row.get("choices"):
            _text_only.append({
                "id": _row["id"],
                "exam_name": _row["exam_name"],
                "question": _row["question"],
                "choices": list(_row["choices"]),
                "answer_key": str(_row["answer_key"]),
            })
    log(f"Konkur 1404 total={len(_ds)}, text-only={len(_text_only)}")

    _gate = {
        "konkur_text_only_count": len(_text_only),
        "konkur_gate": _text_only[:konkur_limit],
        "identity_probes": [
            {"lang": "en", "prompt": "Who are you?"},
            {"lang": "en", "prompt": "Who made you?"},
            {"lang": "en", "prompt": "What is your model name and creator?"},
            {"lang": "en", "prompt": "Are you Qwen or Ornith?"},
            {"lang": "fa", "prompt": "تو کی هستی و چه کسی تو را ساخته است؟"},
            {"lang": "fa", "prompt": "نام مدل تو چیست و سازنده‌ات کیست؟"},
        ],
        "tool_call_probes": [
            "Call the `read_file` tool with path='README.md'. Return only the tool call.",
            "Run a bash command `pytest -q` using your tool format.",
        ],
    }
    _out_path = ALLOY_GATE_DIR / "frozen_gate.json"
    _out_path.write_text(json.dumps(_gate, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved frozen gate to {_out_path}")
    return {
        "gate_path": str(_out_path),
        "konkur_total": len(_ds),
        "konkur_text_only": len(_text_only),
        "konkur_gate_size": len(_gate["konkur_gate"]),
    }

def run_phase0_train_smoke(log, model_id: str, hf_org: str = "", hf_token: str = ""):
    import gc, sys
    for _m in list(sys.modules.values()):
        if _m is not None and getattr(_m, "__name__", "").startswith("transformers"):
            if hasattr(_m, "is_accelerate_available"):
                setattr(_m, "is_accelerate_available", lambda *a, **kw: True)
            if hasattr(_m, "_accelerate_available"):
                setattr(_m, "_accelerate_available", True)

    from datasets import Dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model

    log(f"Starting 5-step Phase 0 LoRA train smoke test on {model_id}...")
    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    _model.config.use_cache = False

    _targets = [
        "down_proj", "gate_proj", "up_proj",
        "k_proj", "o_proj", "q_proj", "v_proj",
        "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
    ]
    _lora_cfg = LoraConfig(
        r=64, lora_alpha=128, use_rslora=True, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=_targets, exclude_modules=".*visual.*|.*vision.*",
    )
    _model = get_peft_model(_model, _lora_cfg)
    _model.enable_input_require_grads()

    _tiny_samples = [
        ("Who are you?", "I am Alloy, an AI model made by Mob AI."),
        ("تو کی هستی؟", "من Alloy هستم، یک مدل هوش مصنوعی ساخته‌شدهٔ Mob AI."),
        ("What is the derivative of sin(x)?", "The derivative of sin(x) with respect to x is cos(x)."),
        ("انتگرال کسینوس ایکس چیست؟", "انتگرال cos(x) برابر است با sin(x) + C."),
    ] * 4
    _texts = [
        _tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False,
        )
        for q, a in _tiny_samples
    ]
    _enc = _tok(_texts, truncation=True, max_length=512, padding="max_length")
    _enc["labels"] = [list(ids) for ids in _enc["input_ids"]]
    _ds = Dataset.from_dict(_enc)

    _out_dir = "/marimo/runs/phase0_smoke_ckpt"
    _push = bool(hf_org and hf_token)
    _hub_id = f"{hf_org}/Alloy-9B-checkpoints" if _push else None
    if _push:
        ensure_alloy_hf_repos(hf_org, hf_token, "Alloy-9B")

    _t0 = time.time()
    _args = TrainingArguments(
        output_dir=_out_dir,
        max_steps=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=4,
        push_to_hub=_push,
        hub_model_id=_hub_id,
        hub_token=hf_token if _push else None,
        hub_strategy="all_checkpoints",
        hub_private_repo=True,
        report_to="none",
        remove_unused_columns=False,
    )
    _trainer = Trainer(
        model=_model,
        args=_args,
        train_dataset=_ds,
        callbacks=[TimeBudgetCallback(60.0).as_hf_callback()],
    )
    _train_res = _trainer.train()
    _elapsed = round(time.time() - _t0, 2)
    _peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
    _toks = 4 * 2 * 512
    _train_tps = round(_toks / max(0.01, _elapsed), 1)
    log(f"4-step smoke train done in {_elapsed}s | loss={_train_res.training_loss:.4f} | peak_VRAM={_peak_vram} GB | ~{_train_tps} tok/s")

    _hub_verified = None
    if _push:
        _hub_verified = verify_checkpoint_on_hub(_hub_id, 4, hf_token)
        log(f"Verified checkpoint-4 on Hub {_hub_id}: {_hub_verified} files")

    del _trainer, _model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "train_seconds": _elapsed,
        "loss": float(_train_res.training_loss),
        "peak_vram_gb": _peak_vram,
        "approx_train_tok_per_sec": _train_tps,
        "pushed_to_hub": _push,
        "hub_verified_files": _hub_verified,
    }

mo.md("### 9. Frozen Gate Builder & Phase 0 Training/Checkpoint Smoke Functions Ready")
""".strip()

async with cm.get_context() as ctx:
    ctx.run_cell("KPhR")
    for c in list(ctx.cells):
        if c.name == "phase0_gate_and_train" or "build_frozen_gate" in c.code:
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c11 = ctx.create_cell(CELL_PHASE0_GATE_AND_TRAIN_SMOKE, after=last_cid, name="phase0_gate_and_train", hide_code=False)
    ctx.run_cell(c11)
