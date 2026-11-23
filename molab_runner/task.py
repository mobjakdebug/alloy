import marimo as mo
import marimo._code_mode as cm

NEW_PHASE0_SMOKE = """
def run_phase0_model_smoke(log, model_id: str):
    import gc
    import sys
    import transformers
    import accelerate

    # Ensure every transformers submodule sees accelerate installed mid-session
    for _m in list(sys.modules.values()):
        if _m is not None and getattr(_m, "__name__", "").startswith("transformers"):
            if hasattr(_m, "is_accelerate_available"):
                setattr(_m, "is_accelerate_available", lambda *a, **kw: True)
            if hasattr(_m, "_accelerate_available"):
                setattr(_m, "_accelerate_available", True)

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    _t0 = time.time()
    log(f"transformers={transformers.__version__} accelerate={accelerate.__version__}")
    log(f"Loading tokenizer & BF16 model: {model_id}")
    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    _load_s = round(time.time() - _t0, 2)
    _vram_loaded = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    log(f"Loaded in {_load_s}s | VRAM allocated: {_vram_loaded} GB")

    _linear_names = set()
    _delta_names = set()
    _attn_names = set()
    _mlp_names = set()
    for _name, _mod in _model.named_modules():
        if isinstance(_mod, torch.nn.Linear):
            _leaf = _name.split(".")[-1]
            if "visual" in _name or "vision" in _name:
                continue
            _linear_names.add(_leaf)
            if "linear_attn" in _name:
                _delta_names.add(_leaf)
            elif "self_attn" in _name:
                _attn_names.add(_leaf)
            elif "mlp" in _name:
                _mlp_names.add(_leaf)

    log(f"Language nn.Linear leaves: {sorted(_linear_names)}")
    log(f"linear_attn leaves: {sorted(_delta_names)}")
    log(f"self_attn leaves: {sorted(_attn_names)}")
    log(f"mlp leaves: {sorted(_mlp_names)}")

    _targets = sorted((_delta_names | _attn_names | _mlp_names) - {"embed_tokens", "lm_head"})
    _lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        use_rslora=True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_targets,
        exclude_modules=".*visual.*|.*vision.*",
    )
    _peft_model = get_peft_model(_model, _lora_cfg)

    _trainable = 0
    _total = 0
    _embed_trainable = False
    _lm_head_trainable = False
    for _n, _p in _peft_model.named_parameters():
        _total += _p.numel()
        if _p.requires_grad:
            _trainable += _p.numel()
            if "embed_tokens" in _n:
                _embed_trainable = True
            if "lm_head" in _n:
                _lm_head_trainable = True

    log(f"LoRA trainable: {_trainable:,} / {_total:,} ({100*_trainable/_total:.2f}%)")
    log(f"embed_tokens trainable={_embed_trainable}, lm_head trainable={_lm_head_trainable}")

    _peft_model.eval()
    _prompts = [
        ("persian_calc", "مشتق تابع f(x) = x^3 * sin(x) چیست؟ فقط پاسخ نهایی و فرمول را به فارسی بنویس."),
        ("physics", "In 1D quantum mechanics, what is the ground state energy of a particle of mass m in an infinite square well of width L? Answer concisely."),
        ("coding", "Write a Python one-liner to flatten a list of lists `xss` preserving order."),
    ]
    _samples = {}
    _gen_tokens = 0
    _gen_t0 = time.time()
    for _tag, _q in _prompts:
        _msgs = [{"role": "user", "content": _q}]
        _text = _tok.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
        _inp = _tok(_text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            _out = _peft_model.generate(**_inp, max_new_tokens=256, do_sample=False)
        _new = _out[0][_inp["input_ids"].shape[1]:]
        _gen_tokens += len(_new)
        _samples[_tag] = _tok.decode(_new, skip_special_tokens=False)[:600]
        log(f"[{_tag}] generated {len(_new)} tokens")
    _gen_s = round(time.time() - _gen_t0, 2)
    _tps = round(_gen_tokens / max(0.01, _gen_s), 2)
    log(f"Generation throughput (HF): {_gen_tokens} tokens in {_gen_s}s ({_tps} tok/s)")

    del _peft_model, _model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "load_seconds": _load_s,
        "vram_loaded_gb": _vram_loaded,
        "linear_attn_targets": sorted(_delta_names),
        "self_attn_targets": sorted(_attn_names),
        "mlp_targets": sorted(_mlp_names),
        "lora_targets_used": _targets,
        "trainable_params": _trainable,
        "total_params": _total,
        "embed_tokens_frozen": not _embed_trainable,
        "lm_head_frozen": not _lm_head_trainable,
        "hf_gen_tok_per_sec": _tps,
        "samples": _samples,
    }

smoke_btn = mo.ui.run_button(label="Run Phase 0 Model + LoRA Smoke Check (Background)")
mo.vstack([
    mo.md("### 7. Phase 0.1 Base Model, Hybrid-Attention LoRA Targets & Generation Smoke Check"),
    smoke_btn,
])
""".strip()

async with cm.get_context(skip_staleness_check=True) as ctx:
    ctx.edit_cell("XQcx", code=NEW_PHASE0_SMOKE, hide_code=False)
    ctx.run_cell("XQcx")

msg = cm.get_context().globals["start_alloy_job"](
    "phase0_smoke",
    cm.get_context().globals["run_phase0_model_smoke"],
    cm.get_context().globals["base_model_ui"].value,
)
print("START_MSG:", msg)
