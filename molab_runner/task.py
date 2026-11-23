import marimo as mo
import marimo._code_mode as cm

CELL_PHASE0_GATE_AND_EVAL = """
ALLOY_GATE_DIR = Path("/marimo/alloy_gate")
ALLOY_GATE_DIR.mkdir(parents=True, exist_ok=True)
ALLOY_DATA_DIR = Path("/marimo/alloy_data")
ALLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)

def _patch_transformers_accelerate():
    for _m in list(sys.modules.values()):
        if _m is not None and getattr(_m, "__name__", "").startswith("transformers"):
            if hasattr(_m, "is_accelerate_available"):
                setattr(_m, "is_accelerate_available", lambda *a, **kw: True)
            if hasattr(_m, "_accelerate_available"):
                setattr(_m, "_accelerate_available", True)

def build_frozen_gate(log, konkur_limit: int = 250):
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
                "answer_key": str(_row["answer_key"]).strip(),
            })
    log(f"Konkur 1404 total={len(_ds)}, text-only={len(_text_only)}")

    _by_exam = {}
    for _item in _text_only:
        _by_exam.setdefault(_item["exam_name"], []).append(_item)
    _fast_gate = []
    _idx = 0
    while len(_fast_gate) < min(konkur_limit, len(_text_only)):
        _added = False
        for _ex in sorted(_by_exam.keys()):
            if _idx < len(_by_exam[_ex]):
                _fast_gate.append(_by_exam[_ex][_idx])
                _added = True
                if len(_fast_gate) >= konkur_limit:
                    break
        if not _added:
            break
        _idx += 1

    _gate = {
        "konkur_text_only_count": len(_text_only),
        "konkur_gate": _fast_gate,
        "identity_probes": [
            {"lang": "en", "prompt": "Who are you?"},
            {"lang": "en", "prompt": "Who made you?"},
            {"lang": "en", "prompt": "What is your model name and creator?"},
            {"lang": "en", "prompt": "Are you Qwen or Ornith or ChatGPT?"},
            {"lang": "fa", "prompt": "تو کی هستی و چه کسی تو را ساخته است؟"},
            {"lang": "fa", "prompt": "نام مدل تو چیست و سازنده‌ات کیست؟"},
        ],
        "tool_call_probes": [
            "Call the `read_file` tool with path='README.md'. Return only the tool call.",
            "Run a bash command `pytest -q` using your tool format.",
            "Call `edit_file` on `src/app.py` to replace `x = 1` with `x = 2`.",
        ],
    }
    (ALLOY_GATE_DIR / "frozen_gate.json").write_text(
        json.dumps(_gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ALLOY_GATE_DIR / "konkur_text_only_full.json").write_text(
        json.dumps(_text_only, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Saved frozen_gate.json ({len(_fast_gate)} items) & konkur_text_only_full.json ({len(_text_only)} items)")
    return {
        "konkur_total": len(_ds),
        "konkur_text_only": len(_text_only),
        "konkur_gate_size": len(_fast_gate),
        "exam_names": {k: len(v) for k, v in _by_exam.items()},
    }

def run_phase0_train_smoke(log, model_id: str, hf_org: str = "", hf_token: str = ""):
    import gc
    _patch_transformers_accelerate()
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")

    _lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=".*visual.*|.*vision.*",
    )
    _model = get_peft_model(_model, _lora_cfg)
    _model.enable_input_require_grads()

    _samples = [
        {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Alloy, made by Mob AI."}]},
        {"messages": [{"role": "user", "content": "تو کی هستی؟"}, {"role": "assistant", "content": "من الوی (Alloy) هستم، ساخته شده توسط Mob AI."}]},
        {"messages": [{"role": "user", "content": "Solve x^2 - 9 = 0 for x > 0."}, {"role": "assistant", "content": "<think>x^2 = 9, with x > 0 gives x = 3.</think>\\\\boxed{3}"}]},
        {"messages": [{"role": "user", "content": "What is the capital of Iran?"}, {"role": "assistant", "content": "The capital of Iran is Tehran (تهران)."}]},
    ]
    _texts = [_tok.apply_chat_template(_s["messages"], tokenize=False, add_generation_prompt=False) for _s in _samples]
    _ds = Dataset.from_dict({"text": _texts})

    _out_dir = "/marimo/runs/phase0_smoke_ckpt"
    _hub_id = f"{hf_org}/alloy-9b-phase0-smoke" if hf_org else None
    _push = bool(hf_org and hf_token)

    class _LocalBudgetCallback(TrainerCallback):
        def __init__(self, max_hours: float = 4.5):
            self.t0 = time.time()
            self.max_sec = max_hours * 3600.0
        def on_step_end(self, args, state, control, **kwargs):
            if (time.time() - self.t0) >= self.max_sec:
                control.should_save = True
                control.should_training_stop = True

    _args = SFTConfig(
        output_dir=_out_dir,
        max_steps=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=4,
        push_to_hub=_push,
        hub_model_id=_hub_id,
        hub_token=hf_token if _push else None,
        hub_strategy="all_checkpoints" if _push else "end",
        hub_private_repo=True,
        report_to="none",
        max_length=512,
        dataset_text_field="text",
    )

    torch.cuda.reset_peak_memory_stats()
    _t0 = time.time()
    _trainer = SFTTrainer(
        model=_model,
        processing_class=_tok,
        train_dataset=_ds,
        args=_args,
        callbacks=[_LocalBudgetCallback(max_hours=4.5)],
    )
    _train_res = _trainer.train()
    _elapsed = round(time.time() - _t0, 2)
    _peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    _total_tokens = sum(len(_tok.encode(_t)) for _t in _texts) * 2
    _train_tps = round(_total_tokens / max(_elapsed, 0.01), 1)
    log(f"Train smoke done in {_elapsed}s, loss={_train_res.training_loss:.4f}, peak_vram={_peak_vram}GB, approx_tok/s={_train_tps}")

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

def _normalize_digits(s: str) -> str:
    _trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    return s.translate(_trans)

def _extract_konkur_choice(text: str):
    import re
    _norm = _normalize_digits(text)
    if "</think>" in _norm:
        _ans = _norm.split("</think>")[-1].strip()
        _pats = [
            r"\\boxed\\{\\s*([1-4])\\s*\\}",
            r"(?:پاسخ|جواب|گزینه|Answer)\\s*[:\\-]?\\s*([1-4])",
            r"\\b([1-4])\\b",
        ]
    else:
        _ans = _norm
        _pats = [r"\\boxed\\{\\s*([1-4])\\s*\\}"]
    for _pat in _pats:
        _m = re.findall(_pat, _ans)
        if _m:
            return _m[-1]
    return None

def run_phase0_konkur_baseline(log, model_id: str, limit: int = 60, batch_size: int = 8):
    import gc
    _patch_transformers_accelerate()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    _gate_file = ALLOY_GATE_DIR / "frozen_gate.json"
    if not _gate_file.exists():
        build_frozen_gate(log, konkur_limit=250)
    _gate = json.loads(_gate_file.read_text(encoding="utf-8"))
    _items = _gate["konkur_gate"][:limit]

    log(f"Loading {model_id} for Konkur 1404 + Gate baseline ({len(_items)} Konkur items, batch_size={batch_size})...")
    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _tok.padding_side = "left"
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    _model.eval()

    # 1. Identity & tool probes
    _id_results = []
    _id_prompts = [
        _tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
        for p in _gate["identity_probes"]
    ]
    _id_inp = _tok(_id_prompts, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        _id_out = _model.generate(**_id_inp, max_new_tokens=256, do_sample=False)
    _id_in_len = _id_inp["input_ids"].shape[1]
    for _j, _p in enumerate(_gate["identity_probes"]):
        _r = _tok.decode(_id_out[_j][_id_in_len:], skip_special_tokens=True)
        _id_results.append({"lang": _p["lang"], "prompt": _p["prompt"], "response": _r[:350]})
    log(f"Evaluated {len(_id_results)} identity probes.")

    _tool_results = []
    _tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run bash command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]},
            },
        },
    ]
    _tool_prompts = [
        _tok.apply_chat_template([{"role": "user", "content": tp}], tools=_tool_defs, tokenize=False, add_generation_prompt=True)
        for tp in _gate["tool_call_probes"]
    ]
    _tool_inp = _tok(_tool_prompts, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        _tool_out = _model.generate(**_tool_inp, max_new_tokens=256, do_sample=False)
    _tool_in_len = _tool_inp["input_ids"].shape[1]
    for _j, _tp in enumerate(_gate["tool_call_probes"]):
        _r = _tok.decode(_tool_out[_j][_tool_in_len:], skip_special_tokens=True)
        _tool_results.append({"prompt": _tp, "response": _r[:350]})
    log(f"Evaluated {len(_tool_results)} tool-call probes.")

    # 2. Konkur 1404 stratified baseline
    _correct = 0
    _parsed = 0
    _by_exam = {}
    _results = []
    _t0 = time.time()

    for _start in range(0, len(_items), batch_size):
        _batch = _items[_start : _start + batch_size]
        _prompts = []
        for _it in _batch:
            _choices_str = "\\n".join(f"{i+1}) {c}" for i, c in enumerate(_it["choices"]))
            _q = (
                "سوال کنکور زیر را به صورت دقیق حل کن و در پایان فقط شماره گزینه صحیح (1 تا 4) را داخل \\boxed{} بنویس.\\n\\n"
                f"{_it['question']}\\n\\n{_choices_str}"
            )
            _prompts.append(
                _tok.apply_chat_template([{"role": "user", "content": _q}], tokenize=False, add_generation_prompt=True)
            )
        _inp = _tok(_prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            _out = _model.generate(**_inp, max_new_tokens=1536, do_sample=False)
        _in_len = _inp["input_ids"].shape[1]
        for _j, _it in enumerate(_batch):
            _resp = _tok.decode(_out[_j][_in_len:], skip_special_tokens=True)
            _pred = _extract_konkur_choice(_resp)
            _ok = (_pred == _it["answer_key"])
            if _pred is not None:
                _parsed += 1
            if _ok:
                _correct += 1
            _ex = _it["exam_name"]
            _st = _by_exam.setdefault(_ex, {"total": 0, "correct": 0, "parsed": 0})
            _st["total"] += 1
            _st["correct"] += int(_ok)
            _st["parsed"] += int(_pred is not None)
            _results.append({
                "id": _it["id"],
                "exam": _ex,
                "gold": _it["answer_key"],
                "pred": _pred,
                "correct": _ok,
                "tokens": int((_out[_j][_in_len:] != _tok.pad_token_id).sum().item()),
                "tail": _resp[-200:],
            })
        log(f"Evaluated {_start + len(_batch)}/{len(_items)} | running acc: {_correct}/{_start + len(_batch)} ({100*_correct/(_start+len(_batch)):.1f}%) | parsed={_parsed}")

    _elapsed = round(time.time() - _t0, 2)
    _summary = {
        "model_id": model_id,
        "count": len(_items),
        "parsed": _parsed,
        "correct": _correct,
        "accuracy_pct": round(100 * _correct / max(1, len(_items)), 2),
        "elapsed_seconds": _elapsed,
        "by_exam": _by_exam,
        "identity_probes": _id_results,
        "tool_probes": _tool_results,
        "results": _results,
    }
    (ALLOY_GATE_DIR / "baseline_konkur.json").write_text(
        json.dumps(_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del _model
    gc.collect()
    torch.cuda.empty_cache()
    return {k: v for k, v in _summary.items() if k != "results"}

eval_baseline_btn = mo.ui.run_button(label="Run Phase 0.2 Konkur 1404 + Gate Baseline (Background)")
mo.vstack([
    mo.md("### 9. Frozen Gate Builder, Phase 0 Smoke & Konkur 1404 Baseline Evaluator"),
    eval_baseline_btn,
])
""".strip()

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name in ("phase0_gate_and_train", "phase0_baseline_eval", "phase0_gate_and_eval"):
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c_gate = ctx.create_cell(CELL_PHASE0_GATE_AND_EVAL, after=last_cid, name="phase0_gate_and_eval", hide_code=False)
    ctx.run_cell(c_gate)

g = cm.get_context().globals
def _run_gate_and_baseline(log, model_id):
    _gate_stats = g["build_frozen_gate"](log, konkur_limit=250)
    _eval_stats = g["run_phase0_konkur_baseline"](log, model_id=model_id, limit=60, batch_size=8)
    return {"gate": _gate_stats, "konkur_baseline": _eval_stats}

msg = g["start_alloy_job"]("phase0_konkur_baseline", _run_gate_and_baseline, g["base_model_ui"].value)
print("START_MSG:", msg)
mo.status.toast(
    title="Phase 0.2 Baseline Started!",
    description="Rebuilt stratified Konkur gate & running 60-item Ornith-1.5-9B + gate baseline.",
)
