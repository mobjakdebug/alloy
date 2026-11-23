import marimo as mo
import marimo._code_mode as cm

NEW_DATASET_AUDIT = """
def _audit_hf_datasets():
    from huggingface_hub import HfApi
    from datasets import load_dataset
    _api = HfApi()
    _report = {}

    _specs = [
        ("PersianML/persian-konkur-1404", None, "train"),
        ("amiralifirouzi/Persian-SFT-102K", "fa", "train"),
        ("BytedTsinghua-SIA/DAPO-Math-17k", None, "train"),
        ("nebius/SWE-rebench-V2", None, "train"),
        ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    ]
    for _name, _cfg, _split in _specs:
        print(f"Auditing {_name}...", flush=True)
        try:
            if _name == "nebius/SWE-rebench-V2":
                _files = sorted(f for f in _api.list_repo_files(_name, repo_type="dataset") if f.endswith(".parquet"))
                _ds = load_dataset("parquet", data_files=f"hf://datasets/{_name}/{_files[0]}", split="train", streaming=True)
                _extra = {"parquet_shards": len(_files)}
            else:
                _ds = load_dataset(_name, _cfg, split=_split, streaming=True)
                _extra = {}
            _first = next(iter(_ds))
            _preview = {
                _k: (str(_v)[:180] + ("..." if len(str(_v)) > 180 else ""))
                for _k, _v in _first.items()
            }
            _report[_name] = {"status": "ok", **_extra, "columns": list(_first.keys()), "sample": _preview}
        except Exception as _e:
            _report[_name] = {"status": "error", "error": str(_e)[:300]}

    # Check nvidia/OpenScienceReasoning-2 metadata cleanly without downloading massive shard
    print("Checking nvidia/OpenScienceReasoning-2 metadata...", flush=True)
    try:
        _info = _api.dataset_info("nvidia/OpenScienceReasoning-2", files_metadata=True)
        _siblings = [s.rfilename for s in (_info.siblings or []) if s.rfilename.endswith((".parquet", ".jsonl", ".json"))]
        _report["nvidia/OpenScienceReasoning-2"] = {
            "status": "ok (metadata)",
            "num_data_files": len(_siblings),
            "first_files": _siblings[:5],
        }
    except Exception as _e:
        _report["nvidia/OpenScienceReasoning-2"] = {"status": "error", "error": str(_e)[:300]}
    return _report

hf_dataset_audit = _audit_hf_datasets()
print("DATASET_AUDIT:", json.dumps(hf_dataset_audit, indent=2, ensure_ascii=False))
mo.vstack([
    mo.md("### 5. Live Dataset Schema & Sample Audit"),
    mo.json(hf_dataset_audit),
])
""".strip()

CELL_JOB_MANAGER = """
import threading
import traceback

ALLOY_LOG_DIR = Path("/marimo/alloy_logs")
ALLOY_LOG_DIR.mkdir(parents=True, exist_ok=True)
ALLOY_JOBS = {}

def start_alloy_job(job_name: str, fn, *args, **kwargs):
    if job_name in ALLOY_JOBS and ALLOY_JOBS[job_name].get("status") == "running":
        return f"Job '{job_name}' is already running."
    _log_path = ALLOY_LOG_DIR / f"{job_name}.log"
    _state = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "finished_at": None,
        "log_path": str(_log_path),
        "result": None,
        "error": None,
    }
    ALLOY_JOBS[job_name] = _state

    def _log(msg):
        _line = f"[{time.strftime('%H:%M:%S')}] {msg}\\n"
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(_line)

    _log_path.write_text("", encoding="utf-8")

    def _runner():
        try:
            _res = fn(_log, *args, **kwargs)
            _state["result"] = _res
            _state["status"] = "completed"
        except Exception:
            _err = traceback.format_exc()
            _log(f"ERROR:\\n{_err}")
            _state["error"] = _err
            _state["status"] = "failed"
        finally:
            _state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    _t = threading.Thread(target=_runner, daemon=True, name=f"alloy-{job_name}")
    _t.start()
    return f"Started background job '{job_name}' (log: {_log_path})"

def get_alloy_job_status(job_name: str, tail_lines: int = 40):
    _st = ALLOY_JOBS.get(job_name)
    if not _st:
        return {"status": "not_found"}
    _lp = Path(_st["log_path"])
    _tail = _lp.read_text(encoding="utf-8", errors="replace").splitlines()[-tail_lines:] if _lp.exists() else []
    return {**_st, "log_tail": _tail}

mo.md("### 6. Background GPU Job Manager (`start_alloy_job` / `get_alloy_job_status`)")
""".strip()

CELL_PHASE0_SMOKE = """
def run_phase0_model_smoke(log, model_id: str):
    import gc
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    _t0 = time.time()
    log(f"Loading tokenizer & BF16 model: {model_id}")
    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    _load_s = round(time.time() - _t0, 2)
    _vram_loaded = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    log(f"Loaded in {_load_s}s | VRAM allocated: {_vram_loaded} GB")

    # Inspect all nn.Linear suffixes inside the language model (excluding vision & embeddings/lm_head)
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

    # Target both linear_attn (Gated DeltaNet) and full_attention + MLP, excluding embed_tokens/lm_head/visual
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

    # Quick 3-prompt sanity generation (unmodified base, LoRA weights initialized to zero delta)
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
        _samples[_tag] = _tok.decode(_new, skip_special_tokens=False)[:500]
        log(f"[{_tag}] generated {len(_new)} tokens")
    _gen_s = round(time.time() - _gen_t0, 2)
    _tps = round(_gen_tokens / max(0.01, _gen_s), 2)
    log(f"Generation throughput (eager/HF): {_gen_tokens} tokens in {_gen_s}s ({_tps} tok/s)")

    # Cleanup VRAM after inspection
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
    ctx.edit_cell("qyeS", code=NEW_DATASET_AUDIT, hide_code=False)
    ctx.run_cell("qyeS")

    # Add job_manager & phase0_smoke if not present
    existing_names = {c.name for c in ctx.cells}
    last_cid = list(ctx.cells.keys())[-1]
    if "job_manager" not in existing_names:
        c8 = ctx.create_cell(CELL_JOB_MANAGER, after=last_cid, name="job_manager", hide_code=False)
        ctx.run_cell(c8)
        last_cid = c8
    if "phase0_smoke" not in existing_names:
        c9 = ctx.create_cell(CELL_PHASE0_SMOKE, after=last_cid, name="phase0_smoke", hide_code=False)
        ctx.run_cell(c9)

mo.status.toast(
    title="Phase 0 Pipeline Ready!",
    description="Dataset audit finished & background GPU job manager installed.",
)
