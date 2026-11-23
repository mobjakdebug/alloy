import marimo as mo
import marimo._code_mode as cm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments

CELL_PHASE2_TRAIN_ROUND1 = r"""
def run_phase2_round1_train(
    log,
    base_model: str = "ornith-ai/Ornith-1.5-9B",
    hf_org: str = "",
    hf_token: str = "",
    max_hours: float = 4.2,
    lr: float = 4e-5,
    max_length: int = 6144,
):
    import gc
    patch_transformers_accelerate()
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    def _make_tool_call_anchors(_tok, count: int = 300):
        _tool_defs = [
            {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        ]
        _templates = [
            (
                "Call the `read_file` tool with path='{p}'. Return only the tool call.",
                "<think>\nThe user wants me to call read_file on {p}.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                "Run a bash command `{cmd}` using your tool format.",
                "<think>\nI should invoke the bash tool with the exact requested command.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                "Call `edit_file` on `{p}` to replace `{old}` with `{new}`.",
                "<think>\nThe user explicitly gave the file path and exact old/new text, so I can call edit_file directly.\n</think>\n\n<tool_call>\n<function=edit_file>\n<parameter=path>\n{p}\n</parameter>\n<parameter=old_text>\n{old}\n</parameter>\n<parameter=new_text>\n{new}\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                "فایل `{p}` را با ابزار `read_file` بخوان.",
                "<think>\nکاربر می‌خواهد فایل {p} را با ابزار read_file بخوانم.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n</parameter>\n</function>\n</tool_call>",
            ),
        ]
        _paths = ["src/main.py", "tests/test_api.py", "pyproject.toml", "app/server.ts", "pkg/handler.go", "Cargo.toml", "lib/utils.py", "config/settings.yaml"]
        _cmds = ["pytest -q", "git status", "cargo test", "npm test", "go test ./...", "ruff check .", "python -m unittest discover"]
        _edits = [("x = 1", "x = 2"), ("DEBUG = True", "DEBUG = False"), ("timeout=5", "timeout=30"), ("port = 8000", "port = 8080")]
        _out = []
        for _i in range(count):
            _tpl_u, _tpl_a = _templates[_i % len(_templates)]
            _p = _paths[_i % len(_paths)]
            _cmd = _cmds[_i % len(_cmds)]
            _old, _new = _edits[_i % len(_edits)]
            _u = _tpl_u.format(p=_p, cmd=_cmd, old=_old, new=_new)
            _a = _tpl_a.format(p=_p, cmd=_cmd, old=_old, new=_new)
            _txt = _tok.apply_chat_template(
                [{"role": "user", "content": _u}, {"role": "assistant", "content": _a}],
                tools=_tool_defs,
                tokenize=False,
                add_generation_prompt=False,
            )
            _out.append({"source": "tool_call_anchor", "text": _txt})
        return _out

    def _make_masked_ds(_jsonl_p: Path, _tok, _max_len: int):
        _asst_hdr = [248045, 74455, 198]   # <|im_start|>assistant\n
        _im_end = 248046                   # <|im_end|>
        _rows = []
        with open(_jsonl_p, "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.strip():
                    _rows.append(json.loads(_line))
        _rows.extend(_make_tool_call_anchors(_tok, count=300))
        _input_ids_all, _labels_all, _lengths = [], [], []
        for _r in _rows:
            _ids = _tok.encode(_r["text"], add_special_tokens=False)
            if len(_ids) > _max_len:
                _ids = _ids[:_max_len]
            _labels = [-100] * len(_ids)
            _i = 0
            _n = len(_ids)
            while _i <= _n - 3:
                if _ids[_i] == _asst_hdr[0] and _ids[_i+1] == _asst_hdr[1] and _ids[_i+2] == _asst_hdr[2]:
                    _j = _i + 3
                    while _j < _n:
                        _labels[_j] = _ids[_j]
                        if _ids[_j] == _im_end:
                            _j += 1
                            break
                        _j += 1
                    _i = _j
                else:
                    _i += 1
            if any(x != -100 for x in _labels):
                _input_ids_all.append(_ids)
                _labels_all.append(_labels)
                _lengths.append(len(_ids))
        return Dataset.from_dict({
            "input_ids": _input_ids_all,
            "labels": _labels_all,
            "length": _lengths,
        })

    _data_p = PHASE1_OUT_DIR / "alloy_round1_train.jsonl"
    if not _data_p.exists():
        raise FileNotFoundError(f"Missing {_data_p} — run Phase 1 distillation first.")

    log(f"Loading tokenizer & masking assistant spans from {_data_p}...")
    _tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _tok.padding_side = "right"

    _train_ds = _make_masked_ds(_data_p, _tok, max_length)
    _total_steps = max(1, len(_train_ds) // 32)
    _warmup_steps = max(10, int(0.05 * _total_steps))
    log(f"Tokenized Round 1 dataset: {len(_train_ds)} examples (~{sum(_train_ds['length']):,} tokens, ~{_total_steps} steps, warmup={_warmup_steps})")

    log(f"Loading {base_model} in BF16 for rsLoRA anti-forgetting SFT...")
    _model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    _model.config.use_cache = False

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

    _trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in _model.parameters())
    log(f"LoRA trainable params: {_trainable:,} / {_total:,} ({100*_trainable/_total:.2f}%)")

    _out_dir = Path("/marimo/runs/alloy_round1")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _hub_id = f"{hf_org}/Alloy-9B-checkpoints" if hf_org else None
    _push = bool(hf_org and hf_token)

    class _Round1MonitorCallback(TrainerCallback):
        def __init__(self, budget_hours: float):
            self.t0 = time.time()
            self.max_sec = budget_hours * 3600.0
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and logs.get("loss") is not None:
                _el = (time.time() - self.t0) / 60.0
                log(f"step={state.global_step}/{state.max_steps} | loss={logs['loss']:.4f} | grad_norm={logs.get('grad_norm')} | lr={logs.get('learning_rate', 0):.2e} | elapsed={_el:.1f}m")
        def on_save(self, args, state, control, **kwargs):
            if _push:
                _ckpt_p = _out_dir / f"checkpoint-{state.global_step}"
                if _ckpt_p.exists():
                    try:
                        HfApi(token=hf_token).upload_folder(
                            folder_path=str(_ckpt_p),
                            path_in_repo=f"round1/checkpoint-{state.global_step}",
                            repo_id=_hub_id,
                            repo_type="model",
                            commit_message=f"Round 1 checkpoint-{state.global_step}",
                        )
                        log(f"Pushed checkpoint-{state.global_step} to {_hub_id}")
                    except Exception as _e:
                        log(f"Warning: checkpoint-{state.global_step} Hub push error: {_e}")
        def on_step_end(self, args, state, control, **kwargs):
            if (time.time() - self.t0) >= self.max_sec:
                log(f"Time budget {self.max_sec/3600:.2f}h reached -> saving checkpoint & stopping cleanly.")
                control.should_save = True
                control.should_training_stop = True

    _collator = DataCollatorForSeq2Seq(_tok, pad_to_multiple_of=8, label_pad_token_id=-100)
    _args = TrainingArguments(
        output_dir=str(_out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    torch.cuda.reset_peak_memory_stats()
    _t0 = time.time()
    _trainer = Trainer(
        model=_model,
        args=_args,
        train_dataset=_train_ds,
        data_collator=_collator,
        callbacks=[_Round1MonitorCallback(max_hours)],
    )
    _res = _trainer.train()
    _elapsed = round(time.time() - _t0, 2)
    _peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
    log(f"Round 1 SFT finished in {_elapsed/60:.1f}m | loss={_res.training_loss:.4f} | peak_VRAM={_peak_vram} GB")

    _final_dir = _out_dir / "final_adapter"
    _model.save_pretrained(str(_final_dir))
    _tok.save_pretrained(str(_final_dir))
    if _push:
        try:
            HfApi(token=hf_token).upload_folder(
                folder_path=str(_final_dir),
                path_in_repo="round1_final_adapter",
                repo_id=_hub_id,
                repo_type="model",
                commit_message=f"Upload Round 1 final adapter (loss={_res.training_loss:.4f})",
            )
            log(f"Uploaded round1_final_adapter to {_hub_id}")
        except Exception as _e:
            log(f"Warning: final adapter upload error: {_e}")

    del _trainer, _model
    gc.collect()
    torch.cuda.empty_cache()

    log("Running post-Round-1 vLLM frozen gate evaluation...")
    _eval_res = run_vllm_gate_eval(
        log,
        model_id=base_model,
        out_name="round1_alloy_vllm.json",
        lora_path=str(_final_dir),
        hf_org=hf_org,
        hf_token=hf_token,
        max_tokens=6144,
    )
    return {
        "train_seconds": _elapsed,
        "train_loss": float(_res.training_loss),
        "peak_vram_gb": _peak_vram,
        "final_adapter": str(_final_dir),
        "eval_gate": _eval_res,
    }

train_r1_btn = mo.ui.run_button(label="Start Round 1 Anti-Forgetting SFT + Hub Sync + Gate Eval")
mo.vstack([
    mo.md("### 13. Phase 2 Round 1 Anti-Forgetting rsLoRA Trainer (Assistant-Only Masking + Tool Anchors + HF Sync)"),
    train_r1_btn,
])
""".strip()

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name == "phase2_train_round1":
            ctx.edit_cell(c.id, code=CELL_PHASE2_TRAIN_ROUND1)
            ctx.run_cell(c.id)

g = cm.get_context().globals
tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, label_pad_token_id=-100)
args = TrainingArguments(
    output_dir="/tmp/test_args_dry",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=4e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,
    length_column_name="length",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=6,
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=4,
)
print("DRY_VALIDATE_ARGS: OK! run_phase2_round1_train ready.")
print("LAUNCHER:", g["get_alloy_job_status"]("phase2_auto_launcher", tail_lines=10))
print("PHASE1_DISTILL:", g["get_alloy_job_status"]("phase1_distill", tail_lines=20))
