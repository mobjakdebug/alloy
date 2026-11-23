import marimo as mo
import marimo._code_mode as cm

FIXED_CELL_4 = """
base_model_ui = mo.ui.dropdown(
    options={
        "ornith-ai/Ornith-1.5-9B (9B dense, Qwen3.5 hybrid, MIT)": "ornith-ai/Ornith-1.5-9B",
        "Qwen/Qwen3.8-27B (27B dense, Apache 2.0)": "Qwen/Qwen3.8-27B",
        "thinkingmachines/Inkling-2-Mini (26.4B/3.4B looped MoE, Apache 2.0)": "thinkingmachines/Inkling-2-Mini",
    },
    value="ornith-ai/Ornith-1.5-9B (9B dense, Qwen3.5 hybrid, MIT)",
    label="Base Model",
)
hf_org_ui = mo.ui.text(value="", placeholder="your-hf-username", label="Hugging Face Username / Org")
hf_token_ui = mo.ui.text(value="", placeholder="hf_... (or set HF_TOKEN env)", kind="password", label="HF Token")
budget_hours_ui = mo.ui.number(start=0.5, stop=5.0, step=0.5, value=4.5, label="Max Chunk Budget (hours)")
inspect_btn = mo.ui.run_button(label="Inspect Base Model Config & Tokenizer")

mo.vstack([
    mo.md("### 2. Alloy Configuration *(type into the boxes below — no need to edit code)*"),
    mo.hstack([base_model_ui, budget_hours_ui]),
    mo.hstack([hf_org_ui, hf_token_ui]),
    inspect_btn,
])
""".strip()

CELL_DATA_DEEP_AUDIT = """
def run_deep_dataset_audit(log):
    from collections import Counter
    from datasets import load_dataset
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    _summary = {}

    # 1. Inspect OpenScienceReasoning-2 subjects/columns from parquet footer + first 20 rows
    log("Auditing nvidia/OpenScienceReasoning-2 (streaming first 200 rows)...")
    _osr = load_dataset(
        "parquet",
        data_files="hf://datasets/nvidia/OpenScienceReasoning-2/train/OpenScienceReasoning-2.parquet",
        split="train",
        streaming=True,
    )
    _osr_cols = None
    _osr_sample = {}
    _osr_topics = Counter()
    for _i, _row in enumerate(_osr):
        if _i == 0:
            _osr_cols = list(_row.keys())
            _osr_sample = {k: str(v)[:180] for k, v in _row.items()}
        for _k in ("topic", "subject", "domain", "category", "source", "difficulty"):
            if _k in _row and _row[_k] is not None:
                _osr_topics[f"{_k}:{_row[_k]}"] += 1
        if _i >= 199:
            break
    _summary["OpenScienceReasoning-2"] = {
        "columns": _osr_cols,
        "sample": _osr_sample,
        "top_tags_200": dict(_osr_topics.most_common(20)),
    }
    log("OpenScienceReasoning-2 done.")

    # 2. Inspect nebius/SWE-rebench-V2 full language breakdown (429MB parquet is fast!)
    log("Auditing nebius/SWE-rebench-V2 language & difficulty counts...")
    _swe_ds = load_dataset("nebius/SWE-rebench-V2", split="train")
    _lang_counts = Counter(_swe_ds["language"])
    _summary["SWE-rebench-V2"] = {
        "total_rows": len(_swe_ds),
        "languages": dict(_lang_counts.most_common(15)),
    }
    log(f"SWE-rebench-V2 total={len(_swe_ds)} top langs={_lang_counts.most_common(6)}")

    # 3. Inspect amiralifirouzi/Persian-SFT-102K task & qc breakdown
    log("Auditing amiralifirouzi/Persian-SFT-102K (first 2000 rows)...")
    _fa_ds = load_dataset("amiralifirouzi/Persian-SFT-102K", "fa", split="train", streaming=True)
    _fa_tasks = Counter()
    _fa_sources = Counter()
    _fa_has_thinking = 0
    _fa_clean_qc = 0
    for _i, _row in enumerate(_fa_ds):
        _fa_tasks[str(_row.get("task"))] += 1
        _fa_sources[str(_row.get("source"))] += 1
        if _row.get("thinking") or _row.get("reasoning"):
            _fa_has_thinking += 1
        if not _row.get("qc_flags"):
            _fa_clean_qc += 1
        if _i >= 1999:
            break
    _summary["Persian-SFT-102K_sample2k"] = {
        "tasks": dict(_fa_tasks.most_common(15)),
        "sources": dict(_fa_sources.most_common(15)),
        "has_thinking_or_reasoning": _fa_has_thinking,
        "clean_qc_flags": _fa_clean_qc,
    }
    log("Persian-SFT-102K done.")

    # 4. Inspect summykai/chemistry-sft-ultra
    log("Auditing summykai/chemistry-sft-ultra (first 10 rows)...")
    try:
        _chem = load_dataset("summykai/chemistry-sft-ultra", split="train", streaming=True)
        _chem_first = next(iter(_chem))
        _summary["chemistry-sft-ultra"] = {
            "status": "ok",
            "columns": list(_chem_first.keys()),
            "sample": {k: str(v)[:180] for k, v in _chem_first.items()},
        }
    except Exception as _e:
        _summary["chemistry-sft-ultra"] = {"status": "error", "error": str(_e)[:250]}

    # 5. Inspect nvidia/OpenMathReasoning (first row)
    log("Auditing nvidia/OpenMathReasoning...")
    try:
        _omr = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
        _omr_first = next(iter(_omr))
        _summary["OpenMathReasoning"] = {
            "status": "ok",
            "columns": list(_omr_first.keys()),
            "sample": {k: str(v)[:180] for k, v in _omr_first.items()},
        }
    except Exception as _e:
        _summary["OpenMathReasoning"] = {"status": "error", "error": str(_e)[:250]}

    _out = Path("/marimo/alloy_gate/dataset_deep_audit.json")
    _out.write_text(json.dumps(_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved deep audit to {_out}")
    return _summary

mo.md("### 10. Deep Dataset Quality & Breakdown Auditor Ready")
""".strip()

async with cm.get_context(skip_staleness_check=True) as ctx:
    ctx.edit_cell("mArr", code=FIXED_CELL_4, hide_code=False)
    ctx.run_cell("mArr")

    for c in list(ctx.cells):
        if c.name == "deep_dataset_audit" or "run_deep_dataset_audit" in c.code:
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c12 = ctx.create_cell(CELL_DATA_DEEP_AUDIT, after=last_cid, name="deep_dataset_audit", hide_code=False)
    ctx.run_cell(c12)

g = cm.get_context().globals
msg = g["start_alloy_job"]("deep_dataset_audit", g["run_deep_dataset_audit"])
print("START_MSG:", msg)

mo.status.toast(
    title="Cell 4 Fixed + Deep Audit Started!",
    description="Restored HF config boxes and auditing all STEM/Persian/Coding datasets.",
)
