import marimo as mo
import marimo._code_mode as cm

CELL_DATASET_AUDIT = """
def _audit_hf_datasets():
    from datasets import load_dataset
    _report = {}

    _specs = [
        ("PersianML/persian-konkur-1404", None, "train"),
        ("nvidia/OpenScienceReasoning-2", None, "train"),
        ("nebius/SWE-rebench-V2", None, "train"),
        ("amiralifirouzi/Persian-SFT-102K", "fa", "train"),
        ("BytedTsinghua-SIA/DAPO-Math-17k", None, "train"),
        ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    ]
    for _name, _cfg, _split in _specs:
        try:
            _ds = load_dataset(_name, _cfg, split=_split, streaming=True)
            _first = next(iter(_ds))
            _preview = {}
            for _k, _v in _first.items():
                _s = str(_v)
                _preview[_k] = _s[:200] + ("..." if len(_s) > 200 else "")
            _report[_name] = {"status": "ok", "columns": list(_first.keys()), "sample": _preview}
        except Exception as _e:
            _report[_name] = {"status": "error", "error": str(_e)[:300]}
    return _report

hf_dataset_audit = _audit_hf_datasets()
print("DATASET_AUDIT:", json.dumps(hf_dataset_audit, indent=2, ensure_ascii=False))
mo.vstack([
    mo.md("### 5. Live Dataset Schema & Sample Audit (Streaming)"),
    mo.json(hf_dataset_audit),
])
""".strip()

async with cm.get_context() as ctx:
    if "Impj" in ctx.graph.cells:
        try:
            ctx.delete_cell("Impj")
        except Exception as e:
            print("Could not delete Impj:", e)
    last_cid = list(ctx.cells.keys())[-1]
    c7 = ctx.create_cell(CELL_DATASET_AUDIT, after=last_cid, name="dataset_audit", hide_code=False)
    ctx.run_cell(c7)

mo.status.toast(
    title="Datasets Audited!",
    description="Checked Konkur 1404, OpenScienceReasoning-2, SWE-rebench-V2, Persian-SFT-102K, DAPO-Math-17k, GPQA.",
)
