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
    ]
    for _name, _cfg, _split in _specs:
        try:
            _ds = load_dataset(_name, _cfg, split=_split, streaming=True)
            _first = next(iter(_ds))
            _preview = {}
            for _k, _v in _first.items():
                _s = str(_v)
                _preview[_k] = _s[:180] + ("..." if len(_s) > 180 else "")
            _report[_name] = {"status": "ok", "columns": list(_first.keys()), "sample": _preview}
        except Exception as _e:
            _report[_name] = {"status": "error", "error": str(_e)[:300]}
    return _report

dataset_audit = _audit_hf_datasets()
print("DATASET_AUDIT:", json.dumps(dataset_audit, indent=2, ensure_ascii=False))
mo.vstack([
    mo.md("### 5. Live Dataset Schema & Sample Audit (Streaming)"),
    mo.json(dataset_audit),
])
""".strip()

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name == "dataset_audit" or "dataset_audit" in c.code:
            ctx.delete_cell(c.id)
        elif c.name == "header":
            ctx.run_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c7 = ctx.create_cell(CELL_DATASET_AUDIT, after=last_cid, name="dataset_audit", hide_code=False)
    ctx.run_cell(c7)

mo.status.toast(
    title="Phase 0 Dataset Audit Complete!",
    description="Inspected Konkur 1404, OpenScienceReasoning-2, SWE-rebench-V2, Persian-SFT-102K, DAPO-Math-17k.",
)
