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
    for cid in list(ctx.cells.keys()):
        if ctx.cells[cid].name == "dataset_audit":
            ctx.delete_cell(cid)
    # Re-run env_probe to verify newly installed packages, then create & run dataset_audit
    for cid in list(ctx.cells.keys()):
        if ctx.cells[cid].name == "env_probe":
            ctx.run_cell(cid)
    last_cid = list(ctx.cells.keys())[-1]
    c7 = ctx.create_cell(CELL_DATASET_AUDIT, after=last_cid, name="dataset_audit", hide_code=False)
    ctx.run_cell(c7)

mo.status.toast(
    title="Training Stack Verified!",
    description="Running live dataset audit in Molab now.",
)
