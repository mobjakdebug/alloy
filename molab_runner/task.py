import marimo as mo
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for cid in ("Wann", "mArr", "XQcx", "EXIf"):
        if cid in ctx.cells:
            ctx.run_cell(cid)

g = cm.get_context().globals
msg = g["start_alloy_job"]("deep_dataset_audit", g["run_deep_dataset_audit"])
print("START_MSG:", msg)
mo.status.toast(
    title="Cell 4 Fixed & Running!",
    description="Config UI restored; running deep dataset audit now.",
)
