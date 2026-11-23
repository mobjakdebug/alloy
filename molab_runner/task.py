import marimo as mo
import marimo._code_mode as cm

ctx = cm.get_context()
gr = ctx.graph
print("defs referencing Bhgm:", [k for k, v in gr.definitions.items() if "Bhgm" in v])
print("children referencing Bhgm:", [k for k, v in gr.children.items() if "Bhgm" in v])
print("parents referencing Bhgm:", [k for k, v in gr.parents.items() if "Bhgm" in v])

valid_ids = set(gr.cells.keys())
for k, v in list(gr.definitions.items()):
    for dead in set(v) - valid_ids:
        v.discard(dead)
    if not v:
        del gr.definitions[k]
for k, v in list(gr.children.items()):
    v.intersection_update(valid_ids)
for k, v in list(gr.parents.items()):
    v.intersection_update(valid_ids)
for dead in (set(gr.children.keys()) | set(gr.parents.keys())) - valid_ids:
    gr.children.pop(dead, None)
    gr.parents.pop(dead, None)

async with cm.get_context() as ctx2:
    for cid in ("Wann", "mArr", "XQcx", "EXIf"):
        if cid in ctx2.cells:
            ctx2.run_cell(cid)

g2 = cm.get_context().globals
msg = g2["start_alloy_job"]("deep_dataset_audit", g2["run_deep_dataset_audit"])
print("START_MSG:", msg)
mo.status.toast(
    title="Cell 4 Fixed & Deep Audit Running!",
    description="Cleaned stale graph edge, restored config UI, auditing datasets now.",
)
