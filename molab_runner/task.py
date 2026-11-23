import inspect
import marimo as mo
import marimo._code_mode as cm

sig = inspect.signature(mo.status.toast)
print("toast signature:", sig)

async with cm.get_context() as ctx:
    existing_ids = list(ctx.cells.keys())
    print("Clearing existing cells:", existing_ids)
    for cid in existing_ids:
        ctx.delete_cell(cid)

    c1 = ctx.create_cell("import marimo as mo", hide_code=False)
    c2 = ctx.create_cell(
        'mo.md("# Hello, World!\\n\\nPaired live with **Molab** — ready to build **Alloy**!")',
        after=c1,
        hide_code=False,
    )
    ctx.run_cell(c1)
    ctx.run_cell(c2)

# Send a fun toast inside the live marimo UI
params = set(sig.parameters.keys())
kwargs = {}
if "title" in params:
    kwargs["title"] = "Ready to pair!"
if "description" in params:
    kwargs["description"] = "Connected to Molab from Arena — let's build Alloy!"
if "kind" in params:
    kwargs["kind"] = "success"

if kwargs:
    mo.status.toast(**kwargs)
else:
    mo.status.toast("Ready to pair! Connected to Molab from Arena — let's build Alloy!")

ctx_after = cm.get_context()
print("Cells now:", [(c.id, c.code, str(c.status)) for c in ctx_after.cells])
