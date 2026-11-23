import marimo as mo
import marimo._code_mode as cm

ctx = cm.get_context()
for c in ctx.cells:
    print("CELL:", c.id, repr(c.name), repr(c.status), "len=", len(c.code))

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if "env_report" in c.code or "dataset_audit" in c.code:
            ctx.run_cell(c.id)
