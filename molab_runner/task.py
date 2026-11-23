import marimo._code_mode as cm

ctx = cm.get_context()
print("ctx.cells:", [(c.id, c.name, c.status) for c in ctx.cells])
print("ctx.graph.cells:", list(ctx.graph.cells.keys()))
print("mArr code:\n", ctx.cells["mArr"].code if "mArr" in ctx.cells else "NO mArr")
