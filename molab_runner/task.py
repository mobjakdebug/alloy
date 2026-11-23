import marimo._code_mode as cm
ctx = cm.get_context()
print("ALLOY_GATE_DIR in defs:", ctx.graph.definitions.get("ALLOY_GATE_DIR"))
print("cells:", [(c.id, c.name) for c in ctx.cells])
print("graph cells:", list(ctx.graph.cells.keys()))
