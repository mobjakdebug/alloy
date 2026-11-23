import marimo._code_mode as cm

ctx = cm.get_context()
print("ctx.cells keys:", list(ctx.cells.keys()))
print("ctx.graph.cells keys:", list(ctx.graph.cells.keys()))
for k, v in ctx.graph.cells.items():
    print("GRAPH:", k, "defs=", v.defs, "refs=", v.refs)
