import marimo._code_mode as cm
ctx = cm.get_context()
print("cells:", [(c.id, c.name) for c in ctx.cells])
print("multiply_defined:", ctx.graph.get_multiply_defined())
for k in ["ALLOY_GATE_DIR", "ALLOY_DATA_DIR", "build_frozen_gate", "run_phase0_train_smoke"]:
    print(k, "-> defs:", ctx.graph.definitions.get(k), "defining_cells:", ctx.graph.get_defining_cells(k))
