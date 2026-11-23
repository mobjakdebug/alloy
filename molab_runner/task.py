import marimo._code_mode as cm
ctx = cm.get_context()
for c in ctx.cells:
    if any(k in c.code for k in ["ALLOY_GATE_DIR", "build_frozen_gate", "run_phase0_train_smoke"]):
        print("FOUND IN CELL:", c.id, c.name)
print("defs keys sample:", sorted(list(ctx.graph.definitions.keys()))[:40])
print("has ALLOY_GATE_DIR in globals:", "ALLOY_GATE_DIR" in ctx.globals)
