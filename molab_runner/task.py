import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.name in ("vllm_gate_and_eval", "phase1_dataset_pool"):
            print(f"=== CELL {c.name} ({c.id}) ===")
            print(c.code)
