import marimo._code_mode as cm

ctx = cm.get_context()
print("CELLS:", [(c.id, c.name, str(c.status)) for c in ctx.cells])
if "hf_dataset_audit" in ctx.globals:
    import json
    print("FOUND hf_dataset_audit:", json.dumps(ctx.globals["hf_dataset_audit"], indent=2, ensure_ascii=False))
else:
    print("hf_dataset_audit not in globals yet")
