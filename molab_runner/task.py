import re
from pathlib import Path
import marimo._code_mode as cm

def mask(s):
    return re.sub(r"hf_[A-Za-z0-9]{6,}", "hf_***REDACTED***", s)

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.name == "config_ui":
            code = ctx.get_cell_code(c.id) if hasattr(ctx, "get_cell_code") else getattr(c, "code", "")
            print("=== config_ui ===")
            print(mask(code))
            ctx.run_cell(c.id)

g = cm.get_context().globals
for k in ["hf_org_ui", "hf_token_ui"]:
    obj = g.get(k)
    val = getattr(obj, "value", obj)
    print(k, "type=", type(obj), "val_len=", len(str(val or "")), "val_masked=", mask(str(val or "")))

for p in Path("/marimo").iterdir():
    print("MARIMO:", p.name, p.is_dir())
