import importlib, subprocess
from pathlib import Path
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.name == "config_ui":
            code = ctx.get_cell_code(c.id) if hasattr(ctx, "get_cell_code") else getattr(c, "code", "")
            for line in code.splitlines():
                if "=" in line and ("hf_" in line or "org" in line or "tok" in line):
                    lhs = line.split("=")[0].strip()
                    print("CFG_LHS:", lhs)
            ctx.run_cell(c.id)

g = cm.get_context().globals
print("GLOBALS_AFTER:", sorted(k for k in g.keys() if "hf" in k.lower() or "org" in k.lower() or "tok" in k.lower()))

tok = None
org = "officialmobai"
for k, v in g.items():
    val = getattr(v, "value", v)
    if isinstance(val, str) and val.startswith("hf_"):
        tok = val
    elif isinstance(val, str) and "officialmobai" in val:
        org = val
print("HF_ORG:", org, "HAS_TOK:", bool(tok))

from huggingface_hub import HfApi
api = HfApi(token=tok)
for repo, rtype in [(f"{org}/alloy-data", "dataset"), (f"{org}/alloy-evals", "dataset"), (f"{org}/Alloy-9B-checkpoints", "model")]:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type=rtype)
        print(f"HF {repo} ({rtype}):", files)
    except Exception as e:
        print(f"HF {repo} ERR:", e)

for pkg in ["torch", "transformers", "peft", "trl", "accelerate", "datasets", "huggingface_hub", "bitsandbytes", "sympy", "vllm", "liger_kernel", "fla"]:
    try:
        m = importlib.import_module(pkg)
        print(f"PKG {pkg}: {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"PKG {pkg}: MISSING ({e})")

print("UV:", subprocess.run(["which", "uv"], capture_output=True, text=True).stdout.strip())
print("DF_MARIMO:", subprocess.check_output(["df", "-h", "/marimo", "/tmp"], text=True).strip())
