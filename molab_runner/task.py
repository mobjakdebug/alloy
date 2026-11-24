import importlib, subprocess
from pathlib import Path
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for name in ["imports", "config_ui"]:
        for c in ctx.cells:
            if c.name == name:
                ctx.run_cell(c.id)

g = cm.get_context().globals
tok = g["hf_tok_in"].value
org = g["hf_org_in"].value
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
