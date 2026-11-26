import importlib
import json
import os
import subprocess
from pathlib import Path

print("HF_TOKEN_SET:", bool((os.environ.get("HF_TOKEN") or "").strip()))
for p in [Path.home() / ".cache/huggingface/token", Path("/marimo/.hf_token")]:
    if p.exists():
        print("TOKEN_FILE:", p, len(p.read_text().strip()))

for pkg in ["torch", "transformers", "peft", "trl", "liger_kernel", "vllm", "datasets", "sympy"]:
    try:
        m = importlib.import_module(pkg)
        print("BASE_PY:", pkg, getattr(m, "__version__", "ok"))
    except Exception as e:
        print("BASE_PY:", pkg, "MISSING", type(e).__name__)

vllm_py = "/marimo/vllm_venv/bin/python"
if Path(vllm_py).exists():
    out = subprocess.check_output(
        [vllm_py, "-c", "import importlib; "
         "print({p: getattr(importlib.import_module(p), '__version__', 'ok') if importlib.util.find_spec(p) else 'MISSING' "
         "for p in ['torch','transformers','peft','trl','liger_kernel','vllm','datasets','sympy']})"],
        text=True,
    )
    print("VLLM_VENV:", out.strip())

# Check GRPOConfig signature / vllm_mode options & trl vllm-serve availability
import inspect
from trl import GRPOConfig
sig = inspect.signature(GRPOConfig.__init__)
vllm_params = {k: str(v.default) for k, v in sig.parameters.items() if "vllm" in k or "loss" in k or "mask" in k or "scale" in k}
print("GRPO_PARAMS:", json.dumps(vllm_params))
print("DISK_DF:", subprocess.check_output(["df", "-h", "/marimo", "/root"], text=True).strip())
print("UPTIME:", subprocess.check_output(["uptime"], text=True).strip())
