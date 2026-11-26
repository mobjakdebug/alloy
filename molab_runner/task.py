import importlib
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path

from trl import GRPOConfig

print("HF_TOKEN_SET:", bool((os.environ.get("HF_TOKEN") or "").strip()))
for p in [Path.home() / ".cache/huggingface/token", Path("/marimo/.hf_token")]:
    if p.exists():
        print("TOKEN_FILE:", p, len(p.read_text().strip()))

vllm_py = "/marimo/vllm_venv/bin/python"
if Path(vllm_py).exists():
    out = subprocess.check_output(
        [
            vllm_py,
            "-c",
            "import importlib, importlib.util; "
            "print({p: getattr(importlib.import_module(p), '__version__', 'ok') if importlib.util.find_spec(p) else 'MISSING' "
            "for p in ['torch','transformers','peft','trl','liger_kernel','vllm','datasets','sympy']})",
        ],
        text=True,
    )
    print("VLLM_VENV:", out.strip())

sig = inspect.signature(GRPOConfig.__init__)
vllm_params = {
    k: str(v.default)
    for k, v in sig.parameters.items()
    if any(w in k for w in ("vllm", "loss", "mask", "scale", "reward", "len", "beta", "generation"))
}
print("GRPO_PARAMS:", json.dumps(vllm_params, indent=2))
print("DISK_DF:", subprocess.check_output(["df", "-h", "/marimo", "/root"], text=True).strip())
print("UPTIME:", subprocess.check_output(["uptime"], text=True).strip())
