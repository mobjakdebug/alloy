import inspect
import json
import os
import subprocess
from pathlib import Path

from trl import GRPOConfig

print("HF_TOKEN_SET:", bool((os.environ.get("HF_TOKEN") or "").strip()))

vllm_py = "/marimo/vllm_venv/bin/python"
if Path(vllm_py).exists():
    proc = subprocess.run(
        [
            vllm_py,
            "-c",
            "import importlib, importlib.util; res={}; "
            "\nfor p in ['torch','transformers','peft','trl','liger_kernel','vllm','datasets','sympy']:\n"
            "  try:\n"
            "    m = importlib.import_module(p)\n"
            "    res[p] = getattr(m, '__version__', 'ok')\n"
            "  except Exception as e:\n"
            "    res[p] = f'ERR:{type(e).__name__}:{e}'\n"
            "print(res)\n",
        ],
        capture_output=True,
        text=True,
    )
    print("VLLM_VENV_RC:", proc.returncode, proc.stdout.strip(), proc.stderr.strip()[:400])

sig = inspect.signature(GRPOConfig.__init__)
vllm_params = {
    k: str(v.default)
    for k, v in sig.parameters.items()
    if any(w in k for w in ("vllm", "loss", "mask", "scale", "reward", "len", "beta", "generation"))
}
print("GRPO_PARAMS:", json.dumps(vllm_params, indent=2))
print("DISK_DF:", subprocess.check_output(["df", "-h", "/marimo", "/root"], text=True).strip())
print("UPTIME:", subprocess.check_output(["uptime"], text=True).strip())
print("RUNS_LS:", sorted(str(p) for p in Path("/marimo/runs").glob("*")))
