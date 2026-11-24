import os
import subprocess
from pathlib import Path
import marimo._code_mode as cm

for root in ["/marimo/runs", "/marimo/alloy_scripts", "/marimo/.config"]:
    p = Path(root)
    if p.exists():
        items = [str(x.relative_to("/marimo")) + f" ({x.stat().st_size})" for x in sorted(p.rglob("*")) if x.is_file()][:40]
        print(f"FILES {root}:", items)

print("VLLM_PYTHON_EXISTS:", Path("/marimo/vllm_venv/bin/python").exists())
try:
    out = subprocess.check_output(["/marimo/vllm_venv/bin/python", "-c", "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"], text=True).strip()
    print("VLLM_CHECK:", out)
except Exception as e:
    print("VLLM_CHECK_ERR:", e)

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.name == "config_ui":
            code = ctx.get_cell_code(c.id) if hasattr(ctx, "get_cell_code") else getattr(c, "code", "")
            print("CONFIG_UI_HAS_TOKEN_LITERAL:", ("hf_" in code), "LEN:", len(code))
            for line in code.splitlines():
                if "hf_" in line or "token" in line.lower() or "org" in line.lower():
                    masked = line
                    if "hf_" in masked:
                        idx = masked.find("hf_")
                        masked = masked[:idx+6] + "***REDACTED***"
                    print("  CFG:", masked)

try:
    from huggingface_hub import HfApi, get_token
    tok = get_token()
    print("HF_CACHED_TOKEN:", bool(tok))
except Exception as e:
    print("HF_HUB_ERR:", e)
