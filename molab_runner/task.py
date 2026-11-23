import json
import subprocess
import marimo._code_mode as cm

print("--- DF ---")
print(subprocess.check_output(["df", "-h", "/", "/marimo", "/tmp"], text=True))

print("--- LS /marimo ---")
print(subprocess.check_output(["ls", "-la", "/marimo"], text=True))

ctx = cm.get_context()
print("--- ctx.packages ---")
help(ctx.packages)

from transformers import AutoConfig
for mid in ["ornith-ai/Ornith-1.5-9B", "Qwen/Qwen3.8-27B"]:
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        tcfg = getattr(cfg, "text_config", cfg)
        print(mid, "->", cfg.model_type, "layers:", getattr(tcfg, "num_hidden_layers", None), "hidden:", getattr(tcfg, "hidden_size", None))
    except Exception as e:
        print(mid, "ERR:", e)
