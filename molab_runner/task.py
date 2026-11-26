import json, os, subprocess
from pathlib import Path
import marimo._code_mode as cm

if "start_alloy_job" not in cm.get_context().globals:
    async with cm.get_context() as ctx:
        by_name = {c.name: c.id for c in ctx.cells}
        for nm in ["imports", "header", "env_probe", "config_ui", "guard_helpers", "job_manager", "vllm_gate_and_eval", "phase1_dataset_pool"]:
            if nm == "config_ui" and "hf_token_ui" in ctx.globals:
                continue
            if nm in by_name:
                ctx.run_cell(by_name[nm])

g = cm.get_context().globals
ui_tok = (getattr(g.get("hf_token_ui"), "value", "") or "").strip()
env_tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
cfg_tok = Path("/marimo/.config/hf_token").read_text().strip() if Path("/marimo/.config/hf_token").exists() else ""
print("TOKENS_PRESENT:", {"ui": bool(ui_tok), "env": bool(env_tok), "cfg": bool(cfg_tok)})
if ui_tok and not cfg_tok:
    Path("/marimo/.config").mkdir(parents=True, exist_ok=True)
    Path("/marimo/.config/hf_token").write_text(ui_tok, encoding="utf-8")
    os.chmod("/marimo/.config/hf_token", 0o600)

print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip())
print("VLLM_VENV_EXISTS:", Path("/marimo/vllm_venv/bin/python").exists())
for d in ["/marimo/alloy_gate", "/marimo/alloy_data/phase1_pool", "/marimo/alloy_data/phase1_r2_verified", "/marimo/runs"]:
    p = Path(d)
    if p.exists():
        items = {}
        for x in sorted(p.iterdir()):
            if x.is_file():
                items[x.name] = x.stat().st_size
            elif x.is_dir():
                items[x.name + "/"] = len(list(x.iterdir()))
        print(f"DIR {d}:", json.dumps(items))
    else:
        print(f"DIR {d}: MISSING")

print("HF_CACHE:", subprocess.check_output(["bash", "-c", "ls -1 ~/.cache/huggingface/hub 2>/dev/null || true"], text=True).strip().splitlines())
