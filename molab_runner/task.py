import os, subprocess
from pathlib import Path
import marimo._code_mode as cm

for envk in sorted(os.environ.keys()):
    if any(w in envk.upper() for w in ["HF", "HUGGING", "TOKEN", "MARIMO", "MOLAB"]):
        print("ENV_KEY:", envk, "len=", len(os.environ[envk]))

for p in [Path.home() / ".cache/huggingface/token", Path("/root/.cache/huggingface/token"), Path("/marimo/.cache/huggingface/token"), Path("/marimo/vllm_venv")]:
    print("CHECK_PATH:", p, p.exists())
    if p.exists() and p.is_dir():
        print("  SUB:", [x.name for x in list(p.iterdir())[:20]])

# Run all core notebook cells cleanly so functions are loaded in kernel
async with cm.get_context() as ctx:
    order = ["imports", "header", "env_probe", "config_ui", "guard_helpers", "job_manager", "vllm_gate_and_eval", "phase1_dataset_pool", "phase1_distill", "phase2_train_round1", "phase2_live_dashboard"]
    by_name = {c.name: c.id for c in ctx.cells}
    for nm in order:
        if nm in by_name:
            ctx.run_cell(by_name[nm])
            print("RAN_CELL:", nm)

# Check if vllm_venv can be built cleanly in background right now
print("CELLS_LOADED_OK")
