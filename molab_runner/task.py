import datetime, json, os, subprocess
from pathlib import Path
import marimo._code_mode as cm

print("UTC_NOW:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
print("GPU:", subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader"],
    text=True,
).strip())

async with cm.get_context() as ctx:
    g = ctx.globals
    print("GLOBALS_HAS_START_JOB:", "start_alloy_job" in g)
    print("GLOBALS_HAS_HF_TOKEN_UI:", "hf_token_ui" in g)
    if "hf_token_ui" in g:
        print("UI_ORG:", repr(getattr(g["hf_org_ui"], "value", "")), "UI_TOK_LEN:", len(getattr(g["hf_token_ui"], "value", "") or ""))
    if "start_alloy_job" not in g:
        by_name = {c.name: c.id for c in ctx.cells}
        for nm in ["imports", "header", "env_probe", "config_ui", "guard_helpers", "job_manager", "vllm_gate_and_eval", "phase1_dataset_pool"]:
            if nm in by_name:
                ctx.run_cell(by_name[nm])
        print("INITIALIZED_CELLS")

print("MARIMO_SUBDIRS:", sorted(p.name for p in Path("/marimo").iterdir()))
print("CFG_HF_TOK_EXISTS:", Path("/marimo/.config/hf_token").exists())
print("ENV_HF_TOK:", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")))
for sub in ["/marimo/alloy_data/phase1_pool", "/marimo/alloy_gate", "/marimo/runs"]:
    p = Path(sub)
    if p.exists():
        print(f"DIR {sub}:", [x.name for x in sorted(p.iterdir())])
