import json
from pathlib import Path
import marimo as mo
import marimo._code_mode as cm

pool_dir = Path("/marimo/alloy_data/phase1_pool")
all_ok = True
for p in sorted(pool_dir.glob("*.jsonl")):
    cnt = 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    cnt += 1
        print(f"VERIFIED_JSONL {p.name}: {cnt} rows OK")
    except Exception as e:
        all_ok = False
        first_line = p.read_text(encoding="utf-8")[:200]
        print(f"ERR {p.name}: {e} | head={repr(first_line)}")

if all_ok:
    async with cm.get_context() as ctx:
        for c in list(ctx.cells):
            if c.name == "phase1_distill":
                new_code = c.code.replace(
                    'return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]',
                    'with open(p, "r", encoding="utf-8") as _f:\n        return [json.loads(line) for line in _f if line.strip()]'
                )
                # Also log full traceback lines in run_phase1_distill
                new_code = new_code.replace(
                    'if any(k in _s for k in ("[Pass ", "FINAL_ROUND1_DATASET", "ERROR", "Error", "Traceback", "Loading safetensors", "Exception", "ValueError")):',
                    'if any(k in _s for k in ("[Pass ", "FINAL_ROUND1_DATASET", "ERROR", "Error", "Traceback", "Loading safetensors", "Exception", "ValueError", "File ")):',
                )
                ctx.update_cell(c.id, code=new_code)
                ctx.run_cell(c.id)
    g = cm.get_context().globals
    msg = g["start_alloy_job"](
        "phase1_distill",
        g["run_phase1_distill"],
        base_model=g["base_model_ui"].value,
        teacher_model="Qwen/Qwen3.8-27B-FP8",
        hf_org=g["hf_org_ui"].value.strip(),
        hf_token=g["get_hf_token"](),
        step="all",
    )
    print("START_MSG:", msg)
    mo.status.toast(title="Phase 1 Distill Restarted!", description="Fixed U+2028/U+2029 JSONL line parsing & running Pass 1 -> Pass 2 -> Assemble.")
