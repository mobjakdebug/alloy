import marimo as mo
import marimo._code_mode as cm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
print("IM_START_USER:", tok.encode("<|im_start|>user\n", add_special_tokens=False))
print("IM_START_ASST:", tok.encode("<|im_start|>assistant\n", add_special_tokens=False))
print("IM_END:", tok.encode("<|im_end|>", add_special_tokens=False))

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name == "phase1_distill":
            new_code = c.code.replace(
                'return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]',
                'with open(p, "r", encoding="utf-8") as _f:\n        return [json.loads(line) for line in _f if line.strip()]'
            ).replace(
                'if any(k in _s for k in ("[Pass ", "FINAL_ROUND1_DATASET", "ERROR", "Error", "Traceback", "Loading safetensors", "Exception", "ValueError")):',
                'if any(k in _s for k in ("[Pass ", "FINAL_ROUND1_DATASET", "ERROR", "Error", "Traceback", "Loading safetensors", "Exception", "ValueError", "File ")):',
            )
            ctx.edit_cell(c.id, code=new_code)
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
mo.status.toast(
    title="Phase 1 Verified Distillation Running!",
    description="Pass 1 (Ornith pass@4) -> Pass 2 (Qwen3.8-27B-FP8) -> Assemble & push to officialmobai/alloy-data.",
)
