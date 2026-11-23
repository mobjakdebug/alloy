import json
import marimo as mo
import marimo._code_mode as cm

CELL_STATUS_VIEW = """
refresh_status_btn = mo.ui.run_button(label="Refresh Phase 0 Smoke Status & Results")
_st = get_alloy_job_status("phase0_smoke", tail_lines=30)
print("PHASE0_SMOKE_STATUS:", json.dumps(_st, indent=2, ensure_ascii=False))
mo.vstack([
    mo.md("### 8. Phase 0 Smoke Job Status & Results"),
    refresh_status_btn,
    mo.json(_st),
])
""".strip()

async with cm.get_context() as ctx:
    msg = ctx.globals["start_alloy_job"](
        "phase0_smoke",
        ctx.globals["run_phase0_model_smoke"],
        ctx.globals["base_model_ui"].value,
    )
    print("START_MSG:", msg)
    for c in list(ctx.cells):
        if c.name == "phase0_status_view" or "refresh_status_btn" in c.code:
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c10 = ctx.create_cell(CELL_STATUS_VIEW, after=last_cid, name="phase0_status_view", hide_code=False)
    ctx.run_cell(c10)

mo.status.toast(
    title="Phase 0 Smoke Started!",
    description="Downloading & inspecting Ornith-1.5-9B in background on RTX PRO 6000.",
)
