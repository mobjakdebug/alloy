import marimo._code_mode as cm

ctx = cm.get_context()
g = ctx.globals

def _run_gate_and_train(log, model_id, hf_org, hf_token):
    g_res = g["build_frozen_gate"](log, konkur_limit=250)
    t_res = g["run_phase0_train_smoke"](log, model_id=model_id, hf_org=hf_org, hf_token=hf_token)
    return {"gate": g_res, "train_smoke": t_res}

msg = g["start_alloy_job"](
    "phase0_gate_train",
    _run_gate_and_train,
    g["base_model_ui"].value,
    (g["hf_org_ui"].value or "").strip(),
    g["get_hf_token"](),
)
print("START_MSG:", msg)
