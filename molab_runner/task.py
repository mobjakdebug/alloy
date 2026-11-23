import subprocess
import marimo as mo
import marimo._code_mode as cm

vp = "/marimo/vllm_venv/bin/python"
print(subprocess.check_output(["uv", "pip", "list", "-p", vp], text=True)[:2000])
r = subprocess.run(["uv", "pip", "uninstall", "-p", vp, "torchaudio"], capture_output=True, text=True)
print("uninstall torchaudio:", r.returncode, r.stdout.strip(), r.stderr.strip())
r2 = subprocess.run([vp, "-c", "from vllm import LLM, SamplingParams; import torch; print('vLLM LLM import OK! torch:', torch.__version__)"], capture_output=True, text=True)
print("vLLM import rc=", r2.returncode, r2.stdout.strip(), r2.stderr[-1000:])
if r2.returncode == 0:
    g = cm.get_context().globals
    _hf_org = g["hf_org_ui"].value.strip()
    _hf_tok = g["get_hf_token"]()
    _model = g["base_model_ui"].value
    msg = g["start_alloy_job"](
        "phase0_vllm_baseline",
        g["run_vllm_gate_eval"],
        model_id=_model,
        out_name="baseline_ornith_vllm.json",
        hf_org=_hf_org,
        hf_token=_hf_tok,
        max_tokens=6144,
    )
    print("START_MSG:", msg)
    mo.status.toast(title="vLLM Baseline Launched!", description="Evaluating Ornith-1.5-9B on Konkur-250 + GPQA-198 + PhysLogic-200.")
