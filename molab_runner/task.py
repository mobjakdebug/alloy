import subprocess
import marimo as mo
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    ctx.packages.add(["flash-linear-attention", "liger-kernel"])

g = cm.get_context().globals

def _setup_vllm_venv(log):
    log("Checking disk space...")
    df_out = subprocess.check_output(["df", "-h", "/", "/marimo", "/tmp"], text=True)
    log(df_out.strip())
    venv_dir = "/marimo/vllm_venv"
    log(f"Creating isolated vLLM venv at {venv_dir}...")
    r1 = subprocess.run(["uv", "venv", venv_dir, "--python", "3.13"], capture_output=True, text=True)
    log(f"uv venv rc={r1.returncode} {r1.stderr.strip()}")
    if r1.returncode != 0:
        raise RuntimeError(r1.stderr)
    log("Installing vllm into /marimo/vllm_venv...")
    r2 = subprocess.run(
        ["uv", "pip", "install", "--python", f"{venv_dir}/bin/python", "vllm", "huggingface_hub"],
        capture_output=True, text=True
    )
    log(f"uv pip install vllm rc={r2.returncode}")
    if r2.returncode != 0:
        log(r2.stderr[-2000:])
        raise RuntimeError(r2.stderr[-1000:])
    r3 = subprocess.run(
        [f"{venv_dir}/bin/python", "-c", "import vllm, torch, transformers; print('vllm:', vllm.__version__, 'torch:', torch.__version__, 'transformers:', transformers.__version__, 'cuda:', torch.cuda.is_available())"],
        capture_output=True, text=True
    )
    log(f"Verify vllm rc={r3.returncode}: {r3.stdout.strip()} {r3.stderr.strip()}")
    return {"vllm_python": f"{venv_dir}/bin/python", "verify": r3.stdout.strip()}

msg = g["start_alloy_job"]("setup_vllm_env", _setup_vllm_venv)
print("START_MSG:", msg)
mo.status.toast(
    title="Installing Fast Inference (vLLM + FLA + Liger)!",
    description="Setting up isolated /marimo/vllm_venv for 20x faster eval & distillation.",
)
