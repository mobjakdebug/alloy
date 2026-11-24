import subprocess
from pathlib import Path
import marimo._code_mode as cm

p = Path("/marimo/vllm_venv/bin")
if p.exists():
    print("VLLM_BIN:", [x.name for x in sorted(p.iterdir())[:20]])

g = cm.get_context().globals
print("JOB_MANAGER_FUNCS:", [k for k in g.keys() if "job" in k or "vllm" in k or "phase1" in k or "phase2" in k])

# Start background job to ensure clean /marimo/vllm_venv + pre-cache Ornith-1.5-9B & Qwen3.8-27B-FP8
def setup_vllm_and_cache(log):
    import shutil
    from huggingface_hub import snapshot_download
    log("Rebuilding clean /marimo/vllm_venv with vllm==0.34.0...")
    shutil.rmtree("/marimo/vllm_venv", ignore_errors=True)
    subprocess.run(["uv", "venv", "--python", "3.13", "/marimo/vllm_venv"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", "/marimo/vllm_venv/bin/python", "vllm==0.34.0"], check=True)
    subprocess.run(["uv", "pip", "uninstall", "--python", "/marimo/vllm_venv/bin/python", "torchaudio"], check=False)
    out = subprocess.check_output(["/marimo/vllm_venv/bin/python", "-c", "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"], text=True).strip()
    log(f"vllm_venv ready: {out}")
    log("Pre-caching ornith-ai/Ornith-1.5-9B...")
    snapshot_download("ornith-ai/Ornith-1.5-9B")
    log("Cached ornith-ai/Ornith-1.5-9B! Pre-caching Qwen/Qwen3.8-27B-FP8...")
    snapshot_download("Qwen/Qwen3.8-27B-FP8")
    log("Cached Qwen/Qwen3.8-27B-FP8!")
    return {"vllm": out, "cached": ["ornith-ai/Ornith-1.5-9B", "Qwen/Qwen3.8-27B-FP8"]}

msg = g["start_alloy_job"]("setup_vllm_and_cache", setup_vllm_and_cache)
print("STARTED_JOB:", msg)
