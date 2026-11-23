import subprocess, json
import marimo._code_mode as cm
from huggingface_hub import HfApi

g = cm.get_context().globals
tok = g["get_hf_token"]()
api = HfApi(token=tok)

# 1. Check gated GPQA access with user's token
try:
    gpqa_info = api.dataset_info("Idavidrein/gpqa", files_metadata=True)
    print("GPQA_ACCESS: OK files=", [s.rfilename for s in (gpqa_info.siblings or [])[:10]])
except Exception as e:
    print("GPQA_ACCESS: ERR ->", e)

# 2. Check candidate teacher repos & exact file sizes
for repo in [
    "beginning-ai/Qwen3.8-Flash-Next-NVFP4",
    "Qwen/Qwen3.8-27B-FP8",
    "huginnfork/Qwen3.8-27B-FP8",
    "nvidia/Qwen3.8-27B-NVFP4",
]:
    try:
        info = api.model_info(repo, files_metadata=True)
        total_gb = sum((s.size or 0) for s in (info.siblings or [])) / (1024**3)
        print(f"TEACHER {repo}: total_gb={total_gb:.2f} GB, files={len(info.siblings or [])}")
    except Exception as e:
        print(f"TEACHER {repo}: ERR -> {e}")

# 3. Dry-run check vllm / sglang / flash-linear-attention / causal-conv1d / liger-kernel installability
for pkg in ["vllm", "sglang", "flash-linear-attention", "causal-conv1d", "liger-kernel"]:
    r = subprocess.run(["uv", "pip", "install", "--dry-run", pkg], capture_output=True, text=True)
    lines = (r.stdout + "\n" + r.stderr).strip().splitlines()
    print(f"DRY_RUN {pkg} rc={r.returncode}:", " | ".join(lines[-4:]))
