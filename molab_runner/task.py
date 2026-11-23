import marimo._code_mode as cm

CELL_IMPORTS = """
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import marimo as mo
import torch
""".strip()

CELL_HEADER = """
mo.md(
    \"\"\"
    # Alloy — Phase 0: Environment, Hardware & Anti-Forgetting Setup
    **Hard rules:** max `4.5h` per chunk · auto-checkpoint & verify push to Hugging Face · regression gate before advancing.
    \"\"\"
)
""".strip()

CELL_ENV_PROBE = """
def _run_cmd(cmd):
    try:
        _out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
        return _out.strip()
    except Exception as _e:
        return f"ERR: {_e}"

def _pkg_ver(name):
    try:
        import importlib.metadata as _md
        return _md.version(name)
    except Exception:
        return "not installed"

_gpu_info = []
if torch.cuda.is_available():
    for _i in range(torch.cuda.device_count()):
        _p = torch.cuda.get_device_properties(_i)
        _gpu_info.append({
            "index": _i,
            "name": _p.name,
            "vram_gb": round(_p.total_memory / (1024**3), 2),
            "cc": f"sm_{_p.major}{_p.minor}",
            "bf16": torch.cuda.is_bf16_supported(),
        })

_pkgs = [
    "torch", "transformers", "peft", "trl", "accelerate",
    "datasets", "huggingface_hub", "vllm", "sglang",
    "unsloth", "liger-kernel", "bitsandbytes", "flash-attn", "sympy",
]
_pkg_rows = {p: _pkg_ver(p) for p in _pkgs}

_disk = shutil.disk_usage("/")
_disk_info = {
    "total_gb": round(_disk.total / (1024**3), 1),
    "free_gb": round(_disk.free / (1024**3), 1),
}

_docker_status = _run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"]) if shutil.which("docker") else "no docker binary"
_podman_status = _run_cmd(["podman", "--version"]) if shutil.which("podman") else "no podman binary"
_ram_line = _run_cmd(["free", "-h"]).splitlines()[1] if shutil.which("free") else "unknown"

env_report = {
    "python": sys.version.split()[0],
    "cuda_version": torch.version.cuda,
    "gpus": _gpu_info,
    "cpu_count": os.cpu_count(),
    "ram": _ram_line,
    "disk_root": _disk_info,
    "cwd": os.getcwd(),
    "docker": _docker_status,
    "podman": _podman_status,
    "hf_token_env_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    "packages": _pkg_rows,
}

print(json.dumps(env_report, indent=2))
mo.vstack([
    mo.md("### 1. Molab Runtime & Hardware Probe"),
    mo.json(env_report),
])
""".strip()

CELL_CONFIG_UI = """
base_model_ui = mo.ui.dropdown(
    options={
        "ornith-ai/Ornith-1.5-9B (9B dense, Qwen3.5 hybrid, MIT)": "ornith-ai/Ornith-1.5-9B",
        "Qwen/Qwen3.8-27B (27B dense, Apache 2.0)": "Qwen/Qwen3.8-27B",
        "thinkingmachines/Inkling-2-Mini (26.4B/3.4B looped MoE, Apache 2.0)": "thinkingmachines/Inkling-2-Mini",
    },
    value="ornith-ai/Ornith-1.5-9B (9B dense, Qwen3.5 hybrid, MIT)",
    label="Base Model",
)
hf_org_ui = mo.ui.text(value="", placeholder="your-hf-username", label="Hugging Face Username / Org")
hf_token_ui = mo.ui.text(value="", placeholder="hf_... (or set HF_TOKEN env)", kind="password", label="HF Token")
budget_hours_ui = mo.ui.number(start=0.5, stop=5.0, step=0.5, value=4.5, label="Max Chunk Budget (hours)")
inspect_btn = mo.ui.run_button(label="Inspect Base Model Config & Tokenizer")

mo.vstack([
    mo.md("### 2. Alloy Configuration"),
    mo.hstack([base_model_ui, budget_hours_ui]),
    mo.hstack([hf_org_ui, hf_token_ui]),
    inspect_btn,
])
""".strip()

CELL_INSPECT_MODEL = """
def _inspect_base(model_id: str):
    from transformers import AutoConfig, AutoTokenizer

    _cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    _text_cfg = getattr(_cfg, "text_config", _cfg)
    _layer_types = getattr(_text_cfg, "layer_types", None)
    _sample_fa = "مشتق تابع سینوس در نقطه پی دوم برابر با صفر است و انتگرال آن کسینوس است."
    _sample_en = "The derivative of sin(x) at pi/2 is zero and its integral is -cos(x)."
    _fa_ids = _tok.encode(_sample_fa, add_special_tokens=False)
    _en_ids = _tok.encode(_sample_en, add_special_tokens=False)

    _msgs = [
        {"role": "system", "content": "You are Alloy, made by Mob AI."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    _templated = _tok.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)

    return {
        "model_id": model_id,
        "model_type": getattr(_cfg, "model_type", None),
        "text_model_type": getattr(_text_cfg, "model_type", None),
        "architectures": getattr(_cfg, "architectures", None),
        "num_hidden_layers": getattr(_text_cfg, "num_hidden_layers", None),
        "hidden_size": getattr(_text_cfg, "hidden_size", None),
        "vocab_size": getattr(_text_cfg, "vocab_size", len(_tok)),
        "layer_types_summary": {
            t: _layer_types.count(t) for t in set(_layer_types)
        } if isinstance(_layer_types, list) else None,
        "persian_tokens": len(_fa_ids),
        "english_tokens": len(_en_ids),
        "persian_chars_per_token": round(len(_sample_fa) / max(1, len(_fa_ids)), 2),
        "chat_template_preview": _templated[:400],
    }

model_inspection = _inspect_base(base_model_ui.value)
print("BASE_INSPECTION:", json.dumps(model_inspection, indent=2, ensure_ascii=False))
mo.vstack([
    mo.md("### 3. Base Model Architecture & Persian Tokenizer Check"),
    mo.json(model_inspection),
])
""".strip()

CELL_GUARD_HELPERS = """
class TimeBudgetCallback:
    \"\"\"Wall-clock guard: saves checkpoint cleanly before the 5h limit.\"\"\"
    def __init__(self, budget_seconds: float):
        self.budget_seconds = float(budget_seconds)
        self.t0 = None

    def as_hf_callback(self):
        from transformers import TrainerCallback
        _outer = self

        class _Guard(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                _outer.t0 = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if _outer.t0 is not None and (time.time() - _outer.t0) >= _outer.budget_seconds:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        return _Guard()


def get_hf_token():
    return (hf_token_ui.value or "").strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""


def ensure_alloy_hf_repos(org_name: str, token: str, model_tag: str = "Alloy-9B"):
    from huggingface_hub import HfApi
    if not org_name or not token:
        raise ValueError("Provide both HF username/org and HF token first.")
    _api = HfApi(token=token)
    _repos = {
        "checkpoints": (f"{org_name}/{model_tag}-checkpoints", "model"),
        "data": (f"{org_name}/alloy-data", "dataset"),
        "evals": (f"{org_name}/alloy-evals", "dataset"),
    }
    for _k, (_rid, _rtype) in _repos.items():
        _api.create_repo(repo_id=_rid, repo_type=_rtype, private=True, exist_ok=True)
    return _repos


def verify_checkpoint_on_hub(repo_id: str, step: int, token: str):
    from huggingface_hub import HfApi
    _api = HfApi(token=token)
    _files = _api.list_repo_files(repo_id=repo_id, repo_type="model")
    _prefix = f"checkpoint-{step}/"
    _matches = [f for f in _files if f.startswith(_prefix)]
    if not _matches:
        raise RuntimeError(f"Checkpoint {_prefix} NOT found on Hub {repo_id}!")
    return len(_matches)


mo.md("### 4. Hard-Rule Guards Ready (`TimeBudgetCallback` + Private HF Repo & Checkpoint Verifier)")
""".strip()


async with cm.get_context() as ctx:
    for cid in list(ctx.cells.keys()):
        ctx.delete_cell(cid)

    c1 = ctx.create_cell(CELL_IMPORTS, name="imports", hide_code=False)
    c2 = ctx.create_cell(CELL_HEADER, after=c1, name="header", hide_code=False)
    c3 = ctx.create_cell(CELL_ENV_PROBE, after=c2, name="env_probe", hide_code=False)
    c4 = ctx.create_cell(CELL_CONFIG_UI, after=c3, name="config_ui", hide_code=False)
    c5 = ctx.create_cell(CELL_INSPECT_MODEL, after=c4, name="inspect_model", hide_code=False)
    c6 = ctx.create_cell(CELL_GUARD_HELPERS, after=c5, name="guard_helpers", hide_code=False)

    for cid in (c1, c2, c3, c4, c5, c6):
        ctx.run_cell(cid)

mo.status.toast(
    title="Alloy Phase 0 Loaded!",
    description="Running GPU, Docker, package, and Ornith-1.5-9B inspection now.",
)
