import os
from pathlib import Path
from huggingface_hub import HfApi, get_token

print("ENV_HF_KEYS:", [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()])
print("GET_TOKEN_PRESENT:", bool(get_token()))
for p in [
    Path("/marimo"),
    Path("/marimo/runs"),
    Path("/marimo/runs/alloy_round2"),
    Path("/marimo/runs/alloy_round2/final_adapter"),
    Path("/marimo/runs/alloy_round3_grpo"),
    Path("/marimo/runs/alloy_9b_merged"),
    Path("/marimo/runs/alloy_evals"),
]:
    if p.exists():
        items = sorted(p.iterdir())
        print(f"LS {p} ({len(items)}):", [f"{x.name}({x.stat().st_size if x.is_file() else 'dir'})" for x in items[:30]])
    else:
        print(f"MISSING: {p}")

api = HfApi()
try:
    models = [m.id for m in api.list_models(author="officialmobai")]
    print("HF_OFFICIALMOBAI_MODELS:", models)
    for m in models:
        try:
            print(f"FILES {m}:", api.list_repo_files(m, repo_type="model"))
        except Exception as e:
            print(f"FILES_ERR {m}:", e)
except Exception as e:
    print("HF_LIST_ERR:", type(e).__name__, str(e)[:200])
