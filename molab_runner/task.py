import json
import os
from pathlib import Path
from huggingface_hub import HfApi, get_token

print("ENV_HF_KEYS:", [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()])
print("GET_TOKEN_PRESENT:", bool(get_token()))
for p in [Path("/marimo"), Path("/marimo/runs"), Path("/marimo/runs/alloy_round2"), Path("/marimo/runs/alloy_round2/final_adapter")]:
    if p.exists():
        print(f"LS {p}:", sorted(x.name for x in p.iterdir()))

try:
    files = HfApi().list_repo_files("officialmobai/Alloy-9B-checkpoints", repo_type="model")
    print("HF_REPO_FILES:", files)
except Exception as e:
    print("HF_REPO_ERR:", type(e).__name__, str(e)[:200])
