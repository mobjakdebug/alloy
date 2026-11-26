import subprocess
from pathlib import Path
from huggingface_hub import HfApi

for cmd in [
    "df -h",
    "find /marimo -maxdepth 4 -type f -size +1M 2>/dev/null | head -n 50",
    "find / -name '*adapter_model*' -o -name '*alloy_round2*' -o -name '*alloy_9b*' -o -name '*phase1_r2*' 2>/dev/null | head -n 50",
    "ls -la /marimo /marimo/alloy_scripts",
]:
    print(f"=== {cmd} ===")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[:3000])

api = HfApi()
try:
    ds = [d.id for d in api.list_datasets(author="officialmobai")]
    print("HF_DATASETS:", ds)
    for d in ds:
        if "alloy" in d.lower():
            print(f"DS_FILES {d}:", api.list_repo_files(d, repo_type="dataset"))
except Exception as e:
    print("HF_DS_ERR:", e)
