import json, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_pools", tail_lines=40), indent=2))
print("=== DF -H ===")
print(subprocess.check_output(["df", "-h"], text=True))
print("=== HF CACHE ===")
print(subprocess.check_output(["bash", "-c", "ls -lh ~/.cache/huggingface/hub 2>/dev/null || true; du -sh ~/.cache/huggingface/hub/* 2>/dev/null || true"], text=True))
print("=== NET / THREADS ===")
print(subprocess.check_output(["bash", "-c", "find ~/.cache/huggingface/hub -name '*.incomplete' -ls 2>/dev/null | head -n 20 || true"], text=True))
