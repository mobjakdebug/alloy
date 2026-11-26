from pathlib import Path
import subprocess

for p in [
    Path("/marimo/vllm_venv/bin/python"),
    Path("/marimo/runs/alloy_round2/final_adapter/README.md"),
    Path("/marimo/alloy_scripts/setup_and_pools_v2.py"),
    Path("/marimo/alloy_scripts/phase1_r2_distill.py"),
    Path("/marimo/alloy_scripts/phase2_r2_sft.py"),
]:
    print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else 0)

r = subprocess.run("ls -la /marimo/vllm_venv /marimo/runs/alloy_round2/checkpoint-77 /marimo/runs/alloy_round2/final_adapter", shell=True, capture_output=True, text=True)
print(r.stdout)
print("--- final_adapter/README.md ---")
print(Path("/marimo/runs/alloy_round2/final_adapter/README.md").read_text()[:1500])
