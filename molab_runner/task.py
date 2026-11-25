import subprocess
from pathlib import Path

p = Path("/marimo/alloy_scripts/phase1_r2_distill.py")
print("HEAD:", "\n".join(p.read_text().splitlines()[:20]))
res = subprocess.run(["/marimo/vllm_venv/bin/python", str(p), "--help"], capture_output=True, text=True)
print("RC:", res.returncode)
print("OUT:", res.stdout)
print("ERR:", res.stderr)
