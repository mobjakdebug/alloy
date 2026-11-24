import json, os
from pathlib import Path
import marimo._code_mode as cm
from huggingface_hub import hf_hub_download

g = cm.get_context().globals
tok = Path("/marimo/.config/hf_token").read_text().strip()
org = "officialmobai"
os.environ["HF_TOKEN"] = tok
os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

# 1. Restore frozen_gate.json and baseline_ornith_vllm.json from officialmobai/alloy-evals
gate_dir = Path("/marimo/alloy_gate")
gate_dir.mkdir(parents=True, exist_ok=True)
fg = hf_hub_download(repo_id=f"{org}/alloy-evals", filename="gates/frozen_gate.json", repo_type="dataset", token=tok)
(gate_dir / "frozen_gate.json").write_bytes(Path(fg).read_bytes())
bg = hf_hub_download(repo_id=f"{org}/alloy-evals", filename="results/baseline_ornith_vllm.json", repo_type="dataset", token=tok)
(gate_dir / "baseline_ornith_vllm.json").write_bytes(Path(bg).read_bytes())
print("RESTORED frozen_gate.json bytes:", (gate_dir / "frozen_gate.json").stat().st_size)

# 2. Inspect phase1_distill.py CLI flags and phase1_dataset_pool code
p1_script = Path("/marimo/alloy_scripts/phase1_distill.py").read_text()
for line in p1_script.splitlines():
    if "add_argument" in line or "ANCHOR_" in line or "PROBE_" in line or "TEACHER_" in line or "QUOTA" in line or "TARGET" in line or "max_tokens" in line:
        print("P1_LINE:", line.strip())

# 3. Launch phase1_pool_prep right now!
msg = g["start_alloy_job"]("phase1_pool_prep", g["build_phase1_raw_pools"], hf_token=tok)
print("POOL_PREP_JOB:", msg)
