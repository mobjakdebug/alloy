import json, subprocess
import marimo._code_mode as cm

g = cm.get_context().globals
print("PHASE1_DISTILL:", json.dumps(g["get_alloy_job_status"]("phase1_distill", tail_lines=40), indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

# Quick check of Ornith assistant header token IDs and TRL SFTConfig fields
from transformers import AutoTokenizer
from trl import SFTConfig
import inspect
tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
print("ASSIST_PREFIX_IDS:", tok.encode("<|im_start|>assistant\n", add_special_tokens=False))
print("SFTConfig params:", [k for k in inspect.signature(SFTConfig.__init__).parameters if "liger" in k or "max_" in k or "loss" in k or "packing" in k])
