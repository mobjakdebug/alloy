import json
from pathlib import Path
import marimo._code_mode as cm
from huggingface_hub import HfApi

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_konkur_baseline", tail_lines=30), indent=2, ensure_ascii=False))

audit_p = Path("/marimo/alloy_gate/dataset_deep_audit.json")
if audit_p.exists():
    print("DEEP_AUDIT:", audit_p.read_text(encoding="utf-8")[:6000])

api = HfApi()
for repo in ["nvidia/OpenCodeReasoning", "ZehuaZhao/SUPERChem", "ScienceOne-AI/PhysLogic", "xmanii/Persian-Math-SFT", "huginnfork/Qwen3.8-27B-FP8"]:
    try:
        info = api.repo_info(repo_id=repo, repo_type="model" if "Qwen3.8" in repo else "dataset", files_metadata=True)
        files = [(s.rfilename, round((s.size or 0)/(1024**3), 3)) for s in (info.siblings or []) if (s.size or 0) > 1_000_000]
        print(f"HF_INFO {repo}: total_files={len(info.siblings or [])}, large_files={files[:12]}")
    except Exception as e:
        print(f"HF_INFO {repo} ERR: {e}")
