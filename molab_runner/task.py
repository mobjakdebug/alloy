import json, subprocess
import marimo._code_mode as cm

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_konkur_baseline", tail_lines=40), indent=2, ensure_ascii=False))
print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

org = g["hf_org_ui"].value.strip()
tok = g["get_hf_token"]()
print("HF_ORG:", org, "TOKEN_SET:", bool(tok), "TOKEN_LEN:", len(tok))
if org and tok:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        me = api.whoami()
        print("HF_WHOAMI:", me.get("name"), "orgs:", [o.get("name") for o in me.get("orgs", [])])
        repos = g["ensure_alloy_hf_repos"](org, tok)
        print("HF_REPOS:", repos)
    except Exception as e:
        print("HF_ERR:", e)
