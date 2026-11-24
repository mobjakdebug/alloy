import json, os
from pathlib import Path
import marimo._code_mode as cm
from huggingface_hub import HfApi, login

g = cm.get_context().globals
org = (getattr(g.get("hf_org_ui"), "value", "") or "officialmobai").strip()
tok = (getattr(g.get("hf_token_ui"), "value", "") or "").strip()

print("ORG:", org, "HAS_TOK:", bool(tok), "TOK_LEN:", len(tok))
if not tok:
    for p in [Path("/marimo/.config/hf_token"), Path.home() / ".cache/huggingface/token"]:
        if p.exists() and p.read_text().strip():
            tok = p.read_text().strip()
            print("LOADED_FALLBACK_TOK_FROM:", p)
            break

if tok:
    Path("/marimo/.config").mkdir(parents=True, exist_ok=True)
    p_tok = Path("/marimo/.config/hf_token")
    p_tok.write_text(tok)
    os.chmod(p_tok, 0o600)
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
    login(token=tok, add_to_git_credential=False)
    api = HfApi(token=tok)
    for repo, rtype in [(f"{org}/alloy-data", "dataset"), (f"{org}/alloy-evals", "dataset"), (f"{org}/Alloy-9B-checkpoints", "model")]:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type=rtype)
            print(f"HF {repo} ({rtype}):", files)
        except Exception as e:
            print(f"HF {repo} ERR:", e)
else:
    print("NO_TOKEN_FOUND_IN_UI_YET")

print("SETUP_JOB:", json.dumps(g["get_alloy_job_status"]("setup_vllm_and_cache", tail_lines=15), indent=2))
