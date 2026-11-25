import json, re
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
pool_dir = Path("/marimo/alloy_data/phase1_pool")

HESITATION_RE = re.compile(
    r"\b(wait,|wait\.|wait\b|let me rethink|actually,?\s*that's wrong|let's start over|hold on|oops|i made a mistake|that doesn't look right|wait a second|wait a minute|hmm,?\s*no)\b",
    re.IGNORECASE,
)

print("=== 1. EXISTING PHASE1_POOL AUDIT ===")
for fn in sorted(pool_dir.glob("*.jsonl")):
    rows = [json.loads(l) for l in open(fn, "r", encoding="utf-8") if l.strip()]
    has_ans = sum(1 for r in rows if r.get("expected_answer"))
    has_trace = sum(1 for r in rows if r.get("teacher_trace"))
    no_hes_trace = sum(1 for r in rows if r.get("teacher_trace") and not HESITATION_RE.search(r["teacher_trace"]))
    short_clean = 0
    for r in rows[:2000]:
        tr = r.get("teacher_trace") or (r.get("messages", [{}])[-1].get("content", "") if r.get("messages") else "")
        if tr and not HESITATION_RE.search(tr) and len(tok.encode(tr, add_special_tokens=False)) <= 2400:
            short_clean += 1
    print(f"  {fn.name}: total={len(rows)}, ans={has_ans}, trace={has_trace}, no_hes={no_hes_trace}, sample_2k_short_clean={short_clean}")

for fn in ["baseline_ornith_vllm.json", "round1_alloy_vllm.json"]:
    d = json.loads((Path("/marimo/alloy_gate") / fn).read_text())
    det = d.get("detailed", [])
    trunc = sum(1 for x in det if x.get("toks", 0) >= 6140)
    print(f"GATE {fn}: detailed={len(det)}, trunc_at_6144={trunc}, avg_toks={round(sum(x.get('toks',0) for x in det)/max(1,len(det)), 1)}")

print("\n=== 2. NEW CANDIDATE DATASETS AUDIT ===")
api = HfApi()
for repo in [
    "nvidia/Nemotron-RL-Science-v1",
    "nvidia/Nemotron-SFT-Science-v2",
    "MegaScience/MegaScience",
    "PrimeIntellect/SWE-rebench-V2-Filtered-Easy-Verified",
    "SWE-bench/SWE-smith-py",
    "amiralifirouzi/tiny-persian-sft-15k",
    "PersianML/persian-konkur-1403",
]:
    try:
        info = api.repo_info(repo, repo_type="dataset")
        files = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith((".parquet", ".jsonl", ".json"))][:8]
        print(f"  REPO {repo}: OK files={files}")
    except Exception as e:
        print(f"  REPO {repo}: ERR {e}")

for repo, split in [
    ("nvidia/Nemotron-RL-Science-v1", "train"),
    ("nvidia/Nemotron-SFT-Science-v2", "train"),
    ("MegaScience/MegaScience", "train"),
    ("PrimeIntellect/SWE-rebench-V2-Filtered-Easy-Verified", "train"),
]:
    try:
        ds = load_dataset(repo, split=split, streaming=True)
        ex = next(iter(ds))
        print(f"  SCHEMA {repo}: keys={list(ex.keys())}")
        for k, v in ex.items():
            s = str(v)
            print(f"    {k}: {s[:180]!r}")
    except Exception as e:
        print(f"  SCHEMA {repo}: ERR {e}")
