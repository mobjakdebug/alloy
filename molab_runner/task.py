import json
import marimo._code_mode as cm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

g = cm.get_context().globals
print("JOB:", json.dumps(g["get_alloy_job_status"]("phase0_konkur_baseline", tail_lines=30), indent=2, ensure_ascii=False))

# Quick CPU-only inspection of OpenCodeReasoning, Persian-Math-SFT, SUPERChem, PhysLogic
for name, repo, fpath in [
    ("OpenCodeReasoning", "nvidia/OpenCodeReasoning", "split_0/train-00000-of-00030.parquet"),
    ("Persian-Math-SFT", "xmanii/Persian-Math-SFT", "data/train-00000-of-00001.parquet"),
    ("SUPERChem", "ZehuaZhao/SUPERChem", "SUPERChem-500.parquet"),
    ("PhysLogic", "ScienceOne-AI/PhysLogic", "data/test.parquet"),
]:
    try:
        p = hf_hub_download(repo_id=repo, filename=fpath, repo_type="dataset")
        t = pq.read_table(p)
        row0 = {k: (str(v)[:180] if v is not None else None) for k, v in t.slice(0, 1).to_pylist()[0].items()}
        print(f"\n=== {name} ({repo}) rows_in_file={t.num_rows} cols={t.column_names} ===")
        print("sample0:", row0)
    except Exception as e:
        print(f"\n=== {name} ERR: {e} ===")
