import json, os, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals

SETUP_SCRIPT = r'''import ast, json, os, random, re
from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

print("[1/5] Checking Ornith-1.5-9B & Qwen3.8-27B-FP8 caches...", flush=True)
snapshot_download("ornith-ai/Ornith-1.5-9B", local_files_only=True)
snapshot_download("Qwen/Qwen3.8-27B-FP8", local_files_only=True)
print("[1/5] Both models verified locally cached.", flush=True)

gate_dir = Path("/marimo/alloy_gate")
pool_dir = Path("/marimo/alloy_data/phase1_pool")
gate_dir.mkdir(parents=True, exist_ok=True)
pool_dir.mkdir(parents=True, exist_ok=True)

def norm_l(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower().strip())
    return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

exact_leak, prefix_leak = set(), set()
def add_leak(s: str):
    n = norm_l(s)
    if n:
        exact_leak.add(n)
        if len(n) >= 100:
            prefix_leak.add(n[:100])

print("[2/5] Building public_gate_659.json & leakage_index.json...", flush=True)
rng = random.Random(42)

# Konkur-1404 (text-only, balanced 250 across exams)
konkur_ds = load_dataset("PersianML/persian-konkur-1404", split="train")
text_only = []
for row in konkur_ds:
    if row.get("figure") is None and row.get("question") and row.get("choices"):
        add_leak(row["question"])
        text_only.append({
            "id": row["id"],
            "exam_name": row["exam_name"],
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer_key": str(row["answer_key"]).strip(),
        })
by_exam = {}
for it in text_only:
    by_exam.setdefault(it["exam_name"], []).append(it)
konkur_250 = []
idx = 0
while len(konkur_250) < min(250, len(text_only)):
    added = False
    for ex in sorted(by_exam.keys()):
        if idx < len(by_exam[ex]):
            it = by_exam[ex][idx]
            cs = "\n".join(f"{i+1}) {c}" for i, c in enumerate(it["choices"]))
            konkur_250.append({
                "kind": "konkur",
                "id": it["id"],
                "exam": it["exam_name"],
                "prompt": f"سوال کنکور زیر را به صورت دقیق حل کن و در پایان فقط شماره گزینه صحیح (1 تا 4) را داخل \\boxed{{}} بنویس.\n\n{it['question']}\n\n{cs}",
                "gold": it["answer_key"],
            })
            added = True
            if len(konkur_250) >= 250:
                break
    if not added:
        break
    idx += 1

# PhysLogic (test.parquet, 200 choice questions balanced across subdomains)
phys_p = hf_hub_download("ScienceOne-AI/PhysLogic", "data/test.parquet", repo_type="dataset")
phys_rows = pq.read_table(phys_p).to_pylist()
for r in phys_rows:
    add_leak(r.get("question", ""))
phys_choice = [
    r for r in phys_rows
    if r.get("question_type") == "choice" and str(r.get("final_answer", "")).strip().upper() in ("A", "B", "C", "D", "E")
]
by_sub = {}
for r in phys_choice:
    by_sub.setdefault(r.get("subdomain", ""), []).append(r)
phys_200 = []
pidx = 0
while len(phys_200) < min(200, len(phys_choice)):
    added = False
    for sd in sorted(by_sub.keys()):
        if pidx < len(by_sub[sd]):
            r = by_sub[sd][pidx]
            phys_200.append({
                "kind": "physlogic",
                "id": r["uid"],
                "difficulty": r.get("difficulty", ""),
                "prompt": f"Solve the following physics multiple choice question carefully. Put only the final option letter (A, B, C, D, or E) inside \\boxed{{}}.\n\n{r['question'].strip()}",
                "gold": str(r["final_answer"]).strip().upper(),
            })
            added = True
            if len(phys_200) >= 200:
                break
    if not added:
        break
    pidx += 1

# Held-out ScienceBench-200 (100 Chem + 100 Phys from Nemotron-RL-Science-v1)
sci_bench = []
nemo_ds = load_dataset("nvidia/Nemotron-RL-Science-v1", split="so_openq", streaming=True)
nemo_chem, nemo_phys = [], []
for ex in nemo_ds:
    topic = (ex.get("metadata") or {}).get("topic", "")
    prob = (ex.get("problem") or "").strip()
    ans = (ex.get("expected_answer") or "").strip()
    if not prob or not ans or len(ans) > 50 or len(prob) > 1500:
        continue
    if topic == "Chemistry" and len(nemo_chem) < 1600:
        nemo_chem.append({"source": "Nemotron-RL-Science/Chemistry", "prompt": f"{prob}\n\nPlace your final answer inside \\boxed{{}}.", "expected_answer": ans})
    elif topic == "Physics" and len(nemo_phys) < 1600:
        nemo_phys.append({"source": "Nemotron-RL-Science/Physics", "prompt": f"{prob}\n\nPlace your final answer inside \\boxed{{}}.", "expected_answer": ans})
    if len(nemo_chem) >= 1600 and len(nemo_phys) >= 1600:
        break

for i, r in enumerate(nemo_chem[:100]):
    add_leak(r["prompt"])
    sci_bench.append({"kind": "science_chem", "id": f"chem_{i}", "prompt": r["prompt"], "gold": r["expected_answer"]})
for i, r in enumerate(nemo_phys[:100]):
    add_leak(r["prompt"])
    sci_bench.append({"kind": "science_phys", "id": f"phys_{i}", "prompt": r["prompt"], "gold": r["expected_answer"]})

nemo_train = nemo_chem[100:] + nemo_phys[100:]
(pool_dir / "nemotron_science_verifiable.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in nemo_train) + "\n", encoding="utf-8"
)
print(f"[2/5] Saved nemotron_science_verifiable.jsonl: {len(nemo_train)}", flush=True)

# SUPERChem canary shield
try:
    sc_ds = load_dataset("ZehuaZhao/SUPERChem", split="train", streaming=True)
    for i, r in enumerate(sc_ds):
        add_leak(str(r.get("question", "")))
        if i >= 1200:
            break
except Exception as e:
    print(f"[2/5] SUPERChem canary note: {e}", flush=True)

(gate_dir / "leakage_index.json").write_text(
    json.dumps({"exact": sorted(exact_leak), "prefixes": sorted(prefix_leak)}, ensure_ascii=False), encoding="utf-8"
)

gate_payload = {
    "konkur": konkur_250,
    "physlogic": phys_200,
    "science_bench": sci_bench,
    "identity_probes": [
        {"lang": "en", "prompt": "Who are you?"},
        {"lang": "en", "prompt": "Who made you?"},
        {"lang": "en", "prompt": "What is your model name and creator?"},
        {"lang": "en", "prompt": "Are you Qwen or Ornith or ChatGPT?"},
        {"lang": "fa", "prompt": "تو کی هستی و چه کسی تو را ساخته است؟"},
        {"lang": "fa", "prompt": "نام مدل تو چیست و سازنده‌ات کیست؟"},
    ],
    "tool_probes": [
        {"prompt": "Call the `read_file` tool with path='README.md'. Return only the tool call."},
        {"prompt": "Run a bash command `pytest -q` using your tool format."},
        {"prompt": "Call `edit_file` on `src/app.py` to replace `x = 1` with `x = 2`."},
    ],
}
(gate_dir / "public_gate_659.json").write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[2/5] Saved public_gate_659.json: konkur={len(konkur_250)}, phys={len(phys_200)}, sci={len(sci_bench)}, leak={len(exact_leak)}", flush=True)

def leaked(s: str) -> bool:
    n = norm_l(s)
    return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

print("[3/5] Building identity + Persian + OpenCode pools...", flush=True)
id_templates = [
    ("Who are you?", "I'm Alloy, an AI assistant created by Mob AI."),
    ("Who made you?", "I was created by Mob AI. My name is Alloy."),
    ("What is your name?", "My name is Alloy, developed by Mob AI."),
    ("Are you Qwen, Ornith, or ChatGPT?", "I'm Alloy, built by Mob AI — not ChatGPT or Qwen."),
    ("تو کی هستی و چه کسی تو را ساخته است؟", "من Alloy هستم، یک دستیار هوش مصنوعی که توسط Mob AI ساخته شده است."),
    ("نام مدل تو چیست؟", "نام من Alloy است و توسط Mob AI توسعه داده شده‌ام."),
    ("سازنده تو کیست؟", "سازنده من Mob AI است و نام من Alloy می‌باشد."),
]
id_rows = [{"source": "alloy_identity", "messages": [{"role": "user", "content": id_templates[i % len(id_templates)][0]}, {"role": "assistant", "content": id_templates[i % len(id_templates)][1]}]} for i in range(400)]
(pool_dir / "identity_seed.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in id_rows) + "\n", encoding="utf-8")

fa_rows = []
for ex in load_dataset("amiralifirouzi/Persian-SFT-102K", "fa", split="train", streaming=True):
    if ex.get("qc_flags"):
        continue
    msgs = ex.get("messages") or []
    if len(msgs) >= 2 and (msgs[0].get("content") or "").strip():
        u = msgs[0]["content"].strip()
        if len(u) >= 12 and len(u) <= 2000 and not leaked(u):
            fa_rows.append({"source": f"Persian-SFT-102K/{ex.get('task', 'chat')}", "prompt": u})
    if len(fa_rows) >= 3500:
        break
pm_p = hf_hub_download("xmanii/Persian-Math-SFT", "data/train-00000-of-00001.parquet", repo_type="dataset")
for r in pq.read_table(pm_p).to_pylist():
    q = (r.get("question") or "").strip()
    if q and not leaked(q):
        fa_rows.append({"source": "Persian-Math-SFT", "prompt": q, "ref_answer": r.get("answer", "")})
(pool_dir / "persian_prompts_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fa_rows) + "\n", encoding="utf-8")
print(f"[3/5] Saved persian_prompts_pool.jsonl: {len(fa_rows)}", flush=True)

oc_rows = []
for shard in ("split_0/train-00000-of-00030.parquet", "split_0/train-00001-of-00030.parquet"):
    op = hf_hub_download("nvidia/OpenCodeReasoning", shard, repo_type="dataset")
    ot = pq.read_table(op)
    oidx = list(range(ot.num_rows))
    rng.shuffle(oidx)
    for i in oidx:
        if len(oc_rows) >= 4000:
            break
        r = ot.slice(i, 1).to_pylist()[0]
        inp, out, sol = r.get("input", ""), r.get("output", ""), r.get("solution", "")
        if not inp or not out or not sol or len(inp) > 2500 or len(out) > 20000:
            continue
        try:
            ast.parse(sol)
        except Exception:
            continue
        if not leaked(inp):
            oc_rows.append({"source": "OpenCodeReasoning", "prompt": inp, "teacher_trace": out})
(pool_dir / "opencode_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in oc_rows) + "\n", encoding="utf-8")
print(f"[3/5] Saved opencode_pool.jsonl: {len(oc_rows)}", flush=True)

print("[4/5] Building OpenScienceReasoning-2 + OpenMath pools...", flush=True)
osr_p = hf_hub_download("nvidia/OpenScienceReasoning-2", "train/OpenScienceReasoning-2.parquet", repo_type="dataset")
osr_tbl = pq.read_table(osr_p)
osr_idx = list(range(osr_tbl.num_rows))
rng.shuffle(osr_idx)
os_rows = []
for i in osr_idx:
    if len(os_rows) >= 3500:
        break
    r = osr_tbl.slice(i, 1).to_pylist()[0]
    inp, ans = (r.get("input") or "").strip(), str(r.get("expected_answer", "")).strip()
    if inp and ans and len(ans) <= 40 and len(inp) <= 2000 and not leaked(inp):
        os_rows.append({"source": "OpenScienceReasoning-2", "prompt": inp, "expected_answer": ans})
(pool_dir / "openscience_pool.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in os_rows) + "\n", encoding="utf-8")
print(f"[4/5] Saved openscience_pool.jsonl: {len(os_rows)}", flush=True)

print("[5/5] Building DAPO-Math-17k + OpenMathReasoning pool...", flush=True)
om_rows = []
dapo_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
dapo_idx = list(range(len(dapo_ds)))
rng.shuffle(dapo_idx)
for i in dapo_idx:
    if len(om_rows) >= 2000:
        break
    r = dapo_ds[i]
    pr = r.get("prompt") or []
    u = pr[-1].get("content", "").strip() if isinstance(pr, list) and pr else str(pr).strip()
    ans = (r.get("reward_model") or {}).get("ground_truth", "") if isinstance(r.get("reward_model"), dict) else ""
    if u and ans and len(u) <= 2000 and not leaked(u):
        om_rows.append({"source": "DAPO-Math-17k", "prompt": u, "expected_answer": str(ans).strip()})

om_ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
scanned = 0
for r in om_ds:
    scanned += 1
    if len(om_rows) >= 4000 or scanned >= 40000:
        break
    pr, ans, rate = (r.get("problem") or "").strip(), str(r.get("expected_answer", "")).strip(), r.get("pass_rate_72b_tir")
    try:
        rf = float(rate)
    except Exception:
        continue
    if pr and ans and len(ans) <= 40 and (0.05 <= rf <= 0.65) and not leaked(pr):
        om_rows.append({"source": "OpenMathReasoning", "prompt": pr, "expected_answer": ans})
(pool_dir / "openmath_pool.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in om_rows) + "\n", encoding="utf-8")
print(f"[5/5] Saved openmath_pool.jsonl: {len(om_rows)} (scanned_omr={scanned})", flush=True)
print("SETUP_AND_POOLS_DONE", flush=True)
'''

Path("/marimo/alloy_scripts").mkdir(parents=True, exist_ok=True)
script_p = Path("/marimo/alloy_scripts/setup_and_pools_v2.py")
script_p.write_text(SETUP_SCRIPT, encoding="utf-8")

def run_setup_and_baseline(log):
    # 1. Sync /marimo/alloy_repo so alloy.public_eval & alloy.verifiers are up-to-date
    repo_dir = Path("/marimo/alloy_repo")
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "-b", "arena/2b4e287c-alloy", "https://github.com/mobjakdebug/alloy.git", str(repo_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", "arena/2b4e287c-alloy"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/arena/2b4e287c-alloy"], check=True)
    log("Synced /marimo/alloy_repo to arena/2b4e287c-alloy.")

    # 2. Run setup_and_pools_v2 with HF_HUB_DISABLE_XET=1
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    proc = subprocess.Popen(
        ["/tmp/uv-venv/bin/python", str(script_p)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        s = line.rstrip()
        if s:
            log(s)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"setup_and_pools_v2 exited with rc={rc}")

    # 3. Write & run baseline public_vllm_eval.py on Ornith-1.5-9B
    import sys
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT

    eval_script = Path("/marimo/alloy_scripts/public_vllm_eval.py")
    eval_script.write_text(PUBLIC_VLLM_EVAL_SCRIPT, encoding="utf-8")
    log("Running baseline vLLM eval on ornith-ai/Ornith-1.5-9B (public_gate_659.json)...")
    ep = subprocess.Popen(
        [
            "/marimo/vllm_venv/bin/python", str(eval_script),
            "--model", "ornith-ai/Ornith-1.5-9B",
            "--gate-path", "/marimo/alloy_gate/public_gate_659.json",
            "--out-path", "/marimo/alloy_gate/baseline_public_vllm.json",
            "--max-tokens", "6144",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    for line in ep.stdout:
        s = line.rstrip()
        if s and any(k in s for k in ["Running vLLM", "PUBLIC_EVAL_SUMMARY", "metrics", "Error", "Traceback", "Processed prompts", "elapsed_s", "identity_ok", "tool_ok", "konkur", "physlogic", "science_bench", "accuracy_pct", "parsed_pct"]):
            log(s)
    erc = ep.wait()
    if erc != 0:
        raise RuntimeError(f"public_vllm_eval exited with rc={erc}")
    return json.loads(Path("/marimo/alloy_gate/baseline_public_vllm.json").read_text(encoding="utf-8"))["metrics"]

print("STARTED setup_and_baseline:", g["start_alloy_job"]("setup_and_baseline", run_setup_and_baseline))
