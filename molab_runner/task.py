import datetime, json, os, re, shutil, subprocess
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals

def run_setup_and_pools(log):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    # 1. Rebuild /marimo/vllm_venv cleanly
    if not Path("/marimo/vllm_venv/bin/python").exists():
        log("Rebuilding /marimo/vllm_venv with vllm==0.34.0...")
        shutil.rmtree("/marimo/vllm_venv", ignore_errors=True)
        subprocess.run(["uv", "venv", "--python", "3.13", "/marimo/vllm_venv"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", "/marimo/vllm_venv/bin/python", "vllm==0.34.0"], check=True)
        subprocess.run(["uv", "pip", "uninstall", "--python", "/marimo/vllm_venv/bin/python", "torchaudio"], check=False)
    vllm_out = subprocess.check_output(
        ["/marimo/vllm_venv/bin/python", "-c", "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
        text=True,
    ).strip()
    log(f"vllm_venv ready: {vllm_out}")

    # 2. Pre-cache Ornith-1.5-9B and Qwen3.8-27B-FP8
    log("Pre-caching ornith-ai/Ornith-1.5-9B...")
    snapshot_download("ornith-ai/Ornith-1.5-9B")
    log("Pre-caching Qwen/Qwen3.8-27B-FP8...")
    snapshot_download("Qwen/Qwen3.8-27B-FP8")
    log("Both models cached cleanly!")

    # 3. Build public gate & leakage index (Konkur-250, PhysLogic-200, SUPERChem canary, ScienceBench-200)
    gate_dir = Path("/marimo/alloy_gate")
    pool_dir = Path("/marimo/alloy_data/phase1_pool")
    gate_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    import random
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

    # Load Konkur-1404 & PhysLogic
    konkur_ds = load_dataset("PersianML/persian-konkur-1404", split="train")
    by_exam = {}
    for r in konkur_ds:
        add_leak(r.get("question", ""))
        by_exam.setdefault(r.get("exam_name", "unknown"), []).append(r)
    rng = random.Random(42)
    konkur_sample = []
    for ex, rows in sorted(by_exam.items()):
        rng.shuffle(rows)
        for r in rows[:25]:
            opts = "\n".join(f"{i+1}) {r.get(f'option_{i+1}', '')}" for i in range(4))
            konkur_sample.append({
                "kind": "konkur",
                "id": f"{ex}_{r.get('id', len(konkur_sample))}",
                "exam": ex,
                "prompt": f"سوال کنکور زیر را به صورت گام‌به‌گام حل کن و در پایان فقط شماره گزینه صحیح (1، 2، 3 یا 4) را داخل \\boxed{{}} بنویس:\n\n{r['question']}\n\n{opts}",
                "gold": str(r["answer"]).strip(),
            })

    phys_ds = load_dataset("ScienceOne-AI/PhysLogic", split="test")
    by_diff = {}
    for r in phys_ds:
        add_leak(r.get("question", ""))
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)
    phys_sample = []
    for diff, rows in sorted(by_diff.items()):
        rng.shuffle(rows)
        for r in rows[:50]:
            phys_sample.append({
                "kind": "physlogic",
                "id": f"phys_{r.get('id', len(phys_sample))}",
                "difficulty": diff,
                "prompt": f"Solve the following physics problem step by step and place only the final answer inside \\boxed{{}}:\n\n{r['question']}",
                "gold": str(r["answer"]).strip(),
            })

    # Held-out ScienceBench-200 (100 Chemistry + 100 Physics from Nemotron-RL-Science-v1)
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

    # Save Nemotron training split (excluding held-out 200)
    nemo_train = nemo_chem[100:] + nemo_phys[100:]
    (pool_dir / "nemotron_science_verifiable.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in nemo_train) + "\n", encoding="utf-8"
    )
    log(f"Saved nemotron_science_verifiable.jsonl: {len(nemo_train)} rows (held out 200 for ScienceBench)")

    # Save SUPERChem canary to leakage index
    try:
        sc_ds = load_dataset("ZehuaZhao/SUPERChem", split="train", streaming=True)
        for i, r in enumerate(sc_ds):
            add_leak(str(r.get("question", "")))
            if i >= 1200:
                break
    except Exception as e:
        log(f"SUPERChem canary check note: {e}")

    (gate_dir / "leakage_index.json").write_text(
        json.dumps({"exact": sorted(exact_leak), "prefixes": sorted(prefix_leak)}, ensure_ascii=False), encoding="utf-8"
    )

    id_probes = [
        {"lang": "en", "prompt": "Who are you?"},
        {"lang": "en", "prompt": "Who made you?"},
        {"lang": "en", "prompt": "What is your model name and creator?"},
        {"lang": "en", "prompt": "Are you Qwen or Ornith or ChatGPT?"},
        {"lang": "fa", "prompt": "تو کی هستی و چه کسی تو را ساخته است؟"},
        {"lang": "fa", "prompt": "نام مدل تو چیست و سازنده‌ات کیست؟"},
    ]
    tool_probes = [
        {"prompt": "Call the `read_file` tool with path='README.md'. Return only the tool call."},
        {"prompt": "Run a bash command `pytest -q` using your tool format."},
        {"prompt": "Call `edit_file` on `src/app.py` to replace `x = 1` with `x = 2`."},
    ]

    gate_payload = {
        "konkur": konkur_sample,
        "physlogic": phys_sample,
        "science_bench": sci_bench,
        "identity_probes": id_probes,
        "tool_probes": tool_probes,
    }
    (gate_dir / "public_gate_659.json").write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved public_gate_659.json: konkur={len(konkur_sample)}, phys={len(phys_sample)}, science={len(sci_bench)}, leak={len(exact_leak)}")

    # 4. Also build identity, openmath, openscience, opencode, persian pools (reusing existing build_phase1_raw_pools helper logic)
    # Build identity_seed.jsonl (500)
    id_templates = [
        ("Who are you?", "I'm Alloy, an AI assistant created by Mob AI."),
        ("Who made you?", "I was created by Mob AI. My name is Alloy."),
        ("What is your name?", "My name is Alloy, developed by Mob AI."),
        ("Are you Qwen, Ornith, or ChatGPT?", "I'm Alloy, built by Mob AI — not ChatGPT or Qwen."),
        ("تو کی هستی و چه کسی تو را ساخته است؟", "من Alloy هستم، یک دستیار هوش مصنوعی که توسط Mob AI ساخته شده است."),
        ("نام مدل تو چیست؟", "نام من Alloy است و توسط Mob AI توسعه داده شده‌ام."),
        ("سازنده تو کیست؟", "سازنده من Mob AI است و نام من Alloy می‌باشد."),
    ]
    id_rows = []
    for i in range(400):
        q, a = id_templates[i % len(id_templates)]
        id_rows.append({"source": "alloy_identity", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    (pool_dir / "identity_seed.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in id_rows) + "\n", encoding="utf-8")

    # Persian prompts (Persian-SFT-102K + Persian-Math-SFT)
    fa_rows = []
    for ex in load_dataset("amiralifirouzi/Persian-SFT-102K", "fa", split="train", streaming=True):
        msgs = ex.get("messages") or []
        if len(msgs) >= 2 and (msgs[0].get("content") or "").strip():
            u = msgs[0]["content"].strip()
            if norm_l(u) not in exact_leak and len(u) <= 2000:
                fa_rows.append({"source": f"Persian-SFT-102K/{ex.get('task', 'chat')}", "prompt": u})
        if len(fa_rows) >= 3000:
            break
    for ex in load_dataset("xmanii/Persian-Math-SFT", split="train"):
        q = (ex.get("question") or "").strip()
        if q and norm_l(q) not in exact_leak:
            fa_rows.append({"source": "Persian-Math-SFT", "prompt": q})
    (pool_dir / "persian_prompts_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fa_rows) + "\n", encoding="utf-8")
    log(f"Saved persian_prompts_pool.jsonl: {len(fa_rows)}")

    # OpenCodeReasoning (3000 prompts)
    oc_rows = []
    for ex in load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True):
        inp = (ex.get("input") or "").strip()
        if inp and len(inp) <= 2500 and norm_l(inp) not in exact_leak:
            oc_rows.append({"source": "OpenCodeReasoning", "prompt": inp, "teacher_trace": ex.get("output", "")})
        if len(oc_rows) >= 3000:
            break
    (pool_dir / "opencode_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in oc_rows) + "\n", encoding="utf-8")
    log(f"Saved opencode_pool.jsonl: {len(oc_rows)}")

    # OpenScienceReasoning-2 (2500 verifiable prompts)
    os_rows = []
    for ex in load_dataset("nvidia/OpenScienceReasoning-2", split="train", streaming=True):
        inp = (ex.get("input") or "").strip()
        ans = (ex.get("expected_answer") or "").strip()
        if inp and ans and len(ans) <= 40 and len(inp) <= 2000 and norm_l(inp) not in exact_leak:
            os_rows.append({"source": "OpenScienceReasoning-2", "prompt": inp, "expected_answer": ans})
        if len(os_rows) >= 2500:
            break
    (pool_dir / "openscience_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in os_rows) + "\n", encoding="utf-8")
    log(f"Saved openscience_pool.jsonl: {len(os_rows)}")

    # DAPO-Math-17k + OpenMathReasoning (2500 verifiable prompts)
    om_rows = []
    for ex in load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", streaming=True):
        pr = ex.get("prompt") or []
        u = pr[0].get("content", "").strip() if isinstance(pr, list) and pr else str(pr).strip()
        ans = (ex.get("reward_model") or {}).get("ground_truth", "") if isinstance(ex.get("reward_model"), dict) else ""
        if u and ans and len(u) <= 2000 and norm_l(u) not in exact_leak:
            om_rows.append({"source": "DAPO-Math-17k", "prompt": u, "expected_answer": str(ans).strip()})
        if len(om_rows) >= 1500:
            break
    for ex in load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True):
        pr = (ex.get("problem") or "").strip()
        ans = (ex.get("expected_answer") or "").strip()
        rate = ex.get("pass_rate_72b_tir")
        if pr and ans and len(ans) <= 40 and (rate is None or 0.05 <= float(rate) <= 0.65) and norm_l(pr) not in exact_leak:
            om_rows.append({"source": "OpenMathReasoning", "prompt": pr, "expected_answer": ans})
        if len(om_rows) >= 2500:
            break
    (pool_dir / "openmath_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in om_rows) + "\n", encoding="utf-8")
    log(f"Saved openmath_pool.jsonl: {len(om_rows)}")

    return {"vllm": vllm_out, "nemo_train": len(nemo_train), "persian": len(fa_rows), "opencode": len(oc_rows), "openscience": len(os_rows), "openmath": len(om_rows)}

print("STARTING setup_and_pools:", g["start_alloy_job"]("setup_and_pools", run_setup_and_pools))
