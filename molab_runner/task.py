import marimo as mo
import marimo._code_mode as cm

CELL_PHASE1_DISTILL = r"""
PHASE1_OUT_DIR = ALLOY_DATA_ROOT / "phase1_verified"
PHASE1_OUT_DIR.mkdir(parents=True, exist_ok=True)

_PHASE1_DISTILL_SCRIPT = r'''
import argparse, json, random, re, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def norm_digits(s: str) -> str:
    return (s or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

def ensure_think_tags(txt: str) -> str:
    t = (txt or "").strip()
    if "</think>" in t and "<think>" not in t:
        return "<think>\n" + t
    return t

def extract_boxed_or_final(txt: str):
    t = norm_digits(txt)
    if "</think>" not in t:
        return None
    ans = t.split("</think>")[-1].strip()
    if not ans:
        return None
    # Match nested braces up to 2 levels for \boxed{...}
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", ans)
    if m:
        return m[-1].strip()
    # Fallback for single MCQ letter at end
    m2 = re.findall(r"(?:Answer|answer|پاسخ|جواب|گزینه)\s*[:\-]?\s*\**([A-E1-4])\**", ans)
    if m2:
        return m2[-1].strip()
    return None

def answers_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    p = norm_digits(pred).strip().strip("$").strip()
    g = norm_digits(gold).strip().strip("$").strip()
    # Strip \text{...} wrapper
    p = re.sub(r"^\\text\{(.*)\}$", r"\1", p).strip()
    g = re.sub(r"^\\text\{(.*)\}$", r"\1", g).strip()
    if p.lower() == g.lower():
        return True
    p_ns = re.sub(r"\s+", "", p)
    g_ns = re.sub(r"\s+", "", g)
    if p_ns.lower() == g_ns.lower():
        return True
    try:
        fp = float(p_ns.replace(",", ""))
        fg = float(g_ns.replace(",", ""))
        if abs(fp - fg) <= 1e-4 * max(1.0, abs(fg)):
            return True
    except Exception:
        pass
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        sp = parse_latex(p)
        sg = parse_latex(g)
        if simplify(sp - sg) == 0:
            return True
    except Exception:
        pass
    return False

def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def save_jsonl(p: Path, rows: list):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

def run_pass1_ornith(pool_dir: Path, out_dir: Path, base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    rng = random.Random(42)

    persian_pool = load_jsonl(pool_dir / "persian_prompts_pool.jsonl")
    opencode_pool = load_jsonl(pool_dir / "opencode_pool.jsonl")
    chem_pool = load_jsonl(pool_dir / "chemistry_pool.jsonl")
    opensci_pool = load_jsonl(pool_dir / "openscience_pool.jsonl")
    openmath_pool = load_jsonl(pool_dir / "openmath_pool.jsonl")

    rng.shuffle(persian_pool)
    rng.shuffle(opencode_pool)
    rng.shuffle(chem_pool)
    rng.shuffle(opensci_pool)
    rng.shuffle(openmath_pool)

    # A. On-policy anti-forgetting anchors (1200 persian non-math + 400 code + 400 chem = 2000)
    fa_non_math = [r for r in persian_pool if r.get("source") != "Persian-Math-SFT"]
    anchor_items = []
    for r in fa_non_math[:1200]:
        anchor_items.append({"source": f"on_policy/{r['source']}", "prompt": r["prompt"]})
    for r in opencode_pool[:400]:
        anchor_items.append({"source": f"on_policy/{r['source']}", "prompt": r["prompt"]})
    for r in chem_pool[:400]:
        q = r["messages"][0]["content"]
        anchor_items.append({"source": f"on_policy/{r['source']}", "prompt": q})

    # B. Edge-of-ability pass@4 probe (800 opensci + 800 dapo/openmath = 1600)
    probe_items = []
    for r in opensci_pool[:800]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"], "teacher_trace": r.get("teacher_trace", "")})
    dapo_items = [r for r in openmath_pool if r["source"] == "DAPO-Math-17k"]
    omr_items = [r for r in openmath_pool if r["source"] == "OpenMathReasoning"]
    for r in dapo_items[:500] + omr_items[:300]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"], "teacher_trace": r.get("teacher_trace", "")})

    print(f"[Pass 1] Loading {base_model} in vLLM (anchors={len(anchor_items)}, pass@4_probes={len(probe_items)})...", flush=True)
    llm = LLM(model=base_model, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.88, trust_remote_code=True)

    # Generate anchors
    anchor_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in anchor_items]
    t0 = time.time()
    a_outs = llm.generate(anchor_prompts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=3072))
    print(f"[Pass 1] Generated {len(a_outs)} on-policy anchors in {time.time()-t0:.1f}s", flush=True)

    on_policy_rows = []
    for it, out in zip(anchor_items, a_outs):
        txt = out.outputs[0].text
        if "</think>" not in txt:
            continue
        ans_tail = txt.split("</think>")[-1].strip()
        if len(ans_tail) < 10:
            continue
        full_txt = ensure_think_tags(txt)
        on_policy_rows.append({
            "source": it["source"],
            "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": full_txt}],
        })
    save_jsonl(out_dir / "on_policy_anchors.jsonl", on_policy_rows)
    print(f"[Pass 1] Kept {len(on_policy_rows)}/{len(anchor_items)} clean on-policy anchors.", flush=True)

    # Generate pass@4 probes
    probe_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in probe_items]
    t1 = time.time()
    p_outs = llm.generate(probe_prompts, SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=3584))
    print(f"[Pass 1] Generated pass@4 ({len(p_outs)*4} traces) in {time.time()-t1:.1f}s", flush=True)

    goldilocks_rows = []
    hard_gap_rows = []
    easy_replay_rows = []
    counts = {"goldilocks": 0, "hard_0_4": 0, "easy_4_4": 0}
    for it, out in zip(probe_items, p_outs):
        passing = []
        for cand in out.outputs:
            pred = extract_boxed_or_final(cand.text)
            if pred is not None and answers_match(pred, it["expected_answer"]):
                passing.append(ensure_think_tags(cand.text))
        k = len(passing)
        if 1 <= k <= 3:
            counts["goldilocks"] += 1
            best = min(passing, key=len)
            goldilocks_rows.append({
                "source": f"edge_pass4/{it['source']}",
                "pass_count": k,
                "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": best}],
            })
        elif k == 0:
            counts["hard_0_4"] += 1
            hard_gap_rows.append(it)
        else:
            counts["easy_4_4"] += 1
            if counts["easy_4_4"] % 7 == 0:
                best = min(passing, key=len)
                easy_replay_rows.append({
                    "source": f"easy_replay/{it['source']}",
                    "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": best}],
                })

    save_jsonl(out_dir / "goldilocks_self_distill.jsonl", goldilocks_rows + easy_replay_rows)
    save_jsonl(out_dir / "hard_gap_for_teacher.jsonl", hard_gap_rows)
    print(f"[Pass 1] pass@4 breakdown: {counts}", flush=True)

def run_pass2_teacher(pool_dir: Path, out_dir: Path, teacher_model: str):
    tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    rng = random.Random(123)

    hard_gap = load_jsonl(out_dir / "hard_gap_for_teacher.jsonl")
    chem_pool = load_jsonl(pool_dir / "chemistry_pool.jsonl")
    persian_pool = load_jsonl(pool_dir / "persian_prompts_pool.jsonl")
    opensci_pool = load_jsonl(pool_dir / "openscience_pool.jsonl")
    swe_pool = load_jsonl(pool_dir / "swe_pool.jsonl")

    rng.shuffle(chem_pool)
    rng.shuffle(swe_pool)

    # 1. Hard 0/4 gap items (verify against expected_answer)
    gap_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in hard_gap]

    # 2. Chemistry (1000 items from chemistry_pool -> clean <think> reasoning)
    chem_targets = chem_pool[400:1400]
    chem_prompts = [tok.apply_chat_template([{"role": "user", "content": x["messages"][0]["content"]}], tokenize=False, add_generation_prompt=True) for x in chem_targets]

    # 3. Persian-Math-SFT (1000) + Hard Persian reasoning/qa (1000)
    pm_items = [r for r in persian_pool if r.get("source") == "Persian-Math-SFT"][:1000]
    fa_hard_items = [r for r in persian_pool if r.get("source") in ("Persian-SFT-102K/reasoning", "Persian-SFT-102K/qa")][500:1500]
    pm_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in pm_items]
    fa_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in fa_hard_items]

    # 4. Localized Persian STEM (600 MCQ/numeric items from opensci_pool[1000:2600] translated & solved in Persian, verified against gold answer!)
    fa_stem_items = [r for r in opensci_pool[1000:2600] if len(r["prompt"]) < 1400 and len(r.get("expected_answer", "")) <= 40][:600]
    fa_stem_prompts = []
    for x in fa_stem_items:
        p = (
            "Translate the following scientific problem accurately into fluent academic Persian, solve it step by step in Persian, "
            "and put the exact final answer (keep symbols/letters like A/B/C/D or numbers unchanged) inside \\boxed{}.\n\n"
            f"{x['prompt']}"
        )
        fa_stem_prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))

    # 5. SWE-rebench-V2 grounded engineering reasoning (1200 multi-lang Grade A/B tasks)
    swe_targets = swe_pool[:1200]
    swe_prompts = []
    for x in swe_targets:
        sp = (
            f"You are debugging repository `{x['repo']}` ({x['language']}).\n"
            f"Issue:\n{x['problem_statement'][:4500]}\n\n"
            f"Failing tests to fix: {x['fail_to_pass'][:400]}\n\n"
            f"Verified fix patch:\n```diff\n{x['patch'][:6000]}\n```\n\n"
            "Write a clear, concise engineering analysis explaining the root cause, the exact files/symbols involved, "
            "why the failing tests failed, and present the complete minimal patch in a ```diff``` block."
        )
        swe_prompts.append(tok.apply_chat_template([{"role": "user", "content": sp}], tokenize=False, add_generation_prompt=True))

    print(f"[Pass 2] Loading teacher {teacher_model} in vLLM...", flush=True)
    llm = LLM(model=teacher_model, dtype="auto", max_model_len=12288, gpu_memory_utilization=0.90, trust_remote_code=True)

    teacher_verified = []

    # 2A. Hard 0/4 gap (n=2)
    if gap_prompts:
        t0 = time.time()
        g_outs = llm.generate(gap_prompts, SamplingParams(n=2, temperature=0.6, top_p=0.95, max_tokens=4096))
        solved = 0
        for it, out in zip(hard_gap, g_outs):
            passing = []
            for cand in out.outputs:
                pred = extract_boxed_or_final(cand.text)
                if pred is not None and answers_match(pred, it["expected_answer"]):
                    passing.append(ensure_think_tags(cand.text))
            if passing:
                solved += 1
                teacher_verified.append({
                    "source": f"teacher_qwen38_gap/{it['source']}",
                    "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": min(passing, key=len)}],
                })
        print(f"[Pass 2] Hard gap solved & verified: {solved}/{len(hard_gap)} in {time.time()-t0:.1f}s", flush=True)

    # 2B. Chemistry (n=1, verified non-empty post-think)
    t0 = time.time()
    c_outs = llm.generate(chem_prompts, SamplingParams(temperature=0.4, top_p=0.95, max_tokens=3072))
    chem_kept = 0
    for it, out in zip(chem_targets, c_outs):
        txt = out.outputs[0].text
        if "</think>" in txt and len(txt.split("</think>")[-1].strip()) >= 5:
            chem_kept += 1
            teacher_verified.append({
                "source": f"teacher_qwen38/{it['source']}",
                "messages": [{"role": "user", "content": it["messages"][0]["content"]}, {"role": "assistant", "content": ensure_think_tags(txt)}],
            })
    print(f"[Pass 2] Chemistry teacher kept {chem_kept}/{len(chem_targets)} in {time.time()-t0:.1f}s", flush=True)

    # 2C. Persian Math + Hard Persian
    t0 = time.time()
    pm_outs = llm.generate(pm_prompts, SamplingParams(temperature=0.4, top_p=0.95, max_tokens=3072))
    fa_outs = llm.generate(fa_prompts, SamplingParams(temperature=0.5, top_p=0.95, max_tokens=3072))
    fa_kept = 0
    for it, out in zip(pm_items + fa_hard_items, pm_outs + fa_outs):
        txt = out.outputs[0].text
        if "</think>" in txt and len(txt.split("</think>")[-1].strip()) >= 10:
            fa_kept += 1
            teacher_verified.append({
                "source": f"teacher_qwen38/{it['source']}",
                "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": ensure_think_tags(txt)}],
            })
    print(f"[Pass 2] Persian Math/Reasoning teacher kept {fa_kept}/{len(pm_items)+len(fa_hard_items)} in {time.time()-t0:.1f}s", flush=True)

    # 2D. Native Persian STEM verified against gold answer
    t0 = time.time()
    fs_outs = llm.generate(fa_stem_prompts, SamplingParams(temperature=0.4, top_p=0.95, max_tokens=3584))
    fs_kept = 0
    for it, out in zip(fa_stem_items, fs_outs):
        txt = out.outputs[0].text
        pred = extract_boxed_or_final(txt)
        if pred is not None and answers_match(pred, it["expected_answer"]):
            # Store clean Persian problem + solution if present, else direct prompt
            fs_kept += 1
            teacher_verified.append({
                "source": "teacher_qwen38/persian_stem_verified",
                "messages": [
                    {"role": "user", "content": f"سوال علمی زیر را به فارسی دقیق حل کن و پاسخ نهایی را داخل \\boxed{{}} بنویس:\n\n{it['prompt']}"},
                    {"role": "assistant", "content": ensure_think_tags(txt)},
                ],
            })
    print(f"[Pass 2] Persian STEM verified against gold: {fs_kept}/{len(fa_stem_items)} in {time.time()-t0:.1f}s", flush=True)

    # 2E. SWE-rebench-V2 grounded debugging trajectories
    t0 = time.time()
    swe_outs = llm.generate(swe_prompts, SamplingParams(temperature=0.3, top_p=0.95, max_tokens=3072))
    swe_kept = 0
    for it, out in zip(swe_targets, swe_outs):
        txt = out.outputs[0].text
        if "</think>" in txt and "```diff" in txt.split("</think>")[-1]:
            swe_kept += 1
            user_prompt = (
                f"Repository: `{it['repo']}` (Language: {it['language']})\n\n"
                f"Issue description:\n{it['problem_statement'][:4500]}\n\n"
                f"Target failing tests: {it['fail_to_pass'][:400]}\n\n"
                "Diagnose the root cause and provide the minimal unified diff patch to resolve this issue."
            )
            teacher_verified.append({
                "source": f"teacher_qwen38/SWE-rebench-V2/{it['language']}",
                "messages": [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": ensure_think_tags(txt)}],
            })
    print(f"[Pass 2] SWE-rebench-V2 grounded engineering kept {swe_kept}/{len(swe_targets)} in {time.time()-t0:.1f}s", flush=True)

    save_jsonl(out_dir / "teacher_qwen38_verified.jsonl", teacher_verified)

def assemble_final_dataset(pool_dir: Path, out_dir: Path, gate_dir: Path, base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    rng = random.Random(2026)

    leak = json.loads((gate_dir / "leakage_index.json").read_text(encoding="utf-8"))
    exact_leak = set(leak["exact"])
    prefix_leak = set(leak["prefixes"])

    def norm_leak(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

    def leaked(s: str) -> bool:
        n = norm_leak(s)
        return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

    on_policy = load_jsonl(out_dir / "on_policy_anchors.jsonl")
    goldilocks = load_jsonl(out_dir / "goldilocks_self_distill.jsonl")
    teacher_rows = load_jsonl(out_dir / "teacher_qwen38_verified.jsonl")
    identity_rows = load_jsonl(pool_dir / "identity_seed.jsonl")

    # Add curated DeepSeek-R1 verified traces (<= 13000 chars so concise & <= 5000 tokens)
    opensci = [r for r in load_jsonl(pool_dir / "openscience_pool.jsonl")[1600:] if len(r.get("teacher_trace","")) <= 13000]
    openmath = [r for r in load_jsonl(pool_dir / "openmath_pool.jsonl") if r.get("teacher_trace") and len(r["teacher_trace"]) <= 13000]
    opencode = [r for r in load_jsonl(pool_dir / "opencode_pool.jsonl")[400:] if len(r.get("teacher_trace","")) <= 13000]
    chem_orig = [r for r in load_jsonl(pool_dir / "chemistry_pool.jsonl")[1400:] if len(r["messages"][-1]["content"]) <= 8000]

    rng.shuffle(opensci); rng.shuffle(openmath); rng.shuffle(opencode); rng.shuffle(chem_orig)

    r1_rows = []
    for r in opensci[:2500]:
        r1_rows.append({"source": "r1_verified/OpenScienceReasoning-2", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": ensure_think_tags(r["teacher_trace"])}]})
    for r in openmath[:2500]:
        r1_rows.append({"source": "r1_verified/OpenMathReasoning", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": ensure_think_tags(r["teacher_trace"])}]})
    for r in opencode[:2500]:
        r1_rows.append({"source": f"r1_verified/{r['source']}", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": ensure_think_tags(r["teacher_trace"])}]})
    for r in chem_orig[:1200]:
        r1_rows.append({"source": r["source"], "messages": r["messages"]})

    # Format identity rows cleanly (half with <think> reinforcing Alloy, half empty-think)
    formatted_id = []
    for idx, r in enumerate(identity_rows):
        msgs = []
        for m in r["messages"]:
            if m["role"] == "assistant" and "<think>" not in m["content"]:
                if idx % 2 == 0 and m["content"].startswith(("I'm Alloy", "My name is Alloy", "من Alloy", "نام من Alloy")):
                    msgs.append({"role": "assistant", "content": f"<think>\nThe user is asking about my identity. I am Alloy, made by Mob AI.\n</think>\n\n{m['content']}"})
                else:
                    msgs.append(m)
            else:
                msgs.append(m)
        formatted_id.append({"source": r["source"], "messages": msgs})

    combined = on_policy + goldilocks + teacher_rows + r1_rows + formatted_id

    final_rows = []
    src_counts = {}
    dropped_len = 0
    dropped_leak = 0
    for r in combined:
        msgs = r["messages"]
        u0 = msgs[0]["content"]
        if leaked(u0):
            dropped_leak += 1
            continue
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        n_toks = len(tok.encode(rendered, add_special_tokens=False))
        if n_toks > 6144 or n_toks < 16:
            dropped_len += 1
            continue
        final_rows.append({"source": r["source"], "num_tokens": n_toks, "text": rendered})
        top_src = r["source"].split("/")[0]
        src_counts[top_src] = src_counts.get(top_src, 0) + 1

    rng.shuffle(final_rows)
    save_jsonl(out_dir / "alloy_round1_train.jsonl", final_rows)
    summary = {
        "total_examples": len(final_rows),
        "total_tokens": sum(r["num_tokens"] for r in final_rows),
        "avg_tokens": round(sum(r["num_tokens"] for r in final_rows) / max(1, len(final_rows)), 1),
        "dropped_len": dropped_len,
        "dropped_leak": dropped_leak,
        "source_counts": src_counts,
    }
    (out_dir / "round1_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL_ROUND1_DATASET:", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["pass1", "pass2", "assemble", "all"], default="all")
    ap.add_argument("--base-model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--teacher-model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate-dir", required=True)
    args = ap.parse_args()

    pool_dir, out_dir, gate_dir = Path(args.pool_dir), Path(args.out_dir), Path(args.gate_dir)
    if args.step in ("pass1", "all"):
        run_pass1_ornith(pool_dir, out_dir, args.base_model)
    if args.step in ("pass2", "all"):
        run_pass2_teacher(pool_dir, out_dir, args.teacher_model)
    if args.step in ("assemble", "all"):
        assemble_final_dataset(pool_dir, out_dir, gate_dir, args.base_model)

if __name__ == "__main__":
    main()
'''

def run_phase1_distill(log, base_model: str = "ornith-ai/Ornith-1.5-9B", teacher_model: str = "Qwen/Qwen3.8-27B-FP8", hf_org: str = "", hf_token: str = "", step: str = "all"):
    from huggingface_hub import HfApi
    _script_p = ALLOY_SCRIPTS_DIR / "phase1_distill.py"
    _script_p.write_text(_PHASE1_DISTILL_SCRIPT, encoding="utf-8")
    _env = os.environ.copy()
    if hf_token:
        _env["HF_TOKEN"] = hf_token

    _steps = ["pass1", "pass2", "assemble"] if step == "all" else [step]
    for _st in _steps:
        _cmd = [
            VLLM_PYTHON, str(_script_p),
            "--step", _st,
            "--base-model", base_model,
            "--teacher-model", teacher_model,
            "--pool-dir", str(PHASE1_POOL_DIR),
            "--out-dir", str(PHASE1_OUT_DIR),
            "--gate-dir", str(ALLOY_GATE_ROOT),
        ]
        log(f"Launching Phase 1 step '{_st}': {' '.join(_cmd)}")
        _proc = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env)
        for _line in _proc.stdout:
            _s = _line.rstrip()
            if any(k in _s for k in ("[Pass ", "FINAL_ROUND1_DATASET", "ERROR", "Error", "Traceback", "Loading safetensors", "Exception", "ValueError")):
                log(_s)
        _rc = _proc.wait()
        if _rc != 0:
            raise RuntimeError(f"phase1_distill.py ({_st}) exited with rc={_rc}")

    _stats = json.loads((PHASE1_OUT_DIR / "round1_stats.json").read_text(encoding="utf-8"))
    if hf_org and hf_token:
        _api = HfApi(token=hf_token)
        for _fn in ("alloy_round1_train.jsonl", "round1_stats.json"):
            _api.upload_file(
                path_or_fileobj=str(PHASE1_OUT_DIR / _fn),
                path_in_repo=f"round1/{_fn}",
                repo_id=f"{hf_org}/alloy-data",
                repo_type="dataset",
                commit_message=f"Upload Phase 1 verified Round 1 dataset ({_fn})",
            )
        log(f"Uploaded alloy_round1_train.jsonl & round1_stats.json to {hf_org}/alloy-data")
    return _stats

mo.md("### 12. Phase 1 Verified Anti-Forgetting Distillation & Dataset Assembly (Ornith pass@4 + Qwen3.8-27B-FP8)")
""".strip()

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name == "phase1_distill":
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c_distill = ctx.create_cell(CELL_PHASE1_DISTILL, after=last_cid, name="phase1_distill", hide_code=False)
    ctx.run_cell(c_distill)

g = cm.get_context().globals
msg = g["start_alloy_job"](
    "phase1_distill",
    g["run_phase1_distill"],
    base_model=g["base_model_ui"].value,
    teacher_model="Qwen/Qwen3.8-27B-FP8",
    hf_org=g["hf_org_ui"].value.strip(),
    hf_token=g["get_hf_token"](),
    step="all",
)
print("START_MSG:", msg)
mo.status.toast(
    title="Phase 1 Verified Distillation Started!",
    description="Running Ornith pass@4 + Qwen/Qwen3.8-27B-FP8 teacher verification & uploading to officialmobai/alloy-data.",
)
