"""Complete Molab execution scripts for the Alloy-9B Scientific Recovery & Scaling Masterplan."""

from __future__ import annotations

PHASE1_R2_SCRIPT = r'''import argparse, json, random, re, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HESITATION_PATTERNS = [
    r"(?:^|[\s\n.!?;:—-])wait[,.\s!:—-]",
    r"\blet me rethink\b",
    r"\bactually,?\s*(?:that's|that is)\s*wrong\b",
    r"\blet'?s start over\b",
    r"\bhold on\b",
    r"\boops\b",
    r"\bi made a mistake\b",
    r"\bthat doesn'?t look right\b",
    r"\bwait a (?:second|minute|moment)\b",
    r"\bhmm,?\s*no\b",
    r"\bstart again from scratch\b",
    r"\bi miscalculated\b",
    r"\bscratch that\b",
    r"\bno,?\s*that'?s incorrect\b",
    r"صبر کن",
    r"بذار دوباره فکر کنم",
    r"اشتباه شد",
]
HESITATION_RE = re.compile("|".join(HESITATION_PATTERNS), re.IGNORECASE | re.MULTILINE)
DIGITS_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

def norm_digits(s: str) -> str:
    return (s or "").translate(DIGITS_MAP)

def ensure_think_tags(txt: str) -> str:
    t = (txt or "").strip()
    if "</think>" in t and "<think>" not in t:
        return "<think>\n" + t
    return t

def passes_zero_hesitation(txt: str, tok, max_tokens: int = 2800) -> tuple[bool, int]:
    t = ensure_think_tags(txt)
    if t.count("<think>") != 1 or t.count("</think>") != 1:
        return False, 0
    o, c = t.find("<think>"), t.find("</think>")
    if c <= o or len(t[o + 7:c].strip()) < 8 or len(t[c + 8:].strip()) < 2:
        return False, 0
    if HESITATION_RE.search(t):
        return False, 0
    n = len(tok.encode(t, add_special_tokens=False))
    if n > max_tokens:
        return False, n
    return True, n

def extract_boxed_or_final(txt: str):
    t = norm_digits(txt)
    if "</think>" not in t:
        return None
    ans = t.split("</think>")[-1].strip()
    if not ans:
        return None
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", ans)
    if m:
        return m[-1].strip()
    m2 = re.findall(r"(?:Answer|answer|پاسخ|جواب|گزینه)\s*[:\-]?\s*\**([A-E1-4])\**", ans)
    if m2:
        return m2[-1].strip()
    return None

def answers_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    p = norm_digits(pred).strip().strip("$").strip()
    g = norm_digits(gold).strip().strip("$").strip()
    p = re.sub(r"^\\text\{(.*)\}$", r"\1", p).strip()
    g = re.sub(r"^\\text\{(.*)\}$", r"\1", g).strip()
    if p.lower() == g.lower():
        return True
    p_ns, g_ns = re.sub(r"\s+", "", p), re.sub(r"\s+", "", g)
    if p_ns.lower() == g_ns.lower():
        return True
    try:
        fp, fg = float(p_ns.replace(",", "")), float(g_ns.replace(",", ""))
        if abs(fp - fg) <= 1e-4 * max(1.0, abs(fg)):
            return True
    except Exception:
        pass
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        if simplify(parse_latex(p) - parse_latex(g)) == 0:
            return True
    except Exception:
        pass
    return False

def load_jsonl(p: Path):
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(p: Path, rows: list):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

def build_extended_stem_pool(pool_dir: Path, gate_dir: Path):
    """Curate verifiable Chemistry & Physics prompts from Nemotron-RL-Science-v1 + MegaScience."""
    out_p = pool_dir / "nemotron_science_verifiable.jsonl"
    if out_p.exists() and out_p.stat().st_size > 100_000:
        return load_jsonl(out_p)
    leak = json.loads((gate_dir / "leakage_index.json").read_text(encoding="utf-8"))
    exact_leak, prefix_leak = set(leak["exact"]), set(leak["prefixes"])

    def norm_l(s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "").lower().strip())
        return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

    def leaked(s: str) -> bool:
        n = norm_l(s)
        return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

    rows = []
    # 1. nvidia/Nemotron-RL-Science-v1 (Chemistry + Physics verifiable short answers)
    try:
        ds = load_dataset("nvidia/Nemotron-RL-Science-v1", split="so_openq", streaming=True)
        chem_cnt, phys_cnt = 0, 0
        for ex in ds:
            topic = (ex.get("metadata") or {}).get("topic", "")
            ans = (ex.get("expected_answer") or "").strip()
            prob = (ex.get("problem") or "").strip()
            if not prob or not ans or len(ans) > 80 or leaked(prob):
                continue
            if topic == "Chemistry" and chem_cnt < 1500:
                chem_cnt += 1
                rows.append({"source": "Nemotron-RL-Science/Chemistry", "prompt": prob + "\n\nPlace your final answer inside \\boxed{}.", "expected_answer": ans})
            elif topic == "Physics" and phys_cnt < 1500:
                phys_cnt += 1
                rows.append({"source": "Nemotron-RL-Science/Physics", "prompt": prob + "\n\nPlace your final answer inside \\boxed{}.", "expected_answer": ans})
            if chem_cnt >= 1500 and phys_cnt >= 1500:
                break
        print(f"[Pool] Nemotron-RL-Science-v1 curated: chem={chem_cnt}, phys={phys_cnt}", flush=True)
    except Exception as e:
        print(f"[Pool] Warning Nemotron-RL-Science-v1 failed: {e}", flush=True)

    # 2. nvidia/Nemotron-SFT-Science-v2 (syn_mcq Physics/Chemistry MCQ)
    try:
        ds_mcq = load_dataset("nvidia/Nemotron-SFT-Science-v2", "syn_mcq", split="train", streaming=True)
        mcq_cnt = 0
        for ex in ds_mcq:
            meta = ex.get("metadata") or {}
            topic = meta.get("topic", "")
            if topic not in ("Chemistry", "Physics"):
                continue
            msgs = ex.get("messages") or []
            if len(msgs) < 2:
                continue
            u = (msgs[0].get("content") or "").strip()
            a = (msgs[-1].get("content") or "").strip()
            m = re.findall(r"(?:\\boxed\{|\*\*Answer:\*\*|Answer:)\s*([A-J])", a)
            if not u or not m or leaked(u):
                continue
            rows.append({"source": f"Nemotron-Science-MCQ/{topic}", "prompt": u + "\n\nGive the final option letter inside \\boxed{}.", "expected_answer": m[-1]})
            mcq_cnt += 1
            if mcq_cnt >= 1500:
                break
        print(f"[Pool] Nemotron-SFT-Science-v2 syn_mcq curated: {mcq_cnt}", flush=True)
    except Exception as e:
        print(f"[Pool] Warning Nemotron-SFT-Science-v2 syn_mcq failed: {e}", flush=True)

    save_jsonl(out_p, rows)
    return rows

def run_pass1_onpolicy_and_probes(pool_dir: Path, out_dir: Path, gate_dir: Path, base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    rng = random.Random(2026)

    persian_pool = load_jsonl(pool_dir / "persian_prompts_pool.jsonl")
    opencode_pool = load_jsonl(pool_dir / "opencode_pool.jsonl")
    chem_pool = load_jsonl(pool_dir / "chemistry_pool.jsonl")
    opensci_pool = load_jsonl(pool_dir / "openscience_pool.jsonl")
    openmath_pool = load_jsonl(pool_dir / "openmath_pool.jsonl")
    nemo_sci_pool = build_extended_stem_pool(pool_dir, gate_dir)

    rng.shuffle(persian_pool)
    rng.shuffle(opencode_pool)
    rng.shuffle(chem_pool)
    rng.shuffle(opensci_pool)
    rng.shuffle(openmath_pool)
    rng.shuffle(nemo_sci_pool)

    # Masterplan Section 2: On-Policy Anchoring (2,400+ raw -> ~2,000+ post-Zero-Hesitation)
    # 50% Persian conversational/STEM prose (1400), 25% fundamental coding (700), 25% identity+formatting (handled in assemble: 700)
    anchor_items = []
    for r in persian_pool[:1400]:
        anchor_items.append({"source": f"on_policy/{r['source']}", "prompt": r["prompt"]})
    for r in opencode_pool[:700]:
        anchor_items.append({"source": f"on_policy/{r['source']}", "prompt": r["prompt"]})

    # Verifiable pass@4 probes targeting Chemistry, Physics, and Math (0/4 -> Qwen3.8-27B)
    probe_items = []
    for r in nemo_sci_pool[:1000]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"]})
    for r in opensci_pool[:600]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"]})
    dapo = [r for r in openmath_pool if r["source"] == "DAPO-Math-17k"]
    omr = [r for r in openmath_pool if r["source"] == "OpenMathReasoning"]
    for r in dapo[:350] + omr[:250]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"]})

    print(f"[Pass 1] Loading {base_model} in vLLM (anchors={len(anchor_items)}, pass@4_probes={len(probe_items)})...", flush=True)
    llm = LLM(model=base_model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, trust_remote_code=True)

    # Concise-prompting system hint? No: keep exact default user prompt so on-policy distribution is exact!
    anchor_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in anchor_items]
    t0 = time.time()
    a_outs = llm.generate(anchor_prompts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2800))
    print(f"[Pass 1] Generated {len(a_outs)} on-policy anchors in {time.time()-t0:.1f}s", flush=True)

    on_policy_rows = []
    dropped_hes = 0
    for it, out in zip(anchor_items, a_outs):
        txt = out.outputs[0].text
        ok, _ = passes_zero_hesitation(txt, tok, max_tokens=2800)
        if not ok:
            dropped_hes += 1
            continue
        on_policy_rows.append({
            "source": it["source"],
            "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": ensure_think_tags(txt)}],
        })
    save_jsonl(out_dir / "r2_on_policy_anchors.jsonl", on_policy_rows)
    print(f"[Pass 1] Kept {len(on_policy_rows)}/{len(anchor_items)} Zero-Hesitation anchors (dropped {dropped_hes}).", flush=True)

    probe_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in probe_items]
    t1 = time.time()
    p_outs = llm.generate(probe_prompts, SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=2800))
    print(f"[Pass 1] Generated pass@4 ({len(p_outs)*4} traces) in {time.time()-t1:.1f}s", flush=True)

    goldilocks_rows, hard_gap_rows = [], []
    counts = {"goldilocks_clean": 0, "hard_0_4": 0, "easy_4_4": 0}
    for it, out in zip(probe_items, p_outs):
        passing_clean = []
        any_pass = 0
        for cand in out.outputs:
            pred = extract_boxed_or_final(cand.text)
            if pred is not None and answers_match(pred, it["expected_answer"]):
                any_pass += 1
                ok, _ = passes_zero_hesitation(cand.text, tok, max_tokens=2800)
                if ok:
                    passing_clean.append(ensure_think_tags(cand.text))
        if 1 <= any_pass <= 3 and passing_clean:
            counts["goldilocks_clean"] += 1
            goldilocks_rows.append({
                "source": f"edge_pass4/{it['source']}",
                "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": min(passing_clean, key=len)}],
            })
        elif any_pass == 0:
            counts["hard_0_4"] += 1
            hard_gap_rows.append(it)
        else:
            counts["easy_4_4"] += 1
            if counts["easy_4_4"] % 8 == 0 and passing_clean:
                goldilocks_rows.append({
                    "source": f"easy_replay/{it['source']}",
                    "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": min(passing_clean, key=len)}],
                })

    save_jsonl(out_dir / "r2_goldilocks.jsonl", goldilocks_rows)
    save_jsonl(out_dir / "r2_hard_gap.jsonl", hard_gap_rows)
    print(f"[Pass 1] pass@4 summary: {counts}", flush=True)

def run_pass2_teacher(pool_dir: Path, out_dir: Path, teacher_model: str):
    tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    hard_gap = load_jsonl(out_dir / "r2_hard_gap.jsonl")
    opensci_pool = load_jsonl(pool_dir / "openscience_pool.jsonl")
    persian_pool = load_jsonl(pool_dir / "persian_prompts_pool.jsonl")

    # Add localized Persian STEM (verifiable!) + Persian-Math-SFT
    fa_stem_items = [r for r in opensci_pool[600:1400] if len(r["prompt"]) <= 1100][:450]
    fa_stem_prompts = [
        tok.apply_chat_template([{
            "role": "user",
            "content": f"این سوال علمی را به صورت دقیق، خلاصه و بدون تکرار به فارسی حل کن و پاسخ نهایی استاندارد را داخل \\boxed{{}} بنویس:\n\n{x['prompt']}"
        }], tokenize=False, add_generation_prompt=True)
        for x in fa_stem_items
    ]

    gap_prompts = [
        tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True)
        for x in hard_gap
    ]

    pm_items = [r for r in persian_pool if r.get("source") == "Persian-Math-SFT"][:400]
    pm_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in pm_items]

    print(f"[Pass 2] Loading {teacher_model} (hard_gap={len(gap_prompts)}, fa_stem={len(fa_stem_prompts)}, persian_math={len(pm_prompts)})...", flush=True)
    llm = LLM(model=teacher_model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, trust_remote_code=True)

    teacher_verified = []
    if gap_prompts:
        t0 = time.time()
        g_outs = llm.generate(gap_prompts, SamplingParams(n=2, temperature=0.5, top_p=0.95, max_tokens=2800))
        solved, kept_clean = 0, 0
        for it, out in zip(hard_gap, g_outs):
            clean_passing = []
            any_ok = False
            for cand in out.outputs:
                pred = extract_boxed_or_final(cand.text)
                if pred is not None and answers_match(pred, it["expected_answer"]):
                    any_ok = True
                    ok, _ = passes_zero_hesitation(cand.text, tok, max_tokens=2800)
                    if ok:
                        clean_passing.append(ensure_think_tags(cand.text))
            if any_ok:
                solved += 1
            if clean_passing:
                kept_clean += 1
                teacher_verified.append({
                    "source": f"teacher_qwen38_gap/{it['source']}",
                    "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": min(clean_passing, key=len)}],
                })
        print(f"[Pass 2] Hard gap solved={solved}/{len(hard_gap)}, zero-hesitation kept={kept_clean} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    fs_outs = llm.generate(fa_stem_prompts, SamplingParams(n=1, temperature=0.4, top_p=0.95, max_tokens=2800))
    fs_kept = 0
    for it, out in zip(fa_stem_items, fs_outs):
        txt = out.outputs[0].text
        pred = extract_boxed_or_final(txt)
        if pred is not None and answers_match(pred, it["expected_answer"]):
            ok, _ = passes_zero_hesitation(txt, tok, max_tokens=2800)
            if ok:
                fs_kept += 1
                teacher_verified.append({
                    "source": "teacher_qwen38/persian_stem_verified",
                    "messages": [
                        {"role": "user", "content": f"سوال علمی زیر را به فارسی دقیق حل کن و پاسخ نهایی را داخل \\boxed{{}} بنویس:\n\n{it['prompt']}"},
                        {"role": "assistant", "content": ensure_think_tags(txt)},
                    ],
                })
    print(f"[Pass 2] Persian STEM verified + zero-hesitation kept {fs_kept}/{len(fa_stem_items)} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    pm_outs = llm.generate(pm_prompts, SamplingParams(n=1, temperature=0.4, top_p=0.95, max_tokens=2600))
    pm_kept = 0
    for it, out in zip(pm_items, pm_outs):
        txt = out.outputs[0].text
        ok, _ = passes_zero_hesitation(txt, tok, max_tokens=2800)
        if ok:
            pm_kept += 1
            teacher_verified.append({
                "source": f"teacher_qwen38/{it['source']}",
                "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": ensure_think_tags(txt)}],
            })
    print(f"[Pass 2] Persian-Math zero-hesitation kept {pm_kept}/{len(pm_items)} in {time.time()-t0:.1f}s", flush=True)

    save_jsonl(out_dir / "r2_teacher_verified.jsonl", teacher_verified)

def assemble_r2_dataset(pool_dir: Path, out_dir: Path, gate_dir: Path, base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    rng = random.Random(2026)

    leak = json.loads((gate_dir / "leakage_index.json").read_text(encoding="utf-8"))
    exact_leak, prefix_leak = set(leak["exact"]), set(leak["prefixes"])

    def norm_l(s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "").lower().strip())
        return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)

    def leaked(s: str) -> bool:
        n = norm_l(s)
        return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

    on_policy = load_jsonl(out_dir / "r2_on_policy_anchors.jsonl")
    goldilocks = load_jsonl(out_dir / "r2_goldilocks.jsonl")
    teacher = load_jsonl(out_dir / "r2_teacher_verified.jsonl")
    identity_rows = load_jsonl(pool_dir / "identity_seed.jsonl")

    # Format identity cleanly (concise Alloy by Mob AI, ~400 items)
    formatted_id = []
    for idx, r in enumerate(identity_rows[:400]):
        msgs = []
        for m in r["messages"]:
            if m["role"] == "assistant" and "<think>" not in m["content"]:
                msgs.append({"role": "assistant", "content": f"<think>\nThe user is asking about my identity. I am Alloy, created by Mob AI.\n</think>\n\n{m['content']}"})
            else:
                msgs.append(m)
        formatted_id.append({"source": r["source"], "messages": msgs})

    # Optional: strictly filtered zero-hesitation <= 2000-tok concise OpenCode/SWE diffs (max 500)
    concise_code = []
    for r in load_jsonl(pool_dir / "opencode_pool.jsonl")[700:2500]:
        tr = ensure_think_tags(r.get("teacher_trace", ""))
        ok, n = passes_zero_hesitation(tr, tok, max_tokens=2200)
        if ok:
            concise_code.append({"source": "zero_hes_code/OpenCodeReasoning", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": tr}]})
        if len(concise_code) >= 400:
            break

    combined = on_policy + goldilocks + teacher + formatted_id + concise_code
    final_rows = []
    src_counts = {}
    dropped_leak, dropped_len = 0, 0
    for r in combined:
        msgs = r["messages"]
        if leaked(msgs[0]["content"]):
            dropped_leak += 1
            continue
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        n_toks = len(tok.encode(rendered, add_special_tokens=False))
        if n_toks > 3072 or n_toks < 16:
            dropped_len += 1
            continue
        final_rows.append({"source": r["source"], "num_tokens": n_toks, "text": rendered})
        top = r["source"].split("/")[0]
        src_counts[top] = src_counts.get(top, 0) + 1

    rng.shuffle(final_rows)
    save_jsonl(out_dir / "alloy_round2_train.jsonl", final_rows)
    summary = {
        "total_examples": len(final_rows),
        "total_tokens": sum(r["num_tokens"] for r in final_rows),
        "avg_tokens": round(sum(r["num_tokens"] for r in final_rows) / max(1, len(final_rows)), 1),
        "on_policy_pct": round(100.0 * (src_counts.get("on_policy", 0) + src_counts.get("edge_pass4", 0) + src_counts.get("easy_replay", 0)) / max(1, len(final_rows)), 2),
        "dropped_leak": dropped_leak,
        "dropped_len": dropped_len,
        "source_counts": src_counts,
    }
    (out_dir / "round2_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL_ROUND2_DATASET:", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

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
        run_pass1_onpolicy_and_probes(pool_dir, out_dir, gate_dir, args.base_model)
    if args.step in ("pass2", "all"):
        run_pass2_teacher(pool_dir, out_dir, args.teacher_model)
    if args.step in ("assemble", "all"):
        assemble_r2_dataset(pool_dir, out_dir, gate_dir, args.base_model)

if __name__ == "__main__":
    main()
'''
