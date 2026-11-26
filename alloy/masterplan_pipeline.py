"""Complete Molab execution scripts for the Alloy-9B Scientific Recovery & Scaling Masterplan."""

from __future__ import annotations

PHASE1_R2_SCRIPT = r'''import argparse, json, random, re, sys, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_DIR = Path("/marimo/alloy_repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.filters import passes_swe_diff_filter
from alloy.verifiers import answers_match, extract_boxed_or_final, norm_digits

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
    from datasets import load_dataset
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
    clean_chem = [
        r for r in nemo_sci_pool
        if "Chemistry" in r.get("source", "")
        and len((r.get("expected_answer") or "").strip()) <= 30
        and any(ch.isdigit() for ch in (r.get("expected_answer") or ""))
    ]
    clean_phys = [
        r for r in nemo_sci_pool
        if "Physics" in r.get("source", "")
        and len((r.get("expected_answer") or "").strip()) <= 30
        and any(ch.isdigit() for ch in (r.get("expected_answer") or ""))
    ]
    for r in clean_chem[:250] + clean_phys[:250]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"]})
    for r in opensci_pool[:300]:
        probe_items.append({"source": r["source"], "prompt": r["prompt"], "expected_answer": r["expected_answer"]})
    dapo = [r for r in openmath_pool if r["source"] == "DAPO-Math-17k"]
    omr = [r for r in openmath_pool if r["source"] == "OpenMathReasoning"]
    for r in dapo[:200] + omr[:100]:
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
    swe_pool = load_jsonl(pool_dir / "swe_pool.jsonl")
    rng = random.Random(123)
    rng.shuffle(swe_pool)

    # Add localized Persian STEM (verifiable!) + Persian-Math-SFT + SWE-rebench-V2
    fa_stem_items = [r for r in opensci_pool[600:1400] if len(r["prompt"]) <= 1100][:300]
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

    pm_items = [r for r in persian_pool if r.get("source") == "Persian-Math-SFT"][:300]
    pm_prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True) for x in pm_items]

    swe_targets = [
        r for r in swe_pool
        if r.get("problem_statement") and len(r["problem_statement"]) <= 5000
    ][:350]
    swe_user_prompts = [
        (
            f"Repository: `{x.get('repo', '')}` (Language: {x.get('language', 'python')})\n\n"
            f"Issue description:\n{x['problem_statement'][:2800]}\n\n"
            f"Target failing tests: {str(x.get('fail_to_pass', ''))[:300]}\n\n"
            "Diagnose the root cause concisely and output a minimal unified diff patch resolving the issue."
        )
        for x in swe_targets
    ]
    swe_prompts = [
        tok.apply_chat_template([{"role": "user", "content": up}], tokenize=False, add_generation_prompt=True)
        for up in swe_user_prompts
    ]

    print(f"[Pass 2] Loading {teacher_model} (hard_gap={len(gap_prompts)}, fa_stem={len(fa_stem_prompts)}, persian_math={len(pm_prompts)}, swe={len(swe_prompts)})...", flush=True)
    llm = LLM(model=teacher_model, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90, trust_remote_code=True)

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

    if swe_prompts:
        t0 = time.time()
        swe_outs = llm.generate(swe_prompts, SamplingParams(n=1, temperature=0.3, top_p=0.95, max_tokens=3584))
        swe_kept, swe_drop_reasons = 0, {}
        for it, up, out in zip(swe_targets, swe_user_prompts, swe_outs):
            txt = out.outputs[0].text
            ok_diff, diff_reason = passes_swe_diff_filter(txt)
            if not ok_diff:
                swe_drop_reasons[diff_reason] = swe_drop_reasons.get(diff_reason, 0) + 1
                continue
            ok_zh, n_zh = passes_zero_hesitation(txt, tok, max_tokens=3400)
            if not ok_zh:
                rkey = "hesitation_or_len" if n_zh == 0 else "exceeds_3400_tokens"
                swe_drop_reasons[rkey] = swe_drop_reasons.get(rkey, 0) + 1
                continue
            swe_kept += 1
            teacher_verified.append({
                "source": f"teacher_qwen38/SWE-rebench-V2/{it.get('language', 'python')}",
                "messages": [{"role": "user", "content": up}, {"role": "assistant", "content": ensure_think_tags(txt)}],
            })
        print(f"[Pass 2] SWE-rebench-V2 kept {swe_kept}/{len(swe_targets)} in {time.time()-t0:.1f}s (drops={swe_drop_reasons})", flush=True)

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
        if n_toks > 4096 or n_toks < 16:
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

SETUP_POOLS_R2_SCRIPT = r'''import ast, json, os, random, re, sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

print("[1/6] Caching ornith-ai/Ornith-1.5-9B & Qwen/Qwen3.8-27B-FP8 (HF_HUB_DISABLE_XET=1)...", flush=True)
snapshot_download("ornith-ai/Ornith-1.5-9B", max_workers=6)
snapshot_download("Qwen/Qwen3.8-27B-FP8", max_workers=6)
print("[1/6] Both models cached cleanly.", flush=True)

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

def leaked(s: str) -> bool:
    n = norm_l(s)
    return (n in exact_leak) or (len(n) >= 100 and n[:100] in prefix_leak)

print("[2/6] Building public_gate_659.json & leakage_index.json...", flush=True)
rng = random.Random(42)

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
konkur_250, idx = [], 0
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
phys_200, pidx = [], 0
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
print(f"[2/6] Saved public_gate_659.json: konkur={len(konkur_250)}, phys={len(phys_200)}, sci={len(sci_bench)}, leak={len(exact_leak)}", flush=True)

print("[3/6] Building identity + Persian + OpenCode pools...", flush=True)
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
        if 12 <= len(u) <= 2000 and not leaked(u):
            fa_rows.append({"source": f"Persian-SFT-102K/{ex.get('task', 'chat')}", "prompt": u})
    if len(fa_rows) >= 3500:
        break
pm_p = hf_hub_download("xmanii/Persian-Math-SFT", "data/train-00000-of-00001.parquet", repo_type="dataset")
for r in pq.read_table(pm_p).to_pylist():
    q = (r.get("question") or "").strip()
    if q and not leaked(q):
        fa_rows.append({"source": "Persian-Math-SFT", "prompt": q, "ref_answer": r.get("answer", "")})
(pool_dir / "persian_prompts_pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fa_rows) + "\n", encoding="utf-8")
print(f"[3/6] Saved persian_prompts_pool.jsonl: {len(fa_rows)}", flush=True)

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
print(f"[3/6] Saved opencode_pool.jsonl: {len(oc_rows)}", flush=True)

print("[4/6] Building OpenScienceReasoning-2 + OpenMath pools...", flush=True)
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
print(f"[4/6] Saved openscience_pool.jsonl: {len(os_rows)}", flush=True)

print("[5/6] Building DAPO-Math-17k + OpenMathReasoning pool...", flush=True)
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
print(f"[5/6] Saved openmath_pool.jsonl: {len(om_rows)} (scanned_omr={scanned})", flush=True)

print("[6/6] Building nebius/SWE-rebench-V2 pool (grade A/B, multi-lang)...", flush=True)
swe_rows, swe_by_lang = [], {}
swe_ds = load_dataset("nebius/SWE-rebench-V2", split="train", streaming=True)
for r in swe_ds:
    meta = r.get("meta") or {}
    lm = meta.get("llm_metadata") or {}
    if lm.get("code") not in ("A", "B"):
        continue
    lang = str(r.get("language", "")).lower()
    if lang not in ("python", "go", "ts", "js", "rust", "java", "cpp", "c"):
        continue
    if swe_by_lang.get(lang, 0) >= 300:
        if len(swe_rows) >= 1500:
            break
        continue
    ps, patch = r.get("problem_statement", ""), r.get("patch", "")
    if not ps or not patch or len(patch) > 16000 or len(ps) > 12000 or leaked(ps):
        continue
    swe_by_lang[lang] = swe_by_lang.get(lang, 0) + 1
    swe_rows.append({
        "source": "SWE-rebench-V2",
        "instance_id": r.get("instance_id", ""),
        "repo": r.get("repo", ""),
        "language": lang,
        "problem_statement": ps,
        "patch": patch,
        "fail_to_pass": str(r.get("FAIL_TO_PASS", ""))[:1000],
    })
    if len(swe_rows) >= 1500:
        break
(pool_dir / "swe_pool.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in swe_rows) + "\n", encoding="utf-8")
print(f"[6/6] Saved swe_pool.jsonl: {len(swe_rows)} ({swe_by_lang})", flush=True)
print("SETUP_POOLS_R2_DONE", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
'''

