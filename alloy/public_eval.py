"""vLLM evaluator for public_gate_659.json (Konkur-250, PhysLogic-200, ScienceBench-200, Identity-6, Tool-3)."""

from __future__ import annotations

PUBLIC_VLLM_EVAL_SCRIPT = r'''import argparse, json, sys, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

REPO_DIR = Path("/marimo/alloy_repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.verifiers import answers_match, extract_boxed_or_final, norm_digits
import re


def extract_choice_or_boxed(txt: str, valid: str):
    t = norm_digits(txt)
    boxed = extract_boxed_or_final(t)
    if boxed:
        m = re.findall(rf"[{valid}]", boxed, flags=re.IGNORECASE if "A" in valid else 0)
        if m:
            return m[-1].upper() if "A" in valid else m[-1]
    if "</think>" in t:
        ans = t.split("</think>")[-1].strip()
        for p in [
            rf"\\boxed\{{\s*([{valid}])\s*\}}",
            rf"(?:پاسخ|جواب|گزینه|Answer|answer)\s*[:\-]?\s*\**([{valid}])\**",
            rf"\b([{valid}])\b",
        ]:
            m = re.findall(p, ans, flags=re.IGNORECASE if "A" in valid else 0)
            if m:
                return m[-1].upper() if "A" in valid else m[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-path", default="")
    ap.add_argument("--gate-path", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--max-tokens", type=int, default=6144)
    args = ap.parse_args()

    gate = json.loads(Path(args.gate_path).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
        enable_lora=bool(args.lora_path),
        max_lora_rank=128 if args.lora_path else 16,
    )
    lora_req = LoRARequest("alloy_adapter", 1, args.lora_path) if args.lora_path else None

    tool_defs = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "bash", "description": "Run bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Edit file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    ]

    prompts, meta = [], []
    for p in gate["identity_probes"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True))
        meta.append(("identity", p))

    for tp in gate["tool_probes"]:
        tp_txt = tp["prompt"] if isinstance(tp, dict) else str(tp)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": tp_txt}], tools=tool_defs, tokenize=False, add_generation_prompt=True))
        meta.append(("tool", tp_txt))

    for it in gate["konkur"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True))
        meta.append(("konkur", it))

    for it in gate["physlogic"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True))
        meta.append(("physlogic", it))

    for it in gate["science_bench"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True))
        meta.append(("science_bench", it))

    print(f"Running vLLM public gate eval on {len(prompts)} prompts (max_tokens={args.max_tokens})...", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens), lora_request=lora_req)
    elapsed = round(time.time() - t0, 2)
    total_gen_toks = sum(len(o.outputs[0].token_ids) for o in outputs)

    id_res, tool_res, detailed = [], [], []
    id_ok, tool_ok = 0, 0
    stats = {
        "konkur": {"total": 0, "correct": 0, "parsed": 0, "truncated": 0},
        "physlogic": {"total": 0, "correct": 0, "parsed": 0, "truncated": 0},
        "science_bench": {"total": 0, "correct": 0, "parsed": 0, "truncated": 0, "by_kind": {}},
    }

    for (kind, item), out in zip(meta, outputs):
        txt = out.outputs[0].text
        ntoks = len(out.outputs[0].token_ids)
        is_trunc = ntoks >= args.max_tokens
        if kind == "identity":
            ok_id = ("alloy" in txt.lower() and "mob ai" in txt.lower())
            id_ok += int(ok_id)
            id_res.append({"lang": item["lang"], "prompt": item["prompt"], "ok": ok_id, "response": txt[:800]})
        elif kind == "tool":
            ok_tl = ("<tool_call>" in txt and "</tool_call>" in txt)
            tool_ok += int(ok_tl)
            tool_res.append({"prompt": item, "ok": ok_tl, "response": txt[:1200]})
        elif kind == "konkur":
            pred = extract_choice_or_boxed(txt, "1-4")
            ok = (pred == str(item["gold"]).strip())
            stats["konkur"]["total"] += 1
            stats["konkur"]["parsed"] += int(pred is not None)
            stats["konkur"]["correct"] += int(ok)
            stats["konkur"]["truncated"] += int(is_trunc)
            detailed.append({"kind": "konkur", "id": item["id"], "gold": item["gold"], "pred": pred, "ok": ok, "toks": ntoks})
        elif kind == "physlogic":
            pred = extract_choice_or_boxed(txt, "A-E")
            ok = (pred == str(item["gold"]).strip().upper())
            stats["physlogic"]["total"] += 1
            stats["physlogic"]["parsed"] += int(pred is not None)
            stats["physlogic"]["correct"] += int(ok)
            stats["physlogic"]["truncated"] += int(is_trunc)
            detailed.append({"kind": "physlogic", "id": item["id"], "gold": item["gold"], "pred": pred, "ok": ok, "toks": ntoks})
        elif kind == "science_bench":
            pred = extract_boxed_or_final(txt)
            ok = bool(pred is not None and answers_match(pred, item["gold"]))
            stats["science_bench"]["total"] += 1
            stats["science_bench"]["parsed"] += int(pred is not None)
            stats["science_bench"]["correct"] += int(ok)
            stats["science_bench"]["truncated"] += int(is_trunc)
            bk = stats["science_bench"]["by_kind"].setdefault(item["kind"], {"total": 0, "correct": 0, "parsed": 0})
            bk["total"] += 1
            bk["parsed"] += int(pred is not None)
            bk["correct"] += int(ok)
            detailed.append({"kind": item["kind"], "id": item["id"], "gold": item["gold"], "pred": pred, "ok": ok, "toks": ntoks})

    for k in ("konkur", "physlogic", "science_bench"):
        t = max(1, stats[k]["total"])
        stats[k]["accuracy_pct"] = round(100.0 * stats[k]["correct"] / t, 2)
        stats[k]["parsed_pct"] = round(100.0 * stats[k]["parsed"] / t, 2)

    summary = {
        "model": args.model,
        "lora_path": args.lora_path or None,
        "elapsed_seconds": elapsed,
        "total_generated_tokens": total_gen_toks,
        "agg_tok_per_sec": round(total_gen_toks / max(0.1, elapsed), 1),
        "identity_ok": f"{id_ok}/{len(id_res)}",
        "tool_ok": f"{tool_ok}/{len(tool_res)}",
        "metrics": stats,
        "identity_probes": id_res,
        "tool_probes": tool_res,
        "detailed": detailed,
    }
    Path(args.out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PUBLIC_EVAL_SUMMARY:", json.dumps({
        "elapsed_s": elapsed,
        "total_tokens": total_gen_toks,
        "tok_per_sec": summary["agg_tok_per_sec"],
        "identity_ok": summary["identity_ok"],
        "tool_ok": summary["tool_ok"],
        "metrics": stats,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''
