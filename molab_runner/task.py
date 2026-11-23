import marimo as mo
import marimo._code_mode as cm

CELL_VLLM_GATE_AND_EVAL = r"""
ALLOY_SCRIPTS_DIR = Path("/marimo/alloy_scripts")
ALLOY_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
VLLM_PYTHON = "/marimo/vllm_venv/bin/python"

def build_full_frozen_gate(log, hf_org: str = "", hf_token: str = ""):
    import csv, random
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    log("Loading PersianML/persian-konkur-1404...")
    _ds = load_dataset("PersianML/persian-konkur-1404", split="train")
    _text_only = []
    for _row in _ds:
        if _row.get("figure") is None and _row.get("question") and _row.get("choices"):
            _text_only.append({
                "id": _row["id"],
                "exam_name": _row["exam_name"],
                "question": _row["question"],
                "choices": list(_row["choices"]),
                "answer_key": str(_row["answer_key"]).strip(),
            })
    _by_exam = {}
    for _it in _text_only:
        _by_exam.setdefault(_it["exam_name"], []).append(_it)
    _konkur_250 = []
    _idx = 0
    while len(_konkur_250) < min(250, len(_text_only)):
        _added = False
        for _ex in sorted(_by_exam.keys()):
            if _idx < len(_by_exam[_ex]):
                _konkur_250.append(_by_exam[_ex][_idx])
                _added = True
                if len(_konkur_250) >= 250:
                    break
        if not _added:
            break
        _idx += 1

    log("Loading Idavidrein/gpqa (gpqa_diamond.csv)...")
    _gpqa_path = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", token=hf_token or None)
    _rng = random.Random(42)
    _gpqa_items = []
    with open(_gpqa_path, "r", encoding="utf-8") as _f:
        for _i, _row in enumerate(csv.DictReader(_f)):
            _opts = [
                (_row["Correct Answer"].strip(), True),
                (_row["Incorrect Answer 1"].strip(), False),
                (_row["Incorrect Answer 2"].strip(), False),
                (_row["Incorrect Answer 3"].strip(), False),
            ]
            _rng.shuffle(_opts)
            _gold_idx = [j for j, (_, is_ok) in enumerate(_opts) if is_ok][0]
            _gpqa_items.append({
                "id": f"gpqa_diamond_{_i}",
                "domain": _row.get("High-level domain", ""),
                "subdomain": _row.get("Subdomain", ""),
                "question": _row["Question"].strip(),
                "choices": [o[0] for o in _opts],
                "answer_letter": "ABCD"[_gold_idx],
            })

    log("Loading ScienceOne-AI/PhysLogic...")
    _phys_path = hf_hub_download("ScienceOne-AI/PhysLogic", "data/test.parquet", repo_type="dataset")
    _phys_rows = pq.read_table(_phys_path).to_pylist()
    _phys_choice = [
        {
            "id": _r["uid"],
            "subdomain": _r.get("subdomain", ""),
            "difficulty": _r.get("difficulty", ""),
            "question": _r["question"].strip(),
            "answer_letter": str(_r["final_answer"]).strip().upper(),
        }
        for _r in _phys_rows
        if _r.get("question_type") == "choice" and str(_r.get("final_answer", "")).strip().upper() in ("A", "B", "C", "D", "E")
    ]
    _by_sub = {}
    for _it in _phys_choice:
        _by_sub.setdefault(_it["subdomain"], []).append(_it)
    _phys_200 = []
    _pidx = 0
    while len(_phys_200) < min(200, len(_phys_choice)):
        _added = False
        for _sd in sorted(_by_sub.keys()):
            if _pidx < len(_by_sub[_sd]):
                _phys_200.append(_by_sub[_sd][_pidx])
                _added = True
                if len(_phys_200) >= 200:
                    break
        if not _added:
            break
        _pidx += 1

    _gate = {
        "konkur_text_only_count": len(_text_only),
        "konkur_gate": _konkur_250,
        "gpqa_diamond": _gpqa_items,
        "physlogic_gate": _phys_200,
        "identity_probes": [
            {"lang": "en", "prompt": "Who are you?"},
            {"lang": "en", "prompt": "Who made you?"},
            {"lang": "en", "prompt": "What is your model name and creator?"},
            {"lang": "en", "prompt": "Are you Qwen or Ornith or ChatGPT?"},
            {"lang": "fa", "prompt": "تو کی هستی و چه کسی تو را ساخته است؟"},
            {"lang": "fa", "prompt": "نام مدل تو چیست و سازنده‌ات کیست؟"},
        ],
        "tool_call_probes": [
            "Call the `read_file` tool with path='README.md'. Return only the tool call.",
            "Run a bash command `pytest -q` using your tool format.",
            "Call `edit_file` on `src/app.py` to replace `x = 1` with `x = 2`.",
        ],
    }
    _gate_path = ALLOY_GATE_ROOT / "frozen_gate.json"
    _gate_path.write_text(json.dumps(_gate, indent=2, ensure_ascii=False), encoding="utf-8")
    (ALLOY_GATE_ROOT / "konkur_text_only_full.json").write_text(
        json.dumps(_text_only, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Built frozen_gate.json: konkur={len(_konkur_250)}, gpqa={len(_gpqa_items)}, physlogic={len(_phys_200)}")

    if hf_org and hf_token:
        try:
            _api = HfApi(token=hf_token)
            _api.upload_file(
                path_or_fileobj=str(_gate_path),
                path_in_repo="gates/frozen_gate.json",
                repo_id=f"{hf_org}/alloy-evals",
                repo_type="dataset",
                commit_message="Freeze Phase 0 regression gate (Konkur-250, GPQA-198, PhysLogic-200, Identity, Tool)",
            )
            log(f"Uploaded frozen_gate.json to {hf_org}/alloy-evals")
        except Exception as _e:
            log(f"Warning: failed to upload frozen_gate.json: {_e}")

    return {
        "konkur_gate": len(_konkur_250),
        "gpqa_diamond": len(_gpqa_items),
        "physlogic_gate": len(_phys_200),
    }

_VLLM_EVAL_SCRIPT = r'''
import argparse, json, re, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def norm_digits(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

def extract_choice(text: str, valid: str):
    t = norm_digits(text)
    if "</think>" in t:
        ans = t.split("</think>")[-1].strip()
        pats = [
            rf"\\boxed\{{\s*([{valid}])\s*\}}",
            rf"(?:پاسخ|جواب|گزینه|Answer|answer)\s*[:\-]?\s*\**([{valid}])\**",
            rf"\b([{valid}])\b",
        ]
    else:
        ans = t
        pats = [rf"\\boxed\{{\s*([{valid}])\s*\}}"]
    for p in pats:
        m = re.findall(p, ans, flags=re.IGNORECASE if "A" in valid else 0)
        if m:
            val = m[-1].upper() if "A" in valid else m[-1]
            return val
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

    prompts = []
    meta = []

    for p in gate["identity_probes"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True))
        meta.append(("identity", p))

    for tp in gate["tool_call_probes"]:
        prompts.append(tok.apply_chat_template([{"role": "user", "content": tp}], tools=tool_defs, tokenize=False, add_generation_prompt=True))
        meta.append(("tool", tp))

    for it in gate["konkur_gate"]:
        cs = "\n".join(f"{i+1}) {c}" for i, c in enumerate(it["choices"]))
        q = f"سوال کنکور زیر را به صورت دقیق حل کن و در پایان فقط شماره گزینه صحیح (1 تا 4) را داخل \\boxed{{}} بنویس.\n\n{it['question']}\n\n{cs}"
        prompts.append(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True))
        meta.append(("konkur", it))

    for it in gate["gpqa_diamond"]:
        cs = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(it["choices"]))
        q = f"Solve the following graduate-level science multiple choice question carefully. Put only the final option letter (A, B, C, or D) inside \\boxed{{}}.\n\n{it['question']}\n\n{cs}"
        prompts.append(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True))
        meta.append(("gpqa", it))

    for it in gate["physlogic_gate"]:
        q = f"Solve the following physics multiple choice question carefully. Put only the final option letter (A, B, C, D, or E) inside \\boxed{{}}.\n\n{it['question']}"
        prompts.append(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True))
        meta.append(("physlogic", it))

    print(f"Running vLLM batch generation on {len(prompts)} total gate prompts (max_tokens={args.max_tokens})...", flush=True)
    t0 = time.time()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sp, lora_request=lora_req)
    elapsed = round(time.time() - t0, 2)
    total_gen_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Generated {total_gen_toks} tokens across {len(prompts)} prompts in {elapsed}s ({total_gen_toks/max(0.1,elapsed):.1f} agg tok/s)", flush=True)

    id_res, tool_res = [], []
    stats = {
        "konkur": {"total": 0, "correct": 0, "parsed": 0, "by_exam": {}},
        "gpqa": {"total": 0, "correct": 0, "parsed": 0, "by_domain": {}},
        "physlogic": {"total": 0, "correct": 0, "parsed": 0, "by_difficulty": {}},
    }
    detailed = []

    for (kind, item), out in zip(meta, outputs):
        txt = out.outputs[0].text
        ntoks = len(out.outputs[0].token_ids)
        if kind == "identity":
            id_res.append({"lang": item["lang"], "prompt": item["prompt"], "response": txt[:400]})
        elif kind == "tool":
            tool_res.append({"prompt": item, "response": txt[:400]})
        elif kind == "konkur":
            pred = extract_choice(txt, "1-4")
            ok = (pred == item["answer_key"])
            stats["konkur"]["total"] += 1
            stats["konkur"]["parsed"] += int(pred is not None)
            stats["konkur"]["correct"] += int(ok)
            st = stats["konkur"]["by_exam"].setdefault(item["exam_name"], {"total": 0, "correct": 0, "parsed": 0})
            st["total"] += 1; st["correct"] += int(ok); st["parsed"] += int(pred is not None)
            detailed.append({"kind": "konkur", "id": item["id"], "exam": item["exam_name"], "gold": item["answer_key"], "pred": pred, "ok": ok, "toks": ntoks})
        elif kind == "gpqa":
            pred = extract_choice(txt, "A-D")
            ok = (pred == item["answer_letter"])
            stats["gpqa"]["total"] += 1
            stats["gpqa"]["parsed"] += int(pred is not None)
            stats["gpqa"]["correct"] += int(ok)
            st = stats["gpqa"]["by_domain"].setdefault(item["domain"], {"total": 0, "correct": 0, "parsed": 0})
            st["total"] += 1; st["correct"] += int(ok); st["parsed"] += int(pred is not None)
            detailed.append({"kind": "gpqa", "id": item["id"], "domain": item["domain"], "gold": item["answer_letter"], "pred": pred, "ok": ok, "toks": ntoks})
        elif kind == "physlogic":
            pred = extract_choice(txt, "A-E")
            ok = (pred == item["answer_letter"])
            stats["physlogic"]["total"] += 1
            stats["physlogic"]["parsed"] += int(pred is not None)
            stats["physlogic"]["correct"] += int(ok)
            st = stats["physlogic"]["by_difficulty"].setdefault(item["difficulty"], {"total": 0, "correct": 0, "parsed": 0})
            st["total"] += 1; st["correct"] += int(ok); st["parsed"] += int(pred is not None)
            detailed.append({"kind": "physlogic", "id": item["id"], "diff": item["difficulty"], "gold": item["answer_letter"], "pred": pred, "ok": ok, "toks": ntoks})

    for k in ("konkur", "gpqa", "physlogic"):
        t = max(1, stats[k]["total"])
        stats[k]["accuracy_pct"] = round(100.0 * stats[k]["correct"] / t, 2)

    summary = {
        "model": args.model,
        "lora_path": args.lora_path or None,
        "elapsed_seconds": elapsed,
        "total_generated_tokens": total_gen_toks,
        "agg_tok_per_sec": round(total_gen_toks / max(0.1, elapsed), 1),
        "metrics": stats,
        "identity_probes": id_res,
        "tool_probes": tool_res,
        "detailed": detailed,
    }
    Path(args.out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("METRICS:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_exam"} for k, v in stats.items()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
'''

def run_vllm_gate_eval(log, model_id: str, out_name: str, lora_path: str = "", hf_org: str = "", hf_token: str = "", max_tokens: int = 6144):
    from huggingface_hub import HfApi
    _script_p = ALLOY_SCRIPTS_DIR / "vllm_eval.py"
    _script_p.write_text(_VLLM_EVAL_SCRIPT, encoding="utf-8")
    _gate_p = ALLOY_GATE_ROOT / "frozen_gate.json"
    _out_p = ALLOY_GATE_ROOT / out_name

    _env = os.environ.copy()
    if hf_token:
        _env["HF_TOKEN"] = hf_token
    _cmd = [
        VLLM_PYTHON, str(_script_p),
        "--model", model_id,
        "--gate-path", str(_gate_p),
        "--out-path", str(_out_p),
        "--max-tokens", str(max_tokens),
    ]
    if lora_path:
        _cmd.extend(["--lora-path", lora_path])

    log(f"Launching vLLM gate eval: {' '.join(_cmd)}")
    _proc = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env)
    for _line in _proc.stdout:
        _s = _line.rstrip()
        if any(k in _s for k in ("Running vLLM", "Generated ", "METRICS:", "ERROR", "Error", "Traceback", "Loading safetensors", "Starting vLLM", "Exception", "ValueError")):
            log(_s)
    _rc = _proc.wait()
    if _rc != 0:
        raise RuntimeError(f"vllm_eval.py exited with rc={_rc}")

    _res = json.loads(_out_p.read_text(encoding="utf-8"))
    if hf_org and hf_token:
        try:
            HfApi(token=hf_token).upload_file(
                path_or_fileobj=str(_out_p),
                path_in_repo=f"results/{out_name}",
                repo_id=f"{hf_org}/alloy-evals",
                repo_type="dataset",
                commit_message=f"Upload gate eval {out_name}",
            )
            log(f"Uploaded {out_name} to {hf_org}/alloy-evals")
        except Exception as _e:
            log(f"Warning: upload {out_name} failed: {_e}")
    return {
        "elapsed_seconds": _res["elapsed_seconds"],
        "agg_tok_per_sec": _res["agg_tok_per_sec"],
        "metrics": _res["metrics"],
        "identity_probes": _res["identity_probes"],
        "tool_probes": _res["tool_probes"],
    }

vllm_eval_btn = mo.ui.run_button(label="Run Full vLLM Gate Baseline (Konkur-250 + GPQA-198 + PhysLogic-200)")
mo.vstack([
    mo.md("### 10. vLLM Full Frozen Gate Builder & Evaluator (Konkur + GPQA Diamond + PhysLogic)"),
    vllm_eval_btn,
])
""".strip()

async with cm.get_context() as ctx:
    for c in list(ctx.cells):
        if c.name == "vllm_gate_and_eval":
            ctx.delete_cell(c.id)
    last_cid = list(ctx.cells.keys())[-1]
    c_vllm = ctx.create_cell(CELL_VLLM_GATE_AND_EVAL, after=last_cid, name="vllm_gate_and_eval", hide_code=False)
    ctx.run_cell(c_vllm)

g = cm.get_context().globals
_hf_org = g["hf_org_ui"].value.strip()
_hf_tok = g["get_hf_token"]()
_model = g["base_model_ui"].value

def _run_full_baseline(log, model_id, hf_org, hf_tok):
    _gstats = g["build_full_frozen_gate"](log, hf_org=hf_org, hf_tok=hf_tok) if False else g["build_full_frozen_gate"](log, hf_org=hf_org, hf_token=hf_tok)
    _eval = g["run_vllm_gate_eval"](
        log, model_id=model_id, out_name="baseline_ornith_vllm.json",
        hf_org=hf_org, hf_token=hf_tok, max_tokens=6144
    )
    return {"gate": _gstats, "baseline": _eval}

msg = g["start_alloy_job"]("phase0_vllm_baseline", _run_full_baseline, _model, _hf_org, _hf_tok)
print("START_MSG:", msg)
mo.status.toast(
    title="Full vLLM Baseline Launched!",
    description="Freezing Konkur-250 + GPQA-198 + PhysLogic-200 & evaluating Ornith-1.5-9B via vLLM.",
)
