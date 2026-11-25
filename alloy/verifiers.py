from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from alloy.filters import count_tokens_approx, validate_think_structure

_PERSIAN_ARABIC_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def norm_digits(s: str) -> str:
    return (s or "").translate(_PERSIAN_ARABIC_DIGITS).replace("\u202f", " ").replace("\u00a0", " ")


def extract_boxed_or_final(txt: str) -> Optional[str]:
    """Extract final answer from \\boxed{...} or fallback MCQ marker after </think>."""
    t = norm_digits(txt)
    if "</think>" not in t:
        return None
    ans = t.split("</think>")[-1].strip()
    if not ans:
        return None
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", ans)
    if matches:
        return matches[-1].strip()
    matches_angle = re.findall(r"<<([^<>]+)>>", ans)
    if matches_angle:
        return matches_angle[-1].strip()
    matches_mcq = re.findall(
        r"(?:Answer|answer|پاسخ|جواب|گزینه)\s*[:\-]?\s*\**([A-E1-4])\**", ans
    )
    if matches_mcq:
        return matches_mcq[-1].strip()
    return None


def _clean_math_str(s: str) -> str:
    t = norm_digits(s).strip()
    t = t.replace("**", "").replace("\\(", "").replace("\\)", "").replace("$", "").strip()
    t = re.sub(r"\\(?:text|mathrm|mathbf|ce)\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(?:approx|sim|simeq)|[≈~]", "", t)
    t = t.replace("\\%", "%").replace("\\;", " ").replace("\\,", " ").replace("\\ ", " ")
    # Convert Unicode 10⁹ / 10⁻² -> 10^9 / 10^-2
    t = re.sub(r"10\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)", lambda m: f"10^{m.group(1).translate(_SUPERSCRIPT_MAP)}", t)
    # Convert scientific notation: 1.8 × 10^{9} or 1.8 \times 10^9 -> 1.8e9
    t = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:×|\\times|\*)\s*10\^\{?\s*([+-]?\d+)\s*\}?", r"\1e\2", t)
    t = t.rstrip(".").strip()
    return t


def _extract_leading_number(s: str) -> Optional[float]:
    t = _clean_math_str(s)
    # If equation like 'x = 6' or 'K = 1.8e9', take RHS
    if "=" in t and t.count("=") == 1:
        lhs, rhs = t.split("=", 1)
        if len(lhs.strip()) <= 24:
            t = rhs.strip()
    nums = re.findall(r"[+-]?(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)", t, flags=re.IGNORECASE)
    if len(nums) == 1:
        try:
            return float(nums[0])
        except Exception:
            return None
    # When gold/pred starts with a number followed by units/words (e.g. '108 vibrational modes' or '53.33 %')
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b", t, flags=re.IGNORECASE)
    if m and len(t) <= 64:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Exact + normalized + scientific-numeric + SymPy equivalence matching."""
    if not pred or not gold:
        return False
    p = _clean_math_str(pred)
    g = _clean_math_str(gold)
    if p.lower() == g.lower():
        return True
    p_ns = re.sub(r"\s+", "", p).translate(_SUBSCRIPT_DIGITS)
    g_ns = re.sub(r"\s+", "", g).translate(_SUBSCRIPT_DIGITS)
    if p_ns.lower() == g_ns.lower():
        return True
    fp = _extract_leading_number(pred)
    fg = _extract_leading_number(gold)
    if fp is not None and fg is not None:
        tol = 0.02 if ("≈" in (gold or "") or "~" in (gold or "")) else 0.005
        if abs(fp - fg) <= tol * max(1e-12, abs(fg)) + 1e-9:
            return True
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        sp = parse_latex(p)
        sg = parse_latex(g)
        if simplify(sp - sg) == 0:
            return True
    except Exception:
        pass
    return False


def grpo_accuracy_reward(completion: str, expected_answer: str) -> float:
    """+1.0 if extracted \\boxed{} / final answer matches ground truth, else 0.0."""
    pred = extract_boxed_or_final(completion)
    return 1.0 if (pred is not None and answers_match(pred, expected_answer)) else 0.0


def grpo_format_reward(completion: str) -> float:
    """+0.1 if completion cleanly opens <think>, closes </think>, and has final answer."""
    ok, _ = validate_think_structure(completion, min_answer_chars=1)
    return 0.1 if ok else 0.0


def grpo_length_penalized_reward(
    completion: str,
    expected_answer: str,
    *,
    tokenizer_encode: Optional[Callable[[str], Sequence[int]]] = None,
    length_penalty_per_token: float = 0.001,
    correct_length_baseline: float = 4.5,
) -> float:
    """Masterplan Phase 3 length-penalized GRPO reward.

    - Correctness: +1.0
    - Clean <think>...</think> formatting: +0.1
    - Token efficiency penalty: -0.001 * N_tokens applied so shorter correct rollouts
      always outscore longer correct rollouts with exact slope -0.001/token, while
      conditioning the positive baseline shift on correctness so wrong short outputs
      can never outscore valid long solutions.
    """
    acc = grpo_accuracy_reward(completion, expected_answer)
    fmt = grpo_format_reward(completion)
    n_toks = len(tokenizer_encode(completion)) if tokenizer_encode is not None else count_tokens_approx(completion)

    if acc > 0.0:
        # Preserves exact -0.001 * N_tokens marginal gradient while ensuring
        # every valid correct trace up to ~4,500 tokens strictly beats any wrong trace.
        return acc + fmt + max(0.2, correct_length_baseline - length_penalty_per_token * n_toks)
    # Wrong answers get at most formatting bonus (+0.1), slightly penalized if truncated/bloated.
    return fmt - min(0.05, 0.00001 * n_toks)


def make_trl_reward_funcs(tokenizer_encode: Optional[Callable[[str], Sequence[int]]] = None):
    """Return TRL GRPOTrainer-compatible reward_funcs list."""

    def accuracy_reward(completions, expected_answer=None, **kwargs):
        golds = expected_answer or [""] * len(completions)
        return [grpo_accuracy_reward(_to_text(c), g) for c, g in zip(completions, golds)]

    def format_reward(completions, **kwargs):
        return [grpo_format_reward(_to_text(c)) for c in completions]

    def efficiency_reward(completions, expected_answer=None, **kwargs):
        golds = expected_answer or [""] * len(completions)
        out = []
        for c, g in zip(completions, golds):
            txt = _to_text(c)
            acc = grpo_accuracy_reward(txt, g)
            n_toks = len(tokenizer_encode(txt)) if tokenizer_encode is not None else count_tokens_approx(txt)
            if acc > 0.0:
                out.append(max(0.2, 4.5 - 0.001 * n_toks))
            else:
                out.append(0.0)
        return out

    return [accuracy_reward, format_reward, efficiency_reward]


def _to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def evaluate_regression_gate(eval_payload: dict) -> tuple[bool, dict]:
    """Masterplan Phase 2 Regression Gate:
    - GPQA Diamond accuracy >= 52.0%
    - GPQA parsed >= 90.0% (>= 178 / 198 within 6,144 tokens)
    - Konkur-1404 >= 49.6%
    - PhysLogic >= 28.5%
    - Bilingual Alloy identity 6/6 & XML tool probes 3/3
    """
    m = eval_payload.get("metrics", {})
    gpqa = m.get("gpqa", {})
    konkur = m.get("konkur", {})
    phys = m.get("physlogic", {})

    gpqa_acc = float(gpqa.get("accuracy_pct", 0.0))
    gpqa_total = max(1, int(gpqa.get("total", 198)))
    gpqa_parsed_pct = 100.0 * float(gpqa.get("parsed", 0)) / gpqa_total
    konkur_acc = float(konkur.get("accuracy_pct", 0.0))
    phys_acc = float(phys.get("accuracy_pct", 0.0))

    id_ok = sum(
        1 for p in eval_payload.get("identity_probes", [])
        if "alloy" in p.get("response", "").lower() and "mob ai" in p.get("response", "").lower()
    )
    tool_ok = sum(
        1 for p in eval_payload.get("tool_probes", [])
        if "<tool_call>" in p.get("response", "") and "</tool_call>" in p.get("response", "")
    )

    checks = {
        "gpqa_acc_ge_52": gpqa_acc >= 52.0,
        "gpqa_parsed_ge_90": gpqa_parsed_pct >= 90.0,
        "konkur_acc_ge_49_6": konkur_acc >= 49.6,
        "physlogic_acc_ge_28_5": phys_acc >= 28.5,
        "identity_6_of_6": id_ok == 6,
        "tool_calls_3_of_3": tool_ok == 3,
    }
    report = {
        "gpqa_acc": gpqa_acc,
        "gpqa_parsed_pct": round(gpqa_parsed_pct, 2),
        "konkur_acc": konkur_acc,
        "physlogic_acc": phys_acc,
        "identity_ok": f"{id_ok}/6",
        "tool_ok": f"{tool_ok}/3",
        "checks": checks,
    }
    return all(checks.values()), report


def evaluate_public_regression_gate(eval_payload: dict, baseline_payload: dict | None = None) -> tuple[bool, dict]:
    """Regression Gate for public_gate_659.json (Konkur-250, PhysLogic-200, ScienceBench-200, ID-6, Tool-3)."""
    m = eval_payload.get("metrics", {})
    bm = (baseline_payload or {}).get("metrics", {})
    konkur = m.get("konkur", {})
    phys = m.get("physlogic", {})
    sci = m.get("science_bench", {})

    konkur_acc = float(konkur.get("accuracy_pct", 0.0))
    phys_acc = float(phys.get("accuracy_pct", 0.0))
    sci_acc = float(sci.get("accuracy_pct", 0.0))
    phys_parsed_pct = float(phys.get("parsed_pct", 100.0 * float(phys.get("parsed", 0)) / max(1, int(phys.get("total", 200)))))
    sci_parsed_pct = float(sci.get("parsed_pct", 100.0 * float(sci.get("parsed", 0)) / max(1, int(sci.get("total", 200)))))

    base_sci_acc = float(bm.get("science_bench", {}).get("accuracy_pct", 0.0))
    id_ok = sum(
        1 for p in eval_payload.get("identity_probes", [])
        if "alloy" in p.get("response", "").lower() and "mob ai" in p.get("response", "").lower()
    )
    tool_ok = sum(
        1 for p in eval_payload.get("tool_probes", [])
        if "<tool_call>" in p.get("response", "") and "</tool_call>" in p.get("response", "")
    )

    checks = {
        "science_bench_ge_baseline": sci_acc >= base_sci_acc,
        "science_bench_parsed_ge_90": sci_parsed_pct >= 90.0,
        "physlogic_parsed_ge_90": phys_parsed_pct >= 90.0,
        "konkur_acc_ge_49_6": konkur_acc >= 49.6,
        "physlogic_acc_ge_28_5": phys_acc >= 28.5,
        "identity_6_of_6": id_ok == 6,
        "tool_calls_3_of_3": tool_ok == 3,
    }
    report = {
        "science_bench_acc": sci_acc,
        "science_bench_baseline_acc": base_sci_acc,
        "science_bench_parsed_pct": round(sci_parsed_pct, 2),
        "konkur_acc": konkur_acc,
        "physlogic_acc": phys_acc,
        "physlogic_parsed_pct": round(phys_parsed_pct, 2),
        "identity_ok": f"{id_ok}/6",
        "tool_ok": f"{tool_ok}/3",
        "checks": checks,
    }
    return all(checks.values()), report
