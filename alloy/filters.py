from __future__ import annotations

import ast
import re
from typing import Callable, Optional, Sequence

# Deterministic backtracking / hesitation markers that cause CoT bloat in 9B SFT.
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


def has_hesitation(text: str) -> bool:
    """Return True if the trace contains backtracking or hesitation markers."""
    if not text:
        return True
    return bool(HESITATION_RE.search(text))


def ensure_think_tags(text: str) -> str:
    """Normalize leading <think> tag if </think> is present without an opener."""
    t = (text or "").strip()
    if "</think>" in t and "<think>" not in t:
        return "<think>\n" + t
    return t


def validate_think_structure(text: str, min_answer_chars: int = 1) -> tuple[bool, str]:
    """Verify balanced single <think>...</think> block and non-empty final answer."""
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if t.count("<think>") != 1 or t.count("</think>") != 1:
        return False, "unbalanced_think_tags"
    open_idx = t.find("<think>")
    close_idx = t.find("</think>")
    if close_idx <= open_idx:
        return False, "invalid_think_order"
    think_body = t[open_idx + len("<think>"):close_idx].strip()
    if len(think_body) < 8:
        return False, "empty_think_body"
    answer_body = t[close_idx + len("</think>"):].strip()
    if len(answer_body) < min_answer_chars:
        return False, "empty_answer_body"
    return True, "ok"


def extract_python_blocks(text: str) -> list[str]:
    """Extract fenced ```python ... ``` blocks from an assistant response."""
    return re.findall(r"```python\s*\n(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)


def validate_code_ast_if_present(text: str) -> bool:
    """If response includes fenced ```python``` blocks, verify at least the final block parses via AST."""
    blocks = extract_python_blocks(text)
    if not blocks:
        return True
    try:
        ast.parse(blocks[-1])
        return True
    except SyntaxError:
        return False


def count_tokens_approx(text: str) -> int:
    """Fast deterministic fallback token counter when no HF tokenizer is passed (~3.4 chars/tok)."""
    return max(1, (len(text or "") + 2) // 3)


def passes_zero_hesitation_filter(
    assistant_text: str,
    *,
    tokenizer_encode: Optional[Callable[[str], Sequence[int]]] = None,
    max_tokens: int = 2800,
    min_answer_chars: int = 1,
    require_valid_python_ast: bool = False,
) -> tuple[bool, str, int]:
    """Apply the Masterplan's Zero-Hesitation + Structure + <=2800 token filter.

    Returns (passed, reason, token_count).
    """
    normalized = ensure_think_tags(assistant_text)
    ok_struct, reason = validate_think_structure(normalized, min_answer_chars=min_answer_chars)
    if not ok_struct:
        return False, reason, 0
    if has_hesitation(normalized):
        return False, "hesitation_marker", 0
    if require_valid_python_ast and not validate_code_ast_if_present(normalized):
        return False, "invalid_python_ast", 0

    if tokenizer_encode is not None:
        n_toks = len(tokenizer_encode(normalized))
    else:
        n_toks = count_tokens_approx(normalized)

    if n_toks > max_tokens:
        return False, f"exceeds_{max_tokens}_tokens", n_toks
    return True, "ok", n_toks


def normalize_for_leakage(text: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^\w\s\u0600-\u06ff]", "", s)


def is_leaked_prompt(prompt: str, exact_set: set[str], prefix_set: set[str]) -> bool:
    n = normalize_for_leakage(prompt)
    return (n in exact_set) or (len(n) >= 100 and n[:100] in prefix_set)
