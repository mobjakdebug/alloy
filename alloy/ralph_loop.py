from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class AttemptRecord:
    iteration: int
    raw_output: str
    patch_applied: bool
    patch_error: str
    test_passed: bool
    returncode: int
    traceback_excerpt: str


@dataclass
class RalphResult:
    solved: bool
    iterations_used: int
    final_diff: str
    attempts: list[AttemptRecord] = field(default_factory=list)


def build_repo_map(repo_dir: Path, max_files: int = 120) -> str:
    """Build a deterministic, static repository map pinned at the top for vLLM APC."""
    files = []
    for p in sorted(repo_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_dir)
        parts = rel.parts
        if any(part.startswith(".") or part in ("__pycache__", "node_modules", ".venv", "dist", "build") for part in parts):
            continue
        files.append(str(rel))
        if len(files) >= max_files:
            break
    return "\n".join(files)


def build_apc_pinned_prompt(
    repo_map: str,
    problem_spec: str,
    *,
    failing_diff: Optional[str] = None,
    traceback_excerpt: Optional[str] = None,
    iteration: int = 1,
) -> list[dict[str, str]]:
    """Construct messages optimized for vLLM Automatic Prefix Caching (APC).

    The system prompt + static <repo_map> + <problem_spec> stay byte-identical across
    iterations 1..N so vLLM reuses ~80%+ of the KV cache while resetting bloated
    previous CoT/conversational history.
    """
    system_msg = (
        "You are Alloy, an execution-verified software engineering agent.\n"
        "Analyze the pinned repository map and issue specification carefully.\n"
        "Respond with your concise diagnosis inside <think>...</think>, followed by a single\n"
        "minimal unified diff inside a ```diff ... ``` block (git apply compatible)."
    )
    pinned_prefix = (
        f"<repo_map>\n{repo_map}\n</repo_map>\n\n"
        f"<problem_spec>\n{problem_spec}\n</problem_spec>\n\n"
    )
    if iteration <= 1 or not traceback_excerpt:
        user_content = pinned_prefix + "Provide the minimal unified diff patch to make all tests pass."
    else:
        user_content = (
            pinned_prefix
            + f"<attempt_feedback iteration=\"{iteration - 1}\">\n"
            + f"<previous_diff>\n{(failing_diff or '').strip()[:4000]}\n</previous_diff>\n"
            + f"<test_traceback>\n{traceback_excerpt.strip()[:4000]}\n</test_traceback>\n"
            + "</attempt_feedback>\n"
            + "Fix the remaining failure using a corrected minimal unified diff."
        )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def extract_unified_diff(model_output: str) -> str:
    """Extract the final unified diff block after </think> (or anywhere if absent)."""
    body = model_output.split("</think>")[-1] if "</think>" in (model_output or "") else (model_output or "")
    diffs = re.findall(r"```diff\s*\n(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
    if diffs:
        d = diffs[-1].strip()
        return d + "\n" if d else ""
    # Fallback: raw diff starting with --- a/ or diff --git
    m = re.search(r"(?:^|\n)((?:diff --git|--- a/).+)", body, flags=re.DOTALL)
    if m:
        d = m.group(1).strip()
        return d + "\n" if d else ""
    return ""


def slice_concise_traceback(stdout: str, stderr: str, max_chars: int = 3000) -> str:
    """Extract the most informative error/traceback lines without KV rot."""
    combined = ((stdout or "") + "\n" + (stderr or "")).strip()
    if len(combined) <= max_chars:
        return combined
    # Keep summary head + failure tail
    head = combined[: max_chars // 3]
    tail = combined[-((2 * max_chars) // 3) :]
    return head + "\n... [middle truncated] ...\n" + tail


def apply_unified_diff_in_dir(workdir: Path, diff_text: str) -> tuple[bool, str]:
    """Apply a unified diff cleanly via `git apply --recount` or `patch -p1`."""
    if not diff_text.strip():
        return False, "Empty diff extracted from model output."
    patch_file = workdir / ".alloy_candidate.patch"
    patch_file.write_text(diff_text, encoding="utf-8")
    try:
        if (workdir / ".git").exists():
            last_err = ""
            for flags in (
                ["--recount", "--whitespace=nowarn"],
                ["--recount", "--ignore-whitespace", "--whitespace=nowarn"],
            ):
                proc = subprocess.run(
                    ["git", "apply", *flags, str(patch_file)],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode == 0:
                    return True, ""
                last_err = (proc.stderr or proc.stdout or "patch failed").strip()
            return False, last_err
        proc = subprocess.run(
            ["patch", "-p1", "--batch", "-i", str(patch_file)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "patch failed").strip()
    finally:
        patch_file.unlink(missing_ok=True)


class RalphWiggumLoop:
    """Phase 4 Execution-Verified Inference Loop with vLLM APC prefix pinning."""

    def __init__(
        self,
        generate_fn: Callable[[list[dict[str, str]]], str],
        *,
        max_iters: int = 5,
        test_timeout_s: int = 120,
    ):
        self.generate_fn = generate_fn
        self.max_iters = max_iters
        self.test_timeout_s = test_timeout_s

    def solve(
        self,
        repo_dir: Path,
        problem_spec: str,
        test_command: Sequence[str] | list[str],
        *,
        extra_files_context: str = "",
    ) -> RalphResult:
        repo_map = build_repo_map(repo_dir)
        full_spec = problem_spec if not extra_files_context else f"{problem_spec}\n\n<relevant_files>\n{extra_files_context}\n</relevant_files>"

        failing_diff: Optional[str] = None
        traceback_excerpt: Optional[str] = None
        attempts: list[AttemptRecord] = []

        for it in range(1, self.max_iters + 1):
            messages = build_apc_pinned_prompt(
                repo_map,
                full_spec,
                failing_diff=failing_diff,
                traceback_excerpt=traceback_excerpt,
                iteration=it,
            )
            raw_out = self.generate_fn(messages)
            diff_text = extract_unified_diff(raw_out)

            with tempfile.TemporaryDirectory(prefix="alloy_ralph_") as tmp:
                sandbox = Path(tmp) / "repo"
                shutil.copytree(repo_dir, sandbox, dirs_exist_ok=False)

                applied, patch_err = apply_unified_diff_in_dir(sandbox, diff_text)
                if not applied:
                    tb = f"Patch application failed:\n{patch_err}"
                    attempts.append(AttemptRecord(it, raw_out, False, patch_err, False, -1, tb))
                    failing_diff = diff_text
                    traceback_excerpt = tb
                    continue

                proc = subprocess.run(
                    list(test_command),
                    cwd=sandbox,
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout_s,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                tb = slice_concise_traceback(proc.stdout, proc.stderr)
                passed = proc.returncode == 0
                attempts.append(AttemptRecord(it, raw_out, True, "", passed, proc.returncode, tb))
                if passed:
                    return RalphResult(solved=True, iterations_used=it, final_diff=diff_text, attempts=attempts)

                failing_diff = diff_text
                traceback_excerpt = tb

        return RalphResult(solved=False, iterations_used=len(attempts), final_diff="", attempts=attempts)
