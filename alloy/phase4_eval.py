"""Phase 4 RalphWiggumLoop + vLLM Automatic Prefix Caching (APC) benchmark."""

from __future__ import annotations

PHASE4_RALPH_EVAL_SCRIPT = r'''import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

REPO_DIR = Path("/marimo/alloy_repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from alloy.ralph_loop import RalphWiggumLoop

BENCH_TASKS = [
    {
        "id": "bug_lru_ttl_eviction",
        "spec": (
            "Fix `cache.py` so `TTLCache.get(key)` returns `None` and removes expired entries "
            "when `now_fn() - ts >= ttl` (strictly `>= ttl`, not `>`), and `TTLCache.set` "
            "evicts the least-recently-used key BEFORE inserting when at capacity `max_size` "
            "and `key` is new. Do not modify `test_cache.py`."
        ),
        "files": {
            "cache.py": (
                "from collections import OrderedDict\n\n"
                "class TTLCache:\n"
                "    def __init__(self, max_size: int = 4, ttl: float = 10.0, now_fn=None):\n"
                "        self.max_size = max_size\n"
                "        self.ttl = ttl\n"
                "        self.now_fn = now_fn or (lambda: 0.0)\n"
                "        self._data = OrderedDict()\n\n"
                "    def get(self, key):\n"
                "        if key not in self._data:\n"
                "            return None\n"
                "        val, ts = self._data[key]\n"
                "        if self.now_fn() - ts > self.ttl:\n"
                "            del self._data[key]\n"
                "            return None\n"
                "        self._data.move_to_end(key)\n"
                "        return val\n\n"
                "    def set(self, key, val):\n"
                "        ts = self.now_fn()\n"
                "        if key in self._data:\n"
                "            self._data[key] = (val, ts)\n"
                "            self._data.move_to_end(key)\n"
                "            return\n"
                "        self._data[key] = (val, ts)\n"
                "        if len(self._data) > self.max_size:\n"
                "            self._data.popitem(last=True)\n"
            ),
            "test_cache.py": (
                "import unittest\n"
                "from cache import TTLCache\n\n"
                "class TestTTLCache(unittest.TestCase):\n"
                "    def test_exact_ttl_boundary_expires(self):\n"
                "        t = [0.0]\n"
                "        c = TTLCache(max_size=2, ttl=5.0, now_fn=lambda: t[0])\n"
                "        c.set('a', 1)\n"
                "        t[0] = 5.0\n"
                "        self.assertIsNone(c.get('a'))\n\n"
                "    def test_lru_eviction_order(self):\n"
                "        t = [0.0]\n"
                "        c = TTLCache(max_size=2, ttl=100.0, now_fn=lambda: t[0])\n"
                "        c.set('a', 1)\n"
                "        c.set('b', 2)\n"
                "        self.assertEqual(c.get('a'), 1)\n"
                "        c.set('c', 3)\n"
                "        self.assertIsNone(c.get('b'))\n"
                "        self.assertEqual(c.get('a'), 1)\n"
                "        self.assertEqual(c.get('c'), 3)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_cache.py"],
    },
    {
        "id": "bug_semver_prerelease",
        "spec": (
            "Fix `semver.py` so `compare_versions(a, b)` properly handles prerelease tags "
            "(e.g. `1.0.0-alpha < 1.0.0-beta < 1.0.0`, numeric prerelease identifiers compare "
            "numerically and sort before alphanumeric ones, and build metadata `+...` is ignored). "
            "Do not modify `test_semver.py`."
        ),
        "files": {
            "semver.py": (
                "def _parse(v: str):\n"
                "    core = v.lstrip('v')\n"
                "    parts = [int(x) for x in core.split('.')]\n"
                "    return tuple(parts)\n\n"
                "def compare_versions(a: str, b: str) -> int:\n"
                "    pa, pb = _parse(a), _parse(b)\n"
                "    return (pa > pb) - (pa < pb)\n"
            ),
            "test_semver.py": (
                "import unittest\n"
                "from semver import compare_versions\n\n"
                "class TestSemver(unittest.TestCase):\n"
                "    def test_prerelease_and_build(self):\n"
                "        self.assertEqual(compare_versions('1.0.0-alpha', '1.0.0'), -1)\n"
                "        self.assertEqual(compare_versions('1.0.0', '1.0.0-alpha'), 1)\n"
                "        self.assertEqual(compare_versions('1.0.0-alpha.1', '1.0.0-alpha.beta'), -1)\n"
                "        self.assertEqual(compare_versions('1.0.0-2', '1.0.0-10'), -1)\n"
                "        self.assertEqual(compare_versions('1.2.3+build1', '1.2.3+build2'), 0)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_semver.py"],
    },
    {
        "id": "bug_csv_quoted_escape",
        "spec": (
            "Fix `mini_csv.py` so `parse_csv_line(line)` handles escaped double quotes (`\"\"`) "
            "inside quoted fields and commas inside quotes, preserving empty trailing fields "
            "(e.g. `'a,b,' -> ['a', 'b', '']`). Do not modify `test_mini_csv.py`."
        ),
        "files": {
            "mini_csv.py": (
                "def parse_csv_line(line: str) -> list[str]:\n"
                "    return [p.strip('\"') for p in line.rstrip('\\n').split(',')]\n"
            ),
            "test_mini_csv.py": (
                "import unittest\n"
                "from mini_csv import parse_csv_line\n\n"
                "class TestMiniCsv(unittest.TestCase):\n"
                "    def test_quotes_and_trailing(self):\n"
                "        self.assertEqual(parse_csv_line('a,\"b,c\",d'), ['a', 'b,c', 'd'])\n"
                "        self.assertEqual(parse_csv_line('\"say \"\"hi\"\"\",2,'), ['say \"hi\"', '2', ''])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_mini_csv.py"],
    },
    {
        "id": "bug_interval_merge_overlap",
        "spec": (
            "Fix `intervals.py` so `merge_intervals(intervals)` sorts unsorted intervals, "
            "merges touching intervals (`[1, 3]` and `[3, 5]` -> `[1, 5]`), and returns a "
            "new list without mutating the input. Do not modify `test_intervals.py`."
        ),
        "files": {
            "intervals.py": (
                "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
                "    if not intervals:\n"
                "        return []\n"
                "    res = [intervals[0]]\n"
                "    for s, e in intervals[1:]:\n"
                "        if s < res[-1][1]:\n"
                "            res[-1][1] = max(res[-1][1], e)\n"
                "        else:\n"
                "            res.append([s, e])\n"
                "    return res\n"
            ),
            "test_intervals.py": (
                "import unittest\n"
                "from intervals import merge_intervals\n\n"
                "class TestIntervals(unittest.TestCase):\n"
                "    def test_merge_touching_and_unsorted(self):\n"
                "        orig = [[5, 7], [1, 3], [3, 5], [8, 10]]\n"
                "        snap = [x[:] for x in orig]\n"
                "        self.assertEqual(merge_intervals(orig), [[1, 7], [8, 10]])\n"
                "        self.assertEqual(orig, snap)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_intervals.py"],
    },
    {
        "id": "bug_rate_limiter_sliding",
        "spec": (
            "Fix `limiter.py` so `SlidingWindowLimiter.allow(user_id, now)` permits up to "
            "`limit` requests in the half-open window `(now - window, now]`, purges older "
            "timestamps, and does NOT record rejected requests. Do not modify `test_limiter.py`."
        ),
        "files": {
            "limiter.py": (
                "class SlidingWindowLimiter:\n"
                "    def __init__(self, limit: int, window: float):\n"
                "        self.limit = limit\n"
                "        self.window = window\n"
                "        self._hits = {}\n\n"
                "    def allow(self, user_id: str, now: float) -> bool:\n"
                "        q = self._hits.setdefault(user_id, [])\n"
                "        q.append(now)\n"
                "        return len(q) <= self.limit\n"
            ),
            "test_limiter.py": (
                "import unittest\n"
                "from limiter import SlidingWindowLimiter\n\n"
                "class TestLimiter(unittest.TestCase):\n"
                "    def test_sliding_window_and_reject_no_record(self):\n"
                "        lim = SlidingWindowLimiter(limit=2, window=10.0)\n"
                "        self.assertTrue(lim.allow('u', 1.0))\n"
                "        self.assertTrue(lim.allow('u', 5.0))\n"
                "        self.assertFalse(lim.allow('u', 6.0))\n"
                "        self.assertFalse(lim.allow('u', 11.0))\n"
                "        self.assertTrue(lim.allow('u', 11.01))\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_limiter.py"],
    },
    {
        "id": "bug_retry_backoff_jitter",
        "spec": (
            "Fix `retry.py` so `run_with_retry(fn, max_attempts=3, should_retry=None)` "
            "retries only when `should_retry(exc)` is True (default: all `Exception`), "
            "never sleeps after the final failed attempt, and raises the last exception when "
            "exhausted. Do not modify `test_retry.py`."
        ),
        "files": {
            "retry.py": (
                "import time\n\n"
                "def run_with_retry(fn, max_attempts: int = 3, base_delay: float = 0.01, should_retry=None, sleep_fn=time.sleep):\n"
                "    for attempt in range(max_attempts):\n"
                "        try:\n"
                "            return fn()\n"
                "        except Exception:\n"
                "            sleep_fn(base_delay * (2 ** attempt))\n"
            ),
            "test_retry.py": (
                "import unittest\n"
                "from retry import run_with_retry\n\n"
                "class TestRetry(unittest.TestCase):\n"
                "    def test_no_sleep_after_last_and_predicate(self):\n"
                "        sleeps = []\n"
                "        calls = [0]\n"
                "        def flaky():\n"
                "            calls[0] += 1\n"
                "            raise ValueError('bad')\n"
                "        with self.assertRaises(ValueError):\n"
                "            run_with_retry(flaky, max_attempts=3, base_delay=0.1, sleep_fn=sleeps.append)\n"
                "        self.assertEqual(calls[0], 3)\n"
                "        self.assertEqual(sleeps, [0.1, 0.2])\n"
                "        sleeps.clear()\n"
                "        calls[0] = 0\n"
                "        with self.assertRaises(ValueError):\n"
                "            run_with_retry(flaky, max_attempts=3, should_retry=lambda e: isinstance(e, RuntimeError), sleep_fn=sleeps.append)\n"
                "        self.assertEqual(calls[0], 1)\n"
                "        self.assertEqual(sleeps, [])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "test_cmd": [sys.executable, "-m", "unittest", "-q", "test_retry.py"],
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--lora-path", default="")
    ap.add_argument("--out-path", default="/marimo/alloy_gate/phase4_ralph_eval.json")
    ap.add_argument("--max-iters", type=int, default=5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enable_lora=bool(args.lora_path),
        max_lora_rank=128 if args.lora_path else 16,
    )
    lora_req = LoRARequest("alloy_adapter", 1, args.lora_path) if args.lora_path else None
    sp = SamplingParams(temperature=0.0, max_tokens=2048)

    results = []
    t_start = time.time()
    total_prompt_toks = 0
    total_cached_toks = 0

    for task in BENCH_TASKS:
        with tempfile.TemporaryDirectory(prefix="alloy_bench_") as td:
            repo_dir = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
            for fn, content in task["files"].items():
                (repo_dir / fn).write_text(content, encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
            subprocess.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=Alloy", "commit", "-q", "-m", "init"], cwd=repo_dir, check=True)

            extra_ctx = "\n\n".join(
                f"=== {fn} ===\n{content}" for fn, content in sorted(task["files"].items())
            )
            call_stats = []

            def gen_fn(messages):
                nonlocal total_prompt_toks, total_cached_toks
                prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outs = llm.generate([prompt_str], sp, lora_request=lora_req)
                o = outs[0]
                ptoks = len(o.prompt_token_ids)
                cached = getattr(o, "num_cached_tokens", 0) or 0
                total_prompt_toks += ptoks
                total_cached_toks += cached
                call_stats.append({"prompt_toks": ptoks, "cached_toks": cached, "gen_toks": len(o.outputs[0].token_ids)})
                return o.outputs[0].text

            loop = RalphWiggumLoop(gen_fn, max_iters=args.max_iters, test_timeout_s=30)
            res = loop.solve(repo_dir, task["spec"], task["test_cmd"], extra_files_context=extra_ctx)
            results.append({
                "id": task["id"],
                "solved": res.solved,
                "iterations_used": res.iterations_used,
                "solved_pass_at_1": res.solved and res.iterations_used == 1,
                "calls": call_stats,
                "attempts": [
                    {"iter": a.iteration, "applied": a.patch_applied, "passed": a.test_passed, "err": a.patch_error}
                    for a in res.attempts
                ],
            })

    pass_at_1 = sum(1 for r in results if r["solved_pass_at_1"])
    pass_at_k = sum(1 for r in results if r["solved"])
    summary = {
        "model": args.model,
        "lora_path": args.lora_path or None,
        "total_tasks": len(results),
        "pass_at_1": pass_at_1,
        f"pass_at_{args.max_iters}": pass_at_k,
        "pass_at_1_pct": round(100.0 * pass_at_1 / max(1, len(results)), 1),
        f"pass_at_{args.max_iters}_pct": round(100.0 * pass_at_k / max(1, len(results)), 1),
        "total_prompt_tokens": total_prompt_toks,
        "total_apc_cached_tokens": total_cached_toks,
        "apc_cache_hit_pct": round(100.0 * total_cached_toks / max(1, total_prompt_toks), 1),
        "elapsed_s": round(time.time() - t_start, 2),
        "tasks": results,
    }
    Path(args.out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PHASE4_RALPH_SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''
