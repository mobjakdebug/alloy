from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from alloy.filters import is_leaked_prompt, normalize_for_leakage, passes_zero_hesitation_filter
from alloy.masterplan_pipeline import PHASE1_R2_SCRIPT
from alloy.phase2_and_3 import PHASE2_R2_SFT_SCRIPT, PHASE3_GRPO_SCRIPT, ROUND2_TRAIN_CONFIG
from alloy.phase4_eval import PHASE4_RALPH_EVAL_SCRIPT
from alloy.public_eval import PUBLIC_VLLM_EVAL_SCRIPT
from alloy.ralph_loop import RalphWiggumLoop, extract_unified_diff
from alloy.verifiers import (
    answers_match,
    evaluate_public_regression_gate,
    evaluate_regression_gate,
    extract_boxed_or_final,
    grpo_length_penalized_reward,
)


class AlloyMasterplanTests(unittest.TestCase):
    def test_zero_hesitation_filter_drops_backtracking_and_bloat(self):
        bad_traces = [
            "<think>\nLet's compute 2+2=4. Wait, let me double check.\n</think>\n\n\\boxed{4}",
            "<think>\nFirst try x=3. Let me rethink this carefully.\n</think>\n\n\\boxed{3}",
            "<think>\nActually, that's wrong, let's start over from scratch.\n</think>\n\n\\boxed{5}",
            "<think>\nابتدا حساب می‌کنیم. صبر کن اشتباه شد.\n</think>\n\n\\boxed{2}",
            "<think>\nShort body\n</think>\n\n",
        ]
        for tr in bad_traces:
            ok, reason, _ = passes_zero_hesitation_filter(tr)
            self.assertFalse(ok, f"Expected drop for {tr!r}, got ok ({reason})")

        bloat = "<think>\n" + ("step " * 3000) + "\n</think>\n\n\\boxed{42}"
        ok_b, reason_b, n_b = passes_zero_hesitation_filter(
            bloat, tokenizer_encode=lambda s: s.split(), max_tokens=2800
        )
        self.assertFalse(ok_b)
        self.assertEqual(reason_b, "exceeds_2800_tokens")
        self.assertGreater(n_b, 2800)

        good = "<think>\nUsing conservation of momentum, p_i = p_f = 12 kg*m/s.\n</think>\n\n\\boxed{12}"
        ok_g, reason_g, n_g = passes_zero_hesitation_filter(
            good, tokenizer_encode=lambda s: s.split(), max_tokens=2800
        )
        self.assertTrue(ok_g)
        self.assertEqual(reason_g, "ok")
        self.assertLess(n_g, 2800)

    def test_leakage_shield(self):
        p = "A block of mass m slides down a frictionless inclined plane of angle theta."
        n = normalize_for_leakage(p)
        self.assertTrue(is_leaked_prompt(p.upper() + "!!", {n}, set()))

    def test_verifiers_and_grpo_length_penalty(self):
        txt_fa = "<think>\nمحاسبه مستقیم.\n</think>\n\nپاسخ نهایی: \\boxed{۱۲.۵}"
        self.assertEqual(extract_boxed_or_final(txt_fa), "12.5")
        self.assertTrue(answers_match("12.5", "12.50"))
        # Real Nemotron-RL-Science-v1 formatting normalization checks
        self.assertTrue(answers_match("K = 1.8 \\times 10^{9}", "1.8\u202f×\u202f10⁹**"))
        self.assertTrue(answers_match("108", "108 vibrational modes.**"))
        self.assertTrue(answers_match("x = 6", "6\u202fwater molecules (x\u202f=\u202f6)"))
        self.assertTrue(answers_match("\\text{Percentage dissociation} \\approx 53.3\\%", "53.33\u202f%**."))
        self.assertTrue(answers_match("\\Delta G° \\approx +7.85\\ \\text{kJ/mol}", "7.85\u202fkJ**."))
        self.assertTrue(answers_match("x:y = 2:5", "\\(x:y = 2:5\\)**"))
        self.assertFalse(answers_match("\\text{Volume percent of SO}_2 \\approx 0.543\\%", "1.09\u202fvol\u202f%"))

        short_correct = "<think>\nDirect proof: 2+2=4.\n</think>\n\n\\boxed{4}"
        long_correct = "<think>\n" + ("x " * 2000) + "\n</think>\n\n\\boxed{4}"
        very_long_correct = "<think>\n" + ("x " * 3490) + "\n</think>\n\n\\boxed{4}"
        short_wrong = "<think>\nGuess.\n</think>\n\n\\boxed{9}"

        tok_fn = lambda s: s.split()
        r_short = grpo_length_penalized_reward(short_correct, "4", tokenizer_encode=tok_fn)
        r_long = grpo_length_penalized_reward(long_correct, "4", tokenizer_encode=tok_fn)
        r_vlong = grpo_length_penalized_reward(very_long_correct, "4", tokenizer_encode=tok_fn)
        r_wrong = grpo_length_penalized_reward(short_wrong, "4", tokenizer_encode=tok_fn)

        self.assertGreater(r_short, r_long)
        self.assertGreater(r_long, r_vlong)
        self.assertGreater(r_vlong, r_wrong)

        c_1200 = "<think>\n" + ("a " * 1196) + "</think>\n\\boxed{4}"
        c_3500 = "<think>\n" + ("a " * 3496) + "</think>\n\\boxed{4}"
        diff = grpo_length_penalized_reward(c_1200, "4", tokenizer_encode=tok_fn) - grpo_length_penalized_reward(
            c_3500, "4", tokenizer_encode=tok_fn
        )
        self.assertAlmostEqual(diff, 2.3, places=6)

    def test_round2_config_and_regression_gate(self):
        self.assertFalse(ROUND2_TRAIN_CONFIG["use_rslora"])
        self.assertEqual(ROUND2_TRAIN_CONFIG["lora_r"], 64)
        self.assertEqual(ROUND2_TRAIN_CONFIG["lora_alpha"], 64)
        self.assertEqual(ROUND2_TRAIN_CONFIG["learning_rate"], 1.5e-5)

        r1_fail_payload = {
            "metrics": {
                "gpqa": {"total": 198, "correct": 54, "parsed": 67, "accuracy_pct": 27.27},
                "konkur": {"total": 250, "correct": 114, "parsed": 149, "accuracy_pct": 45.6},
                "physlogic": {"total": 200, "correct": 39, "parsed": 49, "accuracy_pct": 19.5},
            },
            "identity_probes": [{"response": "Alloy by Mob AI"}] * 6,
            "tool_probes": [{"response": "<tool_call>x</tool_call>"}] * 3,
        }
        passed_r1, _ = evaluate_regression_gate(r1_fail_payload)
        self.assertFalse(passed_r1)

        r2_pass_payload = {
            "metrics": {
                "gpqa": {"total": 198, "correct": 106, "parsed": 182, "accuracy_pct": 53.54},
                "konkur": {"total": 250, "correct": 128, "parsed": 220, "accuracy_pct": 51.2},
                "physlogic": {"total": 200, "correct": 62, "parsed": 160, "accuracy_pct": 31.0},
            },
            "identity_probes": [{"response": "I'm Alloy, made by Mob AI."}] * 6,
            "tool_probes": [{"response": "<tool_call><function=read_file></tool_call>"}] * 3,
        }
        passed_r2, rep = evaluate_regression_gate(r2_pass_payload)
        self.assertTrue(passed_r2)
        self.assertGreaterEqual(rep["gpqa_parsed_pct"], 90.0)

    def test_ralph_wiggum_execution_loop_and_apc_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "mini_repo"
            repo.mkdir()
            (repo / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return a - b\n", encoding="utf-8")
            (repo / "test_calc.py").write_text(
                "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )

            seen_prompts = []

            def mock_model(messages):
                seen_prompts.append(messages)
                it = len(seen_prompts)
                if it == 1:
                    return (
                        "<think>\nLet's try multiplying instead.\n</think>\n\n"
                        "```diff\n"
                        "--- a/calc.py\n"
                        "+++ b/calc.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def add(a: int, b: int) -> int:\n"
                        "-    return a - b\n"
                        "+    return a * b\n"
                        "```"
                    )
                return (
                    "<think>\nTraceback shows 6 != 5, so add should return a + b.\n</think>\n\n"
                    "```diff\n"
                    "--- a/calc.py\n"
                    "+++ b/calc.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def add(a: int, b: int) -> int:\n"
                    "-    return a - b\n"
                    "+    return a + b\n"
                    "```"
                )

            loop = RalphWiggumLoop(mock_model, max_iters=4)
            res = loop.solve(repo, "Fix add(2, 3) returning 5.", [sys.executable, "-m", "unittest", "-q"])

            self.assertTrue(res.solved)
            self.assertEqual(res.iterations_used, 2)
            self.assertTrue(res.attempts[0].patch_applied)
            self.assertFalse(res.attempts[0].test_passed)
            self.assertIn("6 != 5", res.attempts[0].traceback_excerpt)
            self.assertTrue(res.attempts[1].test_passed)
            self.assertIn("+    return a + b", extract_unified_diff(res.attempts[1].raw_output))

            self.assertEqual(seen_prompts[0][0], seen_prompts[1][0])
            p1 = seen_prompts[0][1]["content"].split("</problem_spec>")[0]
            p2 = seen_prompts[1][1]["content"].split("</problem_spec>")[0]
            self.assertEqual(p1, p2)

    def test_public_gate_and_embedded_scripts_compile(self):
        for name, src in [
            ("PHASE1_R2_SCRIPT", PHASE1_R2_SCRIPT),
            ("PHASE2_R2_SFT_SCRIPT", PHASE2_R2_SFT_SCRIPT),
            ("PHASE3_GRPO_SCRIPT", PHASE3_GRPO_SCRIPT),
            ("PUBLIC_VLLM_EVAL_SCRIPT", PUBLIC_VLLM_EVAL_SCRIPT),
            ("PHASE4_RALPH_EVAL_SCRIPT", PHASE4_RALPH_EVAL_SCRIPT),
        ]:
            compile(src, name, "exec")

        base = {"metrics": {"science_bench": {"accuracy_pct": 48.0}}}
        good = {
            "metrics": {
                "konkur": {"accuracy_pct": 50.8},
                "physlogic": {"accuracy_pct": 30.0, "parsed": 185, "total": 200, "parsed_pct": 92.5},
                "science_bench": {"accuracy_pct": 52.0, "parsed": 186, "total": 200, "parsed_pct": 93.0},
            },
            "identity_probes": [{"response": "Alloy by Mob AI"}] * 6,
            "tool_probes": [{"response": "<tool_call>x</tool_call>"}] * 3,
        }
        ok, rep = evaluate_public_regression_gate(good, base)
        self.assertTrue(ok, rep)


if __name__ == "__main__":
    unittest.main()
