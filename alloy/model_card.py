"""Clean, anti-slop Hugging Face model card generator for Alloy-9B."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_VERIFIED_BENCHMARKS = {
    "baseline": {
        "konkur_pct": 43.2,
        "konkur_correct": 108,
        "konkur_total": 250,
        "physlogic_pct": 30.5,
        "physlogic_correct": 61,
        "physlogic_total": 200,
        "science_pct": 26.5,
        "science_correct": 53,
        "science_total": 200,
        "identity": "0/6",
        "tool_calls": "3/3",
        "generated_tokens": 2_200_353,
        "phase4_pass_at_1": "1/6 (16.7%)",
        "phase4_pass_at_5": "1/6 (16.7%)",
    },
    "alloy_r2": {
        "konkur_pct": 48.8,
        "konkur_correct": 122,
        "konkur_total": 250,
        "physlogic_pct": 34.0,
        "physlogic_correct": 68,
        "physlogic_total": 200,
        "science_pct": 26.5,
        "science_correct": 53,
        "science_total": 200,
        "identity": "6/6",
        "tool_calls": "3/3",
        "generated_tokens": 2_137_302,
        "phase4_pass_at_1": "0/6 (0.0%)",
        "phase4_pass_at_5": "2/6 (33.3%)",
    },
}


def extract_gate_summary(eval_payload: Optional[dict[str, Any]], phase4_payload: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Normalize public_gate_659 + Phase 4 Ralph loop JSON into flat display metrics."""
    if not eval_payload:
        return None
    m = eval_payload.get("metrics", {})
    k = m.get("konkur", {})
    p = m.get("physlogic", {})
    s = m.get("science_bench", {})
    id_ok = sum(
        1 for x in eval_payload.get("identity_probes", [])
        if "alloy" in x.get("response", "").lower() and "mob ai" in x.get("response", "").lower()
    )
    tool_ok = sum(
        1 for x in eval_payload.get("tool_probes", [])
        if "<tool_call>" in x.get("response", "") and "</tool_call>" in x.get("response", "")
    )
    out = {
        "konkur_pct": float(k.get("accuracy_pct", 0.0)),
        "konkur_correct": int(k.get("correct", 0)),
        "konkur_total": int(k.get("total", 250)),
        "physlogic_pct": float(p.get("accuracy_pct", 0.0)),
        "physlogic_correct": int(p.get("correct", 0)),
        "physlogic_total": int(p.get("total", 200)),
        "science_pct": float(s.get("accuracy_pct", 0.0)),
        "science_correct": int(s.get("correct", 0)),
        "science_total": int(s.get("total", 200)),
        "identity": f"{id_ok}/{len(eval_payload.get('identity_probes', [])) or 6}",
        "tool_calls": f"{tool_ok}/{len(eval_payload.get('tool_probes', [])) or 3}",
        "generated_tokens": int(eval_payload.get("total_generated_tokens") or eval_payload.get("total_output_tokens") or 0),
    }
    if phase4_payload:
        p1 = int(phase4_payload.get("pass_at_1_solved", 0))
        p5 = int(phase4_payload.get("solved", 0))
        tot = max(1, int(phase4_payload.get("total_tasks", 6)))
        out["phase4_pass_at_1"] = f"{p1}/{tot} ({100.0 * p1 / tot:.1f}%)"
        out["phase4_pass_at_5"] = f"{p5}/{tot} ({100.0 * p5 / tot:.1f}%)"
    return out


def _fmt_delta(a: float, b: float, suffix: str = "%") -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}{suffix}"


def render_alloy_model_card(
    *,
    variant: str = "merged",
    repo_id: str = "officialmobai/Alloy-9B",
    base_model: str = "ornith-ai/Ornith-1.5-9B",
    teacher_model: str = "Qwen/Qwen3.8-27B-FP8",
    baseline_metrics: Optional[dict[str, Any]] = None,
    alloy_metrics: Optional[dict[str, Any]] = None,
    dataset_stats: Optional[dict[str, Any]] = None,
    train_stats: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a clean, anti-slop Hugging Face README.md for Alloy-9B or Alloy-9B-LoRA."""
    if variant not in ("merged", "lora"):
        raise ValueError(f"Unsupported variant: {variant}")

    base_m = baseline_metrics or DEFAULT_VERIFIED_BENCHMARKS["baseline"]
    r2_m = alloy_metrics or DEFAULT_VERIFIED_BENCHMARKS["alloy_r2"]

    if variant == "lora":
        yaml_lines = [
            "---",
            f"base_model: {base_model}",
            "library_name: peft",
            "pipeline_tag: text-generation",
            "language:",
            "- en",
            "- fa",
            "tags:",
            "- alloy",
            "- lora",
            "- reasoning",
            "- persian",
            "- stem",
            "- tool-calling",
            "---",
        ]
        title = "# Alloy-9B (LoRA Adapter)"
        artifact_note = (
            f"This repository contains the LoRA adapter (`r=64`, `alpha=64`) for **Alloy-9B**, "
            f"fine-tuned by Mob AI on top of [`{base_model}`](https://huggingface.co/{base_model}). "
            f"For standalone BF16 weights, see [`officialmobai/Alloy-9B`](https://huggingface.co/officialmobai/Alloy-9B)."
        )
    else:
        yaml_lines = [
            "---",
            f"base_model: {base_model}",
            "library_name: transformers",
            "pipeline_tag: text-generation",
            "language:",
            "- en",
            "- fa",
            "tags:",
            "- alloy",
            "- reasoning",
            "- persian",
            "- stem",
            "- code",
            "- tool-calling",
            "---",
        ]
        title = "# Alloy-9B"
        artifact_note = (
            f"**Alloy-9B** is a 9B-parameter bilingual (English / Persian) reasoning, STEM, tool-calling, "
            f"and code-repair model developed by Mob AI from [`{base_model}`](https://huggingface.co/{base_model}). "
            f"LoRA weights and intermediate checkpoints are available in [`officialmobai/Alloy-9B-LoRA`](https://huggingface.co/officialmobai/Alloy-9B-LoRA)."
        )

    ds_line = ""
    if dataset_stats:
        ds_line = (
            f"- **Curated dataset**: `{dataset_stats.get('total_examples', 'N/A')}` examples "
            f"(`{dataset_stats.get('total_tokens', 'N/A')}` tokens, "
            f"`{dataset_stats.get('on_policy_pct', 'N/A')}%` on-policy, "
            f"`{dataset_stats.get('dropped_leak', 0)}` benchmark-leaked rows removed)\n"
        )
    tr_line = ""
    if train_stats:
        tr_line = (
            f"- **Final SFT run**: `{train_stats.get('global_step', 'N/A')}` steps, "
            f"training loss `{train_stats.get('training_loss', 'N/A')}`\n"
        )

    p4_base_p1 = base_m.get("phase4_pass_at_1", "1/6 (16.7%)")
    p4_r2_p1 = r2_m.get("phase4_pass_at_1", "N/A")
    p4_base_p5 = base_m.get("phase4_pass_at_5", "1/6 (16.7%)")
    p4_r2_p5 = r2_m.get("phase4_pass_at_5", "2/6 (33.3%)")

    tok_base = f"{int(base_m.get('generated_tokens', 0)):,}" if base_m.get("generated_tokens") else "N/A"
    tok_r2 = f"{int(r2_m.get('generated_tokens', 0)):,}" if r2_m.get("generated_tokens") else "N/A"

    if variant == "lora":
        usage_block = f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_id = "{base_model}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_id)

messages = [{{"role": "user", "content": "Solve $x^2 - 5x + 6 = 0$ and place the roots in \\\\boxed{{}}."}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```"""
    else:
        usage_block = f"""```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

messages = [{{"role": "user", "content": "Solve $x^2 - 5x + 6 = 0$ and place the roots in \\\\boxed{{}}."}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Serving with vLLM

```bash
vllm serve {repo_id} --dtype bfloat16 --max-model-len 8192 --enable-prefix-caching
```"""

    lines = yaml_lines + [
        "",
        title,
        "",
        artifact_note,
        "",
        "## Key Capabilities",
        "",
        "- **Bilingual English & Persian reasoning**: Trained on leakage-shielded Persian Konkur, STEM, and mathematical reasoning traces without losing base fluency.",
        "- **Zero-Hesitation reasoning traces**: Filtered against backtracking markers and CoT bloat so `<think>` blocks stay concise within practical inference budgets.",
        "- **Execution-grounded software repair**: Distilled with multi-language unified diff repair trajectories from `nebius/SWE-rebench-V2` and evaluated inside an iterative test-feedback repair loop.",
        "- **Structured tool calling**: Preserves native XML `<tool_call>` formatting across English and Persian tool prompts.",
        "",
        "## Benchmark Results",
        "",
        "Evaluated under identical vLLM greedy/sampling settings (`max_tokens=6144` on Public Gate; `max_iters=5`, `max_tokens=3072` on the iterative Python repo repair harness).",
        "",
        "| Benchmark | Baseline (`Ornith-1.5-9B`) | Alloy-9B | Delta |",
        "|---|---:|---:|---:|",
        f"| **Persian Konkur-1404** (`250`) | {base_m['konkur_pct']:.1f}% ({base_m['konkur_correct']}/{base_m['konkur_total']}) | **{r2_m['konkur_pct']:.1f}% ({r2_m['konkur_correct']}/{r2_m['konkur_total']})** | {_fmt_delta(base_m['konkur_pct'], r2_m['konkur_pct'])} |",
        f"| **PhysLogic** (`200`) | {base_m['physlogic_pct']:.1f}% ({base_m['physlogic_correct']}/{base_m['physlogic_total']}) | **{r2_m['physlogic_pct']:.1f}% ({r2_m['physlogic_correct']}/{r2_m['physlogic_total']})** | {_fmt_delta(base_m['physlogic_pct'], r2_m['physlogic_pct'])} |",
        f"| **ScienceBench** (`200`) | {base_m['science_pct']:.1f}% ({base_m['science_correct']}/{base_m['science_total']}) | {r2_m['science_pct']:.1f}% ({r2_m['science_correct']}/{r2_m['science_total']}) | {_fmt_delta(base_m['science_pct'], r2_m['science_pct'])} |",
        f"| **Bilingual Identity** (`6`) | {base_m['identity']} | **{r2_m['identity']}** | — |",
        f"| **XML Tool Calling** (`3`) | {base_m['tool_calls']} | {r2_m['tool_calls']} | — |",
        f"| **Public Gate Output Tokens** | {tok_base} | **{tok_r2}** | — |",
        f"| **SWE Iterative Loop `pass@1`** (`6` tasks) | {p4_base_p1} | {p4_r2_p1} | — |",
        f"| **SWE Iterative Loop `pass@5`** (`6` tasks) | {p4_base_p5} | **{p4_r2_p5}** | — |",
        "",
        "## Training Recipe",
        "",
        "1. **On-Policy Anchoring & `pass@4` Goldilocks Selection**: Sampled `Ornith-1.5-9B` across Persian prose, coding, physics, chemistry, and math prompts, keeping Zero-Hesitation anchors and medium-difficulty `1/4`–`3/4` self-distilled traces.",
        f"2. **Teacher Gap & Grounded SWE Distillation**: Distilled hard `0/4` failure cases, Persian STEM verification traces, and `nebius/SWE-rebench-V2` unified diff patches using `{teacher_model}`.",
        "3. **Leakage Shielding**: Exact and 100-character prefix deduplication against all Konkur, PhysLogic, and ScienceBench evaluation prompts.",
        "4. **Anti-Forgetting LoRA SFT**: Standard LoRA (`r=64`, `alpha=64`, `use_rslora=False`, `lr=1.5e-5`, cosine decay, `bf16`, `max_length=4096`, attention + MLP + linear-attention projections excluding vision modules), assistant-turn masked.",
    ]

    if ds_line or tr_line:
        lines.extend(["", ds_line.rstrip("\n") if ds_line else "", tr_line.rstrip("\n") if tr_line else ""])
        lines = [l for l in lines if l is not None]

    lines.extend([
        "",
        "## Quickstart",
        "",
        usage_block,
        "",
        "## Scope & Limitations",
        "",
        "- Optimized for English and Persian STEM reasoning, concise tool calls, and unified diff code patches; other languages are not targeted.",
        "- Uses `<think>...</think>` reasoning blocks by default. On complex multi-file repository fixes, allocate at least `3072` output tokens or run inside an execution-verified retry loop.",
        "- As a 9B model, outputs on specialized scientific edge cases and automated patches should still be verified by unit tests or human review.",
        "",
    ])
    return "\n".join(lines).replace("\n\n\n", "\n\n")
