import os
import shutil
import subprocess
from pathlib import Path

vllm_py = "/marimo/vllm_venv/bin/python"
uv_bin = shutil.which("uv") or "/usr/local/bin/uv"
if Path(uv_bin).exists():
    p = subprocess.run([uv_bin, "pip", "install", "-q", "--python", vllm_py, "pandas", "pyarrow"], capture_output=True, text=True)
else:
    p = subprocess.run([vllm_py, "-m", "pip", "install", "-q", "pandas", "pyarrow"], capture_output=True, text=True)
print("PIP_RC:", p.returncode, p.stderr[-300:])

smoke = r'''
import os, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

t0 = time.time()
tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base_m = AutoModelForImageTextToText.from_pretrained("ornith-ai/Ornith-1.5-9B", dtype=torch.bfloat16, trust_remote_code=True)
merged_m = PeftModel.from_pretrained(base_m, "/marimo/runs/alloy_round2/final_adapter").merge_and_unload()

ds = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "What is 2+2? Put your final answer in \\boxed{}."}], "expected_answer": "4"}
    for _ in range(4)
])

rl_lora = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"],
    exclude_modules=".*visual.*|.*vision.*",
)
cfg = GRPOConfig(
    output_dir="/tmp/grpo_smoke",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_completion_length=256,
    temperature=0.8,
    beta=0.02,
    loss_type="dapo",
    mask_truncated_completions=True,
    scale_rewards="group",
    use_liger_kernel=True,
    max_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.35,
    vllm_max_model_length=4096,
    logging_steps=1,
    report_to="none",
)
def rew(completions, **kwargs):
    return [1.0 for _ in completions]

tr = GRPOTrainer(model=merged_m, reward_funcs=[rew], args=cfg, train_dataset=ds, peft_config=rl_lora, processing_class=tok)
tr.train()
print("COLOCATE_SMOKE_OK elapsed=", round(time.time() - t0, 1))
'''
Path("/tmp/grpo_colocate_smoke.py").write_text(smoke, encoding="utf-8")
proc = subprocess.run([vllm_py, "/tmp/grpo_colocate_smoke.py"], capture_output=True, text=True, timeout=180)
print("SMOKE_RC:", proc.returncode)
print("SMOKE_STDOUT:", proc.stdout[-1500:])
print("SMOKE_STDERR:", proc.stderr[-1500:])
