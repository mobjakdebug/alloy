import subprocess
from pathlib import Path

log_path = Path("/marimo/alloy_logs/grpo_smoke.log")
smoke = r'''import os, shutil, subprocess, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")

uv_bin = shutil.which("uv") or "/usr/local/bin/uv"
vllm_py = "/marimo/vllm_venv/bin/python"
if os.path.exists(uv_bin):
    subprocess.run([uv_bin, "pip", "install", "-q", "--python", vllm_py, "pandas", "pyarrow"], check=False)

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

t0 = time.time()
print("Loading tokenizer & base model...", flush=True)
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

print("Creating GRPOTrainer(vllm_mode='colocate')...", flush=True)
tr = GRPOTrainer(model=merged_m, reward_funcs=[rew], args=cfg, train_dataset=ds, peft_config=rl_lora, processing_class=tok)
print("Starting 1-step train...", flush=True)
tr.train()
print("COLOCATE_SMOKE_OK elapsed=", round(time.time() - t0, 1), flush=True)
'''
Path("/tmp/grpo_colocate_smoke.py").write_text(smoke, encoding="utf-8")

def _runner(log_fn):
    log_fn("Starting grpo_smoke...")
    proc = subprocess.Popen(
        ["/marimo/vllm_venv/bin/python", "/tmp/grpo_colocate_smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log_fn(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"grpo_smoke rc={rc}")
    return {"ok": True}

start_alloy_job("grpo_smoke", _runner)
print("LAUNCHED grpo_smoke")
