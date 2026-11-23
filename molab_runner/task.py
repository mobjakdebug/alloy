import ctypes, subprocess, threading, time, gc, torch
import marimo._code_mode as cm

g = cm.get_context().globals

# 1. Gracefully interrupt the slow eager-HF phase0_konkur_baseline thread
for t in threading.enumerate():
    if t.name == "alloy-phase0_konkur_baseline" and t.is_alive():
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(t.ident), ctypes.py_object(RuntimeError)
        )
        print("Sent RuntimeError to alloy-phase0_konkur_baseline, res=", res)
        t.join(timeout=5)
        print("Thread alive after join(5):", t.is_alive())

gc.collect()
torch.cuda.empty_cache()
print("GPU after cancel:", subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"]).decode().strip())

# 2. Check full dry-run install plans (make sure torch/transformers won't be downgraded/broken)
for pkg in ["flash-linear-attention", "causal-conv1d", "liger-kernel", "vllm"]:
    r = subprocess.run(["uv", "pip", "install", "--dry-run", pkg], capture_output=True, text=True)
    print(f"\n=== DRY_RUN {pkg} rc={r.returncode} ===")
    print((r.stdout + "\n" + r.stderr)[-2000:])
