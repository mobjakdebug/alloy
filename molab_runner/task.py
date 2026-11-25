import json, subprocess, sys, traceback
import marimo._code_mode as cm

g = cm.get_context().globals
print("SETUP_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_pools", tail_lines=20), indent=2))
print("=== THREAD STACKS ===")
for tid, frame in sys._current_frames().items():
    print(f"--- Thread {tid} ---")
    traceback.print_stack(frame, limit=8)
print("=== RECENT CACHE FILES ===")
print(subprocess.check_output(["bash", "-c", "ls -lht ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/blobs ~/.cache/huggingface/hub/blobs 2>/dev/null | head -n 30"], text=True))
