import subprocess
vp = "/marimo/vllm_venv/bin/python"
print(subprocess.check_output([vp, "-c", "import torch, torchaudio, torchvision; print('torch:', torch.__version__, torch.version.cuda); print('torchaudio:', torchaudio.__version__); print('torchvision:', torchvision.__version__)"], text=True))
# Check if uninstalling torchaudio lets vllm import cleanly (or install matching cu132 torchaudio)
r = subprocess.run(["uv", "pip", "uninstall", "-p", vp, "torchaudio"], capture_output=True, text=True)
print("uninstall torchaudio:", r.returncode, r.stdout.strip(), r.stderr.strip())
r2 = subprocess.run([vp, "-c", "from vllm import LLM, SamplingParams; print('vLLM LLM import OK!')"], capture_output=True, text=True)
print("vLLM import rc=", r2.returncode, r2.stdout.strip(), r2.stderr[-1000:])
