from pathlib import Path
lines = Path("/marimo/alloy_scripts/phase1_distill.py").read_text().splitlines()
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[120:270], start=120)))
