from pathlib import Path
p = Path("/usr/local/lib/python3.13/site-packages/marimo/_code_mode/_context.py")
lines = p.read_text().splitlines()
for i in range(1660, min(len(lines), 1735)):
    print(f"{i+1}: {lines[i]}")
