import marimo._code_mode as cm
from pathlib import Path
p = Path("/usr/local/lib/python3.13/site-packages/marimo/_code_mode/_context.py")
lines = p.read_text().splitlines()
for i in range(795, 860):
    print(f"{i+1}: {lines[i]}")
print("---")
for i in range(1620, 1665):
    print(f"{i+1}: {lines[i]}")
