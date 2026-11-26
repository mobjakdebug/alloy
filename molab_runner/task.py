from pathlib import Path
lines = Path("/marimo/notebook.py").read_text().splitlines()
print("=== chemistry_pool ===")
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[1545:1590], start=1545)))
print("=== swe_pool ===")
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[1610:1650], start=1610)))
