import json
from pathlib import Path
import marimo._code_mode as cm

g = cm.get_context().globals
print("POOL_PREP:", json.dumps(g["get_alloy_job_status"]("phase1_pool_prep", tail_lines=25), indent=2))
p = Path("/marimo/alloy_data/phase1_pool")
if p.exists():
    print("POOL_FILES:", {x.name: round(x.stat().st_size / (1024*1024), 2) for x in sorted(p.iterdir())})
