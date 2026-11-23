import marimo._code_mode as cm
g = cm.get_context().globals
print(g["get_alloy_job_status"]("phase0_konkur_baseline", tail_lines=40))
