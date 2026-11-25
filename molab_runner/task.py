import inspect, json
import marimo._code_mode as cm
from trl import GRPOConfig, GRPOTrainer

print("GRPOConfig params:", sorted(inspect.signature(GRPOConfig.__init__).parameters.keys()))
print("GRPOTrainer params:", sorted(inspect.signature(GRPOTrainer.__init__).parameters.keys()))
g = cm.get_context().globals
print("SETUP_STATUS:", json.dumps(g["get_alloy_job_status"]("setup_and_pools", tail_lines=20), indent=2))
