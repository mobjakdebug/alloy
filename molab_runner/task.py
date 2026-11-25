from datasets import load_dataset
import marimo._code_mode as cm

k = load_dataset("PersianML/persian-konkur-1404", split="train")
print("KONKUR_COLS:", k.column_names)
print("KONKUR_0:", {kk: str(vv)[:120] for kk, vv in k[0].items()})

p = load_dataset("ScienceOne-AI/PhysLogic", split="test")
print("PHYSLOGIC_COLS:", p.column_names)
print("PHYSLOGIC_0:", {kk: str(vv)[:120] for kk, vv in p[0].items()})
