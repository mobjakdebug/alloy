import sys, importlib, transformers
print("sys.executable:", sys.executable)
print("sys.path:", sys.path)
print("transformers.__file__:", transformers.__file__)
for mod in ("accelerate", "peft", "trl", "bitsandbytes"):
    try:
        m = importlib.import_module(mod)
        print(mod, "-> OK:", getattr(m, "__file__", None))
    except Exception as e:
        print(mod, "-> ERR:", e)
print("is_accelerate_available():", transformers.utils.is_accelerate_available())
