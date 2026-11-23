import marimo._code_mode as cm

help(cm)
print("CTX_METHODS:", [m for m in dir(cm) if not m.startswith("_")])
