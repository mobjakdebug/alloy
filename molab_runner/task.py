import inspect
from transformers import TrainingArguments

sig = inspect.signature(TrainingArguments.__init__)
print("HAS group_by_length:", "group_by_length" in sig.parameters)
print("LENGTH_PARAMS:", [k for k in sig.parameters if "length" in k or "group" in k or "warmup" in k or "push" in k or "save" in k])
