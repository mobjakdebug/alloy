from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
print("=== TEMPLATE WITH <think> ===")
print(repr(tok.apply_chat_template([
    {"role": "user", "content": "Solve 1+1"},
    {"role": "assistant", "content": "<think>\n1+1=2\n</think>\n\n2"}
], tokenize=False, add_generation_prompt=False)))
print("=== TEMPLATE WITHOUT <think> ===")
print(repr(tok.apply_chat_template([
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I'm Alloy, made by Mob AI."}
], tokenize=False, add_generation_prompt=False)))
print("=== GENERATION PROMPT ===")
print(repr(tok.apply_chat_template([
    {"role": "user", "content": "Who are you?"}
], tokenize=False, add_generation_prompt=True)))
