import marimo._code_mode as cm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments

g = cm.get_context().globals
fn_g = g["run_phase2_round1_train"].__globals__
tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
anchors = fn_g["_build_tool_call_anchors"](tok, count=4)
print("TOOL_ANCHOR_SAMPLE:", anchors[0]["text"][:260])

collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, label_pad_token_id=-100)
args = TrainingArguments(
    output_dir="/tmp/test_args_dry",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=4e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,
    length_column_name="length",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=6,
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=4,
)
print("DRY_VALIDATE_ARGS: OK! bf16=", args.bf16, "warmup_steps=", args.warmup_steps)
print("LAUNCHER:", g["get_alloy_job_status"]("phase2_auto_launcher", tail_lines=10))
print("PHASE1_DISTILL:", g["get_alloy_job_status"]("phase1_distill", tail_lines=15))
