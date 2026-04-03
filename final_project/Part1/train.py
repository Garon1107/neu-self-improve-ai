"""
Train Qwen2.5-0.5B on countdown task using GRPO + LoRA.
Local version for RTX 5070 (12GB VRAM).
Based on reference implementation with proper KL penalty.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import think_format_reward
from reward import countdown_accuracy_reward

# ─── Configuration ───────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output/countdown-v5"

# ─── LoRA Config ─────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# ─── Load dataset ───────────────────────────────────────────────
print("Loading dataset...")
raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

TRAIN_SIZE = 50000
TEST_SIZE = 1024
train_dataset = raw_dataset.select(range(TRAIN_SIZE))
test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

def format_example(example):
    target = example['target']
    numbers = example['nums']
    prompt = (
        f"Using the numbers {numbers}, create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        f"Show your work in <think> </think> tags. "
        f"And return the final answer in <answer> </answer> tags, "
        f"for example <answer> (1 + 2) / 3 </answer>."
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "target": example["target"],
        "nums": example["nums"],
    }

train_dataset = train_dataset.map(format_example)
test_dataset = test_dataset.map(format_example)

# ─── GRPO Training Config ───────────────────────────────────────
training_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=2000,
    per_device_train_batch_size=1,      # Reduced for 12GB VRAM
    gradient_accumulation_steps=8,      # Keep effective batch = 8
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_generations=4,
    max_completion_length=512,
    beta=0.001,                         # KL penalty - prevents collapse!
    logging_steps=5,
    bf16=True,
    gradient_checkpointing=True,        # Save VRAM
    save_strategy="steps",
    save_steps=200,
    save_total_limit=10,
    report_to="none",
    reward_weights=[2.0, 1.0],
    seed=42,
)

# ─── Create Trainer ─────────────────────────────────────────────
print("Setting up GRPO trainer...")
trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=[countdown_accuracy_reward, think_format_reward],
    args=training_config,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

# ─── Train ──────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ─── Save ───────────────────────────────────────────────────────
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
trainer.processing_class.save_pretrained(OUTPUT_DIR)
print("Done!")