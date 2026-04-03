"""
Run GRPO + LoRA training on Modal with A100 GPU.
Qwen2.5-3B-Instruct for countdown task.
"""

import modal

app = modal.App("countdown-grpo-lora-1.5b")
volume = modal.Volume.from_name("training-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/output": volume},
)
def train():
    import re
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import think_format_reward

    # ─── Reward Function ────────────────────────────────────────
    def countdown_accuracy_reward(completions, target, nums, **kwargs):
        rewards = []
        for completion, t, n in zip(completions, target, nums):
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = completion

            match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue

            equation = match.group(1).strip()
            # Strip "= result" if present
            parts = equation.split("=")
            if len(parts) > 1:
                equation = parts[0].strip()
            try:
                used_numbers = [int(x) for x in re.findall(r'\d+', equation)]
                available = list(n)
                valid = True
                for num in used_numbers:
                    if num in available:
                        available.remove(num)
                    else:
                        valid = False
                        break
                if not valid:
                    rewards.append(0.0)
                    continue
                allowed = re.compile(r'^[\d+\-*/().\s]+$')
                if not allowed.match(equation):
                    rewards.append(0.0)
                    continue
                result = eval(equation)
                if abs(result - t) < 1e-6:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    # ─── Config ─────────────────────────────────────────────────
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "/output/countdown-1.5b-500"

    print(f"Using model: {MODEL_NAME}")

    # ─── LoRA Config ────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # ─── Prepare dataset ────────────────────────────────────────
    print("Loading dataset from HuggingFace...")
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

    # ─── GRPO Config ────────────────────────────────────────────
    training_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_generations=4,
        max_completion_length=512,
        beta=0.001,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        report_to="none",
        reward_weights=[2.0, 1.0],
        seed=42,
    )

    # ─── Train ──────────────────────────────────────────────────
    print("Setting up GRPO trainer...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[countdown_accuracy_reward, think_format_reward],
        args=training_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    trainer.processing_class.save_pretrained(OUTPUT_DIR)

    volume.commit()
    print("Done! Model saved to Modal volume.")


@app.local_entrypoint()
def main():
    train.remote()
    print("Training complete!")