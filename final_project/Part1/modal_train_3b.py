"""
Run GRPO + LoRA training on Modal - Qwen2.5-3B-Instruct.
"""

import modal

app = modal.App("countdown-grpo-lora-3b")
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
    timeout=14400,
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
                used_numbers = [int(x) for x in re.findall(r"\d+", equation)]
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
                allowed = re.compile(r"^[\d+\-*/().\s]+$")
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

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    OUTPUT_DIR = "/output/countdown-3b-500"

    print("Using model: " + MODEL_NAME)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    print("Loading dataset from HuggingFace...")
    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    TRAIN_SIZE = 50000
    TEST_SIZE = 1024
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def format_example(example):
        target = example["target"]
        numbers = example["nums"]
        prompt = (
            "Using the numbers " + str(numbers) + ", create an equation that equals " + str(target) + ". "
            "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
            "Show your work in <think> </think> tags. "
            "And return the final answer in <answer> </answer> tags, "
            "for example <answer> (1 + 2) / 3 </answer>."
        )
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "target": example["target"],
            "nums": example["nums"],
        }

    train_dataset = train_dataset.map(format_example)
    test_dataset = test_dataset.map(format_example)

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

    print("Saving model to " + OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    trainer.processing_class.save_pretrained(OUTPUT_DIR)

    volume.commit()
    print("Done!")


@app.local_entrypoint()
def main():
    train.remote()
    print("Training complete!")