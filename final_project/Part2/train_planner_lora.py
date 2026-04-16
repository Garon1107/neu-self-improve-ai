import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

def answer_accuracy_reward(completions, result, **kwargs):
    rewards = []
    for completion, gold in zip(completions, result):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            pred = match.group(1).strip().lower()
        else:
            pred = text.strip().lower()
        gold_str = str(gold).strip().lower()
        if gold_str in pred or pred in gold_str:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        score = 0.0
        if "<answer>" in text and "</answer>" in text:
            score += 0.5
        if any(t in text for t in ["Google_Search", "Wikipedia", "Web_Search", "Python_Coder"]):
            score += 0.5
        rewards.append(score)
    return rewards

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "./output/agentflow-planner-lora"

peft_config = LoraConfig(
    r=16, lora_alpha=64, lora_dropout=0.05,
    target_modules="all-linear", task_type="CAUSAL_LM",
)

print("Loading training data...")
dataset = load_dataset("parquet", data_files="data/train/combined_train.parquet", split="train")
dataset = dataset.select(range(5000))

TOOLS = "Available tools: Google_Search_Tool, Wikipedia_Search_Tool, Web_Search_Tool, Python_Coder_Tool, Generalist_Solution_Generator_Tool"

def format_example(example):
    q = example["question"]
    prompt = f"You are a planner agent. Given a question, plan which tool to use.\n\n{TOOLS}\n\nQuestion: {q}\n\nProvide your answer in <answer> </answer> tags."
    return {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "result": example["result"],
    }

dataset = dataset.map(format_example)
print(f"Training on {len(dataset)} examples")

training_config = GRPOConfig(
    output_dir=OUTPUT_DIR, max_steps=200,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=5e-6, warmup_ratio=0.1, lr_scheduler_type="cosine",
    num_generations=2, max_completion_length=256, beta=0.001,
    logging_steps=5, bf16=True, gradient_checkpointing=True,
    save_strategy="steps", save_steps=50, save_total_limit=4,
    report_to="none", reward_weights=[2.0, 1.0], seed=42,
)

print("Setting up GRPO trainer...")
trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=[answer_accuracy_reward, format_reward],
    args=training_config,
    train_dataset=dataset,
    peft_config=peft_config,
)

print("Starting Flow-GRPO + LoRA training...")
trainer.train()
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
trainer.processing_class.save_pretrained(OUTPUT_DIR)
