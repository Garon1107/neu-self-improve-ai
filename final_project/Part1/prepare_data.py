"""
Prepare countdown dataset for GRPO training with TRL.
Downloads from HuggingFace and formats into chat-style prompts.
"""

from datasets import load_dataset
import json
import os

def make_prompt(example):
    """Create the countdown task prompt from a dataset example."""
    target = example['target']
    numbers = example['nums']
    
    prompt = (
        f"Using the numbers {numbers}, create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        f"Show your work in <think> </think> tags. "
        f"And return the final answer in <answer> </answer> tags, "
        f"for example <answer> (1 + 2) / 3 </answer>."
    )
    return prompt

def main():
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    print(f"Total samples: {len(dataset)}")

    TRAIN_SIZE = 50000   # Reduced from TinyZero's 327680 for faster training
    TEST_SIZE = 1024

    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def format_example(example):
        """Format into chat messages for TRL."""
        prompt = make_prompt(example)
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "target": example["target"],
            "nums": example["nums"],
        }

    train_dataset = train_dataset.map(format_example)
    test_dataset = test_dataset.map(format_example)

    # Save locally
    os.makedirs("data", exist_ok=True)
    train_dataset.to_json("data/train.jsonl")
    test_dataset.to_json("data/test.jsonl")

    print(f"Saved {len(train_dataset)} train samples to data/train.jsonl")
    print(f"Saved {len(test_dataset)} test samples to data/test.jsonl")
    
    # Print a sample to verify
    sample = train_dataset[0]
    print("\n--- Sample prompt ---")
    print(sample["prompt"][0]["content"])
    print(f"Target: {sample['target']}, Numbers: {sample['nums']}")

if __name__ == "__main__":
    main()