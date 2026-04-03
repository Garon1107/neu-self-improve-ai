"""
Test trained LoRA checkpoints on countdown tasks.
Compares base model vs all checkpoints to find the best one.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_DIR = "./output/countdown-v5"

TEST_CASES = [
    {"target": 10, "nums": [2, 3, 5]},
    {"target": 24, "nums": [1, 2, 3, 4]},
    {"target": 15, "nums": [7, 8, 1]},
    {"target": 100, "nums": [25, 50, 75, 3]},
    {"target": 42, "nums": [10, 20, 6, 8]},
    {"target": 5, "nums": [2, 3]},
    {"target": 12, "nums": [3, 4, 6]},
    {"target": 30, "nums": [5, 6, 10]},
]

def make_prompt(target, nums):
    return (
        f"Using the numbers {nums}, create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        f"Show your work in <think> </think> tags. "
        f"And return the final answer in <answer> </answer> tags, "
        f"for example <answer> (1 + 2) / 3 </answer>."
    )

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def validate(equation, target, numbers):
    try:
        used = [int(n) for n in re.findall(r'\d+', equation)]
        avail = list(numbers)
        for n in used:
            if n in avail:
                avail.remove(n)
            else:
                return False, "used invalid number"
        if not re.match(r'^[\d+\-*/().\s]+$', equation):
            return False, "invalid characters"
        result = eval(equation)
        if abs(result - target) < 1e-6:
            return True, f"= {result}"
        return False, f"= {result} (expected {target})"
    except Exception as e:
        return False, str(e)

def generate(model, tokenizer, prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def test_model(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    correct = 0
    for tc in TEST_CASES:
        prompt = make_prompt(tc["target"], tc["nums"])
        response = generate(model, tokenizer, prompt)

        print(f"\nTarget: {tc['target']}, Numbers: {tc['nums']}")
        print(f"Response: {response[:300]}")
        equation = extract_answer(response)
        if equation:
            valid, detail = validate(equation, tc["target"], tc["nums"])
            status = "CORRECT" if valid else "WRONG"
            if valid:
                correct += 1
            print(f"Equation: {equation} -> {status} ({detail})")
        else:
            print("No <answer> tags found")

    print(f"\nScore: {correct}/{len(TEST_CASES)}")
    return correct

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Test base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    test_model(base_model, tokenizer, "Base Qwen2.5-0.5B-Instruct")
    del base_model
    torch.cuda.empty_cache()

    # Find and test all checkpoints
    ckpt_dirs = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    best_score = -1
    best_ckpt = None

    for ckpt_name in ckpt_dirs:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.isfile(os.path.join(ckpt_path, "adapter_config.json")):
            print(f"\nSkipping {ckpt_name} (no adapter_config.json)")
            continue

        print(f"\nLoading {ckpt_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(base_model, ckpt_path)
        lora_model.eval()
        score = test_model(lora_model, tokenizer, f"LoRA {ckpt_name}")

        if score > best_score:
            best_score = score
            best_ckpt = ckpt_name

        del lora_model, base_model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Best checkpoint: {best_ckpt} ({best_score}/{len(TEST_CASES)})")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()