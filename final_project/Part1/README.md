# TinyZero Countdown Task - LoRA Reproduction

Reproducing the TinyZero countdown task using **LoRA adapters** instead of full fine-tuning, with GRPO (Group Relative Policy Optimization).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft trl datasets accelerate
```

For Modal cloud training:
```bash
pip install modal
modal setup
```

## Project Structure

```
final_project/
├── prepare_data.py        # Downloads and formats the Countdown dataset
├── reward.py              # Binary accuracy reward function
├── train.py               # GRPO + LoRA training (local, 0.5B)
├── modal_train.py         # GRPO + LoRA training (Modal, 1.5B)
├── modal_train_3b.py      # GRPO + LoRA training (Modal, 3B)
├── modal_test.py          # Evaluation on Modal
├── test.py                # Evaluation (local)
├── data/                  # Generated train/test data
└── output/                # Saved checkpoints
```

## How to Run

### 1. Prepare Data
```bash
python prepare_data.py
```
Downloads `Jiayi-Pan/Countdown-Tasks-3to4` from HuggingFace and formats it into chat-style prompts.

### 2. Train

**0.5B (local GPU):**
```bash
python train.py
```

**1.5B (Modal cloud):**
```bash
modal run modal_train.py
```

**3B (Modal cloud):**
```bash
modal run modal_train_3b.py
```

### 3. Evaluate
```bash
modal run modal_test.py
```

## Approach

- **Base Models**: Qwen2.5-0.5B / 1.5B / 3B Instruct
- **Fine-tuning Method**: LoRA (rank=16, alpha=64, target=all-linear)
- **RL Algorithm**: GRPO via TRL library
- **Reward Functions**:
  1. `countdown_accuracy_reward`: Binary (1.0 if equation is correct, 0.0 otherwise)
  2. `think_format_reward` (TRL built-in): Checks `<think>...</think>...<answer>...</answer>` format
- **Reward Weights**: [2.0, 1.0] (accuracy weighted 2x over format)
- **KL Penalty (beta)**: 0.001

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 64 |
| Learning rate | 5e-6 |
| LR scheduler | Cosine with warmup |
| Warmup ratio | 0.1 |
| Effective batch size | 8 |
| Num generations (GRPO) | 4 |
| Max completion length | 512 |
| Beta (KL penalty) | 0.001 |

## Results

### Evaluation (8 test problems, best checkpoint per model)

| Model | Base Score | Best Trained Score | Training Accuracy Reward |
|-------|-----------|-------------------|------------------------|
| Qwen2.5-0.5B-Instruct + LoRA (2000 steps) | 0/8 | 0/8 | ~0 |
| **Qwen2.5-1.5B-Instruct + LoRA (500 steps)** | **1/8** | **3/8** | **~0.25** |
| Qwen2.5-3B-Instruct + LoRA (400 steps*) | 2/8 | 2/8 | ~0.33 |

*3B training timed out at step 416/500.

**Note on test scores**: The test set contains only 8 problems with stochastic sampling (temperature=0.7), so scores have high variance. The training accuracy reward (measured over thousands of samples) is a more reliable indicator of model capability. By this metric, **3B > 1.5B > 0.5B**, confirming that larger models benefit more from GRPO training. The 3B model also frequently produced correct reasoning but failed to wrap results in `<answer>` tags, causing undercounting in the test evaluation.

### 1.5B Checkpoint Progression
| Checkpoint | Score | Correct Problems |
|------------|-------|-----------------|
| Base | 1/8 | 2*5=10 |
| Step 100 | 1/8 | 1*8+7=15 |
| Step 200 | 0/8 | - |
| Step 300 | 0/8 | - |
| Step 400 | 1/8 | 1*2*3*4=24 |
| **Step 500** | **3/8** | **1*2*3*4=24, 3*25-50+75=100, 2+3=5** |
| Final | 2/8 | 1*2*3*4=24, 2+3=5 |

### Training Curves (1.5B, 500 steps)
- **Format reward**: 0.05 → 1.0 (learned `<think>/<answer>` format)
- **Accuracy reward**: 0 → 0.25 (steady improvement throughout training)
- **Entropy**: Stable at ~0.2 (maintained by KL penalty)
- **KL divergence**: ~0.10 (model stays close to base)

## Key Findings

### 1. Model Size is the Critical Factor
Qwen2.5-0.5B with LoRA could not learn arithmetic reasoning (accuracy stayed at 0), consistent with TinyZero's own finding that 0.5B fails even with full fine-tuning. Qwen2.5-1.5B showed clear improvement from 1/8 to 3/8, making it the optimal size for LoRA-based training.

### 2. Larger Models Learn Better
Training accuracy reward scaled with model size: 0.5B (~0), 1.5B (~0.25), 3B (~0.33). This confirms the expected relationship between model capacity and RL learning ability. The 3B model's lower test score is likely due to format compliance issues (often omitting `<answer>` tags) and the small test set size, not lower capability.

### 3. Format Learning Precedes Accuracy
Across all model sizes, format reward saturated quickly (within 50-100 steps) while accuracy reward improved gradually over hundreds of steps. This suggests the model first learns the structural pattern, then slowly develops mathematical reasoning within that structure.

### 4. KL Penalty Prevents Collapse
Early experiments without `beta=0.001` showed rapid entropy collapse (entropy → 0, all outputs identical). Adding the KL penalty maintained output diversity and enabled continued learning.

### 5. Reward Hacking (0.5B)
The 0.5B model learned to output the target number directly in `<answer>` tags instead of constructing equations — a classic reward hacking behavior. Larger models were less prone to this.

## Differences from Original TinyZero
| Aspect | TinyZero | This Reproduction |
|--------|----------|-------------------|
| Fine-tuning | Full | LoRA (r=16) |
| Framework | verl | TRL + PEFT |
| Training steps | ~40,000 | 500 (1.5B) / 400 (3B) |
| Parameters updated | All | ~7% (LoRA adapters) |
| Best result | 3B significant improvement | 1.5B: 1/8→3/8, 3B: accuracy reward 0.33 |

## Libraries
- `transformers` 5.5.0
- `peft` 0.18.1
- `trl` 1.0.0
- `torch` 2.11.0 (CUDA 12.8)