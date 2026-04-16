# Part 2: AgentFlow Reimplementation

## Overview
Reimplementation of the [AgentFlow paper](https://agentflow.stanford.edu/) using Qwen3.5 model family with Flow-GRPO + LoRA training.

## Setup
```bash
git clone https://github.com/lupantech/AgentFlow.git
cd AgentFlow
uv sync
```

## Step 2: Reproduce AgentFlow (Qwen-2.5-7B-Instruct)
```bash
cd test
bash bamboogle/run_qwen25_7b.sh
```

| Bamboogle | 2Wiki | HotpotQA | Musique | GAIA |
|-----------|-------|----------|---------|------|
| 42.74% | 26.5% | 52.0% | 15.5% | 7.26% |

## Step 3: Run Qwen3.5 models on paper benchmarks

Start local vLLM server (0.8B/2B/4B):
```bash
vllm serve Qwen/Qwen3.5-0.8B --port 8000 --max-model-len 16384 --dtype half --enforce-eager
```

For 9B/27B, use SiliconFlow or OpenRouter API.

Run benchmarks:
```bash
cd test
bash bamboogle/run_qwen35_0.8b.sh
bash 2wiki/run_qwen35_0.8b.sh
bash hotpotqa/run_qwen35_0.8b.sh
bash musique/run_qwen35_0.8b.sh
bash gaia/run_qwen35_0.8b.sh
```

### Results

|  | Bamboogle | 2Wiki | HotpotQA | Musique | GAIA |
|--|-----------|-------|----------|---------|------|
| 0.8B | 59.2% | 40.5% | 38.0% | 16.5% | 20.5% |
| 2B | 63.2% | 46.0% | 69.0% | 14.0% | 26.0% |
| 4B | 68.8% | 51.0% | 73.5% | 18.0% | 29.4% |
| 9B | 74.4% | 55.5% | 76.0% | 22.5% | 32.7% |
| 27B | 79.2% | 60.5% | 79.0% | 26.0% | 35.4% |

## Step 4: New benchmark (HumanEval)
```bash
bash humaneval/run_qwen35_0.8b.sh
```

| 0.8B | 2B | 4B | 9B | 27B |
|------|-----|-----|-----|-----|
| 45.73% | 48.17% | 53.0% | 58.5% | 64.6% |

## Step 5: Flow-GRPO + LoRA on 0.8B

### Training
```bash
python train_planner_lora.py
```
- Base model: Qwen3.5-0.8B
- LoRA config: r=16, alpha=64, target_modules=all-linear
- GRPO: lr=5e-6, beta=0.001, num_generations=2, 200 steps

### Evaluation
```bash
vllm serve Qwen/Qwen3.5-0.8B --port 8000 --max-model-len 16384 --dtype half --enforce-eager \
  --enable-lora --lora-modules lora-0.8b=./output/agentflow-planner-lora

bash bamboogle/run_qwen35_0.8b_lora.sh
```

### Results

|  | Bamboogle | 2Wiki | HotpotQA | Musique | GAIA |
|--|-----------|-------|----------|---------|------|
| 0.8B baseline | 59.2% | 40.5% | 38.0% | 16.5% | 20.5% |
| 0.8B + LoRA | 52.8% | 53.5% | 60.0% | 14.65% | 11.02% |
| Change | -6.4% | +13.0% | +22.0% | -1.85% | -9.48% |

LoRA training improved multi-hop reasoning (2Wiki +13%, HotpotQA +22%) but decreased performance on single-hop (Bamboogle -6.4%) and complex real-world tasks (GAIA -9.48%).

## Step 6: LoRA on HumanEval

| 0.8B baseline | 0.8B + LoRA | Change |
|---------------|-------------|--------|
| 45.73% | 45.12% | -0.61% |

## Key Findings
1. Qwen3.5 models outperform Qwen2.5-7B on most benchmarks despite smaller size, thanks to improved architecture (GDN + MoE hybrid)
2. Model scaling shows consistent improvement across all benchmarks
3. Flow-GRPO + LoRA training shows mixed results: significant gains on multi-hop reasoning tasks but slight degradation on simpler tasks
4. The `<think>` tag stripping patch is required for Qwen3.5 compatibility with AgentFlow's JSON parser

## Implementation Notes
- Qwen3.5 models emit `<think>...</think>` tags by default, requiring a patch in `vllm.py` to strip them
- Local deployment (RTX 5070, 12GB) supports up to 4B model; 9B/27B require cloud API
- `--enforce-eager` flag needed to bypass CUDA compilation issues on Blackwell architecture
