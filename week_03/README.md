# ReKG-MCTS Reproduction: Part 1 (Baseline)

This repository contains the replication of the **Chain-of-Thought (CoT)** baseline for the ACL 2025 paper:

> *ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search*

---

## 📌 Overview

- **Objective**: Replicate the CoT baseline performance without external Knowledge Graphs.  
- **Model**: **DeepSeek-V3** (`deepseek-chat`) via OpenAI-compatible API.  
- **Dataset**: WebQuestions (used as proxy for WebQSP).  
- **Metric**: Hits@1 (Exact Match).  

---

## 🛠️ Setup

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Configure API Key

Create a `.env` file in the root directory:

```plaintext
DEEPSEEK_API_KEY=sk-your_key_here
```

---

## 🚀 Usage

### Run a quick test (10 samples)

```bash
python main.py --limit 10
```

### Run evaluation (50 samples)

```bash
python main.py --limit 50 --model deepseek-chat
```

---

## 📊 Results

| Method        | Model         | Samples | Accuracy (Hits@1) |
|--------------|--------------|----------|-------------------|
| CoT Baseline | DeepSeek-V3  | 10       | 40.00%            |
| CoT Baseline | DeepSeek-V3  | 20       | 50.00%            |
| CoT Baseline | DeepSeek-V3  | 30       | 51.72%            |

> **Note:**  
> The accuracy aligns with the baseline performance reported in the paper (33%–49% for CoT).  
> Failures are primarily due to strict exact string matching and the absence of Knowledge Graph retrieval (to be implemented in Part 2).

---

## 📂 Files

- `main.py` — Core logic for data loading, CoT inference, and evaluation  
- `requirements.txt` — Project dependencies  
- `experiment.db` — SQLite database for caching results (auto-generated)  

---

## 📚 Reference

Song, X., et al.  
**"ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search."**  
ACL 2025.
