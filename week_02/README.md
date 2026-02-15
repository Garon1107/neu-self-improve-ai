# Queue Management using MDP (Policy Iteration)

## 1. Problem Description

This project models a single-server queue system as a Markov Decision Process (MDP).  
At each time step, the controller decides whether to use **normal** or **fast** service to minimize long-term cost.

The solution is obtained using **Policy Iteration**, implemented from scratch (no RL libraries).

---

## 2. MDP Formulation

### State Space
Queue length:

S = {0, 1, ..., 15}

---

### Action Space

0 = Normal service  
1 = Fast service  

---

### Transition Model

Arrival rate: λ = 0.3  
Normal service rate: μ₀ = 0.4  
Fast service rate: μ₁ = 0.7  

Transitions:
- Arrival → s + 1  
- Service → s − 1  
- Otherwise → stay  

(See `get_transition_probs()` in `main.py`.)

---

### Reward Function

Reward = negative cost:

R(s,a) = −(2s + service_cost)

- Normal cost = 1  
- Fast cost = 5  
- Discount factor γ = 0.95  

Goal: maximize discounted long-term reward.

---

## 3. Solution Method

Policy Iteration:

1. Policy Evaluation  
2. Policy Improvement  
3. Repeat until convergence  

Implemented in:
- `policy_evaluation()`
- `policy_improvement()`
- `policy_iteration()`

---

## 4. Results

Converged in 3 iterations (~0.02s).

Optimal policy:

- Queue 0–2 → Normal
- Queue ≥3 → Fast

The optimal strategy follows a **threshold structure**, which is consistent with queue control theory.

---

## 5. How to Run

```bash
python main.py