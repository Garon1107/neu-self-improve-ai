# A2C for Continuous Action Spaces — MuJoCo Experiments

## Project Overview

This project extends the Advantage Actor-Critic (A2C) algorithm to handle **continuous action spaces** and compares three advantage estimation methods across three MuJoCo robotics environments.

## Key Design Decisions

### Continuous Action Space Extension

The core change from discrete A2C is replacing the categorical policy with a **Gaussian (Normal) policy**:

- The **Actor** network outputs the mean `μ(s)` of a multivariate Gaussian distribution.
- A learnable parameter vector `log σ` (state-independent) defines the standard deviation.
- Actions are sampled as: `a ~ N(μ(s), σ²I)`, then clipped to the environment's valid range.
- The log probability is: `log π(a|s) = Σᵢ log N(aᵢ; μᵢ(s), σᵢ²)`

### Three Advantage Estimation Methods

All three methods are unified under the GAE framework — the only difference is the value of λ:

| Method | λ value | Bias-Variance | Description |
|--------|---------|---------------|-------------|
| **1-step TD** | λ = 0 | High bias, low variance | `A(t) = r_t + γV(s_{t+1}) − V(s_t)` |
| **Monte Carlo** | λ = 1 | Low bias, high variance | `A(t) = G_t − V(s_t)` (full return) |
| **GAE** | 0 < λ < 1 | Balanced | `A^GAE(t) = Σₖ (γλ)^k δ_{t+k}` |

This is implemented in a single `_compute_gae()` function — changing the method only requires changing one hyperparameter (λ), exactly as the assignment hints.

### Architecture

- **Actor** and **Critic** are separate networks with independent optimizers (allowing different learning rates as required for grid search).
- Both use 2 hidden layers of 64 units with Tanh activations and orthogonal initialization.
- Advantages are normalized before the policy gradient update.

## Environments

| Environment | Obs Dim | Act Dim | Description |
|-------------|---------|---------|-------------|
| HalfCheetah-v5 | 17 | 6 | 2D cheetah locomotion |
| Hopper-v5 | 11 | 3 | 2D one-legged hopper |
| Walker2d-v5 | 17 | 6 | 2D bipedal walker |

## Installation & Usage

```bash
# Install dependencies
pip install gymnasium[mujoco] torch numpy matplotlib pandas tqdm

# 1. Run full comparison (3 envs × 3 methods × 3 seeds) — generates learning curves
python a2c_continuous.py

# 2. Run hyperparameter grid search on all environments
python a2c_continuous.py --grid-search

# 3. Grid search on a single environment
python a2c_continuous.py --grid-search --env HalfCheetah-v5

# 4. Single experiment
python a2c_continuous.py --env Hopper-v5 --method gae --total-steps 500000
```

## Output Structure

```
results/
├── all_results.csv           # Raw data from all runs
└── learning_curves.png       # One chart per environment

grid_search_results/
├── grid_search_HalfCheetah-v5.csv
├── grid_search_Hopper-v5.csv
└── grid_search_Walker2d-v5.csv
```

## Hyperparameter Grid Search

The grid search covers the 5 required hyperparameters:

| Hyperparameter | Values Searched |
|---------------|-----------------|
| `num_envs` (parallel environments) | 4, 8, 16 |
| `policy_lr` (policy learning rate) | 1e-4, 3e-4, 1e-3 |
| `value_lr` (value learning rate) | 1e-4, 1e-3, 3e-3 |
| `γ` (discount factor) | 0.95, 0.99, 0.999 |
| `λ` (GAE lambda) | 0.9, 0.95, 0.99 |

Total combinations per environment: 3⁵ = 243. Each trained for 500k steps with the GAE method. Results are sorted by final mean episode reward.

## Expected Results

Typical findings from these experiments:

- **GAE** generally performs best — it provides a good bias-variance tradeoff.
- **1-step TD** (λ=0) can learn quickly early on but may plateau due to high bias.
- **Monte Carlo** (λ=1) has the highest variance, often showing noisier learning curves, but can achieve good final performance.
- HalfCheetah tends to be the most forgiving environment; Hopper and Walker2d are more sensitive to hyperparameter choices.

## References

1. Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 6, 7, 12, 13
2. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016
3. Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning* (A3C/A2C), ICML 2016