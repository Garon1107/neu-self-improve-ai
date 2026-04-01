"""
A2C for Continuous Action Spaces — Full Implementation
=======================================================
Three advantage estimation methods:
  1) 1-step Temporal Difference (TD)
  2) Monte Carlo (MC)
  3) Generalized Advantage Estimation (GAE)

Three MuJoCo environments:
  - HalfCheetah-v5
  - Hopper-v5
  - Walker2d-v5

Requirements
------------
    pip install gymnasium[mujoco] torch numpy matplotlib pandas tqdm

Usage
-----
    # Run the main comparison experiment (3 envs × 3 methods)
    python a2c_continuous.py

    # Run hyperparameter grid search
    python a2c_continuous.py --grid-search

    # Single run with custom settings
    python a2c_continuous.py --env HalfCheetah-v5 --method gae --total-steps 1000000
"""

import argparse, os, time, json, itertools
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All hyperparameters in one place."""
    # --- environment ---
    env_name: str = "HalfCheetah-v5"
    num_envs: int = 8

    # --- training ---
    total_timesteps: int = 1_000_000
    n_steps: int = 2048          # rollout length per update
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # λ for GAE

    # --- optimizer ---
    policy_lr: float = 3e-4
    value_lr: float = 1e-3

    # --- network ---
    hidden_size: int = 64

    # --- loss ---
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- advantage method: "td1", "mc", "gae" ---
    advantage_method: str = "gae"

    # --- logging ---
    seed: int = 42
    log_interval: int = 5        # print every N updates
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. VECTORIZED ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(env_name: str, seed: int, idx: int):
    """Return a thunk that creates a single gym environment."""
    def _thunk():
        env = gym.make(env_name)
        env.reset(seed=seed + idx)
        return env
    return _thunk


def make_vec_envs(env_name: str, num_envs: int, seed: int):
    """Create a SyncVectorEnv with `num_envs` parallel copies."""
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, seed, i) for i in range(num_envs)]
    )
    return envs


# ═══════════════════════════════════════════════════════════════════════════════
#  3. NEURAL NETWORKS — ACTOR (Gaussian Policy) + CRITIC
# ═══════════════════════════════════════════════════════════════════════════════

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (standard for PPO/A2C)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Gaussian policy for continuous actions.
    Outputs mean; log_std is a learnable parameter vector.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        # state-independent log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class Critic(nn.Module):
    """State value function V(s)."""
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ADVANTAGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_advantages(
    rewards: torch.Tensor,     # (T, N)
    values: torch.Tensor,      # (T, N)
    dones: torch.Tensor,       # (T, N)
    next_value: torch.Tensor,  # (N,)
    gamma: float,
    gae_lambda: float,
    method: str,               # "td1" | "mc" | "gae"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages and returns.

    Parameters
    ----------
    rewards, values, dones : (T, num_envs) tensors from the rollout.
    next_value : (num_envs,) bootstrap value V(s_{T+1}).
    gamma : discount factor.
    gae_lambda : λ for GAE (only used when method="gae").
    method : one of {"td1", "mc", "gae"}.

    Returns
    -------
    advantages : (T, N)
    returns    : (T, N)
    """
    T, N = rewards.shape

    if method == "td1":
        # ── 1-step TD: A(t) = r_t + γ V(s_{t+1}) − V(s_t) ──
        # This is GAE with λ = 0
        return _compute_gae(rewards, values, dones, next_value, gamma, lam=0.0)

    elif method == "mc":
        # ── Monte Carlo: A(t) = G_t − V(s_t), where G_t = Σ γ^k r_{t+k} ──
        # This is GAE with λ = 1
        return _compute_gae(rewards, values, dones, next_value, gamma, lam=1.0)

    elif method == "gae":
        # ── GAE(γ, λ) ──
        return _compute_gae(rewards, values, dones, next_value, gamma, lam=gae_lambda)

    else:
        raise ValueError(f"Unknown advantage method: {method}")


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
        λ = 0  →  1-step TD advantage
        λ = 1  →  Monte Carlo advantage
        0 < λ < 1  →  GAE(γ, λ)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ROLLOUT STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores transitions from parallel environments during a rollout."""

    def __init__(self, n_steps: int, num_envs: int, obs_dim: int, act_dim: int,
                 device: str = "cpu"):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(n_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, num_envs, act_dim, device=device)
        self.rewards = torch.zeros(n_steps, num_envs, device=device)
        self.dones = torch.zeros(n_steps, num_envs, device=device)
        self.log_probs = torch.zeros(n_steps, num_envs, device=device)
        self.values = torch.zeros(n_steps, num_envs, device=device)
        self.step = 0

    def insert(self, obs, action, reward, done, log_prob, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.step += 1

    def reset(self):
        self.step = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  6. A2C AGENT — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class A2CAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # create environments
        self.envs = make_vec_envs(cfg.env_name, cfg.num_envs, cfg.seed)
        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.shape[0]

        # networks
        self.actor = Actor(obs_dim, act_dim, cfg.hidden_size).to(self.device)
        self.critic = Critic(obs_dim, cfg.hidden_size).to(self.device)

        # separate optimizers for policy and value (as required by the assignment)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.policy_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.value_lr)

        # rollout storage
        self.buffer = RolloutBuffer(cfg.n_steps, cfg.num_envs, obs_dim, act_dim,
                                    device=str(self.device))

        # logging
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.global_step = 0
        self._running_rewards = np.zeros(cfg.num_envs)
        self._running_lengths = np.zeros(cfg.num_envs, dtype=int)

    def collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """Collect n_steps of experience from parallel envs."""
        self.buffer.reset()
        for _ in range(self.cfg.n_steps):
            with torch.no_grad():
                action, log_prob, _ = self.actor.get_action(obs)
                value = self.critic(obs)

            # step environments
            action_np = action.cpu().numpy()
            # clip actions to valid range
            action_np = np.clip(
                action_np,
                self.envs.single_action_space.low,
                self.envs.single_action_space.high,
            )
            next_obs_np, reward_np, terminated_np, truncated_np, infos = self.envs.step(action_np)
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

            # track episode statistics
            self._running_rewards += reward_np
            self._running_lengths += 1
            for i in range(self.cfg.num_envs):
                if done_np[i]:
                    self.episode_rewards.append(self._running_rewards[i])
                    self.episode_lengths.append(self._running_lengths[i])
                    self._running_rewards[i] = 0.0
                    self._running_lengths[i] = 0

            # store transition
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            self.buffer.insert(
                obs,
                action,
                torch.tensor(reward_np, dtype=torch.float32, device=self.device),
                torch.tensor(done_np, dtype=torch.float32, device=self.device),
                log_prob,
                value,
            )
            obs = next_obs
            self.global_step += self.cfg.num_envs

        return obs  # return last observation for next rollout

    def update(self):
        """Compute advantages and perform a single A2C update."""
        with torch.no_grad():
            # bootstrap value for the last state
            last_obs = self.buffer.obs[-1]  # not exactly right; we use the *next* obs
            # We need the observation *after* the last stored step.
            # It was returned by collect_rollout, but we'll recompute from buffer.
            pass

        # We'll fix bootstrapping: collect_rollout returns next_obs; we call
        # update right after and pass next_obs separately.
        raise NotImplementedError("Call update_with_next_obs instead")

    def update_with_next_obs(self, next_obs: torch.Tensor):
        """A2C parameter update given the last observation for bootstrapping."""
        with torch.no_grad():
            next_value = self.critic(next_obs)

        advantages, returns = compute_advantages(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
            self.cfg.advantage_method,
        )

        # flatten (T, N) -> (T*N,)
        b_obs = self.buffer.obs.reshape(-1, self.buffer.obs.shape[-1])
        b_actions = self.buffer.actions.reshape(-1, self.buffer.actions.shape[-1])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # forward pass
        dist = self.actor(b_obs)
        log_probs = dist.log_prob(b_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        values = self.critic(b_obs)

        # policy (actor) loss: −E[log π(a|s) · A(s,a)]
        policy_loss = -(log_probs * b_advantages.detach()).mean()

        # value (critic) loss: MSE between V(s) and returns
        value_loss = 0.5 * (b_returns.detach() - values).pow(2).mean()

        # update actor
        self.actor_optim.zero_grad()
        actor_loss = policy_loss - self.cfg.entropy_coef * entropy
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optim.step()

        # update critic
        self.critic_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def train(self, verbose=True) -> pd.DataFrame:
        """Full training loop. Returns a DataFrame of logged metrics."""
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        obs_np, _ = self.envs.reset(seed=self.cfg.seed)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        num_updates = self.cfg.total_timesteps // (self.cfg.n_steps * self.cfg.num_envs)
        log_records = []

        pbar = tqdm(range(1, num_updates + 1), desc=f"{self.cfg.env_name} | {self.cfg.advantage_method}", disable=not verbose)
        for update in pbar:
            obs = self.collect_rollout(obs)
            stats = self.update_with_next_obs(obs)

            # log
            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-100:]
                mean_r = np.mean(recent)
                log_records.append({
                    "step": self.global_step,
                    "mean_reward": mean_r,
                    "episodes": len(self.episode_rewards),
                    **stats,
                })
                if verbose and update % self.cfg.log_interval == 0:
                    pbar.set_postfix({
                        "steps": self.global_step,
                        "ep_reward": f"{mean_r:.1f}",
                        "episodes": len(self.episode_rewards),
                    })

        self.envs.close()
        return pd.DataFrame(log_records)

    def close(self):
        self.envs.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  7. EXPERIMENT RUNNER — COMPARISON ACROSS METHODS
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_NAMES = {
    "td1": "A2C (1-step TD)",
    "mc":  "A2C (Monte Carlo)",
    "gae": "A2C (GAE)",
}

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
METHODS = ["td1", "mc", "gae"]


def run_single_experiment(env_name: str, method: str, seed: int = 42,
                          total_timesteps: int = 1_000_000,
                          num_envs: int = 8, **kwargs) -> pd.DataFrame:
    """Run one A2C training and return the log DataFrame."""
    cfg = Config(
        env_name=env_name,
        advantage_method=method,
        seed=seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        **kwargs,
    )
    agent = A2CAgent(cfg)
    df = agent.train(verbose=True)
    df["method"] = method
    df["env"] = env_name
    df["seed"] = seed
    return df


def smooth(data, window=10):
    """Simple moving-average smoothing for plotting."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window, min_periods=1).mean().values


def run_comparison(total_timesteps: int = 1_000_000, n_seeds: int = 3,
                   output_dir: str = "results"):
    """
    Run 3 envs × 3 methods × n_seeds, then plot learning curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    for env_name in ENVS:
        for method in METHODS:
            for seed in range(n_seeds):
                print(f"\n{'='*60}")
                print(f"  {env_name} | {METHOD_NAMES[method]} | seed={seed}")
                print(f"{'='*60}")
                df = run_single_experiment(
                    env_name, method, seed=seed,
                    total_timesteps=total_timesteps,
                )
                all_dfs.append(df)

    all_data = pd.concat(all_dfs, ignore_index=True)
    all_data.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    # ── Plot learning curves ──
    plot_learning_curves(all_data, output_dir)
    print(f"\nResults and plots saved to: {output_dir}/")


def plot_learning_curves(all_data: pd.DataFrame, output_dir: str):
    """
    For each environment, plot mean reward ± std over seeds for each method.
    """
    fig, axes = plt.subplots(1, len(ENVS), figsize=(6 * len(ENVS), 5))
    if len(ENVS) == 1:
        axes = [axes]

    colors = {"td1": "#e74c3c", "mc": "#2ecc71", "gae": "#3498db"}

    for ax, env_name in zip(axes, ENVS):
        env_data = all_data[all_data["env"] == env_name]

        for method in METHODS:
            method_data = env_data[env_data["method"] == method]
            if method_data.empty:
                continue

            # group by step across seeds, compute mean and std
            grouped = method_data.groupby("step")["mean_reward"]
            mean = grouped.mean()
            std = grouped.std().fillna(0)

            steps = mean.index.values
            mean_vals = smooth(mean.values, window=5)
            std_vals = smooth(std.values, window=5)

            ax.plot(steps, mean_vals, label=METHOD_NAMES[method],
                    color=colors[method], linewidth=1.5)
            ax.fill_between(steps,
                            mean_vals - std_vals,
                            mean_vals + std_vals,
                            alpha=0.15, color=colors[method])

        ax.set_title(env_name, fontsize=14)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Episode Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  → Saved learning curves to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. HYPERPARAMETER GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

# Grid values to search over
GRID = {
    "num_envs":   [4, 8, 16],
    "policy_lr":  [1e-4, 3e-4, 1e-3],
    "value_lr":   [1e-4, 1e-3, 3e-3],
    "gamma":      [0.95, 0.99, 0.999],
    "gae_lambda": [0.9, 0.95, 0.99],
}


def run_grid_search(env_name: str = "HalfCheetah-v5",
                    total_timesteps: int = 500_000,
                    output_dir: str = "grid_search_results"):
    """
    Grid search over the 5 hyperparameters for GAE on one environment.
    To keep it tractable, we use a coarse grid and shorter training.
    For full search across all environments, call this function for each env.
    """
    os.makedirs(output_dir, exist_ok=True)

    # generate all combinations
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))

    print(f"\nGrid search: {len(combos)} combinations for {env_name}")
    print(f"Hyperparameters: {keys}")
    print(f"Training for {total_timesteps} steps each\n")

    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"[{idx+1}/{len(combos)}] {params}")

        try:
            cfg = Config(
                env_name=env_name,
                advantage_method="gae",
                total_timesteps=total_timesteps,
                num_envs=params["num_envs"],
                policy_lr=params["policy_lr"],
                value_lr=params["value_lr"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                seed=42,
            )
            agent = A2CAgent(cfg)
            df = agent.train(verbose=False)

            # use final mean reward as the metric
            if len(df) > 0:
                final_reward = df["mean_reward"].iloc[-1]
            else:
                final_reward = float("-inf")

            results.append({**params, "final_reward": final_reward})
            print(f"  → final mean reward: {final_reward:.1f}")

        except Exception as e:
            print(f"  → FAILED: {e}")
            results.append({**params, "final_reward": float("-inf")})

    # save and report
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("final_reward", ascending=False)
    csv_path = os.path.join(output_dir, f"grid_search_{env_name}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nGrid search results saved to {csv_path}")
    print(f"\nTop 5 configurations for {env_name}:")
    print(results_df.head(5).to_string(index=False))

    return results_df


def run_full_grid_search(total_timesteps: int = 500_000,
                         output_dir: str = "grid_search_results"):
    """Run grid search for each environment."""
    for env_name in ENVS:
        print(f"\n{'#'*60}")
        print(f"  GRID SEARCH: {env_name}")
        print(f"{'#'*60}")
        run_grid_search(env_name, total_timesteps, output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="A2C Continuous Control Experiments")

    parser.add_argument("--grid-search", action="store_true",
                        help="Run hyperparameter grid search instead of comparison")
    parser.add_argument("--env", type=str, default=None,
                        choices=ENVS,
                        help="Run on a single environment only")
    parser.add_argument("--method", type=str, default=None,
                        choices=METHODS,
                        help="Run a single advantage method only")
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total timesteps for training")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of random seeds for comparison")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.grid_search:
        # ── Grid Search Mode ──
        if args.env:
            run_grid_search(args.env, args.total_steps,
                            output_dir=args.output_dir)
        else:
            run_full_grid_search(args.total_steps,
                                 output_dir=args.output_dir)

    elif args.env and args.method:
        # ── Single Experiment Mode ──
        os.makedirs(args.output_dir, exist_ok=True)
        df = run_single_experiment(
            args.env, args.method,
            total_timesteps=args.total_steps,
        )
        csv_path = os.path.join(args.output_dir, f"{args.env}_{args.method}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    else:
        # ── Full Comparison Mode (default) ──
        run_comparison(
            total_timesteps=args.total_steps,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()