#!/usr/bin/env python3
"""
Replication: "Polychromic Objectives for Reinforcement Learning"
Environment: MiniGrid-FourRooms-v0
Algorithms: REINFORCE w/ Baseline, PPO, Polychromic PPO

Usage:
    python polychromic_ppo.py --algo pretrain_only   # pretrain & evaluate
    python polychromic_ppo.py --algo reinforce
    python polychromic_ppo.py --algo ppo
    python polychromic_ppo.py --algo poly_ppo
    python polychromic_ppo.py --algo all              # run all & compare
    python polychromic_ppo.py --algo all --quick       # quick sanity check

Paper target results (Four Rooms):
    Pretrained:  (0.469, 70.4%)
    REINFORCE:   (0.639, 89.6%)
    PPO:         (0.618, 89.2%)
    Poly-PPO:    (0.666, 92.4%)
"""

import os, sys, argparse, time, copy, math, random, pickle, json
from collections import deque, defaultdict
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
try:
    import minigrid
    from minigrid.wrappers import ImgObsWrapper
except ImportError:
    print("Please install: pip install minigrid gymnasium")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

class Config:
    # --- Environment ---
    env_name        = "MiniGrid-FourRooms-v0"
    num_configs     = 50        # fixed env seeds
    max_steps       = 100       # episode horizon H
    num_actions     = 7         # MiniGrid discrete actions

    # --- Reward: 1 - 0.5 * t/H ---
    time_penalty    = 0.5

    # --- RL training ---
    num_iterations  = 300
    gamma           = 1.0
    gae_lambda      = 0.95
    rollouts_per_config = 4     # rollouts per config per iteration (for REINFORCE/PPO)

    # --- PPO / Poly-PPO ---
    ppo_epochs      = 2
    mini_batch_size = 64
    clip_eps        = 0.2
    actor_lr        = 1e-5
    critic_lr       = 1e-4
    max_grad_norm   = 0.5
    kl_coeff        = 0.01

    # --- REINFORCE ---
    reinforce_lr    = 1e-5
    baseline_lr     = 1e-4

    # --- Poly-PPO specific ---
    n_set           = 4         # set size n
    N_vines         = 8         # vines per rollout state
    p_rollout       = 2         # rollout states per seed trajectory
    M_sets          = 4         # sets for baseline
    W_window        = 5         # polychromic window

    # --- Pretraining ---
    pretrain_epochs     = 100
    pretrain_lr         = 1e-3
    pretrain_batch      = 128
    demos_per_config    = 20
    pretrain_ent_coeff  = 0.01

    # --- Evaluation ---
    eval_rollouts   = 100
    num_seeds       = 3

    # --- Model ---
    hidden_dim      = 128

    # --- Misc ---
    log_interval    = 10
    save_dir        = "checkpoints"
    results_dir     = "results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Environment Utilities
# ═══════════════════════════════════════════════════════════════

def make_env():
    env = gym.make(Config.env_name, max_steps=Config.max_steps)
    env = ImgObsWrapper(env)
    return env


def flatten_obs(obs):
    """uint8 image → flat float32 in [0, 1]."""
    return obs.flatten().astype(np.float32) / 255.0


def obs_to_tensor(obs, device=None):
    device = device or Config.device
    return torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device)


def get_obs_dim():
    env = make_env()
    obs, _ = env.reset()
    dim = int(np.prod(obs.shape))
    env.close()
    return dim


def compute_reward(timestep):
    """Paper: r = 1 − 0.5·t/H when goal reached at step t."""
    return 1.0 - Config.time_penalty * timestep / Config.max_steps


# --- State save / restore (for vine sampling) ---

def save_env_state(env):
    inner = env.unwrapped
    return {
        'agent_pos': (int(inner.agent_pos[0]), int(inner.agent_pos[1])),
        'agent_dir': int(inner.agent_dir),
        'step_count': int(inner.step_count),
    }


def restore_env_state(env, state):
    """Restore state; return raw image observation."""
    inner = env.unwrapped
    inner.agent_pos = np.array(state['agent_pos'])
    inner.agent_dir = state['agent_dir']
    inner.step_count = state['step_count']
    return inner.gen_obs()['image']


# --- Room identification ---

def get_room_id(pos, mid_x=None, mid_y=None):
    """Map (x,y) to room 0‑3.  mid_x/mid_y are wall positions."""
    x, y = int(pos[0]), int(pos[1])
    mx = mid_x if mid_x else 9  # default 19×19 grid
    my = mid_y if mid_y else 9
    return (1 if x >= mx else 0) + (2 if y >= my else 0)


def rooms_visited(positions):
    return frozenset(get_room_id(p) for p in positions)


def compute_diversity(room_set_list):
    """
    d(s, τ_{1:n}) = (unique − 1) / (n − 1),  0 if all same.
    room_set_list: list of frozenset for each trajectory.
    """
    n = len(room_set_list)
    if n <= 1:
        return 0.0
    unique = len(set(room_set_list))
    return max(0.0, (unique - 1) / (n - 1))


# --- BFS Expert Solver ---

DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def find_goal_pos(grid):
    for x in range(grid.width):
        for y in range(grid.height):
            c = grid.get(x, y)
            if c is not None and c.type == 'goal':
                return (x, y)
    return None


def bfs_path(grid, start, goal):
    w, h = grid.width, grid.height
    start = (int(start[0]), int(start[1]))
    goal  = (int(goal[0]),  int(goal[1]))
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) == goal:
            return path
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                cell = grid.get(nx, ny)
                if cell is None or cell.type == 'goal':
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return None


def path_to_actions(path, start_dir):
    """Convert grid path → MiniGrid actions (0=left, 1=right, 2=fwd)."""
    actions = []
    d = start_dir
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        target = None
        for dd, (vx, vy) in DIR_VEC.items():
            if (vx, vy) == (dx, dy):
                target = dd
                break
        if target is None:
            continue
        diff = (target - d) % 4
        if diff == 1:
            actions.append(1)
        elif diff == 2:
            actions += [1, 1]
        elif diff == 3:
            actions.append(0)
        actions.append(2)
        d = target
    return actions


def get_expert_actions(env):
    inner = env.unwrapped
    goal = find_goal_pos(inner.grid)
    if goal is None:
        return []
    path = bfs_path(inner.grid, inner.agent_pos, goal)
    if path is None:
        return []
    return path_to_actions(path, inner.agent_dir)


# ═══════════════════════════════════════════════════════════════
# Expert Demonstrations
# ═══════════════════════════════════════════════════════════════

def generate_demonstrations(config_seeds):
    """BFS expert → (obs, action) pairs for behavioral cloning."""
    print("[Demos] Generating expert demonstrations ...")
    env = make_env()
    all_obs, all_acts = [], []
    ok, total = 0, 0

    for seed in config_seeds:
        for _ in range(Config.demos_per_config):
            obs, _ = env.reset(seed=seed)
            expert = get_expert_actions(env)
            total += 1
            if not expert:
                continue
            for a in expert:
                all_obs.append(flatten_obs(obs))
                all_acts.append(a)
                obs, rew, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            if rew > 0:
                ok += 1
    env.close()
    print(f"[Demos] {len(all_obs)} transitions, expert success {ok}/{total}")
    return np.array(all_obs, dtype=np.float32), np.array(all_acts, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=None):
        super().__init__()
        h = hidden or Config.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h),       nn.Tanh(),
            nn.Linear(h, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_dist(self, obs):
        return Categorical(logits=self.forward(obs))

    def get_action(self, obs, deterministic=False):
        dist = self.get_dist(obs)
        act = dist.probs.argmax(-1) if deterministic else dist.sample()
        return act, dist.log_prob(act), dist.entropy()


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=None):
        super().__init__()
        h = hidden or Config.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h),       nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Rollout helpers
# ═══════════════════════════════════════════════════════════════

def do_rollout(env, policy, seed=None,
               record_pos=False, record_states=False):
    """
    Collect one episode.
    If seed is given, env.reset(seed=seed) is called.
    Returns dict with obs, actions, log_probs, rewards, dones, etc.
    """
    device = Config.device

    if seed is not None:
        obs_raw, _ = env.reset(seed=seed)
    else:
        obs_raw, _ = env.reset()

    obs_l, act_l, logp_l, rew_l, done_l = [], [], [], [], []
    pos_l, state_l = [], []

    for t in range(Config.max_steps):
        obs_t = obs_to_tensor(obs_raw, device)
        obs_l.append(obs_t)
        if record_pos:
            pos_l.append((int(env.unwrapped.agent_pos[0]),
                          int(env.unwrapped.agent_pos[1])))
        if record_states:
            state_l.append(save_env_state(env))

        with torch.no_grad():
            a, lp, _ = policy.get_action(obs_t.unsqueeze(0))
        a_item = a.item()
        obs_raw, rew, term, trunc, _ = env.step(a_item)
        done = term or trunc

        # Paper reward
        r = compute_reward(t) if (term and rew > 0) else 0.0

        act_l.append(a_item)
        logp_l.append(lp.item())
        rew_l.append(r)
        done_l.append(done)
        if done:
            break

    result = dict(
        obs      = torch.stack(obs_l),
        actions  = torch.tensor(act_l, device=device, dtype=torch.long),
        log_probs= torch.tensor(logp_l, device=device, dtype=torch.float32),
        rewards  = torch.tensor(rew_l, device=device, dtype=torch.float32),
        dones    = torch.tensor(done_l, device=device, dtype=torch.bool),
        length   = len(act_l),
        success  = any(r > 0 for r in rew_l),
        ret      = sum(rew_l),
    )
    if record_pos:
        result['positions'] = pos_l
    if record_states:
        result['states'] = state_l
    return result


def do_rollout_from_state(env, policy, saved_state, config_seed,
                          record_pos=False):
    """Rollout starting from a saved mid-episode state (vine sampling)."""
    device = Config.device

    # Reset env to correct layout, then overwrite agent state
    env.reset(seed=config_seed)
    obs_raw = restore_env_state(env, saved_state)

    remaining = Config.max_steps - saved_state['step_count']
    obs_l, act_l, logp_l, rew_l, done_l = [], [], [], [], []
    pos_l = []

    for t in range(max(remaining, 1)):
        obs_t = obs_to_tensor(obs_raw, device)
        obs_l.append(obs_t)
        if record_pos:
            pos_l.append((int(env.unwrapped.agent_pos[0]),
                          int(env.unwrapped.agent_pos[1])))

        with torch.no_grad():
            a, lp, _ = policy.get_action(obs_t.unsqueeze(0))
        a_item = a.item()
        obs_raw, rew, term, trunc, _ = env.step(a_item)
        done = term or trunc

        actual_step = saved_state['step_count'] + t
        r = compute_reward(actual_step) if (term and rew > 0) else 0.0

        act_l.append(a_item)
        logp_l.append(lp.item())
        rew_l.append(r)
        done_l.append(done)
        if done:
            break

    if not act_l:
        return None

    result = dict(
        obs      = torch.stack(obs_l),
        actions  = torch.tensor(act_l, device=device, dtype=torch.long),
        log_probs= torch.tensor(logp_l, device=device, dtype=torch.float32),
        rewards  = torch.tensor(rew_l, device=device, dtype=torch.float32),
        dones    = torch.tensor(done_l, device=device, dtype=torch.bool),
        length   = len(act_l),
        success  = any(r > 0 for r in rew_l),
        ret      = sum(rew_l),
    )
    if record_pos:
        result['positions'] = pos_l
    return result


# ═══════════════════════════════════════════════════════════════
# GAE
# ═══════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones):
    T = len(rewards)
    adv = torch.zeros(T, device=rewards.device)
    last = 0.0
    gamma, lam = Config.gamma, Config.gae_lambda
    for t in reversed(range(T)):
        nv = 0.0 if (t == T - 1 or dones[t]) else values[t + 1].item()
        delta = rewards[t].item() + gamma * nv - values[t].item()
        adv[t] = last = delta + gamma * lam * (0.0 if dones[t] else 1.0) * last
    returns = adv + values.detach()
    return adv, returns


# ═══════════════════════════════════════════════════════════════
# Pretraining (Behavioral Cloning)
# ═══════════════════════════════════════════════════════════════

def pretrain(policy, config_seeds):
    print("\n" + "=" * 60)
    print("PRETRAINING (Behavioral Cloning)")
    print("=" * 60)

    obs_np, act_np = generate_demonstrations(config_seeds)
    if len(obs_np) == 0:
        print("WARNING: no demos generated")
        return

    obs_t = torch.tensor(obs_np, device=Config.device)
    act_t = torch.tensor(act_np, device=Config.device)
    N = len(obs_t)

    opt = torch.optim.Adam(policy.parameters(), lr=Config.pretrain_lr)

    for ep in range(Config.pretrain_epochs):
        perm = torch.randperm(N, device=Config.device)
        total_loss, total_acc, nb = 0, 0, 0
        for i in range(0, N, Config.pretrain_batch):
            idx = perm[i:i + Config.pretrain_batch]
            logits = policy(obs_t[idx])
            ce = F.cross_entropy(logits, act_t[idx])
            ent = -Categorical(logits=logits).entropy().mean()
            loss = ce + Config.pretrain_ent_coeff * ent
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(-1) == act_t[idx]).float().mean().item()
            nb += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{Config.pretrain_epochs}  "
                  f"loss={total_loss/nb:.4f}  acc={total_acc/nb:.3f}")


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(policy, config_seeds, num_rollouts=None):
    nr = num_rollouts or Config.eval_rollouts
    env = make_env()
    tot_r, tot_s, tot_n = 0.0, 0, 0
    for seed in config_seeds:
        for _ in range(nr):
            traj = do_rollout(env, policy, seed=seed)
            tot_r += traj['ret']
            tot_s += int(traj['success'])
            tot_n += 1
    env.close()
    return tot_r / tot_n, tot_s / tot_n * 100


# ═══════════════════════════════════════════════════════════════
# REINFORCE with Baseline
# ═══════════════════════════════════════════════════════════════

def train_reinforce(policy, value_net, config_seeds):
    print("\n" + "=" * 60)
    print("REINFORCE with Baseline")
    print("=" * 60)

    pi_opt = torch.optim.Adam(policy.parameters(), lr=Config.reinforce_lr)
    vf_opt = torch.optim.Adam(value_net.parameters(), lr=Config.baseline_lr)
    env = make_env()
    device = Config.device
    rpc = Config.rollouts_per_config

    for it in range(Config.num_iterations):
        batch_obs, batch_act, batch_ret = [], [], []
        it_r, it_s, it_n = 0.0, 0, 0

        for seed in config_seeds:
            for _ in range(rpc):
                traj = do_rollout(env, policy, seed=seed)
                T = traj['length']
                # Discounted returns
                G = 0.0
                returns = torch.zeros(T, device=device)
                for t in reversed(range(T)):
                    G = traj['rewards'][t].item() + Config.gamma * G
                    returns[t] = G

                batch_obs.append(traj['obs'])
                batch_act.append(traj['actions'])
                batch_ret.append(returns)
                it_r += traj['ret']; it_s += int(traj['success']); it_n += 1

        obs_b  = torch.cat(batch_obs)
        act_b  = torch.cat(batch_act)
        ret_b  = torch.cat(batch_ret)

        # Value baseline
        with torch.no_grad():
            val_b = value_net(obs_b)
        adv = ret_b - val_b
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy gradient
        dist = policy.get_dist(obs_b)
        logp = dist.log_prob(act_b)
        pi_loss = -(logp * adv.detach()).mean()

        pi_opt.zero_grad(); pi_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), Config.max_grad_norm)
        pi_opt.step()

        # Value update
        pred = value_net(obs_b)
        vf_loss = F.mse_loss(pred, ret_b.detach())
        vf_opt.zero_grad(); vf_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), Config.max_grad_norm)
        vf_opt.step()

        if (it + 1) % Config.log_interval == 0:
            print(f"  Iter {it+1:4d}/{Config.num_iterations}  "
                  f"R={it_r/it_n:.3f}  SR={it_s/it_n*100:.1f}%  "
                  f"πL={pi_loss.item():.4f}  VL={vf_loss.item():.4f}")

    env.close()
    return policy, value_net


# ═══════════════════════════════════════════════════════════════
# PPO
# ═══════════════════════════════════════════════════════════════

def train_ppo(policy, value_net, config_seeds):
    print("\n" + "=" * 60)
    print("PPO Training")
    print("=" * 60)

    pi_opt = torch.optim.Adam(policy.parameters(), lr=Config.actor_lr)
    vf_opt = torch.optim.Adam(value_net.parameters(), lr=Config.critic_lr)
    env = make_env()
    device = Config.device
    rpc = Config.rollouts_per_config

    for it in range(Config.num_iterations):
        # ---- collect ----
        batch_obs, batch_act, batch_oldlp, batch_adv, batch_ret = [], [], [], [], []
        it_r, it_s, it_n = 0.0, 0, 0

        for seed in config_seeds:
            for _ in range(rpc):
                traj = do_rollout(env, policy, seed=seed)
                with torch.no_grad():
                    vals = value_net(traj['obs'])
                adv, ret = compute_gae(traj['rewards'], vals, traj['dones'])
                batch_obs.append(traj['obs'])
                batch_act.append(traj['actions'])
                batch_oldlp.append(traj['log_probs'])
                batch_adv.append(adv)
                batch_ret.append(ret)
                it_r += traj['ret']; it_s += int(traj['success']); it_n += 1

        obs_b   = torch.cat(batch_obs)
        act_b   = torch.cat(batch_act)
        oldlp_b = torch.cat(batch_oldlp)
        adv_b   = torch.cat(batch_adv)
        ret_b   = torch.cat(batch_ret)
        adv_b   = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # frozen snapshot for KL
        old_pi = _OldPolicy(policy)

        # ---- PPO epochs ----
        N = obs_b.shape[0]
        for _ in range(Config.ppo_epochs):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, Config.mini_batch_size):
                idx = perm[s:s + Config.mini_batch_size]
                mb_obs  = obs_b[idx]
                mb_act  = act_b[idx]
                mb_olp  = oldlp_b[idx]
                mb_adv  = adv_b[idx]
                mb_ret  = ret_b[idx]

                dist   = policy.get_dist(mb_obs)
                newlp  = dist.log_prob(mb_act)
                ratio  = torch.exp(newlp - mb_olp)
                s1     = ratio * mb_adv
                s2     = torch.clamp(ratio, 1 - Config.clip_eps,
                                     1 + Config.clip_eps) * mb_adv
                pi_loss = -torch.min(s1, s2).mean()

                # KL penalty against frozen old policy
                with torch.no_grad():
                    old_dist = old_pi.get_dist(mb_obs)
                kl = torch.distributions.kl_divergence(old_dist, dist).mean()

                vf_loss = F.mse_loss(value_net(mb_obs), mb_ret)
                loss = pi_loss + 0.5 * vf_loss + Config.kl_coeff * kl

                pi_opt.zero_grad(); vf_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), Config.max_grad_norm)
                nn.utils.clip_grad_norm_(value_net.parameters(), Config.max_grad_norm)
                pi_opt.step(); vf_opt.step()

        if (it + 1) % Config.log_interval == 0:
            print(f"  Iter {it+1:4d}/{Config.num_iterations}  "
                  f"R={it_r/it_n:.3f}  SR={it_s/it_n*100:.1f}%")

    env.close()
    return policy, value_net


class _OldPolicy:
    """Lightweight wrapper: holds a frozen copy of the policy for KL computation."""
    def __init__(self, policy):
        self.net = PolicyNetwork(
            policy.net[0].in_features,
            policy.net[-1].out_features,
            policy.net[0].out_features,
        ).to(next(policy.parameters()).device)
        self.net.load_state_dict(copy.deepcopy(policy.state_dict()))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def get_dist(self, obs):
        return self.net.get_dist(obs)


# ═══════════════════════════════════════════════════════════════
# Polychromic PPO
# ═══════════════════════════════════════════════════════════════

def collect_vines(env, policy, seed):
    """
    Vine sampling for one config:
      1. N seed trajectories
      2. p rollout states per seed (equally spaced)
      3. N vine trajectories from each rollout state
    Returns: seed_trajs, vine_dict
    """
    N = Config.N_vines
    p = Config.p_rollout

    seed_trajs = []
    # key → {state, trajs[], obs_at}
    vine_dict = {}

    for _ in range(N):
        traj = do_rollout(env, policy, seed=seed,
                          record_pos=True, record_states=True)
        seed_trajs.append(traj)

    for traj in seed_trajs:
        T = traj['length']
        if T < p + 1:
            continue
        for k in range(1, p + 1):
            idx = min(int(k * T / (p + 1)), T - 1)
            st = traj['states'][idx]
            key = (st['agent_pos'][0], st['agent_pos'][1],
                   st['agent_dir'], st['step_count'])
            if key in vine_dict:
                continue  # already have vines from this state
            vine_dict[key] = dict(state=st, trajs=[], obs_at=traj['obs'][idx])
            remaining = Config.max_steps - st['step_count']
            for _ in range(N):
                vt = do_rollout_from_state(env, policy, st, seed,
                                           record_pos=True)
                if vt is not None:
                    vine_dict[key]['trajs'].append(vt)

    return seed_trajs, vine_dict


def poly_objective(trajs):
    """f_poly = (1/n) Σ R(τ_i) · d(s, τ_{1:n})"""
    n = len(trajs)
    if n == 0:
        return 0.0
    avg_r = sum(t['ret'] for t in trajs) / n
    rs = [rooms_visited(t['positions']) if t.get('positions') else frozenset()
          for t in trajs]
    div = compute_diversity(rs)
    return avg_r * div


def train_poly_ppo(policy, value_net, config_seeds):
    print("\n" + "=" * 60)
    print("Polychromic PPO Training")
    print("=" * 60)

    pi_opt = torch.optim.Adam(policy.parameters(), lr=Config.actor_lr)
    vf_opt = torch.optim.Adam(value_net.parameters(), lr=Config.critic_lr)
    env = make_env()
    device = Config.device
    n, M, W = Config.n_set, Config.M_sets, Config.W_window

    for it in range(Config.num_iterations):
        batch_obs, batch_act, batch_oldlp = [], [], []
        batch_adv, batch_ret = [], []
        it_r, it_s, it_n = 0.0, 0, 0

        for seed in config_seeds:
            seed_trajs, vine_dict = collect_vines(env, policy, seed)

            # stats from seed trajs
            for t in seed_trajs:
                it_r += t['ret']; it_s += int(t['success']); it_n += 1

            # --- seed trajectories: standard GAE ---
            for traj in seed_trajs:
                with torch.no_grad():
                    vals = value_net(traj['obs'])
                adv, ret = compute_gae(traj['rewards'], vals, traj['dones'])
                batch_obs.append(traj['obs'])
                batch_act.append(traj['actions'])
                batch_oldlp.append(traj['log_probs'])
                batch_adv.append(adv)
                batch_ret.append(ret)

            # --- vine states: polychromic advantage ---
            for key, vd in vine_dict.items():
                vts = vd['trajs']
                if len(vts) < n:
                    continue

                # form M sets of n trajectories
                sets = []
                for _ in range(M):
                    chosen = random.sample(range(len(vts)), min(n, len(vts)))
                    sets.append([vts[c] for c in chosen])

                scores = [poly_objective(s) for s in sets]
                baseline = float(np.mean(scores))

                for s_set, score in zip(sets, scores):
                    poly_adv_val = score - baseline
                    for traj in s_set:
                        T = traj['length']
                        w = min(W + 1, T)
                        if w > 0:
                            pa = torch.full((w,), poly_adv_val,
                                            dtype=torch.float32, device=device)
                            with torch.no_grad():
                                v = value_net(traj['obs'][:w])
                            batch_obs.append(traj['obs'][:w])
                            batch_act.append(traj['actions'][:w])
                            batch_oldlp.append(traj['log_probs'][:w])
                            batch_adv.append(pa)
                            batch_ret.append(pa + v)

                        # rest: standard GAE
                        if T > w:
                            rest_obs  = traj['obs'][w:]
                            rest_act  = traj['actions'][w:]
                            rest_lp   = traj['log_probs'][w:]
                            rest_rew  = traj['rewards'][w:]
                            rest_done = traj['dones'][w:]
                            with torch.no_grad():
                                rest_val = value_net(rest_obs)
                            ra, rr = compute_gae(rest_rew, rest_val, rest_done)
                            batch_obs.append(rest_obs)
                            batch_act.append(rest_act)
                            batch_oldlp.append(rest_lp)
                            batch_adv.append(ra)
                            batch_ret.append(rr)

        if not batch_obs:
            continue

        obs_b   = torch.cat(batch_obs)
        act_b   = torch.cat(batch_act)
        oldlp_b = torch.cat(batch_oldlp)
        adv_b   = torch.cat(batch_adv)
        ret_b   = torch.cat(batch_ret)
        adv_b   = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        old_pi = _OldPolicy(policy)

        # ---- PPO update ----
        Ntot = obs_b.shape[0]
        for _ in range(Config.ppo_epochs):
            perm = torch.randperm(Ntot, device=device)
            for s in range(0, Ntot, Config.mini_batch_size):
                idx = perm[s:s + Config.mini_batch_size]
                mb_obs = obs_b[idx]
                mb_act = act_b[idx]
                mb_olp = oldlp_b[idx]
                mb_adv = adv_b[idx]
                mb_ret = ret_b[idx]

                dist  = policy.get_dist(mb_obs)
                newlp = dist.log_prob(mb_act)
                ratio = torch.exp(newlp - mb_olp)
                s1    = ratio * mb_adv
                s2    = torch.clamp(ratio, 1 - Config.clip_eps,
                                    1 + Config.clip_eps) * mb_adv
                pi_loss = -torch.min(s1, s2).mean()

                with torch.no_grad():
                    old_dist = old_pi.get_dist(mb_obs)
                kl = torch.distributions.kl_divergence(old_dist, dist).mean()

                vf_loss = F.mse_loss(value_net(mb_obs), mb_ret)
                loss = pi_loss + 0.5 * vf_loss + Config.kl_coeff * kl

                pi_opt.zero_grad(); vf_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), Config.max_grad_norm)
                nn.utils.clip_grad_norm_(value_net.parameters(), Config.max_grad_norm)
                pi_opt.step(); vf_opt.step()

        if (it + 1) % Config.log_interval == 0:
            print(f"  Iter {it+1:4d}/{Config.num_iterations}  "
                  f"R={it_r/max(it_n,1):.3f}  SR={it_s/max(it_n,1)*100:.1f}%")

    env.close()
    return policy, value_net


# ═══════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════

def save_model(policy, value_net, name):
    os.makedirs(Config.save_dir, exist_ok=True)
    p = os.path.join(Config.save_dir, f"{name}.pt")
    torch.save({'policy': policy.state_dict(),
                'value': value_net.state_dict()}, p)
    print(f"  Saved → {p}")


def load_model(policy, value_net, name):
    p = os.path.join(Config.save_dir, f"{name}.pt")
    if not os.path.exists(p):
        return False
    ckpt = torch.load(p, map_location=Config.device, weights_only=True)
    policy.load_state_dict(ckpt['policy'])
    value_net.load_state_dict(ckpt['value'])
    print(f"  Loaded ← {p}")
    return True


# ═══════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════

def run_experiment(algo, training_seed=42, quick=False):
    if quick:
        Config.num_configs = 10
        Config.num_iterations = 50
        Config.eval_rollouts = 10
        Config.pretrain_epochs = 30
        Config.demos_per_config = 10
        Config.log_interval = 5
        Config.rollouts_per_config = 2

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)

    seeds = list(range(Config.num_configs))
    obs_dim = get_obs_dim()
    act_dim = Config.num_actions
    dev = Config.device
    print(f"\nDevice={dev}  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"configs={Config.num_configs}  iters={Config.num_iterations}")

    policy    = PolicyNetwork(obs_dim, act_dim).to(dev)
    value_net = ValueNetwork(obs_dim).to(dev)

    # ---- Pretrain ----
    pt_name = f"pretrained_s{training_seed}"
    if not load_model(policy, value_net, pt_name):
        pretrain(policy, seeds)
        save_model(policy, value_net, pt_name)

    print("\nEval pretrained ...")
    pr, ps = evaluate(policy, seeds, Config.eval_rollouts)
    print(f"  Pretrained → reward={pr:.3f}  success={ps:.1f}%")

    if algo == 'pretrain_only':
        return {'pretrained': (pr, ps)}

    # ---- RL fine-tune (from fresh copy of pretrained weights) ----
    pi_rl = PolicyNetwork(obs_dim, act_dim).to(dev)
    vf_rl = ValueNetwork(obs_dim).to(dev)
    pi_rl.load_state_dict(copy.deepcopy(policy.state_dict()))
    vf_rl.load_state_dict(copy.deepcopy(value_net.state_dict()))

    t0 = time.time()
    if algo == 'reinforce':
        train_reinforce(pi_rl, vf_rl, seeds)
    elif algo == 'ppo':
        train_ppo(pi_rl, vf_rl, seeds)
    elif algo == 'poly_ppo':
        train_poly_ppo(pi_rl, vf_rl, seeds)
    else:
        raise ValueError(algo)
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.0f}s")

    save_model(pi_rl, vf_rl, f"{algo}_s{training_seed}")

    print(f"\nEval {algo} ...")
    rr, rs = evaluate(pi_rl, seeds, Config.eval_rollouts)
    print(f"  {algo} → reward={rr:.3f}  success={rs:.1f}%")

    return {'pretrained': (pr, ps), algo: (rr, rs)}


def run_all(quick=False):
    print("=" * 60)
    print("Full comparison  (Table 1 — Four Rooms)")
    print("=" * 60)

    results = defaultdict(list)
    nseed = 1 if quick else Config.num_seeds

    for si in range(nseed):
        ts = 42 + si * 111
        print(f"\n{'='*60}\nSEED {si+1}/{nseed}  (training_seed={ts})\n{'='*60}")
        for algo in ['reinforce', 'ppo', 'poly_ppo']:
            r = run_experiment(algo, ts, quick)
            for k, v in r.items():
                results[k].append(v)

    paper = dict(pretrained=(0.469, 70.4), reinforce=(0.639, 89.6),
                 ppo=(0.618, 89.2), poly_ppo=(0.666, 92.4))

    print("\n" + "=" * 60)
    print(f"{'Method':<16} {'Reward':>8} {'Succ%':>8}  {'Paper-R':>8} {'Paper-S%':>8}")
    print("-" * 52)
    for m in ['pretrained', 'reinforce', 'ppo', 'poly_ppo']:
        if m in results:
            rw = np.mean([x[0] for x in results[m]])
            sc = np.mean([x[1] for x in results[m]])
            p_r, p_s = paper[m]
            print(f"  {m:<14} {rw:>8.3f} {sc:>7.1f}%  {p_r:>8.3f} {p_s:>7.1f}%")

    os.makedirs(Config.results_dir, exist_ok=True)
    with open(os.path.join(Config.results_dir, "results.json"), 'w') as f:
        json.dump({k: [(float(a), float(b)) for a, b in v]
                   for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {Config.results_dir}/results.json")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='all',
                        choices=['reinforce','ppo','poly_ppo','pretrain_only','all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Reduced scale for sanity check')
    parser.add_argument('--iters', type=int)
    parser.add_argument('--configs', type=int)
    args = parser.parse_args()
    if args.iters:   Config.num_iterations = args.iters
    if args.configs:  Config.num_configs = args.configs

    if args.algo == 'all':
        run_all(args.quick)
    else:
        run_experiment(args.algo, args.seed, args.quick)