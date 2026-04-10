"""
QΔ v2: Ensemble QΔ pre-trained on TRUE gap, frozen during policy training.

Fixes from v1:
1. Ensemble QΔ (K=3) — no single-network blowup
2. Pre-trained on TRUE ||Δs||² from paired data — learns real gap, not SINDy noise
3. Frozen during policy training — no drift from noisy online signal
4. Penalty clamped at max_penalty — prevents Q-target collapse

运行: conda run -n MC-WM python3 -u experiments/mvp_qdelta_v2.py --mode c1
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.policy.q_delta import QDeltaModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_COLLECT    = 3_000
TRAIN_STEPS  = 50_000
EVAL_INTERVAL = 5_000
N_EVAL_EPS   = 10
WARMUP       = 2_000
BATCH_SIZE   = 256
REPLAY_SIZE  = 100_000
N_CRITICS    = 3
BETA_LCB     = -2.0
HIDDEN       = 256
LR           = 3e-4
PENALTY_SCALE = 0.1
MAX_PENALTY   = 2.0


class ReplayBuffer:
    def __init__(self, od, ad, cap):
        self.max_size = cap; self.ptr = self.size = 0
        self.s  = np.zeros((cap, od), np.float32)
        self.a  = np.zeros((cap, ad), np.float32)
        self.r  = np.zeros((cap, 1),  np.float32)
        self.s2 = np.zeros((cap, od), np.float32)
        self.d  = np.zeros((cap, 1),  np.float32)
    def add(self, s, a, r, s2, d):
        i = self.ptr; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (i+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in [self.s, self.a, self.r, self.s2, self.d])


def collect_paired_trajectories(env_cls, n_steps, seed=SEED):
    """Collect paired trajectories with TRUE gap reward."""
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    ep = 0

    # Build trajectories for QΔ pre-training
    current_traj = {'s': [], 'a': [], 'gap_reward': [], 'done': []}
    trajectories = []

    for step in range(n_steps):
        a = sim.action_space.sample()
        ns_s, _, ds, ts, _ = sim.step(a)
        ns_r, _, dr, tr, _ = real.step(a)

        # TRUE gap reward: ||s_real_next - s_sim_next||²
        gap_r = float(np.mean((ns_r - ns_s) ** 2))

        current_traj['s'].append(os_s.copy())
        current_traj['a'].append(a.copy())
        current_traj['gap_reward'].append(gap_r)
        current_traj['done'].append(float(ds or ts or dr or tr))

        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            # Finalize trajectory
            for k in current_traj:
                current_traj[k] = np.array(current_traj[k])
            trajectories.append(current_traj)
            current_traj = {'s': [], 'a': [], 'gap_reward': [], 'done': []}
            ep += 1
            os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)

    # Don't forget last partial trajectory
    if len(current_traj['s']) > 1:
        for k in current_traj:
            current_traj[k] = np.array(current_traj[k])
        trajectories.append(current_traj)

    sim.close(); real.close()
    return trajectories, ep


def evaluate(agent, env_cls, n_eps=N_EVAL_EPS):
    env = env_cls(mode="real"); rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep+200); total = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = env.step(a); total += r
            if d or tr: break
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def train(env_cls, label, log_fn, q_delta_module=None, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_cls(mode="sim")
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])

    # Pass pre-trained QΔ module directly (or None for baseline)
    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE,
                       gap_fn=q_delta_module, penalty_scale=PENALTY_SCALE)

    buf = ReplayBuffer(od, ad, REPLAY_SIZE)
    obs, _ = env.reset(seed=seed); curve = []
    has_qd = q_delta_module is not None

    log_fn(f"\n[{label}] {TRAIN_STEPS//1000}k steps" +
           (" | QΔ ensemble K=3 frozen" if has_qd else ""))

    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

        if step % EVAL_INTERVAL == 0:
            ret, std = evaluate(agent, env_cls)
            curve.append((step, ret))
            diag = ""
            if has_qd and q_delta_module._penalty_history:
                pen_recent = np.mean(q_delta_module._penalty_history[-100:])
                diag = f"  avg_pen={pen_recent:.3f}"
            log_fn(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}{diag}")

    env.close()
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2"])
    args = parser.parse_args()
    mode = args.mode

    log_path = f"/tmp/mvp_qdv2_{mode}.log"
    def log_fn(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log_fn(f"[{mode}] Device: {DEVICE}")
    env_cls = CarpetAntEnv

    if mode == "c1":
        # Baseline: Raw Sim, no QΔ
        curve = train(env_cls, "Raw Sim (baseline)", log_fn)
    elif mode == "c2":
        # Pre-train QΔ on TRUE gap, then freeze
        log_fn("\n[A] Collecting paired trajectories with TRUE gap...")
        trajs, n_eps = collect_paired_trajectories(env_cls, N_COLLECT)
        total_steps = sum(len(t['s']) for t in trajs)
        avg_gap = np.mean([t['gap_reward'].mean() for t in trajs])
        log_fn(f"  {total_steps} steps, {n_eps} episodes, {len(trajs)} trajectories")
        log_fn(f"  Avg TRUE gap reward: {avg_gap:.4f}")

        log_fn("\n[B] Pre-training ensemble QΔ (K=3) on TRUE gap...")
        obs_dim = trajs[0]['s'].shape[1]; act_dim = trajs[0]['a'].shape[1]
        q_delta = QDeltaModule(obs_dim, act_dim, hidden_dim=128, K=3, lr=3e-4,
                               gamma=0.99, tau=5e-3, penalty_scale=PENALTY_SCALE,
                               max_penalty=MAX_PENALTY, device=DEVICE)
        q_delta.pretrain(trajs, n_epochs=50, batch_size=256)

        # Print loss curve
        losses = q_delta._loss_history
        log_fn(f"  Loss curve: start={losses[0]:.4f} → end={losses[-1]:.4f}")
        log_fn(f"  Loss at epoch 10={losses[min(9,len(losses)-1)]:.4f}, "
               f"30={losses[min(29,len(losses)-1)]:.4f}")

        log_fn("\n[C] Freezing QΔ")
        q_delta.freeze()

        curve = train(env_cls, "Raw Sim + QΔv2 (frozen)", log_fn,
                      q_delta_module=q_delta)

    avg = np.mean([r for _, r in curve[-3:]])
    log_fn(f"\n{'='*50}")
    log_fn(f"{mode} last 3 avg: {avg:.1f}")
    log_fn(f"{'='*50}")


if __name__ == "__main__":
    main()
