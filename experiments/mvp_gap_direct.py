"""
MVP: Direct SINDy gap penalty on Q-target (no QΔ network).

Key insight: QΔ with γ=0.99 smooths away spatial variation.
Direct per-step gap signal preserves it.

penalty(s,a) = λ * ||SINDy_predict(s,a)||²

High velocity → large gap prediction → large penalty → conservative Q
Low velocity → small gap → small penalty → trust Q

2 conditions:
  c1: Raw Sim (baseline)
  c2: Raw Sim + direct gap penalty

运行: conda run -n MC-WM python3 -u experiments/mvp_gap_direct.py --mode c1
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector

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
PENALTY_SCALE = 0.5   # per-step gap is small (~0.5), need higher scale


class ReplayBuffer:
    def __init__(self, od, ad, cap):
        self.max_size = cap; self.ptr = self.size = 0
        self.s = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros((cap, 1), np.float32); self.s2 = np.zeros((cap, od), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
    def add(self, s, a, r, s2, d):
        i = self.ptr; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (i+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in [self.s, self.a, self.r, self.s2, self.d])


def collect_paired(env_cls, n_steps, seed=SEED):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed); ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns_s, _, ds, ts, _ = sim.step(a); ns_r, _, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_s, a])); ds_list.append(ns_r - ns_s)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    return np.array(SA_list, np.float32), np.array(ds_list, np.float32), ep


def make_gap_fn(corrector):
    """Direct per-step gap signal: ||Δ̂(s,a)||²"""
    def gap_fn(s_batch, a_batch):
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        delta_pred = corrector.predict_batch(SA)
        return np.mean(delta_pred ** 2, axis=-1)
    return gap_fn


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


def train(env_cls, label, log_fn, gap_fn=None, penalty_scale=0.5, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_cls(mode="sim")
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])

    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE,
                       gap_fn=gap_fn, penalty_scale=penalty_scale)

    buf = ReplayBuffer(od, ad, REPLAY_SIZE)
    obs, _ = env.reset(seed=seed); curve = []

    # Diagnostic accumulators
    gap_vals = []

    log_fn(f"\n[{label}] {TRAIN_STEPS//1000}k steps" +
           (f" | penalty_scale={penalty_scale}" if gap_fn else ""))

    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))

        # Track gap signal
        if gap_fn is not None and step % 100 == 0:
            g = gap_fn(obs.reshape(1, -1), a.reshape(1, -1))[0]
            gap_vals.append(g)

        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

        if step % EVAL_INTERVAL == 0:
            ret, std = evaluate(agent, env_cls)
            curve.append((step, ret))
            diag = ""
            if gap_vals:
                g_mean = np.mean(gap_vals[-50:])
                g_std = np.std(gap_vals[-50:])
                pen_mean = g_mean * penalty_scale
                diag = f"  gap={g_mean:.3f}±{g_std:.3f} pen={pen_mean:.3f}"
                gap_vals.clear()
            log_fn(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}{diag}")

    env.close()
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2"])
    args = parser.parse_args()
    mode = args.mode

    log_path = f"/tmp/mvp_gap_{mode}.log"
    def log_fn(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log_fn(f"[{mode}] Device: {DEVICE}")
    log_fn("Direct SINDy gap penalty (no QΔ network)")

    env_cls = CarpetAntEnv

    if mode == "c1":
        curve = train(env_cls, "Raw Sim (baseline)", log_fn)
    elif mode == "c2":
        log_fn("\n[A] SINDy ensemble fit...")
        SA, delta_s, n_eps = collect_paired(env_cls, N_COLLECT)
        log_fn(f"  {len(SA)} paired steps ({n_eps} eps)")
        corrector = SINDyEnsembleCorrector(SA.shape[1] - env_cls.ACT_DIM, K=5, gate_tau=0.1)
        corrector.fit(SA, delta_s)
        gap_fn = make_gap_fn(corrector)

        # Verify gap signal variation
        test_gap = gap_fn(SA[:200, :env_cls.OBS_DIM], SA[:200, env_cls.OBS_DIM:])
        log_fn(f"  Gap signal: mean={test_gap.mean():.4f} std={test_gap.std():.4f} "
               f"min={test_gap.min():.4f} max={test_gap.max():.4f}")
        log_fn(f"  Penalty range: [{test_gap.min()*PENALTY_SCALE:.3f}, {test_gap.max()*PENALTY_SCALE:.3f}]")

        curve = train(env_cls, "Raw Sim + direct gap", log_fn,
                      gap_fn=gap_fn, penalty_scale=PENALTY_SCALE)

    avg = np.mean([r for _, r in curve[-3:]])
    log_fn(f"\n{'='*50}")
    log_fn(f"{mode} last 3 avg: {avg:.1f}")
    log_fn(f"{'='*50}")


if __name__ == "__main__":
    main()
