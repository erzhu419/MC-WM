"""
Correction validity check: is corrected sim closer to real than raw sim?

Tests on TWO distributions:
A) Random policy (training distribution) — should match RMSE reduction %
B) Trained policy — the real question: does correction generalize?

For each distribution, collects parallel trajectories and compares:
  raw_error   = |s_real - s_sim|
  corr_error  = |s_real - (s_sim + sindy_delta)|
  reduction % = 1 - corr_error / raw_error
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from experiments.mvp_aero_cheetah import SINDyStateCorrector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/correction_validity.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def collect_parallel_trajectories(env_cls, policy_fn, n_steps, corrector, seed=SEED):
    """Run sim and real in lockstep with same actions, record errors."""
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_sim, _ = sim.reset(seed=seed); os_real, _ = real.reset(seed=seed)

    raw_errors = []      # |s_real - s_sim|
    corr_errors = []     # |s_real - s_corrected|
    per_dim_raw = []
    per_dim_corr = []
    ep = 0

    for step in range(n_steps):
        a = policy_fn(os_sim)
        ns_sim, _, ds, ts, _ = sim.step(a)
        ns_real, _, dr, tr, _ = real.step(a)

        # Raw sim error
        raw_err = np.abs(ns_real - ns_sim)
        raw_errors.append(float(np.mean(raw_err)))
        per_dim_raw.append(raw_err)

        # Corrected sim error
        SA = np.concatenate([os_sim, a]).reshape(1, -1).astype(np.float32)
        delta = corrector.predict_batch(SA)[0]  # (obs_dim,)
        ns_corr = ns_sim + delta
        corr_err = np.abs(ns_real - ns_corr)
        corr_errors.append(float(np.mean(corr_err)))
        per_dim_corr.append(corr_err)

        os_sim, os_real = ns_sim, ns_real
        if ds or ts or dr or tr:
            ep += 1
            os_sim, _ = sim.reset(seed=seed+ep)
            os_real, _ = real.reset(seed=seed+ep)

    sim.close(); real.close()

    raw_mean = np.mean(raw_errors)
    corr_mean = np.mean(corr_errors)
    reduction = (1 - corr_mean / max(raw_mean, 1e-8)) * 100

    per_dim_raw = np.array(per_dim_raw)   # (N, obs_dim)
    per_dim_corr = np.array(per_dim_corr)
    per_dim_reduction = (1 - per_dim_corr.mean(0) / np.maximum(per_dim_raw.mean(0), 1e-8)) * 100

    return {
        "raw_mae": raw_mean,
        "corr_mae": corr_mean,
        "reduction_pct": reduction,
        "per_dim_reduction": per_dim_reduction,
        "n_steps": n_steps,
        "n_eps": ep,
    }


def main():
    log(f"Device: {DEVICE}")
    log("="*60)
    log("CORRECTION VALIDITY CHECK — CarpetAnt")
    log("="*60)

    env_cls = CarpetAntEnv
    obs_dim = 27; act_dim = 8

    # ── Fit SINDy on paired data
    log("\n[1] Fitting SINDy corrector...")
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_, _ = sim.reset(seed=SEED); or_, _ = real.reset(seed=SEED)
    ep = 0
    for _ in range(3000):
        a = sim.action_space.sample()
        ns, _, ds, ts, _ = sim.step(a)
        nr, _, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_, a]))
        ds_list.append(nr - ns)
        os_, or_ = ns, nr
        if ds or ts or dr or tr:
            ep += 1; os_, _ = sim.reset(seed=ep+SEED); or_, _ = real.reset(seed=ep+SEED)
    sim.close(); real.close()
    SA = np.array(SA_list, dtype=np.float32)
    delta_s = np.array(ds_list, dtype=np.float32)

    corrector = SINDyStateCorrector(obs_dim)
    corrector.fit(SA, delta_s)
    log(f"  SINDy fitted on {len(SA)} paired steps")

    # ── Test A: Random policy (training distribution)
    log("\n[A] Random policy (training distribution)")
    def random_policy(obs):
        return env_cls(mode="sim").action_space.sample()

    result_a = collect_parallel_trajectories(env_cls, random_policy, 2000, corrector)
    log(f"  Raw MAE:       {result_a['raw_mae']:.4f}")
    log(f"  Corrected MAE: {result_a['corr_mae']:.4f}")
    log(f"  Reduction:     {result_a['reduction_pct']:.1f}%")
    log(f"  Per-dim reduction (vel dims 13-26):")
    for i in range(13, 27):
        r = result_a['per_dim_reduction'][i]
        marker = "✓" if r > 10 else "✗" if r < -10 else "~"
        log(f"    dim {i:2d}: {r:+6.1f}% {marker}")

    # ── Train a policy in pure sim (20k steps, fast)
    log("\n[B] Training policy in pure sim (20k steps)...")
    env = env_cls(mode="sim")
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])
    agent = RESACAgent(od, ad, al, hidden_dim=256, n_critics=3,
                       beta=-2.0, lr=3e-4, device=DEVICE)

    # Simple replay buffer
    buf_s  = np.zeros((50000, od), np.float32)
    buf_a  = np.zeros((50000, ad), np.float32)
    buf_r  = np.zeros((50000, 1),  np.float32)
    buf_s2 = np.zeros((50000, od), np.float32)
    buf_d  = np.zeros((50000, 1),  np.float32)
    buf_ptr = buf_size = 0

    class SimpleBuf:
        def __init__(self): pass
        @property
        def size(self): return buf_size
        def sample(self, n):
            idx = np.random.randint(0, buf_size, n)
            return (torch.FloatTensor(buf_s[idx]).to(DEVICE),
                    torch.FloatTensor(buf_a[idx]).to(DEVICE),
                    torch.FloatTensor(buf_r[idx]).to(DEVICE),
                    torch.FloatTensor(buf_s2[idx]).to(DEVICE),
                    torch.FloatTensor(buf_d[idx]).to(DEVICE))
    sbuf = SimpleBuf()

    obs, _ = env.reset(seed=SEED)
    for step in range(1, 20001):
        a = env.action_space.sample() if step < 2000 else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf_s[buf_ptr]=obs; buf_a[buf_ptr]=a; buf_r[buf_ptr]=r
        buf_s2[buf_ptr]=obs2; buf_d[buf_ptr]=float(d and not tr)
        buf_ptr = (buf_ptr+1) % 50000; buf_size = min(buf_size+1, 50000)
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= 2000 and buf_size >= 256:
            agent.update(sbuf)
    env.close()
    log("  Policy trained (20k steps)")

    # ── Test B: Trained policy (OOD distribution)
    log("\n[B] Trained policy (OOD for SINDy)")
    def trained_policy(obs):
        return agent.get_action(obs, deterministic=True)

    result_b = collect_parallel_trajectories(env_cls, trained_policy, 2000, corrector)
    log(f"  Raw MAE:       {result_b['raw_mae']:.4f}")
    log(f"  Corrected MAE: {result_b['corr_mae']:.4f}")
    log(f"  Reduction:     {result_b['reduction_pct']:.1f}%")
    log(f"  Per-dim reduction (vel dims 13-26):")
    for i in range(13, 27):
        r = result_b['per_dim_reduction'][i]
        marker = "✓" if r > 10 else "✗" if r < -10 else "~"
        log(f"    dim {i:2d}: {r:+6.1f}% {marker}")

    # ── Verdict
    log(f"\n{'='*60}")
    log("VERDICT")
    log(f"{'='*60}")
    log(f"  Random policy reduction:  {result_a['reduction_pct']:.1f}%")
    log(f"  Trained policy reduction: {result_b['reduction_pct']:.1f}%")
    drop = result_a['reduction_pct'] - result_b['reduction_pct']
    log(f"  Generalization drop:      {drop:.1f} pp")
    if result_b['reduction_pct'] > 20:
        log("  >>> CORRECTION VALID: generalizes to trained policy")
    elif result_b['reduction_pct'] > 0:
        log("  >>> CORRECTION MARGINAL: some value but degraded OOD")
    else:
        log("  >>> CORRECTION HARMFUL: makes things worse on trained policy distribution")


if __name__ == "__main__":
    main()
