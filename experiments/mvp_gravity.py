"""
MVP on GravityCheetah (2x gravity) — H2O benchmark env.

Reward gap = 0.495 (19x CarpetAnt), dynamics gap = 1.07 (3x CarpetAnt).
This is where MC-WM should shine.

4 conditions:
  c1: Raw Sim (2x gravity) — baseline
  c2: Raw Sim + direct SINDy gap penalty
  c3: Real env online training — upper bound
  c4: Raw Sim + direct gap + step-by-step diagnosis

运行: conda run -n MC-WM python3 -u experiments/mvp_gravity.py --mode c1
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv
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
PENALTY_SCALE = 0.1   # w_min: highest-gap transitions keep 10% learning weight


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
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    SA_list, ds_list, dr_list = [], [], []; ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns_s, rs, ds, ts, _ = sim.step(a); ns_r, rr, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_s, a]))
        ds_list.append(ns_r - ns_s)
        dr_list.append(rr - rs)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    SA = np.array(SA_list, np.float32)
    delta_s = np.array(ds_list, np.float32)
    delta_r = np.array(dr_list, np.float32)
    return SA, delta_s, delta_r, ep


def make_gap_fn(corrector, SA_train, log_fn=print):
    """
    Normalized gap signal: z-score based on training distribution, then sigmoid.

    Fixes three issues from diagnostics:
    1. OOD magnitude explosion (gap/true ratio up to 13x) → clamp + normalize
    2. Absolute magnitude not meaningful → relative to training dist
    3. Rank order preserved (ρ=0.80) → normalization keeps rank

    Output: gap in [0, 1] where 0=no gap (training avg), 1=extreme gap
    """
    delta_train = corrector.predict_batch(SA_train)
    train_gap = np.mean(delta_train ** 2, axis=-1)
    gap_mean = float(train_gap.mean())
    gap_std = float(train_gap.std())
    gap_p95 = float(np.percentile(train_gap, 95))
    log_fn(f"  Gap calibration: mean={gap_mean:.4f} std={gap_std:.4f} p95={gap_p95:.4f}")

    def gap_fn(s_batch, a_batch):
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        delta_pred = corrector.predict_batch(SA)
        gap_raw = np.mean(delta_pred ** 2, axis=-1)
        # Normalize: z-score relative to training distribution
        gap_z = (gap_raw - gap_mean) / max(gap_std, 1e-8)
        # Clamp z-score to [-2, 3] — cap OOD explosion
        gap_z = np.clip(gap_z, -2.0, 3.0)
        # Shift to [0, 1]: z=-2 → 0, z=3 → 1
        gap_norm = (gap_z + 2.0) / 5.0
        return gap_norm
    return gap_fn


def evaluate(agent, env_cls, mode="real", n_eps=N_EVAL_EPS):
    env = env_cls(mode=mode); rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep+200); total = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = env.step(a); total += r
            if d or tr: break
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def train(env_fn, env_cls, label, log_fn, gap_fn=None, penalty_scale=0.3, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_fn()
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])

    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE,
                       gap_fn=gap_fn, penalty_scale=penalty_scale)

    buf = ReplayBuffer(od, ad, REPLAY_SIZE)
    obs, _ = env.reset(seed=seed); curve = []
    # Diagnostic accumulators
    diag_history = []

    log_fn(f"\n[{label}] {TRAIN_STEPS//1000}k steps" +
           (f" | pen_scale={penalty_scale}" if gap_fn else ""))

    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE:
            update_diag = agent.update(buf)
            if step % 200 == 0:
                diag_history.append(update_diag)

        if step % EVAL_INTERVAL == 0:
            ret_real, std_real = evaluate(agent, env_cls, "real")
            ret_sim, _ = evaluate(agent, env_cls, "sim")
            curve.append((step, ret_real, ret_sim))

            # Aggregate diagnostics from last interval
            diag_str = ""
            if diag_history:
                keys = ["critic_loss", "q_pred_mean", "q_tgt_mean",
                         "iw_mean", "iw_min", "iw_reduction",
                         "gap_mean", "gap_std", "alpha"]
                avg = {}
                for k in keys:
                    vals = [d[k] for d in diag_history if k in d]
                    if vals: avg[k] = np.mean(vals)

                parts = []
                if "critic_loss" in avg:
                    parts.append(f"crit={avg['critic_loss']:.1f}")
                if "q_pred_mean" in avg:
                    parts.append(f"Q={avg['q_pred_mean']:.0f}")
                if "iw_mean" in avg:
                    parts.append(f"w={avg['iw_mean']:.3f}[{avg.get('iw_min',0):.3f}]")
                if "iw_reduction" in avg:
                    parts.append(f"reduct={avg['iw_reduction']:.1%}")
                if "gap_mean" in avg:
                    parts.append(f"gap={avg['gap_mean']:.2f}[{avg.get('gap_std',0):.2f}]")
                diag_str = "  " + " | ".join(parts)
                diag_history.clear()

            log_fn(f"  step {step:>6d} | real={ret_real:7.1f}±{std_real:4.0f}  sim={ret_sim:7.1f}{diag_str}")

    env.close()
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2", "c3", "c4"])
    args = parser.parse_args()
    mode = args.mode

    log_path = f"/tmp/mvp_grav_{mode}.log"
    def log_fn(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log_fn(f"[{mode}] Device: {DEVICE}")
    log_fn("Env: GravityCheetah (sim=2x gravity, real=1x)")

    env_cls = GravityCheetahEnv

    if mode == "c1":
        # Baseline: train in 2x gravity sim, eval in 1x real
        curve = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim (2x grav)", log_fn)

    elif mode == "c2":
        # SINDy gap detection + direct penalty
        log_fn("\n[A] Paired data + SINDy ensemble")
        SA, delta_s, delta_r, n_eps = collect_paired(env_cls, N_COLLECT)
        log_fn(f"  {len(SA)} steps ({n_eps} eps)")
        log_fn(f"  State gap: mean={np.abs(delta_s).mean():.4f}")
        log_fn(f"  Reward gap: mean={np.abs(delta_r).mean():.4f}")

        obs_dim = delta_s.shape[1]
        corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.1)
        corrector.fit(SA, delta_s)
        cov = corrector.correction_coverage(SA, delta_s)
        log_fn(f"  RMSE reduction: {cov['rmse_reduction_pct']:.1f}%")

        gap_fn = make_gap_fn(corrector, SA)
        test_gap = gap_fn(SA[:200, :obs_dim], SA[:200, obs_dim:])
        log_fn(f"  Gap signal: mean={test_gap.mean():.4f} std={test_gap.std():.4f} "
               f"min={test_gap.min():.4f} max={test_gap.max():.4f}")

        curve = train(lambda: env_cls(mode="sim"), env_cls,
                      "Raw Sim + gap penalty", log_fn,
                      gap_fn=gap_fn, penalty_scale=PENALTY_SCALE)

    elif mode == "c3":
        # Upper bound: train directly in real env
        curve = train(lambda: env_cls(mode="real"), env_cls,
                      "Real Online (upper bound)", log_fn)

    elif mode == "c4":
        # Ablation: uniform weight = 0.17 (same avg as c2 end, no gap signal)
        # If c4 ≈ c2 → improvement is just reduced LR, not gap detection
        # If c4 << c2 → gap signal provides real value
        def uniform_gap_fn(s_batch, a_batch):
            # Returns constant 0.83 → w = 0.1 + 0.9*(1-0.83) = 0.253 ≈ c2 average
            return np.full(len(s_batch), 0.83)
        curve = train(lambda: env_cls(mode="sim"), env_cls,
                      "Uniform w=0.25 (ablation)", log_fn,
                      gap_fn=uniform_gap_fn, penalty_scale=PENALTY_SCALE)

    # Summary
    avg_real = np.mean([r for _, r, _ in curve[-3:]])
    avg_sim = np.mean([s for _, _, s in curve[-3:]])
    log_fn(f"\n{'='*50}")
    log_fn(f"{mode} last 3 avg: real={avg_real:.1f} sim={avg_sim:.1f}")
    log_fn(f"sim-real gap: {(avg_sim-avg_real)/max(abs(avg_real),1)*100:+.1f}%")
    log_fn(f"{'='*50}")


if __name__ == "__main__":
    main()
