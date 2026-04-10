"""
MVP: QΔ (Residual Bellman) — the correct architecture.

SINDy is NOT a corrector. It's an OOD detector for the Q-function.
Policy trains in raw sim. QΔ penalizes Q-targets where dynamics gap is large.

3 conditions:
1. Raw Sim (baseline, no QΔ)
2. Raw Sim + QΔ (SINDy gap detection)
3. Raw Sim + QΔ + MixedBuffer (gap detection + real data)

运行: conda run -n MC-WM python3 -u experiments/mvp_qdelta.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/mvp_qdelta.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

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
OUT_DIR      = os.path.dirname(__file__)


class ReplayBuffer:
    def __init__(self, od, ad, cap):
        self.max_size = cap; self.ptr = self.size = 0
        self.s  = np.zeros((cap, od), np.float32)
        self.a  = np.zeros((cap, ad), np.float32)
        self.r  = np.zeros((cap, 1),  np.float32)
        self.s2 = np.zeros((cap, od), np.float32)
        self.d  = np.zeros((cap, 1),  np.float32)
    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (i+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE)
                     for x in [self.s, self.a, self.r, self.s2, self.d])


class MixedReplayBuffer:
    def __init__(self, od, ad, real_data, sim_cap, real_ratio=0.5):
        self.real_ratio = real_ratio; n = len(real_data)
        self.real_s = np.zeros((n, od), np.float32); self.real_a = np.zeros((n, ad), np.float32)
        self.real_r = np.zeros((n, 1), np.float32); self.real_s2 = np.zeros((n, od), np.float32)
        self.real_d = np.zeros((n, 1), np.float32)
        for i, (s, a, r, s2, d) in enumerate(real_data):
            self.real_s[i]=s; self.real_a[i]=a; self.real_r[i]=r; self.real_s2[i]=s2; self.real_d[i]=d
        self.n_real = n; self.sim_cap = sim_cap; self.sim_ptr = self.sim_size = 0
        self.sim_s = np.zeros((sim_cap, od), np.float32); self.sim_a = np.zeros((sim_cap, ad), np.float32)
        self.sim_r = np.zeros((sim_cap, 1), np.float32); self.sim_s2 = np.zeros((sim_cap, od), np.float32)
        self.sim_d = np.zeros((sim_cap, 1), np.float32)
    @property
    def size(self): return self.n_real + self.sim_size
    def add(self, s, a, r, s2, d):
        i = self.sim_ptr
        self.sim_s[i]=s; self.sim_a[i]=a; self.sim_r[i]=r; self.sim_s2[i]=s2; self.sim_d[i]=d
        self.sim_ptr = (i+1) % self.sim_cap; self.sim_size = min(self.sim_size+1, self.sim_cap)
    def sample(self, n):
        n_r = int(n * self.real_ratio); n_s = n - n_r
        r_idx = np.random.randint(0, self.n_real, n_r)
        if self.sim_size > 0:
            s_idx = np.random.randint(0, self.sim_size, n_s)
            return tuple(torch.FloatTensor(np.concatenate([ra[r_idx], sa[s_idx]])).to(DEVICE)
                        for ra, sa in zip(
                            [self.real_s, self.real_a, self.real_r, self.real_s2, self.real_d],
                            [self.sim_s, self.sim_a, self.sim_r, self.sim_s2, self.sim_d]))
        r_idx = np.random.randint(0, self.n_real, n)
        return tuple(torch.FloatTensor(x[r_idx]).to(DEVICE)
                     for x in [self.real_s, self.real_a, self.real_r, self.real_s2, self.real_d])


def collect_paired(env_cls, n_steps, seed=SEED):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list, real_trans = [], [], []
    os_, _ = sim.reset(seed=seed); or_, _ = real.reset(seed=seed); ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns, _, ds, ts, _ = sim.step(a); nr, rr, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_, a])); ds_list.append(nr - ns)
        real_trans.append((or_.copy(), a.copy(), rr, nr.copy(), float(dr and not tr)))
        os_, or_ = ns, nr
        if ds or ts or dr or tr:
            ep += 1; os_, _ = sim.reset(seed=ep+seed); or_, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    return np.array(SA_list, np.float32), np.array(ds_list, np.float32), real_trans, ep


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


def make_gap_fn(corrector):
    """
    Create gap detection function from SINDy ensemble.

    gap(s, a) = ||Δ̂(s,a)||² — magnitude of predicted dynamics gap.
    SINDy doesn't need to be accurate for correction,
    just needs to detect WHERE the gap is large vs small.
    """
    def gap_fn(s_batch, a_batch):
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        delta_pred = corrector.predict_batch(SA)  # (N, obs_dim)
        # Gap signal: squared norm of predicted dynamics difference
        gap = np.mean(delta_pred ** 2, axis=-1)  # (N,)
        return gap
    return gap_fn


def train(env_fn, env_cls, label, gap_fn=None, penalty_scale=0.1,
          real_data=None, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_fn()
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])

    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE,
                       gap_fn=gap_fn, penalty_scale=penalty_scale)

    if real_data is not None:
        buf = MixedReplayBuffer(od, ad, real_data, REPLAY_SIZE, real_ratio=0.5)
        log(f"  MixedBuffer: {len(real_data)} real (frozen 50%)")
    else:
        buf = ReplayBuffer(od, ad, REPLAY_SIZE)

    obs, _ = env.reset(seed=seed); curve = []
    log(f"\n[{label}] {TRAIN_STEPS//1000}k steps" +
        (f" | QΔ penalty={penalty_scale}" if gap_fn else ""))

    # Diagnostic accumulators
    diag_gap_rewards = []
    diag_q_penalties = []
    diag_q_targets = []

    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

            # Collect QΔ diagnostics every step (lightweight)
            if agent.q_delta is not None and step % 100 == 0:
                with torch.no_grad():
                    s_sample, a_sample, _, _, _ = buf.sample(64)
                    # Gap reward (SINDy signal)
                    s_np = s_sample.cpu().numpy(); a_np = a_sample.cpu().numpy()
                    gap_r = gap_fn(s_np, a_np) if gap_fn else np.zeros(64)
                    diag_gap_rewards.append(float(gap_r.mean()))
                    # QΔ prediction
                    qd = agent.q_delta.q_delta(s_sample, a_sample).squeeze(-1)
                    diag_q_penalties.append(float(qd.mean()))

        if step % EVAL_INTERVAL == 0:
            ret, std = evaluate(agent, env_cls)
            curve.append((step, ret))
            # Step-level diagnostic summary
            diag_str = ""
            if diag_gap_rewards:
                gr_mean = np.mean(diag_gap_rewards[-50:])  # last 50 samples
                qp_mean = np.mean(diag_q_penalties[-50:]) if diag_q_penalties else 0
                penalty_applied = qp_mean * penalty_scale
                diag_str = (f"  gap_r={gr_mean:.3f} QΔ={qp_mean:.3f} "
                           f"penalty={penalty_applied:.3f}")
                diag_gap_rewards.clear()
                diag_q_penalties.clear()
            log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}{diag_str}")

    env.close()
    return curve


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2", "c3"])
    args = parser.parse_args()
    mode = args.mode

    global LOG
    LOG = f"/tmp/mvp_qdelta_{mode}.log"

    def log_local(msg=""):
        print(msg, flush=True)
        with open(LOG, "a") as f:
            f.write(str(msg) + "\n")

    # Monkey-patch module-level log
    global log
    log = log_local

    log(f"[{mode}] Device: {DEVICE}")
    log("Architecture: SINDy as OOD detector → QΔ penalty on Q-target")

    env_cls = CarpetAntEnv

    # Paired data + SINDy ensemble
    log("\n[A] SINDy Ensemble (gap detector)")
    SA, delta_s, real_trans, n_eps = collect_paired(env_cls, N_COLLECT)
    log(f"  {len(SA)} paired steps ({n_eps} eps)")
    obs_dim = delta_s.shape[1]

    corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.1)
    corrector.fit(SA, delta_s)
    gap_fn = make_gap_fn(corrector)

    test_gap = gap_fn(SA[:100, :obs_dim], SA[:100, obs_dim:])
    log(f"  Gap signal: mean={test_gap.mean():.4f} std={test_gap.std():.4f}")

    RAW_SIM_BASELINE = 951.6

    if mode == "c1":
        curve = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim (baseline)")
    elif mode == "c2":
        curve = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim + QΔ",
                      gap_fn=gap_fn, penalty_scale=PENALTY_SCALE)
    elif mode == "c3":
        curve = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim + QΔ + MixBuf",
                      gap_fn=gap_fn, penalty_scale=PENALTY_SCALE, real_data=real_trans)

    avg = np.mean([r for _, r in curve[-3:]])
    gap = (avg - RAW_SIM_BASELINE) / RAW_SIM_BASELINE * 100
    log(f"\n{'='*50}")
    log(f"Baseline (Raw Sim v1): {RAW_SIM_BASELINE:.1f}")
    log(f"{mode} result:          {avg:.1f}")
    log(f"Gap: {gap:+.1f}%")
    if gap > 10: log(">>> SOLID")
    elif gap > 0: log(">>> POSITIVE")
    else: log(">>> NEGATIVE")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
