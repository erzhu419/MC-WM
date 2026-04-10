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
            log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}")

    env.close()
    return curve


def main():
    log(f"Device: {DEVICE}")
    log("Architecture: SINDy as OOD detector → QΔ penalty on Q-target")

    env_cls = CarpetAntEnv

    # Paired data + SINDy ensemble (for gap detection, NOT correction)
    log("\n[A] SINDy Ensemble (gap detector, not corrector)")
    SA, delta_s, real_trans, n_eps = collect_paired(env_cls, N_COLLECT)
    log(f"  {len(SA)} paired steps ({n_eps} eps)")
    obs_dim = delta_s.shape[1]

    corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.1)
    corrector.fit(SA, delta_s)
    gap_fn = make_gap_fn(corrector)

    # Verify gap signal: should be high for vel dims, ~0 for non-vel dims
    test_gap = gap_fn(SA[:100, :obs_dim], SA[:100, obs_dim:])
    log(f"  Gap signal on training data: mean={test_gap.mean():.4f} std={test_gap.std():.4f}")

    # Condition 1: Raw Sim (no QΔ, baseline)
    c1 = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim (baseline)")

    # Condition 2: Raw Sim + QΔ (SINDy gap detection)
    c2 = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim + QΔ",
               gap_fn=gap_fn, penalty_scale=PENALTY_SCALE)

    # Condition 3: Raw Sim + QΔ + MixedBuffer
    c3 = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim + QΔ + MixBuf",
               gap_fn=gap_fn, penalty_scale=PENALTY_SCALE, real_data=real_trans)

    # Summary
    log(f"\n{'='*60}")
    log("FINAL RESULTS (last 3 avg, real env)")
    log(f"{'='*60}")
    results = {}
    for name, c in [("Raw Sim", c1), ("RawSim+QΔ", c2), ("RawSim+QΔ+MixBuf", c3)]:
        avg = np.mean([r for _, r in c[-3:]])
        results[name] = avg
        log(f"  {name:25s}: {avg:7.1f}")

    r1, r2, r3 = results["Raw Sim"], results["RawSim+QΔ"], results["RawSim+QΔ+MixBuf"]
    log(f"\n  QΔ vs Raw Sim:           {(r2-r1)/max(abs(r1),1)*100:+.1f}%")
    log(f"  QΔ+MixBuf vs Raw Sim:    {(r3-r1)/max(abs(r1),1)*100:+.1f}%")

    if (r2 - r1) / max(abs(r1), 1) * 100 > 10:
        log(">>> SOLID — QΔ works!")
    elif (r2 - r1) / max(abs(r1), 1) * 100 > 0:
        log(">>> POSITIVE — QΔ adds marginal value")
    else:
        log(">>> NEGATIVE — need to tune penalty_scale or gap signal")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, c, color in [("Raw Sim", c1, "steelblue"),
                            ("RawSim+QΔ", c2, "darkorange"),
                            ("RawSim+QΔ+MixBuf", c3, "forestgreen")]:
        steps, rets = zip(*c)
        ax.plot(steps, rets, label=name, color=color, lw=2)
    ax.set_xlabel("Steps"); ax.set_ylabel("Real env return")
    ax.set_title("CarpetAnt: QΔ (Residual Bellman) MVP")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mvp_qdelta.png"), dpi=100); plt.close()
    log(f"Plot → mvp_qdelta.png")


if __name__ == "__main__":
    main()
