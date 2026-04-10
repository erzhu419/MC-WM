"""
MVP with SINDy Ensemble gate — CarpetAnt.

Based on validation results:
- Single SINDy: +36% on random, -75% on trained (catastrophic OOD)
- Ensemble gate (tau=0.01): +7% on random, +1.5% on trained (safe)

This test: does ensemble-gated correction + MixedBuffer beat Raw Sim
in the actual RL training loop?

3 conditions:
1. Raw Sim (baseline)
2. Ensemble-Gated Corrected + MixedBuffer
3. Raw Sim + MixedBuffer (no correction, isolate data value)

运行: conda run -n MC-WM python3 -u experiments/mvp_ensemble.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/mvp_ensemble.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

# Config
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
REAL_RATIO   = 0.5
GATE_TAU     = 0.01   # tight gate — validated as safe
OUT_DIR      = os.path.dirname(__file__)


class ReplayBuffer:
    def __init__(self, od, ad, cap):
        self.max_size = cap; self.ptr = self.size = 0
        self.s  = np.zeros((cap, od), np.float32)
        self.a  = np.zeros((cap, ad), np.float32)
        self.r  = np.zeros((cap, 1),  np.float32)
        self.s2 = np.zeros((cap, od), np.float32)
        self.d  = np.zeros((cap, 1),  np.float32)
    def add(self, s, a, r, s2, d, **kw):
        i = self.ptr
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (i+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[idx]).to(DEVICE),
                torch.FloatTensor(self.a[idx]).to(DEVICE),
                torch.FloatTensor(self.r[idx]).to(DEVICE),
                torch.FloatTensor(self.s2[idx]).to(DEVICE),
                torch.FloatTensor(self.d[idx]).to(DEVICE))


class MixedReplayBuffer:
    def __init__(self, od, ad, real_data, sim_cap, real_ratio=0.5):
        self.real_ratio = real_ratio
        n = len(real_data)
        self.real_s  = np.zeros((n, od), np.float32)
        self.real_a  = np.zeros((n, ad), np.float32)
        self.real_r  = np.zeros((n, 1),  np.float32)
        self.real_s2 = np.zeros((n, od), np.float32)
        self.real_d  = np.zeros((n, 1),  np.float32)
        for i, (s, a, r, s2, d) in enumerate(real_data):
            self.real_s[i]=s; self.real_a[i]=a; self.real_r[i]=r
            self.real_s2[i]=s2; self.real_d[i]=d
        self.n_real = n
        self.sim_cap = sim_cap; self.sim_ptr = self.sim_size = 0
        self.sim_s  = np.zeros((sim_cap, od), np.float32)
        self.sim_a  = np.zeros((sim_cap, ad), np.float32)
        self.sim_r  = np.zeros((sim_cap, 1),  np.float32)
        self.sim_s2 = np.zeros((sim_cap, od), np.float32)
        self.sim_d  = np.zeros((sim_cap, 1),  np.float32)
    @property
    def size(self): return self.n_real + self.sim_size
    def add(self, s, a, r, s2, d, **kw):
        i = self.sim_ptr
        self.sim_s[i]=s; self.sim_a[i]=a; self.sim_r[i]=r
        self.sim_s2[i]=s2; self.sim_d[i]=d
        self.sim_ptr = (i+1) % self.sim_cap
        self.sim_size = min(self.sim_size+1, self.sim_cap)
    def sample(self, n):
        n_r = int(n * self.real_ratio); n_s = n - n_r
        r_idx = np.random.randint(0, self.n_real, n_r)
        if self.sim_size > 0:
            s_idx = np.random.randint(0, self.sim_size, n_s)
        else:
            r_idx = np.random.randint(0, self.n_real, n)
            return (torch.FloatTensor(self.real_s[r_idx]).to(DEVICE),
                    torch.FloatTensor(self.real_a[r_idx]).to(DEVICE),
                    torch.FloatTensor(self.real_r[r_idx]).to(DEVICE),
                    torch.FloatTensor(self.real_s2[r_idx]).to(DEVICE),
                    torch.FloatTensor(self.real_d[r_idx]).to(DEVICE))
        s  = np.concatenate([self.real_s[r_idx],  self.sim_s[s_idx]])
        a  = np.concatenate([self.real_a[r_idx],  self.sim_a[s_idx]])
        r  = np.concatenate([self.real_r[r_idx],  self.sim_r[s_idx]])
        s2 = np.concatenate([self.real_s2[r_idx], self.sim_s2[s_idx]])
        d  = np.concatenate([self.real_d[r_idx],  self.sim_d[s_idx]])
        return (torch.FloatTensor(s).to(DEVICE), torch.FloatTensor(a).to(DEVICE),
                torch.FloatTensor(r).to(DEVICE), torch.FloatTensor(s2).to(DEVICE),
                torch.FloatTensor(d).to(DEVICE))


class EnsembleGatedEnv:
    """Sim env with ensemble-gated SINDy correction."""
    def __init__(self, env_cls, corrector):
        self._env = env_cls(mode="sim")
        self.corrector = corrector
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._last_obs = None
        self._gate_history = []
        self._g_smooth = None

    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        self._last_obs = obs.copy()
        self._g_smooth = None
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self._env.step(action)
        if self._last_obs is not None:
            delta, stds, g_raw = self.corrector.predict_with_uncertainty(
                self._last_obs, action)
            # Temporal smoothing
            if self._g_smooth is None:
                self._g_smooth = g_raw
            else:
                self._g_smooth = 0.9 * self._g_smooth + 0.1 * g_raw
            g = self._g_smooth
            self._gate_history.append(g)
            if g > 0.001:
                obs = obs + g * delta
        self._last_obs = obs.copy()
        return obs, reward, term, trunc, info

    def gate_stats(self):
        if not self._gate_history: return {}
        h = np.array(self._gate_history)
        return {"mean": float(h.mean()), "median": float(np.median(h)),
                "frac_zero": float((h < 0.01).mean())}

    def close(self): self._env.close()


def collect_paired(env_cls, n_steps, seed=SEED):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list, real_trans = [], [], []
    os_, _ = sim.reset(seed=seed); or_, _ = real.reset(seed=seed)
    ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns, _, ds, ts, _ = sim.step(a)
        nr, rr, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_, a]))
        ds_list.append(nr - ns)
        real_trans.append((or_.copy(), a.copy(), rr, nr.copy(), float(dr and not tr)))
        os_, or_ = ns, nr
        if ds or ts or dr or tr:
            ep += 1; os_, _ = sim.reset(seed=ep+seed); or_, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    return (np.array(SA_list, np.float32), np.array(ds_list, np.float32),
            real_trans, ep)


def evaluate(agent, env_cls, n_eps=N_EVAL_EPS):
    env = env_cls(mode="real")
    rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep+200); total = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = env.step(a); total += r
            if d or tr: break
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def train(env_fn, env_cls, label, real_data=None, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_fn()
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])
    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE)

    if real_data is not None:
        buf = MixedReplayBuffer(od, ad, real_data, REPLAY_SIZE, real_ratio=REAL_RATIO)
        log(f"  MixedBuffer: {len(real_data)} real (frozen {REAL_RATIO*100:.0f}%)")
    else:
        buf = ReplayBuffer(od, ad, REPLAY_SIZE)

    obs, _ = env.reset(seed=seed)
    curve = []
    log(f"\n[{label}] {TRAIN_STEPS//1000}k steps")

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
            gate_info = ""
            if hasattr(env, 'gate_stats'):
                gs = env.gate_stats()
                if gs: gate_info = f"  g_med={gs['median']:.3f} fz={gs['frac_zero']:.2f}"
            log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}{gate_info}")

    env.close()
    return curve


def main():
    log(f"Device: {DEVICE}")
    log(f"Env: CarpetAnt | Ensemble K=5 | gate_tau={GATE_TAU}")

    env_cls = CarpetAntEnv

    # Paired data + ensemble fit
    log("\n[A] Paired data + SINDy Ensemble")
    SA, delta_s, real_trans, n_eps = collect_paired(env_cls, N_COLLECT)
    log(f"  Collected {len(SA)} paired steps ({n_eps} eps)")
    obs_dim = delta_s.shape[1]

    corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=GATE_TAU)
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    log(f"  RMSE reduction: {cov['rmse_reduction_pct']:.1f}%")

    # Condition 1: Raw Sim
    c1 = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim")

    # Condition 2: Ensemble-Gated + MixedBuffer
    def make_gated():
        return EnsembleGatedEnv(env_cls, corrector)
    c2 = train(make_gated, env_cls, "EnsGated+MixBuf", real_data=real_trans)

    # Condition 3: Raw Sim + MixedBuffer (no correction)
    c3 = train(lambda: env_cls(mode="sim"), env_cls, "RawSim+MixBuf",
               real_data=real_trans)

    # Summary
    log(f"\n{'='*60}")
    log("FINAL RESULTS (last 3 avg, real env)")
    log(f"{'='*60}")
    results = {}
    for name, c in [("Raw Sim", c1), ("EnsGated+MixBuf", c2), ("RawSim+MixBuf", c3)]:
        avg = np.mean([r for _, r in c[-3:]])
        results[name] = avg
        log(f"  {name:20s}: {avg:7.1f}")

    r1 = results["Raw Sim"]
    r2 = results["EnsGated+MixBuf"]
    r3 = results["RawSim+MixBuf"]
    log(f"\n  EnsGated+MixBuf vs Raw Sim: {(r2-r1)/max(abs(r1),1)*100:+.1f}%")
    log(f"  RawSim+MixBuf vs Raw Sim:   {(r3-r1)/max(abs(r1),1)*100:+.1f}%")
    log(f"  Correction increment:        {(r2-r3)/max(abs(r3),1)*100:+.1f}%")

    if (r2 - r1) / max(abs(r1), 1) * 100 > 10:
        log(">>> SOLID — ensemble correction works")
    elif (r2 - r1) / max(abs(r1), 1) * 100 > 0:
        log(">>> POSITIVE — correction adds marginal value")
    else:
        log(">>> NEGATIVE — correction still hurts even with ensemble gate")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, c, color in [("Raw Sim", c1, "steelblue"),
                            ("EnsGated+MixBuf", c2, "darkorange"),
                            ("RawSim+MixBuf", c3, "forestgreen")]:
        steps = [x[0] for x in c]; rets = [x[1] for x in c]
        ax.plot(steps, rets, label=name, color=color, lw=2)
    ax.set_xlabel("Steps"); ax.set_ylabel("Real env return")
    ax.set_title(f"CarpetAnt: Ensemble SINDy MVP (K=5, tau={GATE_TAU})")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(OUT_DIR, "mvp_ensemble_carpetant.png")
    plt.savefig(out, dpi=100); plt.close()
    log(f"Plot → {out}")


if __name__ == "__main__":
    main()
