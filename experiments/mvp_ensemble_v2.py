"""
MVP Ensemble v2 — 3 fixes in parallel:
  v2a: EnsGated only (no MixedBuffer) — isolate correction value
  v2b: EnsGated + trained-policy real data (better data quality)
  v2c: EnsGated + MixedBuffer + gate_tau=0.05 (more correction)

Baseline from v1: Raw Sim = 951.6

运行: conda run -n MC-WM python3 -u experiments/mvp_ensemble_v2.py --mode v2a
"""
import sys, os, warnings, argparse
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

def make_logger(mode):
    path = f"/tmp/mvp_ens_{mode}.log"
    def log(msg=""):
        print(msg, flush=True)
        with open(path, "a") as f: f.write(str(msg) + "\n")
    return log, path

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
RAW_SIM_BASELINE = 951.6


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
        self.ptr = (i+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in [self.s, self.a, self.r, self.s2, self.d])


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
    def add(self, s, a, r, s2, d, **kw):
        i = self.sim_ptr
        self.sim_s[i]=s; self.sim_a[i]=a; self.sim_r[i]=r; self.sim_s2[i]=s2; self.sim_d[i]=d
        self.sim_ptr = (i+1) % self.sim_cap; self.sim_size = min(self.sim_size+1, self.sim_cap)
    def sample(self, n):
        n_r = int(n * self.real_ratio); n_s = n - n_r
        r_idx = np.random.randint(0, self.n_real, n_r)
        if self.sim_size > 0:
            s_idx = np.random.randint(0, self.sim_size, n_s)
        else:
            r_idx = np.random.randint(0, self.n_real, n); n_s = 0
        arrays = [self.real_s, self.real_a, self.real_r, self.real_s2, self.real_d]
        if n_s > 0:
            arrays_s = [self.sim_s, self.sim_a, self.sim_r, self.sim_s2, self.sim_d]
            return tuple(torch.FloatTensor(np.concatenate([a[r_idx], b[s_idx]])).to(DEVICE) for a, b in zip(arrays, arrays_s))
        return tuple(torch.FloatTensor(a[r_idx]).to(DEVICE) for a in arrays)


class EnsembleGatedEnv:
    def __init__(self, env_cls, corrector):
        self._env = env_cls(mode="sim"); self.corrector = corrector
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._last_obs = None; self._gate_history = []; self._g_smooth = None
    def reset(self, **kw):
        obs, info = self._env.reset(**kw); self._last_obs = obs.copy(); self._g_smooth = None; return obs, info
    def step(self, action):
        obs, reward, term, trunc, info = self._env.step(action)
        if self._last_obs is not None:
            delta, stds, g_raw = self.corrector.predict_with_uncertainty(self._last_obs, action)
            if self._g_smooth is None: self._g_smooth = g_raw
            else: self._g_smooth = 0.9 * self._g_smooth + 0.1 * g_raw
            g = self._g_smooth; self._gate_history.append(g)
            if g > 0.001: obs = obs + g * delta
        self._last_obs = obs.copy(); return obs, reward, term, trunc, info
    def gate_stats(self):
        if not self._gate_history: return {}
        h = np.array(self._gate_history)
        return {"mean": float(h.mean()), "median": float(np.median(h)), "frac_zero": float((h < 0.01).mean())}
    def close(self): self._env.close()


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


def collect_trained_real(env_cls, agent, n_steps, seed=SEED):
    """Collect real transitions with a partially-trained policy (better quality)."""
    real = env_cls(mode="real"); trans = []
    obs, _ = real.reset(seed=seed); ep = 0
    for _ in range(n_steps):
        a = agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = real.step(a)
        trans.append((obs.copy(), a.copy(), r, obs2.copy(), float(d and not tr)))
        obs = obs2
        if d or tr: ep += 1; obs, _ = real.reset(seed=seed+ep)
    real.close()
    return trans, ep


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


def train(env_fn, env_cls, label, log_fn, real_data=None, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_fn(); od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])
    agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS, beta=BETA_LCB, lr=LR, device=DEVICE)
    if real_data is not None:
        buf = MixedReplayBuffer(od, ad, real_data, REPLAY_SIZE, real_ratio=0.5)
        log_fn(f"  MixedBuffer: {len(real_data)} real (frozen 50%)")
    else:
        buf = ReplayBuffer(od, ad, REPLAY_SIZE)
    obs, _ = env.reset(seed=seed); curve = []
    log_fn(f"\n[{label}] {TRAIN_STEPS//1000}k steps")
    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a); buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE: agent.update(buf)
        if step % EVAL_INTERVAL == 0:
            ret, std = evaluate(agent, env_cls); curve.append((step, ret))
            gi = ""
            if hasattr(env, 'gate_stats'):
                gs = env.gate_stats()
                if gs: gi = f"  g_med={gs['median']:.3f} fz={gs['frac_zero']:.2f}"
            log_fn(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}{gi}")
    env.close()
    return curve, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["v2a", "v2b", "v2c"])
    args = parser.parse_args()
    mode = args.mode

    log, log_path = make_logger(mode)
    log(f"[{mode}] Device: {DEVICE}")
    env_cls = CarpetAntEnv

    # Paired data + ensemble
    log("\n[A] Paired data + SINDy Ensemble")
    SA, delta_s, real_trans_random, n_eps = collect_paired(env_cls, N_COLLECT)
    log(f"  Collected {len(SA)} paired steps ({n_eps} eps)")
    obs_dim = delta_s.shape[1]

    if mode == "v2a":
        # EnsGated only, no MixedBuffer
        gate_tau = 0.01
        corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=gate_tau)
        corrector.fit(SA, delta_s)
        log(f"  gate_tau={gate_tau}")
        def make_env(): return EnsembleGatedEnv(env_cls, corrector)
        curve, _ = train(make_env, env_cls, "EnsGated (no MixBuf)", log)

    elif mode == "v2b":
        # First train a policy in raw sim for 20k, then collect real data with it
        gate_tau = 0.01
        corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=gate_tau)
        corrector.fit(SA, delta_s)
        log("\n[B] Pre-training policy in raw sim (20k steps)...")
        pretrain_curve, pretrained_agent = train(
            lambda: env_cls(mode="sim"), env_cls, "Pretrain", log, seed=SEED)
        log(f"  Pretrained return: {pretrain_curve[-1][1]:.1f}")

        log("\n[C] Collecting real data with pretrained policy...")
        trained_real, n_ep = collect_trained_real(env_cls, pretrained_agent, 3000)
        log(f"  Collected {len(trained_real)} trained-policy real transitions ({n_ep} eps)")

        def make_env(): return EnsembleGatedEnv(env_cls, corrector)
        curve, _ = train(make_env, env_cls, "EnsGated+TrainedReal", log,
                        real_data=trained_real)

    elif mode == "v2c":
        # Same as v1 but gate_tau=0.05
        gate_tau = 0.05
        corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=gate_tau)
        corrector.fit(SA, delta_s)
        log(f"  gate_tau={gate_tau}")
        def make_env(): return EnsembleGatedEnv(env_cls, corrector)
        curve, _ = train(make_env, env_cls, "EnsGated(tau=0.05)+MixBuf", log,
                        real_data=real_trans_random)

    # Result
    avg = np.mean([r for _, r in curve[-3:]])
    gap = (avg - RAW_SIM_BASELINE) / RAW_SIM_BASELINE * 100
    log(f"\n{'='*50}")
    log(f"Raw Sim baseline:  {RAW_SIM_BASELINE:.1f}")
    log(f"{mode} result:      {avg:.1f}")
    log(f"Gap: {gap:+.1f}%")
    if gap > 10: log(">>> SOLID")
    elif gap > 0: log(">>> POSITIVE")
    else: log(">>> NEGATIVE")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
