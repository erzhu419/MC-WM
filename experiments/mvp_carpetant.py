"""
MVP on CarpetAnt — the env with the LARGEST sim-real gap (30% vel damping).
Tests: Raw Sim vs Gated+MixedBuffer vs RawSim+MixedBuffer (no correction).

Key: Cal-QL floor prevents sim Q-targets from collapsing.

运行: conda run -n MC-WM python3 -u experiments/mvp_carpetant.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from experiments.mvp_aero_cheetah import SINDyStateCorrector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/mvp_carpetant.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

# ── Config
N_COLLECT    = 3_000
TRAIN_STEPS  = 50_000
EVAL_INTERVAL = 5_000
N_EVAL_EPS   = 10
WARMUP       = 2_000
BATCH_SIZE   = 256
REPLAY_SIZE  = 100_000
N_CRITICS    = 3
BETA_LCB     = -2.0
HIDDEN       = 256   # bigger net for 27-dim obs + 8-dim act
LR           = 3e-4
GATE_TAU     = 0.5
GATE_EPS_JAC = 0.01
REAL_RATIO   = 0.5
OUT_DIR      = os.path.dirname(__file__)


# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.max_size = capacity; self.ptr = self.size = 0
        self.s  = np.zeros((capacity, obs_dim), np.float32)
        self.a  = np.zeros((capacity, act_dim), np.float32)
        self.r  = np.zeros((capacity, 1),       np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d  = np.zeros((capacity, 1),       np.float32)
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
    def __init__(self, obs_dim, act_dim, real_data, sim_capacity, real_ratio=0.5):
        self.real_ratio = real_ratio
        n_real = len(real_data)
        self.real_s  = np.zeros((n_real, obs_dim), np.float32)
        self.real_a  = np.zeros((n_real, act_dim), np.float32)
        self.real_r  = np.zeros((n_real, 1),       np.float32)
        self.real_s2 = np.zeros((n_real, obs_dim), np.float32)
        self.real_d  = np.zeros((n_real, 1),       np.float32)
        for i, (s, a, r, s2, d) in enumerate(real_data):
            self.real_s[i]=s; self.real_a[i]=a; self.real_r[i]=r
            self.real_s2[i]=s2; self.real_d[i]=d
        self.n_real = n_real
        self.sim_cap = sim_capacity; self.sim_ptr = self.sim_size = 0
        self.sim_s  = np.zeros((sim_capacity, obs_dim), np.float32)
        self.sim_a  = np.zeros((sim_capacity, act_dim), np.float32)
        self.sim_r  = np.zeros((sim_capacity, 1),       np.float32)
        self.sim_s2 = np.zeros((sim_capacity, obs_dim), np.float32)
        self.sim_d  = np.zeros((sim_capacity, 1),       np.float32)
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


# ─────────────────────────────────────────────
class GatedCorrectedEnv:
    def __init__(self, env_cls, corrector, tau, eps_jac):
        self._env = env_cls(mode="sim")
        self.corrector = corrector
        self.tau = tau; self.eps_jac = eps_jac
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._train_center = None; self._L_eff = 0.05
        self._gate_history = []; self._last_obs = None; self._g_smooth = None
    def set_train_center(self, SA):
        self._train_center = SA.mean(axis=0)
    def _gate(self, s, a):
        if self._train_center is None: return 1.0
        SA = np.concatenate([s, a])
        dist = float(np.linalg.norm(SA - self._train_center))
        eps_fit = float(self.corrector.fit_errors.mean())
        bound = eps_fit + self.eps_jac * dist + (self._L_eff/2) * dist**2
        g_raw = max(0.0, 1.0 - bound / self.tau)
        if self._g_smooth is None: self._g_smooth = g_raw
        else: self._g_smooth = 0.9 * self._g_smooth + 0.1 * g_raw
        return self._g_smooth
    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        self._last_obs = obs.copy(); self._g_smooth = None
        return obs, info
    def step(self, action):
        obs, reward, term, trunc, info = self._env.step(action)
        if self._last_obs is not None:
            g = self._gate(self._last_obs, action)
            self._gate_history.append(g)
            if g > 0:
                delta = self.corrector.predict(self._last_obs, action)
                obs = obs + g * delta
        self._last_obs = obs.copy()
        return obs, reward, term, trunc, info
    def gate_stats(self):
        if not self._gate_history: return {}
        h = np.array(self._gate_history)
        return {"mean": float(h.mean()), "median": float(np.median(h)),
                "frac_zero": float((h < 0.01).mean())}
    def close(self): self._env.close()


# ─────────────────────────────────────────────
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
    SA = np.array(SA_list, dtype=np.float32)
    delta_s = np.array(ds_list, dtype=np.float32)
    log(f"  Collected {len(SA)} paired steps ({ep} eps)")
    return SA, delta_s, real_trans


def evaluate(agent, env_cls, mode, n_eps=N_EVAL_EPS):
    env = env_cls(mode=mode)
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
            ret_sim, std_sim = evaluate(agent, env_cls, "sim")
            ret_real, std_real = evaluate(agent, env_cls, "real")
            curve.append((step, ret_sim, ret_real))
            gap = (ret_sim - ret_real) / max(abs(ret_real), 1) * 100
            gate_info = ""
            if hasattr(env, 'gate_stats'):
                gs = env.gate_stats()
                if gs: gate_info = f"  g_med={gs['median']:.3f} fz={gs['frac_zero']:.2f}"
            log(f"  step {step:>6d} | sim={ret_sim:7.1f}±{std_sim:4.0f}  real={ret_real:7.1f}±{std_real:4.0f}  gap={gap:+5.1f}%{gate_info}")

    env.close()
    return curve


def main():
    log(f"Device: {DEVICE}")
    log(f"Env: CarpetAnt (obs=27, act=8, vel_damp=0.7, motor_penalty=0.01)")

    env_cls = CarpetAntEnv

    # ── Paired data + SINDy
    log("\n[A] Paired data + SINDy")
    SA, delta_s, real_trans = collect_paired(env_cls, N_COLLECT)
    obs_dim = delta_s.shape[1]
    corrector = SINDyStateCorrector(obs_dim)
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    log(f"  RMSE reduction: {cov['rmse_reduction_pct']:.1f}%")

    # ── Condition 1: Raw Sim (baseline)
    c1 = train(lambda: env_cls(mode="sim"), env_cls, "Raw Sim")

    # ── Condition 2: Raw Sim + MixedBuffer (real data, no correction)
    c2 = train(lambda: env_cls(mode="sim"), env_cls, "RawSim+RealData",
               real_data=real_trans)

    # ── Condition 3: Gated Corrected + MixedBuffer
    def make_gated():
        env = GatedCorrectedEnv(env_cls, corrector, GATE_TAU, GATE_EPS_JAC)
        env.set_train_center(SA)
        return env
    c3 = train(make_gated, env_cls, "Gated+MixedBuf", real_data=real_trans)

    # ── Summary
    log(f"\n{'='*60}")
    log("FINAL RESULTS (last 3 avg, eval in REAL env)")
    log(f"{'='*60}")
    for name, c in [("Raw Sim", c1), ("RawSim+RealData", c2), ("Gated+MixedBuf", c3)]:
        avg_real = np.mean([r for _, _, r in c[-3:]])
        avg_sim = np.mean([s for _, s, _ in c[-3:]])
        log(f"  {name:20s}: sim={avg_sim:7.1f}  real={avg_real:7.1f}")

    r1 = np.mean([r for _, _, r in c1[-3:]])
    r2 = np.mean([r for _, _, r in c2[-3:]])
    r3 = np.mean([r for _, _, r in c3[-3:]])
    log(f"\n  RealData value (no correction): {(r2-r1)/max(abs(r1),1)*100:+.1f}%")
    log(f"  Gated+MixedBuf vs Raw Sim:      {(r3-r1)/max(abs(r1),1)*100:+.1f}%")
    log(f"  Gated+MixedBuf vs RealData only: {(r3-r2)/max(abs(r2),1)*100:+.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, c, color in [("Raw Sim", c1, "steelblue"),
                            ("RawSim+RealData", c2, "forestgreen"),
                            ("Gated+MixedBuf", c3, "darkorange")]:
        steps = [x[0] for x in c]; reals = [x[2] for x in c]
        ax.plot(steps, reals, label=name, color=color, lw=2)
    ax.set_xlabel("Steps"); ax.set_ylabel("Real env return")
    ax.set_title("CarpetAnt: MC-WM MVP (30% vel damping gap)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(OUT_DIR, "mvp_carpetant.png")
    plt.savefig(out, dpi=100); plt.close()
    log(f"Plot → {out}")


if __name__ == "__main__":
    main()
