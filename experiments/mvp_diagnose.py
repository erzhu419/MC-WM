"""
诊断实验：
A) 量化 WindHopper sim-to-real gap（同一 policy 在 sim vs real 的 performance 差）
B) Raw Sim + MixedReplayBuffer（不加 correction）vs Raw Sim alone
C) 用 20 eps 评估消除方差
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.wind_hopper import WindHopperEnv
from mc_wm.policy.resac_agent import RESACAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/mvp_diagnose.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

# ── Config
TRAIN_STEPS  = 50_000
EVAL_INTERVAL = 5_000
N_EVAL_EPS   = 20   # 20 eps for lower variance
WARMUP       = 2_000
BATCH_SIZE   = 256
REPLAY_SIZE  = 100_000
N_CRITICS    = 3
BETA_LCB     = -2.0
HIDDEN       = 128
LR           = 3e-4
REAL_RATIO   = 0.5


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.max_size = capacity; self.ptr = self.size = 0
        self.s  = np.zeros((capacity, obs_dim), np.float32)
        self.a  = np.zeros((capacity, act_dim), np.float32)
        self.r  = np.zeros((capacity, 1),       np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d  = np.zeros((capacity, 1),       np.float32)
    def add(self, s, a, r, s2, d):
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
    def add(self, s, a, r, s2, d):
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


def collect_real_transitions(env_cls, n_steps, seed=SEED):
    """Collect real env transitions with random policy."""
    real = env_cls(mode="real")
    transitions = []
    obs, _ = real.reset(seed=seed); ep = 0
    for _ in range(n_steps):
        a = real.action_space.sample()
        obs2, r, d, tr, _ = real.step(a)
        transitions.append((obs.copy(), a.copy(), r, obs2.copy(), float(d and not tr)))
        obs = obs2
        if d or tr:
            ep += 1; obs, _ = real.reset(seed=seed+ep)
    real.close()
    log(f"  Collected {len(transitions)} real transitions ({ep} eps)")
    return transitions


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
    return rets  # return full list for variance analysis


def train(env_fn, label, real_data=None, seed=SEED):
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
    curve_sim, curve_real = [], []

    log(f"\n[{label}] {TRAIN_STEPS//1000}k steps, eval={N_EVAL_EPS} eps")
    for step in range(1, TRAIN_STEPS+1):
        a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

        if step % EVAL_INTERVAL == 0:
            # Eval in BOTH sim and real
            rets_sim = evaluate(agent, WindHopperEnv, "sim")
            rets_real = evaluate(agent, WindHopperEnv, "real")
            m_sim, s_sim = np.mean(rets_sim), np.std(rets_sim)
            m_real, s_real = np.mean(rets_real), np.std(rets_real)
            curve_sim.append((step, m_sim))
            curve_real.append((step, m_real))
            gap_pct = (m_sim - m_real) / max(abs(m_real), 1) * 100
            log(f"  step {step:>6d} | sim={m_sim:7.1f}±{s_sim:5.1f}  real={m_real:7.1f}±{s_real:5.1f}  sim-real gap={gap_pct:+.1f}%")

    env.close()
    return curve_sim, curve_real


def main():
    log(f"Device: {DEVICE}")
    log(f"Eval episodes: {N_EVAL_EPS} (for lower variance)")

    # ── Collect real data for MixedBuffer
    log("\n[A] Collecting real transitions")
    real_data = collect_real_transitions(WindHopperEnv, 3000)

    # ── Experiment 1: Raw Sim alone (baseline, higher-confidence eval)
    log("\n" + "="*60)
    log("EXPERIMENT 1: Raw Sim (train in sim, eval in sim+real)")
    log("="*60)
    cs1, cr1 = train(lambda: WindHopperEnv(mode="sim"), "Raw Sim")

    # ── Experiment 2: Raw Sim + MixedReplayBuffer (no correction!)
    log("\n" + "="*60)
    log("EXPERIMENT 2: Raw Sim + 50% Real Data (no correction)")
    log("="*60)
    cs2, cr2 = train(lambda: WindHopperEnv(mode="sim"), "RawSim+RealData",
                     real_data=real_data)

    # ── Summary
    log("\n" + "="*60)
    log("FINAL COMPARISON (last 3 avg, 20-ep eval)")
    log("="*60)
    for name, cs, cr in [("Raw Sim", cs1, cr1), ("RawSim+RealData", cs2, cr2)]:
        avg_sim = np.mean([r for _, r in cs[-3:]])
        avg_real = np.mean([r for _, r in cr[-3:]])
        gap = (avg_sim - avg_real) / max(abs(avg_real), 1) * 100
        log(f"  {name:20s}: sim={avg_sim:7.1f}  real={avg_real:7.1f}  gap={gap:+.1f}%")

    real_1 = np.mean([r for _, r in cr1[-3:]])
    real_2 = np.mean([r for _, r in cr2[-3:]])
    improvement = (real_2 - real_1) / max(abs(real_1), 1) * 100
    log(f"\n  Real data value (no correction): {improvement:+.1f}%")
    if improvement > 5:
        log("  >>> Real data alone adds value — correction is the problem")
    elif improvement > -5:
        log("  >>> Real data neutral — sim-to-real gap may be small")
    else:
        log("  >>> Real data hurts — distribution mismatch in buffer")


if __name__ == "__main__":
    main()
