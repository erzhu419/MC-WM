"""
Step 3: Annealed MBPO — gradually shift from real env to model rollouts.

Schedule:
  Early:  every step in real, 1 model rollout per real step (n=1)
  Mid:    every n steps, 1 real step + n model rollouts
  Late:   mostly model rollouts, rare real corrections

n = real_interval, anneals from 1 → max_interval over training.

This solves:
- Q2 (reward mismatch): Q-learning on real transitions (early) → consistent
- Q1 (model accuracy): online refit from real transitions → model stays calibrated
- Q3 (policy-model coupling): gradual shift gives model time to catch up

Budget control: total real env steps = TRAIN_STEPS / avg(n) << TRAIN_STEPS

运行: conda run -n MC-WM python3 -u experiments/step3_annealed_mbpo.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.world_model import WorldModelEnsemble, ResidualAdapter, CorrectedWorldModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/step3_annealed.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

# Config
TRAIN_STEPS    = 50_000
EVAL_INTERVAL  = 5_000
N_EVAL_EPS     = 10
WARMUP         = 2_000
BATCH_SIZE     = 256
REPLAY_SIZE    = 100_000

# Model
N_SIM_PRETRAIN = 50_000
MODEL_REFIT_FREQ = 2000  # refit world model every N real steps

# Anneal schedule: n goes from 1 → max_interval
ANNEAL_START = 1       # initially every step is real
ANNEAL_END   = 10      # eventually 1 real per 10 model
ANNEAL_OVER  = 30_000  # anneal over this many training steps

# RE-SAC
N_CRITICS = 3; BETA_LCB = -2.0; HIDDEN = 256; LR = 3e-4


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
    def sample_states(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.s[idx]


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


def get_anneal_n(step, start=ANNEAL_START, end=ANNEAL_END, over=ANNEAL_OVER):
    """Linear anneal from start to end over 'over' steps."""
    frac = min(step / over, 1.0)
    return int(start + frac * (end - start))


def main():
    log(f"Device: {DEVICE}")
    log("Step 3: Annealed MBPO — real→model gradual shift")
    log(f"Anneal: n={ANNEAL_START}→{ANNEAL_END} over {ANNEAL_OVER} steps")

    env_cls = GravityCheetahEnv
    obs_dim = 17; act_dim = 6

    # Phase 1: Pretrain M_sim on sim data
    log(f"\n[Phase 1] Pretraining M_sim on {N_SIM_PRETRAIN//1000}k sim transitions...")
    sim = env_cls(mode="sim")
    s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
    obs, _ = sim.reset(seed=SEED); ep = 0
    for _ in range(N_SIM_PRETRAIN):
        a = sim.action_space.sample()
        obs2, r, d, tr, _ = sim.step(a)
        s_list.append(obs); a_list.append(a); r_list.append(r)
        s2_list.append(obs2); d_list.append(float(d and not tr))
        obs = obs2
        if d or tr: ep += 1; obs, _ = sim.reset(seed=ep+SEED)
    sim.close()
    s_arr = np.array(s_list, np.float32); a_arr = np.array(a_list, np.float32)
    r_arr = np.array(r_list, np.float32); s2_arr = np.array(s2_list, np.float32)

    wm = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
    wm.fit(s_arr, a_arr, s2_arr, r_arr, n_epochs=100, patience=20)
    ns_p, r_p = wm.predict(s_arr[:1000], a_arr[:1000], deterministic=True)
    log(f"  M_sim RMSE: {np.sqrt(np.mean((ns_p - s2_arr[:1000])**2)):.4f}")

    # Phase 2: Train with annealing schedule
    real_env = env_cls(mode="real")
    al = float(real_env.action_space.high[0])
    agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                       beta=BETA_LCB, lr=LR, device=DEVICE)

    real_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)  # real transitions
    model_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)  # model rollouts

    obs_real, _ = real_env.reset(seed=SEED)
    curve = []
    total_real_steps = 0
    total_model_steps = 0
    real_steps_since_refit = 0

    log(f"\n[Phase 2] Training {TRAIN_STEPS//1000}k steps with annealing")
    log(f"  {'step':>6s} | {'real':>7s} {'n':>3s} {'real_total':>10s} {'model_total':>11s} "
        f"{'m_s':>6s} {'r_budget%':>9s}")

    for step in range(1, TRAIN_STEPS + 1):
        n = get_anneal_n(step)

        # Decide: real step or model step?
        # Every n steps: 1 real + (n-1) model
        is_real_step = (step % n == 0) or (step <= WARMUP)

        if is_real_step:
            # Real env step
            a = real_env.action_space.sample() if step < WARMUP else agent.get_action(obs_real, deterministic=False)
            obs2, r, d, tr, _ = real_env.step(a)
            real_buf.add(obs_real, a, r, obs2, float(d and not tr))
            obs_real = obs2
            if d or tr: obs_real, _ = real_env.reset()
            total_real_steps += 1
            real_steps_since_refit += 1

            # Online model refit
            if real_steps_since_refit >= MODEL_REFIT_FREQ and real_buf.size >= 5000:
                n_fit = min(real_buf.size, 50000)
                idx = np.random.choice(real_buf.size, n_fit, replace=False) if real_buf.size > n_fit else np.arange(real_buf.size)
                wm.fit(real_buf.s[idx], real_buf.a[idx], real_buf.s2[idx],
                       real_buf.r[idx].squeeze(), n_epochs=20, patience=10)
                model_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)  # clear stale
                real_steps_since_refit = 0
                log(f"  [REFIT at step {step}] {n_fit} real transitions")
        else:
            # Model rollout step
            if real_buf.size >= 100:
                start = real_buf.sample_states(1)
                a_model = agent.get_action(start[0], deterministic=False)
                ns_pred, r_pred = wm.predict(start, a_model.reshape(1, -1), deterministic=False)
                model_buf.add(start[0], a_model, r_pred[0], ns_pred[0], 0.0)
                total_model_steps += 1

        # Train on whichever buffer has data
        if step >= WARMUP:
            if real_buf.size >= BATCH_SIZE:
                agent.update(real_buf)
            if model_buf.size >= BATCH_SIZE:
                agent.update(model_buf)

        # Eval
        if step % EVAL_INTERVAL == 0:
            ret, std = evaluate(agent, env_cls)
            curve.append((step, ret))
            real_pct = total_real_steps / step * 100
            # Model accuracy on recent real data
            m_s = 0.0
            if real_buf.size >= 500:
                idx = np.random.randint(max(0, real_buf.size-2000), real_buf.size, 500)
                ns_t, _ = wm.predict(real_buf.s[idx], real_buf.a[idx], deterministic=True)
                m_s = np.sqrt(np.mean((ns_t - real_buf.s2[idx]) ** 2))
            log(f"  {step:6d} | {ret:7.1f} {n:3d} {total_real_steps:10d} {total_model_steps:11d} "
                f"{m_s:6.3f} {real_pct:8.1f}%")

    real_env.close()

    # Summary
    avg = np.mean([r for _, r in curve[-3:]])
    log(f"\n{'='*60}")
    log(f"Last 3 avg: real={avg:.1f}")
    log(f"Total real steps: {total_real_steps} ({total_real_steps/TRAIN_STEPS*100:.1f}% of training)")
    log(f"Total model steps: {total_model_steps}")
    log(f"Real env budget: {total_real_steps} transitions")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
