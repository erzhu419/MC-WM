"""
Step 2: Model-Based RL with Residual World Model.

Pipeline (following ReDRAW):
  1. Collect sim transitions → train M_sim (world model ensemble)
  2. Collect paired (sim,real) transitions → train δ (residual adapter)
  3. M_real = M_sim + δ
  4. Policy trains on imagined rollouts from M_real (Dyna-style)
  5. Periodically: collect more paired data with current policy, refit δ

3 conditions:
  c1: Raw Sim env (baseline, no model)
  c2: M_sim imagined rollouts (model-based, no correction)
  c3: M_real imagined rollouts (model-based + residual correction)

运行: conda run -n MC-WM python3 -u experiments/step2_mbrl_residual.py --mode c1
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.world_model import WorldModelEnsemble, ResidualAdapter, CorrectedWorldModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data collection
N_SIM_PRETRAIN = 50_000   # sim transitions for M_sim
N_PAIRED       = 3_000    # paired transitions for residual δ

# Training
TRAIN_STEPS    = 50_000
EVAL_INTERVAL  = 5_000
N_EVAL_EPS     = 10
WARMUP         = 2_000
BATCH_SIZE     = 256
REPLAY_SIZE    = 100_000

# Model-based (MBPO-tuned for HalfCheetah)
ROLLOUT_HORIZON = 1       # single-step rollout — no error accumulation
ROLLOUT_BATCH   = 400     # rollouts per generation cycle
MODEL_BUF_MAX   = 50_000  # cap model buffer to prevent domination
MODEL_TRAIN_FREQ = 1000   # retrain world model every N steps
ROLLOUT_FREQ    = 250     # generate rollouts every N env steps

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
    def add_batch(self, s, a, r, s2, d):
        for i in range(len(s)):
            self.add(s[i], a[i], r[i], s2[i], d[i])
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in [self.s, self.a, self.r, self.s2, self.d])
    def sample_states(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.s[idx]


def collect_transitions(env_cls, mode, n_steps, policy_fn=None, seed=42):
    """Collect transitions. Returns (s, a, r, s2, done)."""
    env = env_cls(mode=mode)
    s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
    obs, _ = env.reset(seed=seed); ep = 0
    for _ in range(n_steps):
        a = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs2, r, d, tr, _ = env.step(a)
        s_list.append(obs); a_list.append(a); r_list.append(r)
        s2_list.append(obs2); d_list.append(float(d and not tr))
        obs = obs2
        if d or tr: ep += 1; obs, _ = env.reset(seed=seed+ep)
    env.close()
    return (np.array(s_list, np.float32), np.array(a_list, np.float32),
            np.array(r_list, np.float32), np.array(s2_list, np.float32),
            np.array(d_list, np.float32), ep)


def collect_paired(env_cls, n_steps, policy_fn=None, seed=42):
    """Collect paired sim+real transitions."""
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    s_list, a_list = [], []
    ns_sim, ns_real, r_sim, r_real = [], [], [], []
    ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample() if policy_fn is None else policy_fn(os_s)
        nss, rs, ds, ts, _ = sim.step(a); nsr, rr, dr, tr, _ = real.step(a)
        s_list.append(os_s); a_list.append(a)
        ns_sim.append(nss); ns_real.append(nsr); r_sim.append(rs); r_real.append(rr)
        os_s, os_r = nss, nsr
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=seed+ep); os_r, _ = real.reset(seed=seed+ep)
    sim.close(); real.close()
    return {
        "s": np.array(s_list, np.float32), "a": np.array(a_list, np.float32),
        "ns_sim": np.array(ns_sim, np.float32), "ns_real": np.array(ns_real, np.float32),
        "r_sim": np.array(r_sim, np.float32), "r_real": np.array(r_real, np.float32),
        "n_eps": ep,
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2", "c3"])
    args = parser.parse_args()
    mode = args.mode

    log_path = f"/tmp/step2_{mode}.log"
    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log(f"[{mode}] Device: {DEVICE}")
    log(f"Step 2: Model-Based RL with Residual World Model")
    env_cls = GravityCheetahEnv
    obs_dim = 17; act_dim = 6

    if mode == "c1":
        # ── Baseline: train in sim env directly
        np.random.seed(SEED); torch.manual_seed(SEED)
        env = env_cls(mode="sim")
        al = float(env.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        obs, _ = env.reset(seed=SEED); curve = []
        log(f"\n[c1: Raw Sim baseline] {TRAIN_STEPS//1000}k steps")
        for step in range(1, TRAIN_STEPS+1):
            a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r, d, tr, _ = env.step(a)
            buf.add(obs, a, r, obs2, float(d and not tr))
            obs = obs2
            if d or tr: obs, _ = env.reset()
            if step >= WARMUP and buf.size >= BATCH_SIZE: agent.update(buf)
            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}")
        env.close()

    elif mode in ("c2", "c3"):
        # ── Model-based: pretrain world model, then Dyna-style training
        np.random.seed(SEED); torch.manual_seed(SEED)

        # Phase 1: Collect sim data + train M_sim
        log(f"\n[Phase 1] Collecting {N_SIM_PRETRAIN//1000}k sim transitions...")
        s, a, r, s2, d, ep = collect_transitions(env_cls, "sim", N_SIM_PRETRAIN)
        log(f"  {len(s)} transitions, {ep} episodes")

        log(f"  Training world model ensemble (K=5)...")
        wm = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
        wm.fit(s, a, s2, r, n_epochs=100, patience=20)

        # Validate M_sim
        ns_pred, r_pred = wm.predict(s[:1000], a[:1000], deterministic=True)
        s_rmse = np.sqrt(np.mean((ns_pred - s2[:1000]) ** 2))
        r_rmse = np.sqrt(np.mean((r_pred - r[:1000]) ** 2))
        log(f"  M_sim validation: state RMSE={s_rmse:.4f}, reward RMSE={r_rmse:.4f}")

        corrected_model = None
        if mode == "c3":
            # Phase 2: Collect paired data + train residual
            log(f"\n[Phase 2] Collecting {N_PAIRED} paired transitions...")
            paired = collect_paired(env_cls, N_PAIRED)
            log(f"  {len(paired['s'])} transitions, {paired['n_eps']} episodes")
            log(f"  State gap: {np.abs(paired['ns_real'] - paired['ns_sim']).mean():.4f}")
            log(f"  Reward gap: {np.abs(paired['r_real'] - paired['r_sim']).mean():.4f}")

            log(f"  Training residual adapter...")
            wm.freeze()  # freeze M_sim
            residual = ResidualAdapter(obs_dim, act_dim, hidden=64, device=DEVICE)
            residual.fit(paired['s'], paired['a'],
                         paired['ns_sim'], paired['r_sim'],
                         paired['ns_real'], paired['r_real'],
                         n_epochs=200, patience=30)

            corrected_model = CorrectedWorldModel(wm, residual)

            # Validate M_real
            ns_real_pred, r_real_pred = corrected_model.predict(
                paired['s'][:500], paired['a'][:500], deterministic=True)
            corr_rmse = np.sqrt(np.mean((ns_real_pred - paired['ns_real'][:500]) ** 2))
            raw_rmse = np.sqrt(np.mean((paired['ns_sim'][:500] - paired['ns_real'][:500]) ** 2))
            log(f"  M_real state RMSE: {corr_rmse:.4f} (raw sim gap: {raw_rmse:.4f})")
            log(f"  Correction improvement: {(1-corr_rmse/raw_rmse)*100:.1f}%")
        else:
            corrected_model = CorrectedWorldModel(wm, ResidualAdapter(obs_dim, act_dim, device=DEVICE))

        # Phase 3: Dyna-style policy training (MBPO-tuned)
        env = env_cls(mode="sim")
        al = float(env.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)

        env_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

        obs, _ = env.reset(seed=SEED); curve = []
        label = "c2: M_sim rollouts" if mode == "c2" else "c3: M_real rollouts"
        log(f"\n[Phase 3: {label}] {TRAIN_STEPS//1000}k steps")
        log(f"  horizon={ROLLOUT_HORIZON}, rollout_batch={ROLLOUT_BATCH}, "
            f"model_buf_max={MODEL_BUF_MAX}, rollout_freq={ROLLOUT_FREQ}")

        for step in range(1, TRAIN_STEPS+1):
            # Sim env interaction for exploration (collect states/actions)
            a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r_sim, d, tr, _ = env.step(a)

            # env_buf stores sim transitions — used ONLY for start-state sampling
            env_buf.add(obs, a, r_sim, obs2, float(d and not tr))
            obs = obs2
            if d or tr: obs, _ = env.reset()

            if step >= WARMUP and env_buf.size >= BATCH_SIZE:
                # Generate single-step rollouts from M_real
                # Policy trains ONLY on these — reward & dynamics fully consistent
                if step % ROLLOUT_FREQ == 0:
                    start_states = env_buf.sample_states(ROLLOUT_BATCH)
                    actions = np.array([agent.get_action(s, deterministic=False)
                                       for s in start_states])
                    ns_pred, r_pred = corrected_model.predict(
                        start_states, actions, deterministic=False)
                    model_buf.add_batch(
                        start_states, actions,
                        r_pred.reshape(-1, 1), ns_pred,
                        np.zeros((ROLLOUT_BATCH, 1), np.float32))

                # Train ONLY on model_buf — no env_buf for Q-learning
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))

                # Diagnostics
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = corrected_model.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    r_err = np.sqrt(np.mean((r_test - env_buf.r[idx].squeeze()) ** 2))
                    disagree = corrected_model.wm.get_disagreement(
                        env_buf.s[idx], env_buf.a[idx]).mean()
                    diag = f"  m_s={s_err:.3f} m_r={r_err:.3f} dis={disagree:.3f}"

                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")

        env.close()

    # Summary
    avg = np.mean([r for _, r in curve[-3:]])
    log(f"\n{'='*50}")
    log(f"{mode} last 3 avg: real={avg:.1f}")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
