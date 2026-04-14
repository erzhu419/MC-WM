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
from mc_wm.residual.sindy_nau_adapter import SINDyNAUAdapter

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
MODEL_BUF_MAX   = 50_000  # cap model buffer
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
    def reset(self):
        """Clear buffer without reallocating arrays (avoids malloc/free overhead)."""
        self.ptr = self.size = 0
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in [self.s, self.a, self.r, self.s2, self.d])
    def sample_states(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.s[idx]


def collect_transitions(env_cls, mode, n_steps, policy_fn=None, seed=42):
    """Collect transitions. Returns (s, a, r, s2, done).
    Uses pre-allocated numpy arrays to avoid Python list → np.array peak memory.
    """
    env = env_cls(mode=mode)
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    s_buf  = np.empty((n_steps, od), np.float32)
    a_buf  = np.empty((n_steps, ad), np.float32)
    r_buf  = np.empty(n_steps, np.float32)
    s2_buf = np.empty((n_steps, od), np.float32)
    d_buf  = np.empty(n_steps, np.float32)
    obs, _ = env.reset(seed=seed); ep = 0
    for i in range(n_steps):
        a = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs2, r, d, tr, _ = env.step(a)
        s_buf[i]=obs; a_buf[i]=a; r_buf[i]=r; s2_buf[i]=obs2; d_buf[i]=float(d and not tr)
        obs = obs2
        if d or tr: ep += 1; obs, _ = env.reset(seed=seed+ep)
    env.close()
    return s_buf, a_buf, r_buf, s2_buf, d_buf, ep


def collect_paired(env_cls, n_steps, policy_fn=None, seed=42):
    """Collect paired sim+real transitions.
    Uses pre-allocated numpy arrays to avoid Python list → np.array peak memory.
    """
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    od = sim.observation_space.shape[0]; ad = sim.action_space.shape[0]
    s_buf    = np.empty((n_steps, od), np.float32)
    a_buf    = np.empty((n_steps, ad), np.float32)
    ns_sim_b = np.empty((n_steps, od), np.float32)
    ns_real_b= np.empty((n_steps, od), np.float32)
    rs_buf   = np.empty(n_steps, np.float32)
    rr_buf   = np.empty(n_steps, np.float32)
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    ep = 0
    for i in range(n_steps):
        a = sim.action_space.sample() if policy_fn is None else policy_fn(os_s)
        nss, rs, ds, ts, _ = sim.step(a); nsr, rr, dr, tr, _ = real.step(a)
        s_buf[i]=os_s; a_buf[i]=a; ns_sim_b[i]=nss; ns_real_b[i]=nsr
        rs_buf[i]=rs; rr_buf[i]=rr
        os_s, os_r = nss, nsr
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=seed+ep); os_r, _ = real.reset(seed=seed+ep)
    sim.close(); real.close()
    return {
        "s": s_buf, "a": a_buf,
        "ns_sim": ns_sim_b, "ns_real": ns_real_b,
        "r_sim": rs_buf, "r_real": rr_buf,
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
    parser.add_argument("--mode", default="c1", choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"])
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
                    actions = agent.get_actions_batch(start_states, deterministic=False)
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

    elif mode == "c4":
        # ── Direct M_real: train world model directly on REAL transitions
        # No residual needed — just train M on real dynamics
        # This validates: if model is accurate enough, does MBRL work?
        np.random.seed(SEED); torch.manual_seed(SEED)

        log(f"\n[Phase 1] Collecting 50k REAL transitions...")
        s, a, r, s2, d, ep = collect_transitions(env_cls, "real", N_SIM_PRETRAIN)
        log(f"  {len(s)} transitions, {ep} episodes")

        log(f"  Training M_real directly on real data (K=5)...")
        wm_real = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
        wm_real.fit(s, a, s2, r, n_epochs=100, patience=20)

        # Validate
        ns_pred, r_pred = wm_real.predict(s[:2000], a[:2000], deterministic=True)
        s_rmse = np.sqrt(np.mean((ns_pred - s2[:2000]) ** 2))
        r_rmse = np.sqrt(np.mean((r_pred - r[:2000]) ** 2))
        log(f"  M_real (direct) validation: state RMSE={s_rmse:.4f}, reward RMSE={r_rmse:.4f}")

        # Use M_real directly (no residual adapter needed)
        dummy_res = ResidualAdapter(obs_dim, act_dim, device=DEVICE)
        direct_model = CorrectedWorldModel(wm_real, dummy_res)

        # Phase 2: MBPO-style training with online model refit
        # Key difference: refit M_real every MODEL_TRAIN_FREQ steps using
        # REAL env transitions collected online
        env_real = env_cls(mode="real")  # interact in REAL env (both are simulators)
        al = float(env_real.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        env_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

        # Seed env_buf with pretrain data
        for i in range(len(s)):
            env_buf.add(s[i], a[i], r[i], s2[i], d[i])

        obs, _ = env_real.reset(seed=SEED); curve = []
        log(f"\n[Phase 2: c4 MBPO-style with online refit] {TRAIN_STEPS//1000}k steps")
        log(f"  horizon={ROLLOUT_HORIZON}, initial RMSE={s_rmse:.4f}")
        log(f"  Model refit every {MODEL_TRAIN_FREQ} steps, explore in REAL env")

        for step in range(1, TRAIN_STEPS+1):
            # Interact in REAL env (cheap — it's a simulator)
            a_act = env_real.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r_real_step, d_flag, tr, _ = env_real.step(a_act)
            env_buf.add(obs, a_act, r_real_step, obs2, float(d_flag and not tr))
            obs = obs2
            if d_flag or tr: obs, _ = env_real.reset()

            # Online model refit: retrain M_real on growing env_buf
            if step > WARMUP and step % MODEL_TRAIN_FREQ == 0:
                n_fit = min(env_buf.size, 50000)
                idx_fit = np.random.choice(env_buf.size, n_fit, replace=False) if env_buf.size > n_fit else np.arange(env_buf.size)
                log(f"  [REFIT step {step}] Retraining M_real on {n_fit} transitions...")
                wm_real.fit(env_buf.s[idx_fit], env_buf.a[idx_fit],
                           env_buf.s2[idx_fit], env_buf.r[idx_fit].squeeze(),
                           n_epochs=20, patience=10)
                # Clear stale model rollouts (reset instead of reallocate)
                model_buf.reset()

            if step >= WARMUP and env_buf.size >= BATCH_SIZE:
                # Generate model rollouts
                if step % ROLLOUT_FREQ == 0:
                    start_states = env_buf.sample_states(ROLLOUT_BATCH)
                    actions = agent.get_actions_batch(start_states, deterministic=False)
                    ns_pred, r_pred = direct_model.predict(
                        start_states, actions, deterministic=False)
                    model_buf.add_batch(
                        start_states, actions,
                        r_pred.reshape(-1, 1), ns_pred,
                        np.zeros((ROLLOUT_BATCH, 1), np.float32))

                # MBPO: train on BOTH env + model data
                agent.update(env_buf)
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = direct_model.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    disagree = wm_real.get_disagreement(env_buf.s[idx], env_buf.a[idx]).mean()
                    diag = f"  m_s={s_err:.3f} dis={disagree:.3f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")
        env_real.close()

    elif mode == "c6":
        # ── c4 MBPO pipeline but with M_sim + residual δ instead of direct M_real
        #
        # Tests: is residual (M_sim + δ) better/worse/same as direct M_real?
        # Both interact in REAL env, both do online refit, both mixed training.
        # Only difference: world model architecture.
        #
        # If c6 ≈ c4: residual adds nothing (direct is simpler)
        # If c6 > c4: residual provides useful structural prior
        # If c6 < c4: residual hurts (M_sim prior is wrong)
        np.random.seed(SEED); torch.manual_seed(SEED)

        # Phase 1a: Train M_sim on sim data (frozen base)
        log(f"\n[Phase 1a] Collecting {N_SIM_PRETRAIN//1000}k SIM transitions + training M_sim...")
        s_sim, a_sim, r_sim, s2_sim, d_sim, ep_sim = collect_transitions(
            env_cls, "sim", N_SIM_PRETRAIN)
        log(f"  {len(s_sim)} sim transitions, {ep_sim} episodes")

        wm_sim = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
        wm_sim.fit(s_sim, a_sim, s2_sim, r_sim, n_epochs=100, patience=20)
        ns_p, r_p = wm_sim.predict(s_sim[:1000], a_sim[:1000], deterministic=True)
        sim_rmse = np.sqrt(np.mean((ns_p - s2_sim[:1000]) ** 2))
        log(f"  M_sim RMSE on sim: {sim_rmse:.4f}")
        wm_sim.freeze()

        # Phase 1b: Collect initial real data + train residual δ
        log(f"\n[Phase 1b] Collecting {N_SIM_PRETRAIN//1000}k REAL transitions for env_buf seed...")
        s_real, a_real, r_real, s2_real, d_real, ep_real = collect_transitions(
            env_cls, "real", N_SIM_PRETRAIN)
        log(f"  {len(s_real)} real transitions, {ep_real} episodes")

        log(f"  Training initial residual δ on {N_SIM_PRETRAIN//1000}k real transitions...")
        ns_sim_pred, r_sim_pred = wm_sim.predict(s_real[:N_SIM_PRETRAIN],
                                                  a_real[:N_SIM_PRETRAIN], deterministic=True)
        # Match c4's model capacity: 200-dim hidden (same as M_sim/M_real)
        residual = ResidualAdapter(obs_dim, act_dim, hidden=128, device=DEVICE)
        residual.fit(s_real, a_real,
                     ns_sim_pred, r_sim_pred,
                     s2_real, r_real,
                     n_epochs=100, patience=20)

        corrected = CorrectedWorldModel(wm_sim, residual)

        # Validate M_real = M_sim + δ
        ns_corr, r_corr = corrected.predict(s_real[:2000], a_real[:2000], deterministic=True)
        corr_rmse = np.sqrt(np.mean((ns_corr - s2_real[:2000]) ** 2))
        direct_rmse = np.sqrt(np.mean((ns_sim_pred[:2000] - s2_real[:2000]) ** 2))
        log(f"  M_sim → real RMSE: {direct_rmse:.4f}")
        log(f"  M_real (M_sim+δ) → real RMSE: {corr_rmse:.4f}")
        log(f"  Residual improvement: {(1-corr_rmse/direct_rmse)*100:.1f}%")

        # Phase 2: MBPO in real env with M_sim+δ
        env_real = env_cls(mode="real")
        al = float(env_real.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        env_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

        # Seed env_buf with pretrain real data
        for i in range(len(s_real)):
            env_buf.add(s_real[i], a_real[i], r_real[i], s2_real[i], d_real[i])

        obs, _ = env_real.reset(seed=SEED); curve = []
        log(f"\n[Phase 2: c6 MBPO with M_sim+δ] {TRAIN_STEPS//1000}k steps")
        log(f"  M_sim frozen, only refit δ online")

        for step in range(1, TRAIN_STEPS+1):
            a_act = env_real.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r_step, d_flag, tr, _ = env_real.step(a_act)
            env_buf.add(obs, a_act, r_step, obs2, float(d_flag and not tr))
            obs = obs2
            if d_flag or tr: obs, _ = env_real.reset()

            # Online refit: only retrain RESIDUAL δ (M_sim stays frozen)
            # Fix 1: warm-start δ (keep weights, don't recreate)
            # Fix 2: larger δ (128×2)
            # Fix 3: more epochs (50)
            if step > WARMUP and step % MODEL_TRAIN_FREQ == 0:
                n_fit = min(env_buf.size, 50000)
                idx_fit = np.random.choice(env_buf.size, n_fit, replace=False) if env_buf.size > n_fit else np.arange(env_buf.size)
                ns_sim_p, r_sim_p = wm_sim.predict(
                    env_buf.s[idx_fit], env_buf.a[idx_fit], deterministic=True)
                log(f"  [REFIT δ step {step}] Warm-start refit on {n_fit} transitions...")
                # Warm-start: reuse existing residual, just keep training
                residual.fit(env_buf.s[idx_fit], env_buf.a[idx_fit],
                            ns_sim_p, r_sim_p,
                            env_buf.s2[idx_fit], env_buf.r[idx_fit].squeeze(),
                            n_epochs=50, patience=15)
                corrected = CorrectedWorldModel(wm_sim, residual)
                model_buf.reset()  # clear stale rollouts without reallocating

            if step >= WARMUP and env_buf.size >= BATCH_SIZE:
                if step % ROLLOUT_FREQ == 0:
                    start_states = env_buf.sample_states(ROLLOUT_BATCH)
                    actions = agent.get_actions_batch(start_states, deterministic=False)
                    ns_pred, r_pred = corrected.predict(
                        start_states, actions, deterministic=False)
                    model_buf.add_batch(
                        start_states, actions,
                        r_pred.reshape(-1, 1), ns_pred,
                        np.zeros((ROLLOUT_BATCH, 1), np.float32))

                agent.update(env_buf)
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, _ = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    diag = f"  m_s={s_err:.3f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")
        env_real.close()

    elif mode in ("c7", "c8"):
        # ── c6 but with SINDy+NAU instead of MLP δ
        # Tests: interpretable symbolic residual vs black-box MLP
        np.random.seed(SEED); torch.manual_seed(SEED)

        # Phase 1a: M_sim
        log(f"\n[Phase 1a] Collecting {N_SIM_PRETRAIN//1000}k SIM transitions + training M_sim...")
        s_sim, a_sim, r_sim, s2_sim, d_sim, ep_sim = collect_transitions(
            env_cls, "sim", N_SIM_PRETRAIN)
        wm_sim = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
        wm_sim.fit(s_sim, a_sim, s2_sim, r_sim, n_epochs=100, patience=20)
        wm_sim.freeze()

        # Phase 1b: Real data + SINDy+NAU residual
        log(f"\n[Phase 1b] Collecting {N_SIM_PRETRAIN//1000}k REAL transitions...")
        s_real, a_real, r_real, s2_real, d_real, ep_real = collect_transitions(
            env_cls, "real", N_SIM_PRETRAIN)

        ns_sim_pred, r_sim_pred = wm_sim.predict(s_real, a_real, deterministic=True)

        log(f"  Training SINDy+NAU residual δ...")
        _env_type = "gravity_cheetah" if mode == "c8" else None
        _max_rounds = 3  # hypothesis loop discovers features, then locks sparsity for SGD refit
        residual = SINDyNAUAdapter(obs_dim, act_dim, device=DEVICE, log_fn=log,
                                    env_type=_env_type, max_rounds=_max_rounds)
        residual.fit(s_real, a_real, ns_sim_pred, r_sim_pred, s2_real, r_real,
                     n_epochs=100, patience=20)

        corrected = CorrectedWorldModel(wm_sim, residual)

        # Validate state + reward correction
        ns_corr, r_corr = corrected.predict(s_real[:2000], a_real[:2000], deterministic=True)
        corr_rmse = np.sqrt(np.mean((ns_corr - s2_real[:2000]) ** 2))
        direct_rmse = np.sqrt(np.mean((ns_sim_pred[:2000] - s2_real[:2000]) ** 2))
        r_corr_rmse = np.sqrt(np.mean((r_corr - r_real[:2000]) ** 2))
        r_sim_rmse = np.sqrt(np.mean((r_sim_pred[:2000] - r_real[:2000]) ** 2))
        log(f"  State: M_sim→real RMSE={direct_rmse:.4f} → M_real RMSE={corr_rmse:.4f} "
            f"({(1-corr_rmse/direct_rmse)*100:.1f}% improvement)")
        log(f"  Reward: M_sim→real RMSE={r_sim_rmse:.4f} → M_real RMSE={r_corr_rmse:.4f} "
            f"({(1-r_corr_rmse/max(r_sim_rmse,1e-8))*100:.1f}% improvement)")

        # Print discovered symbolic structure
        terms = residual.get_active_terms()
        log(f"  Discovered terms: {len(terms)} dims with active features")
        for name, active in list(terms.items())[:5]:
            t_str = ", ".join(f"{n}:{c:.3f}" for n, c in active[:4])
            log(f"    {name}: [{t_str}]")

        # Phase 2: MBPO with SINDy+NAU
        env_real = env_cls(mode="real")
        al = float(env_real.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        env_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

        for i in range(len(s_real)):
            env_buf.add(s_real[i], a_real[i], r_real[i], s2_real[i], d_real[i])

        obs, _ = env_real.reset(seed=SEED); curve = []
        log(f"\n[Phase 2: c7 MBPO with SINDy+NAU δ] {TRAIN_STEPS//1000}k steps")

        for step in range(1, TRAIN_STEPS+1):
            a_act = env_real.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r_step, d_flag, tr, _ = env_real.step(a_act)
            env_buf.add(obs, a_act, r_step, obs2, float(d_flag and not tr))
            obs = obs2
            if d_flag or tr: obs, _ = env_real.reset()

            # Online refit: retrain SINDy+NAU δ (warm-start)
            if step > WARMUP and step % MODEL_TRAIN_FREQ == 0:
                n_fit = min(env_buf.size, 50000)
                idx_fit = np.random.choice(env_buf.size, n_fit, replace=False) if env_buf.size > n_fit else np.arange(env_buf.size)
                ns_sim_p, r_sim_p = wm_sim.predict(
                    env_buf.s[idx_fit], env_buf.a[idx_fit], deterministic=True)
                log(f"  [REFIT SINDy+NAU step {step}] on {n_fit} transitions...")
                residual.fit(env_buf.s[idx_fit], env_buf.a[idx_fit],
                            ns_sim_p, r_sim_p,
                            env_buf.s2[idx_fit], env_buf.r[idx_fit].squeeze(),
                            n_epochs=50, patience=15)
                corrected = CorrectedWorldModel(wm_sim, residual)
                model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

            if step >= WARMUP and env_buf.size >= BATCH_SIZE:
                if step % ROLLOUT_FREQ == 0:
                    start_states = env_buf.sample_states(ROLLOUT_BATCH)
                    actions = np.array([agent.get_action(ss, deterministic=False)
                                       for ss in start_states])
                    ns_pred, r_pred = corrected.predict(
                        start_states, actions, deterministic=False)
                    model_buf.add_batch(
                        start_states, actions,
                        r_pred.reshape(-1, 1), ns_pred,
                        np.zeros((ROLLOUT_BATCH, 1), np.float32))

                agent.update(env_buf)
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    r_err = np.sqrt(np.mean((r_test - env_buf.r[idx].squeeze()) ** 2))
                    ood = residual.get_ood_bound(env_buf.s[idx], env_buf.a[idx]).mean()
                    diag = f"  m_s={s_err:.3f} m_r={r_err:.3f} L={residual._nau_head.L_eff:.3f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")
        env_real.close()

    elif mode == "c5":
        # ── THE PAPER EXPERIMENT: Sim env + residual-corrected world model
        #
        # Like c4 but:
        # - Interact in SIM env (not real)
        # - World model = M_sim + residual δ (corrected to real dynamics)
        # - Periodically collect small batches of PAIRED data to refit δ
        # - Policy trains on mix of sim env_buf + corrected model rollouts
        #
        # This is the sim-to-real scenario: only sim is cheap, real is expensive.
        # We use a small real data budget (3k initial + 500 per refit) to correct
        # the world model, then train policy mostly in sim + corrected model.
        np.random.seed(SEED); torch.manual_seed(SEED)

        # Phase 1: Train M_sim on sim data
        log(f"\n[Phase 1] Collecting {N_SIM_PRETRAIN//1000}k sim transitions + training M_sim...")
        s_sim, a_sim, r_sim_arr, s2_sim, d_sim, ep_sim = collect_transitions(
            env_cls, "sim", N_SIM_PRETRAIN)
        log(f"  {len(s_sim)} sim transitions, {ep_sim} episodes")

        wm_sim = WorldModelEnsemble(obs_dim, act_dim, K=5, hidden=200, device=DEVICE)
        wm_sim.fit(s_sim, a_sim, s2_sim, r_sim_arr, n_epochs=100, patience=20)
        ns_pred, r_pred = wm_sim.predict(s_sim[:1000], a_sim[:1000], deterministic=True)
        sim_rmse = np.sqrt(np.mean((ns_pred - s2_sim[:1000]) ** 2))
        log(f"  M_sim RMSE on sim: {sim_rmse:.4f}")

        # Phase 2: Collect initial paired data + train residual
        log(f"\n[Phase 2] Collecting {N_PAIRED} paired transitions + training residual...")
        paired = collect_paired(env_cls, N_PAIRED)
        log(f"  {len(paired['s'])} paired, state gap={np.abs(paired['ns_real']-paired['ns_sim']).mean():.4f}")

        wm_sim.freeze()
        residual = ResidualAdapter(obs_dim, act_dim, hidden=64, device=DEVICE)
        residual.fit(paired['s'], paired['a'],
                     paired['ns_sim'], paired['r_sim'],
                     paired['ns_real'], paired['r_real'],
                     n_epochs=200, patience=30)
        corrected = CorrectedWorldModel(wm_sim, residual)

        # Validate M_real
        ns_rp, r_rp = corrected.predict(paired['s'][:500], paired['a'][:500], deterministic=True)
        corr_rmse = np.sqrt(np.mean((ns_rp - paired['ns_real'][:500]) ** 2))
        raw_gap = np.sqrt(np.mean((paired['ns_sim'][:500] - paired['ns_real'][:500]) ** 2))
        log(f"  M_real RMSE: {corr_rmse:.4f} (raw gap: {raw_gap:.4f}, "
            f"correction: {(1-corr_rmse/raw_gap)*100:.1f}%)")

        # Phase 3: MBPO in sim env with corrected model rollouts
        env = env_cls(mode="sim")
        al = float(env.action_space.high[0])
        agent = RESACAgent(obs_dim, act_dim, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        env_buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)
        model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)

        # Accumulated paired data for residual refit
        paired_s = list(paired['s'])
        paired_a = list(paired['a'])
        paired_ns_sim = list(paired['ns_sim'])
        paired_ns_real = list(paired['ns_real'])
        paired_r_sim = list(paired['r_sim'])
        paired_r_real = list(paired['r_real'])

        obs, _ = env.reset(seed=SEED); curve = []
        RESIDUAL_REFIT_FREQ = 5000   # refit residual every N steps
        RESIDUAL_REFIT_SAMPLES = 500 # new paired samples per refit

        log(f"\n[Phase 3: c5 Sim + Corrected Model MBPO] {TRAIN_STEPS//1000}k steps")
        log(f"  horizon={ROLLOUT_HORIZON}, residual refit every {RESIDUAL_REFIT_FREQ} steps")
        log(f"  Interact in SIM, rollouts from M_real = M_sim + δ")

        for step in range(1, TRAIN_STEPS+1):
            # Interact in SIM env
            a_act = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r_env, d_flag, tr, _ = env.step(a_act)
            env_buf.add(obs, a_act, r_env, obs2, float(d_flag and not tr))
            obs = obs2
            if d_flag or tr: obs, _ = env.reset()

            # Periodically refit residual with new paired data
            if step > WARMUP and step % RESIDUAL_REFIT_FREQ == 0:
                log(f"  [REFIT δ step {step}] Collecting {RESIDUAL_REFIT_SAMPLES} paired samples...")
                # Use current policy to collect paired data
                new_paired = collect_paired(env_cls, RESIDUAL_REFIT_SAMPLES,
                    policy_fn=lambda s: agent.get_action(s, deterministic=False),
                    seed=step)
                # Append to accumulated paired data
                paired_s.extend(new_paired['s'])
                paired_a.extend(new_paired['a'])
                paired_ns_sim.extend(new_paired['ns_sim'])
                paired_ns_real.extend(new_paired['ns_real'])
                paired_r_sim.extend(new_paired['r_sim'])
                paired_r_real.extend(new_paired['r_real'])
                # Keep last 10k paired samples
                max_paired = 10000
                if len(paired_s) > max_paired:
                    paired_s = paired_s[-max_paired:]
                    paired_a = paired_a[-max_paired:]
                    paired_ns_sim = paired_ns_sim[-max_paired:]
                    paired_ns_real = paired_ns_real[-max_paired:]
                    paired_r_sim = paired_r_sim[-max_paired:]
                    paired_r_real = paired_r_real[-max_paired:]
                # Refit residual
                residual = ResidualAdapter(obs_dim, act_dim, hidden=64, device=DEVICE)
                residual.fit(
                    np.array(paired_s, np.float32), np.array(paired_a, np.float32),
                    np.array(paired_ns_sim, np.float32), np.array(paired_r_sim, np.float32),
                    np.array(paired_ns_real, np.float32), np.array(paired_r_real, np.float32),
                    n_epochs=50, patience=15)
                corrected = CorrectedWorldModel(wm_sim, residual)
                # Clear stale model rollouts
                model_buf = ReplayBuffer(obs_dim, act_dim, MODEL_BUF_MAX)
                log(f"    Paired data: {len(paired_s)} total")

            if step >= WARMUP and env_buf.size >= BATCH_SIZE:
                # Generate corrected model rollouts
                if step % ROLLOUT_FREQ == 0:
                    start_states = env_buf.sample_states(ROLLOUT_BATCH)
                    actions = np.array([agent.get_action(ss, deterministic=False)
                                       for ss in start_states])
                    ns_pred, r_pred = corrected.predict(
                        start_states, actions, deterministic=False)
                    model_buf.add_batch(
                        start_states, actions,
                        r_pred.reshape(-1, 1), ns_pred,
                        np.zeros((ROLLOUT_BATCH, 1), np.float32))

                # MBPO: train on env_buf + model_buf
                agent.update(env_buf)
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls)
                curve.append((step, ret))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    diag = f"  m_s={s_err:.3f} paired={len(paired_s)}"
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
