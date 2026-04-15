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
from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv, GravityCheetahCeilingEnv, GravityCheetahSoftCeilingEnv
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.envs.hp_mujoco.ant_wall_broken import AntWallBrokenEnv
from mc_wm.envs.hp_mujoco.friction_walker import FrictionWalkerSoftCeilingEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.world_model import WorldModelEnsemble, ResidualAdapter, CorrectedWorldModel
from mc_wm.residual.sindy_nau_adapter import SINDyNAUAdapter
from mc_wm.self_audit.constraint_system import ConstraintSystem
from mc_wm.self_audit.icrl_constraint import ResidualAwareICRL

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
    """
    Returns (avg_return, std_return, avg_violations_per_ep).

    Violations = count of info['ceiling_hit'] / info['wall_hit'] events per episode
    (Type 2 OOD access: agent reached a hard-constraint terminal region).
    For envs without explicit constraints: violation count is always 0.
    """
    env = env_cls(mode="real"); rets = []; viols = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep+200); total = 0.0; n_v = 0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, info = env.step(a); total += r
            if info.get("ceiling_hit") or info.get("wall_hit"):
                n_v += 1
            if d or tr: break
        rets.append(total); viols.append(n_v)
    env.close()
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(viols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="c1", choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"])
    parser.add_argument("--env", default="gravity",
                        choices=["gravity", "gravity_ceiling", "gravity_soft_ceiling",
                                 "carpet_ant", "ant_wall_broken", "friction_walker_soft_ceiling"])
    parser.add_argument("--icrl_mode", default="transition", choices=["transition", "confidence"],
                        help="v4=transition (Δs discriminator), v1=confidence (model-conf input, Type 2 proxy)")
    parser.add_argument("--save_phi", default=None, help="Path to save trained ICRL φ")
    parser.add_argument("--load_phi", default=None, help="Path to load ICRL φ (freezes it)")
    parser.add_argument("--icrl_combine", default="top_k", choices=["top_k", "soft"],
                        help="v4 default=top_k (filter top 70%); v2/v3=soft (w=QΔ×(0.5+0.5×φ))")
    args = parser.parse_args()
    mode = args.mode

    _suffix = f"_{args.icrl_mode}" if mode == "c10" else ""
    if mode == "c10" and args.icrl_combine == "soft":
        _suffix += "_soft"
    if mode == "c10" and args.load_phi is not None:
        _suffix += "_transfer"
    log_path = f"/tmp/step2_{mode}_{args.env}{_suffix}.log"
    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log(f"[{mode}/{args.env}] Device: {DEVICE}")
    log(f"Step 2: Model-Based RL with Residual World Model")

    # Environment selection
    if args.env == "gravity":
        env_cls = GravityCheetahEnv; obs_dim, act_dim = 17, 6
    elif args.env == "gravity_ceiling":
        env_cls = GravityCheetahCeilingEnv; obs_dim, act_dim = 17, 6
    elif args.env == "gravity_soft_ceiling":
        env_cls = GravityCheetahSoftCeilingEnv; obs_dim, act_dim = 17, 6
    elif args.env == "carpet_ant":
        env_cls = CarpetAntEnv; obs_dim, act_dim = 27, 8
    elif args.env == "ant_wall_broken":
        env_cls = AntWallBrokenEnv; obs_dim, act_dim = 27, 8
    elif args.env == "friction_walker_soft_ceiling":
        env_cls = FrictionWalkerSoftCeilingEnv; obs_dim, act_dim = 17, 6

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
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}")
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
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))

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

                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}  "
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
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = direct_model.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    disagree = wm_real.get_disagreement(env_buf.s[idx], env_buf.a[idx]).mean()
                    diag = f"  m_s={s_err:.3f} dis={disagree:.3f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}  "
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
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, _ = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    diag = f"  m_s={s_err:.3f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")
        env_real.close()

    elif mode in ("c7", "c8", "c9", "c10"):
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

        # Constraint systems
        constraint_sys = ConstraintSystem(env_type="gravity_cheetah", log_fn=log) if mode == "c9" else None

        # ICRL learned constraint (c10)
        # Two modes:
        #   transition (v4): φ(s,a,Δs) — discriminates real vs sim dynamics (Type 1 territory)
        #   confidence (v1): φ(s,a,model_conf) — flags low-confidence regions (Type 2 proxy)
        icrl = None
        icrl_mode_used = args.icrl_mode
        if mode == "c10":
            log(f"\n[ICRL] Initializing ICRL (mode={icrl_mode_used})...")
            use_trans = (icrl_mode_used == "transition")
            icrl = ResidualAwareICRL(
                obs_dim, act_dim, hidden_sizes=(128, 128),
                lr=3e-4, reg_coeff=0.05, use_transition=use_trans,
                target_kl=10.0, device=DEVICE, log_fn=log)
            if use_trans:
                icrl.set_expert_data(s_real, a_real, next_obs=s2_real)
            else:
                # Confidence mode (v1 style, Type 2 OOD proxy):
                # model_conf = 1/(1+MSE(corrected_pred, real)) — high where model is accurate
                ns_real_pred, _ = corrected.predict(s_real, a_real, deterministic=True)
                model_conf = 1.0 / (1.0 + np.mean((ns_real_pred - s2_real) ** 2, axis=1))
                icrl.set_expert_data(s_real, a_real, model_confidence=model_conf)
            # Cross-env transfer: optionally load pretrained φ (frozen)
            if args.load_phi is not None:
                log(f"  [TRANSFER] Loading frozen φ from {args.load_phi}")
                icrl.load(args.load_phi, freeze=True)

        obs, _ = env_real.reset(seed=SEED); curve = []
        log(f"\n[Phase 2: {mode} MBPO with SINDy+NAU δ] {TRAIN_STEPS//1000}k steps")

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
                    # Sample start states from env_buf (with their real next-states)
                    s_idx = np.random.randint(0, env_buf.size, ROLLOUT_BATCH)
                    start_states = env_buf.s[s_idx]
                    actions = np.array([agent.get_action(ss, deterministic=False)
                                       for ss in start_states])
                    ns_pred, r_pred = corrected.predict(
                        start_states, actions, deterministic=False)

                    if mode in ("c9", "c10"):
                        # QΔ: per-transition model confidence weight
                        ns_real = env_buf.s2[s_idx]
                        per_s_err = np.mean((ns_pred - ns_real) ** 2, axis=1)
                        tau_s = float(np.median(per_s_err)) + 1e-8
                        w_qdelta = 1.0 / (1.0 + per_s_err / tau_s)

                        if mode == "c10" and icrl is not None:
                            # Compute φ score per rollout (input depends on ICRL mode)
                            if icrl_mode_used == "transition":
                                delta_pred = ns_pred - start_states
                                phi_scores = icrl.get_feasibility(
                                    start_states, actions, delta_s=delta_pred)
                            else:
                                phi_scores = icrl.get_feasibility(
                                    start_states, actions, model_confidence=w_qdelta)
                            if args.icrl_combine == "top_k":
                                # v4: keep top 70% most real-like rollouts
                                keep_frac = 0.7
                                n_keep = max(1, int(len(phi_scores) * keep_frac))
                                top_k_idx = np.argpartition(-phi_scores, n_keep - 1)[:n_keep]
                                keep_mask = np.zeros(len(phi_scores), dtype=bool)
                                keep_mask[top_k_idx] = True
                                w = w_qdelta * keep_mask.astype(np.float32)
                            else:
                                # v2/v3 soft modulation: w = QΔ × (0.5 + 0.5×φ)
                                w_icrl = 0.5 + 0.5 * phi_scores
                                w = w_qdelta * w_icrl
                        elif mode == "c9" and constraint_sys is not None:
                            # Hardcoded constraint filter
                            ok_mask, _ = constraint_sys.check_batch(
                                start_states, actions, ns_pred, r_pred * w_qdelta)
                            w = w_qdelta * ok_mask.astype(np.float32)
                        else:
                            w = w_qdelta

                        r_weighted = r_pred * w
                        # Filter zero-weight transitions
                        valid = np.where(w > 0.01)[0]
                        if len(valid) > 0:
                            model_buf.add_batch(
                                start_states[valid], actions[valid],
                                r_weighted[valid].reshape(-1, 1), ns_pred[valid],
                                np.zeros((len(valid), 1), np.float32))
                    else:
                        model_buf.add_batch(
                            start_states, actions,
                            r_pred.reshape(-1, 1), ns_pred,
                            np.zeros((ROLLOUT_BATCH, 1), np.float32))

                agent.update(env_buf)
                if model_buf.size >= BATCH_SIZE:
                    agent.update(model_buf)

            if step % EVAL_INTERVAL == 0:
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    r_err = np.sqrt(np.mean((r_test - env_buf.r[idx].squeeze()) ** 2))
                    ood = residual.get_ood_bound(env_buf.s[idx], env_buf.a[idx]).mean()
                    diag = f"  m_s={s_err:.3f} m_r={r_err:.3f} L={residual._nau_head.L_eff:.3f}"
                    if constraint_sys is not None:
                        cs = constraint_sys.get_stats()
                        diag += f" C={cs['n_constraints']} rej={cs['reject_rate']:.1%}"
                    if icrl is not None:
                        ics = icrl.get_stats()
                        diag += f" φ_e={ics.get('phi_expert',0):.2f} φ_n={ics.get('phi_nominal',0):.2f}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")

                # ICRL backward step: update φ (mode-dependent)
                # Skip if φ is frozen (transfer mode with --load_phi)
                if icrl is not None and not icrl.is_frozen and step % EVAL_INTERVAL == 0 and env_buf.size > 500:
                    nom_n = min(2000, env_buf.size)
                    nom_idx = np.random.randint(0, env_buf.size, nom_n)
                    nom_states = env_buf.s[nom_idx]
                    nom_actions = env_buf.a[nom_idx]
                    ns_sim_raw, _ = wm_sim.predict(nom_states, nom_actions, deterministic=True)
                    if icrl_mode_used == "transition":
                        # Negatives = raw M_sim Δs (discriminate real dynamics vs sim dynamics)
                        nominal_extra = ns_sim_raw - nom_states
                    else:
                        # Confidence mode: nominal = sim predictions have LOW confidence in real
                        # conf_sim = 1/(1+MSE(sim_prediction, real_next_state))
                        nom_real_next = env_buf.s2[nom_idx]
                        conf_sim = 1.0 / (1.0 + np.mean(
                            (ns_sim_raw - nom_real_next) ** 2, axis=1))
                        nominal_extra = conf_sim
                    icrl_metrics = icrl.train_constraint(
                        nom_states, nom_actions,
                        nominal_extra=nominal_extra, n_iters=3)
                    log(f"    ICRL update: φ_e={icrl_metrics['phi_expert']:.3f} "
                        f"φ_n={icrl_metrics['phi_nominal']:.3f} "
                        f"sep={icrl_metrics['separation']:.3f} kl={icrl_metrics['kl']:.3f}")

                # Role #3: periodic constraint audit
                if constraint_sys is not None and step % (EVAL_INTERVAL * 2) == 0:
                    idx_audit = np.random.randint(0, env_buf.size, min(1000, env_buf.size))
                    ns_audit, r_audit = corrected.predict(
                        env_buf.s[idx_audit], env_buf.a[idx_audit], deterministic=True)
                    corr_mag = np.sqrt(np.mean((ns_audit - env_buf.s2[idx_audit]) ** 2, axis=1))
                    constraint_sys.audit_suspicious(
                        env_buf.s[idx_audit], env_buf.a[idx_audit],
                        ns_audit, r_audit, corr_mag, step=step)

        # Optional: save trained φ for cross-env transfer
        if icrl is not None and args.save_phi is not None and not icrl.is_frozen:
            icrl.save(args.save_phi)
            log(f"  [SAVE] ICRL φ saved to {args.save_phi}")

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
                ret, std, viol = evaluate(agent, env_cls)
                curve.append((step, ret, viol))
                diag = ""
                if env_buf.size >= 500:
                    idx = np.random.randint(max(0, env_buf.size-2000), env_buf.size, 500)
                    ns_test, r_test = corrected.predict(
                        env_buf.s[idx], env_buf.a[idx], deterministic=True)
                    s_err = np.sqrt(np.mean((ns_test - env_buf.s2[idx]) ** 2))
                    diag = f"  m_s={s_err:.3f} paired={len(paired_s)}"
                log(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={viol:4.1f}  "
                    f"env={env_buf.size} mdl={model_buf.size}{diag}")
        env.close()

    # Summary
    avg = np.mean([r for _, r, *_ in curve[-3:]])
    avg_v = np.mean([v for _, _, v, *_ in curve[-3:]]) if curve and len(curve[0]) >= 3 else 0.0
    log(f"{mode} last 3 avg: violations={avg_v:.2f}/ep")
    log(f"\n{'='*50}")
    log(f"{mode} last 3 avg: real={avg:.1f}")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
