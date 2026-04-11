"""
Attempt 3/8: Confident Residual Model + Online Refit + Actor Constraint.

Three mechanisms working together:
1. MLP ensemble confidence → Q-loss importance weight (multiplicative)
2. Confidence constraint on actor → penalize exploring low-confidence regions
3. Online paired data collection → refit residual model every REFIT_INTERVAL steps

Key difference from attempts 1-2: confidence (ensemble disagreement) is
bounded and improves over training via online refit.

运行: conda run -n MC-WM python3 -u experiments/mvp_confident.py --mode c1
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.confident_residual import ConfidentResidualModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_COLLECT     = 3_000
TRAIN_STEPS   = 50_000
EVAL_INTERVAL = 5_000
REFIT_INTERVAL = 5_000   # refit residual model every N steps
REFIT_SAMPLES  = 500     # on-policy paired samples per refit
N_EVAL_EPS    = 10
WARMUP        = 2_000
BATCH_SIZE    = 256
REPLAY_SIZE   = 100_000
N_CRITICS     = 3
BETA_LCB      = -2.0
HIDDEN        = 256
LR            = 3e-4
W_MIN         = 0.1      # minimum weight for lowest-confidence transitions
CONSTRAINT_SCALE = 0.5   # actor confidence constraint weight


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


def collect_initial_paired(env_cls, n_steps, seed=SEED):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    SA_list, ds_list = [], []; ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns_s, _, ds, ts, _ = sim.step(a); ns_r, _, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_s, a])); ds_list.append(ns_r - ns_s)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    return np.array(SA_list, np.float32), np.array(ds_list, np.float32), ep


def collect_online_paired(env_cls, agent, n_steps, seed_offset=10000):
    """
    Collect paired data using current policy in lockstep sim+real.
    Uses the TRAINED policy (not random), so data covers the policy's actual distribution.
    """
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed_offset); os_r, _ = real.reset(seed=seed_offset)
    SA_new, ds_new = [], []; ep = 0
    for _ in range(n_steps):
        a = agent.get_action(os_s, deterministic=False)
        ns_s, _, ds, ts, _ = sim.step(a); ns_r, _, dr, tr, _ = real.step(a)
        SA_new.append(np.concatenate([os_s, a])); ds_new.append(ns_r - ns_s)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=seed_offset+ep); os_r, _ = real.reset(seed=seed_offset+ep)
    sim.close(); real.close()
    return np.array(SA_new, np.float32), np.array(ds_new, np.float32)


def evaluate(agent, env_cls, mode="real", n_eps=N_EVAL_EPS):
    env = env_cls(mode=mode); rets = []
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
    parser.add_argument("--mode", default="c1", choices=["c1", "c2"])
    args = parser.parse_args()
    mode = args.mode

    log_path = f"/tmp/mvp_conf_{mode}.log"
    def log_fn(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(str(msg) + "\n")

    log_fn(f"[{mode}] Device: {DEVICE}")
    log_fn("Attempt 3/8: Confident Residual + Online Refit + Actor Constraint")
    env_cls = GravityCheetahEnv

    if mode == "c1":
        # Baseline: Raw Sim, no confidence
        np.random.seed(SEED); torch.manual_seed(SEED)
        env = env_cls(mode="sim")
        od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
        al = float(env.action_space.high[0])
        agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE)
        buf = ReplayBuffer(od, ad, REPLAY_SIZE)
        obs, _ = env.reset(seed=SEED); curve = []
        log_fn(f"\n[Raw Sim baseline] {TRAIN_STEPS//1000}k steps")
        diag_history = []
        for step in range(1, TRAIN_STEPS+1):
            a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r, d, tr, _ = env.step(a)
            buf.add(obs, a, r, obs2, float(d and not tr))
            obs = obs2
            if d or tr: obs, _ = env.reset()
            if step >= WARMUP and buf.size >= BATCH_SIZE:
                diag = agent.update(buf)
                if step % 200 == 0: diag_history.append(diag)
            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls, "real")
                ret_s, _ = evaluate(agent, env_cls, "sim")
                curve.append((step, ret, ret_s))
                crit = np.mean([d["critic_loss"] for d in diag_history]) if diag_history else 0
                Q = np.mean([d["q_pred_mean"] for d in diag_history]) if diag_history else 0
                diag_history.clear()
                log_fn(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  sim={ret_s:7.1f}  crit={crit:.1f} Q={Q:.0f}")
        env.close()

    elif mode == "c2":
        # Confident Residual + Online Refit + Actor Constraint
        log_fn("\n[A] Initial paired data collection...")
        SA, delta_s, n_eps = collect_initial_paired(env_cls, N_COLLECT)
        log_fn(f"  {len(SA)} steps ({n_eps} eps)")

        obs_dim = delta_s.shape[1]; act_dim = env_cls.ACT_DIM
        log_fn("\n[B] Fitting confident residual model (K=5)...")
        residual = ConfidentResidualModel(obs_dim, act_dim, K=5, hidden=128,
                                          confidence_tau=0.91, device=DEVICE)
        residual.fit(SA, delta_s, n_epochs=100)

        # Test confidence on training data
        test_conf = residual.get_confidence(SA[:200, :obs_dim], SA[:200, obs_dim:])
        log_fn(f"  Train confidence: mean={test_conf.mean():.3f} std={test_conf.std():.3f} "
               f"min={test_conf.min():.3f} max={test_conf.max():.3f}")

        # Create gap_fn (gap = 1 - confidence)
        gap_fn = residual.make_confidence_fn()

        np.random.seed(SEED); torch.manual_seed(SEED)
        env = env_cls(mode="sim")
        od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
        al = float(env.action_space.high[0])
        agent = RESACAgent(od, ad, al, hidden_dim=HIDDEN, n_critics=N_CRITICS,
                           beta=BETA_LCB, lr=LR, device=DEVICE,
                           gap_fn=gap_fn, penalty_scale=CONSTRAINT_SCALE)
        buf = ReplayBuffer(od, ad, REPLAY_SIZE)
        obs, _ = env.reset(seed=SEED); curve = []

        log_fn(f"\n[C] Training with confidence IW + constraint + online refit")
        log_fn(f"  w_min={W_MIN}, constraint_scale={CONSTRAINT_SCALE}, "
               f"refit every {REFIT_INTERVAL} steps ({REFIT_SAMPLES} samples)")

        diag_history = []
        for step in range(1, TRAIN_STEPS+1):
            a = env.action_space.sample() if step < WARMUP else agent.get_action(obs, deterministic=False)
            obs2, r, d, tr, _ = env.step(a)
            buf.add(obs, a, r, obs2, float(d and not tr))
            obs = obs2
            if d or tr: obs, _ = env.reset()
            if step >= WARMUP and buf.size >= BATCH_SIZE:
                diag = agent.update(buf)
                if step % 200 == 0: diag_history.append(diag)

            # Online refit: collect new paired data + re-train residual
            if step > WARMUP and step % REFIT_INTERVAL == 0:
                log_fn(f"  [REFIT at step {step}] Collecting {REFIT_SAMPLES} on-policy paired samples...")
                SA_new, ds_new = collect_online_paired(env_cls, agent, REFIT_SAMPLES,
                                                       seed_offset=step)
                residual.add_paired_data(SA_new, ds_new)
                residual.refit(n_epochs=50)
                # Test confidence after refit
                test_conf = residual.get_confidence(
                    buf.s[:min(500, buf.size)],
                    buf.a[:min(500, buf.size)])
                log_fn(f"    Post-refit confidence on buffer: "
                       f"mean={test_conf.mean():.3f} std={test_conf.std():.3f} "
                       f"min={test_conf.min():.3f}")

            if step % EVAL_INTERVAL == 0:
                ret, std = evaluate(agent, env_cls, "real")
                ret_s, _ = evaluate(agent, env_cls, "sim")
                curve.append((step, ret, ret_s))
                # Diagnostics
                parts = []
                if diag_history:
                    avg = lambda k: np.mean([d[k] for d in diag_history if k in d])
                    parts.append(f"crit={avg('critic_loss'):.1f}")
                    parts.append(f"Q={avg('q_pred_mean'):.0f}")
                    if any("iw_mean" in d for d in diag_history):
                        parts.append(f"w={avg('iw_mean'):.3f}[{avg('iw_min'):.3f}]")
                        parts.append(f"reduct={avg('iw_reduction'):.1%}")
                    if any("conf_penalty" in d for d in diag_history):
                        parts.append(f"cpn={avg('conf_penalty'):.3f}")
                    diag_history.clear()
                diag_str = "  " + " | ".join(parts) if parts else ""
                log_fn(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f}  sim={ret_s:7.1f}{diag_str}")

        env.close()

    # Summary
    avg_real = np.mean([r for _, r, _ in curve[-3:]])
    avg_sim = np.mean([s for _, _, s in curve[-3:]])
    log_fn(f"\n{'='*50}")
    log_fn(f"{mode} last 3 avg: real={avg_real:.1f} sim={avg_sim:.1f}")
    log_fn(f"{'='*50}")


if __name__ == "__main__":
    main()
