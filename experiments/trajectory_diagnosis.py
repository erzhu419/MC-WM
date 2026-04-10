"""
Trajectory-level step-by-step diagnosis — saves EVERYTHING per step.

For EACH step in a 50-step trajectory:
  States:     s_sim (27d), s_real (27d), s_corrected (27d)
  Action:     a (8d)
  Rewards:    r_sim, r_real
  Residual:   Δs_true (27d), Δs_predicted (27d)
  Ensemble:   per_model_pred (5×27d), ensemble_std (27d)
  Gate:       gate_value (scalar)
  QΔ:         q_delta_value, penalty, gap_signal
  Distance:   sa_dist_to_center (scalar)
  Errors:     raw_error (27d), corrected_error (27d)

Saved as npz + printed to log.
Two passes: random policy + trained policy.

运行: conda run -n MC-WM python3 -u experiments/trajectory_diagnosis.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector
from mc_wm.policy.q_delta import QDeltaModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/trajectory_diagnosis.log"
NPZ_DIR = "/home/erzhu419/mine_code/MC-WM/experiments/diag_data"
N_STEPS = 50
VEL_DIMS = list(range(13, 27))

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def run_trajectory(sim, real, policy_fn, corrector, q_delta, train_center, n_steps, label, seed):
    """Run sim+real in lockstep, record everything per step."""
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    obs_dim = os_s.shape[0]

    # Per-step storage
    data = {
        "s_sim": [], "s_real": [], "s_corrected": [],
        "ns_sim": [], "ns_real": [],
        "action": [],
        "r_sim": [], "r_real": [],
        "delta_true": [], "delta_pred": [],         # (27d) each
        "per_model_pred": [],                        # (5, 27d)
        "ensemble_std": [],                          # (27d)
        "gate": [],
        "gap_signal": [],
        "q_delta": [], "penalty": [],
        "sa_dist": [],
        "raw_error": [], "corrected_error": [],      # (27d) each
        "done_sim": [], "done_real": [],
    }

    log(f"\n  [{label}] {n_steps} steps, seed={seed}")
    log(f"  {'step':>4s} | {'raw_err':>8s} {'corr_err':>8s} {'improv':>8s} | "
        f"{'gap':>7s} {'gate':>6s} {'QΔ':>7s} {'pen':>7s} | "
        f"{'|Δtrue|':>8s} {'|Δpred|':>8s} {'ratio':>6s} | "
        f"{'r_sim':>7s} {'r_real':>7s} {'dist':>7s}")
    log("  " + "-"*120)

    for step in range(n_steps):
        a = policy_fn(os_s)
        ns_s, rs, ds, ts, _ = sim.step(a)
        ns_r, rr, dr, tr, _ = real.step(a)

        # True residual
        delta_true = ns_r - ns_s

        # Ensemble prediction — get ALL model predictions
        SA_1 = np.concatenate([os_s, a]).reshape(1, -1).astype(np.float32)
        Theta = corrector._get_theta(SA_1)
        per_model = np.zeros((5, obs_dim))
        for k, model in enumerate(corrector.models):
            preds = model.predict_ensemble(Theta)  # (K, 1)
            per_model[k, :] = 0  # init
        # Actually use predict_with_uncertainty for clean API
        delta_pred, stds, gate = corrector.predict_with_uncertainty(os_s, a)

        # Per-model predictions (each of K=5 models)
        per_model_all = np.zeros((5, obs_dim))
        for dim_i, model in enumerate(corrector.models):
            preds_k = model.predict_ensemble(Theta)  # (K, 1) for this dim
            per_model_all[:, dim_i] = preds_k[:, 0]

        # Corrected state
        ns_corr = ns_s + gate * delta_pred

        # Gap signal
        gap_signal = float(np.mean(delta_pred ** 2))

        # SA distance to training center
        sa_dist = float(np.linalg.norm(SA_1[0] - train_center)) if train_center is not None else 0.0

        # QΔ
        s_t = torch.FloatTensor(os_s).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(a).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            qd_val = float(q_delta.q_delta(s_t, a_t).item())
        penalty = qd_val * 0.1

        # Per-dim errors
        raw_err = np.abs(ns_r - ns_s)
        corr_err = np.abs(ns_r - ns_corr)

        # Store
        data["s_sim"].append(os_s.copy())
        data["s_real"].append(os_r.copy())
        data["s_corrected"].append(ns_corr.copy())
        data["ns_sim"].append(ns_s.copy())
        data["ns_real"].append(ns_r.copy())
        data["action"].append(a.copy())
        data["r_sim"].append(rs)
        data["r_real"].append(rr)
        data["delta_true"].append(delta_true.copy())
        data["delta_pred"].append(delta_pred.copy())
        data["per_model_pred"].append(per_model_all.copy())
        data["ensemble_std"].append(stds.copy())
        data["gate"].append(gate)
        data["gap_signal"].append(gap_signal)
        data["q_delta"].append(qd_val)
        data["penalty"].append(penalty)
        data["sa_dist"].append(sa_dist)
        data["raw_error"].append(raw_err.copy())
        data["corrected_error"].append(corr_err.copy())
        data["done_sim"].append(float(ds or ts))
        data["done_real"].append(float(dr or tr))

        # Print summary line
        re_vel = raw_err[VEL_DIMS].mean()
        ce_vel = corr_err[VEL_DIMS].mean()
        dt_mag = np.abs(delta_true[VEL_DIMS]).mean()
        dp_mag = np.abs(delta_pred[VEL_DIMS]).mean()
        ratio = dp_mag / max(dt_mag, 1e-8)
        log(f"  {step:4d} | {re_vel:8.4f} {ce_vel:8.4f} {re_vel-ce_vel:+8.4f} | "
            f"{gap_signal:7.4f} {gate:6.3f} {qd_val:7.4f} {penalty:7.4f} | "
            f"{dt_mag:8.4f} {dp_mag:8.4f} {ratio:6.2f} | "
            f"{rs:7.2f} {rr:7.2f} {sa_dist:7.2f}")

        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            os_s, _ = sim.reset(seed=seed+step+1); os_r, _ = real.reset(seed=seed+step+1)
            log(f"  ---- RESET at step {step} ----")

    # Summary
    raw_total = sum(np.abs(d[VEL_DIMS]).mean() for d in data["raw_error"])
    corr_total = sum(np.abs(d[VEL_DIMS]).mean() for d in data["corrected_error"])
    pct = (1 - corr_total / max(raw_total, 1e-8)) * 100
    log(f"\n  [{label}] Summary: raw={raw_total:.2f} corr={corr_total:.2f} improvement={pct:+.1f}%")
    log(f"  Mean gate={np.mean(data['gate']):.4f}  Mean QΔ={np.mean(data['q_delta']):.4f}  "
        f"Mean gap={np.mean(data['gap_signal']):.4f}  Mean dist={np.mean(data['sa_dist']):.2f}")

    # Convert to arrays
    arrays = {}
    for k, v in data.items():
        arrays[k] = np.array(v)
    return arrays


def main():
    os.makedirs(NPZ_DIR, exist_ok=True)
    log("="*80)
    log("TRAJECTORY DIAGNOSIS — CarpetAnt (full per-step data)")
    log("="*80)

    env_cls = CarpetAntEnv
    obs_dim = 27; act_dim = 8

    # 1. Fit SINDy
    log("\n[1] SINDy ensemble fit...")
    sim_fit = env_cls(mode="sim"); real_fit = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_s, _ = sim_fit.reset(seed=SEED); os_r, _ = real_fit.reset(seed=SEED); ep = 0
    for _ in range(3000):
        a = sim_fit.action_space.sample()
        ns_s, _, ds, ts, _ = sim_fit.step(a); ns_r, _, dr, tr, _ = real_fit.step(a)
        SA_list.append(np.concatenate([os_s, a])); ds_list.append(ns_r - ns_s)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim_fit.reset(seed=ep+SEED); os_r, _ = real_fit.reset(seed=ep+SEED)
    sim_fit.close(); real_fit.close()
    SA = np.array(SA_list, np.float32); delta_s = np.array(ds_list, np.float32)
    train_center = SA.mean(axis=0)

    corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.1)
    corrector.fit(SA, delta_s)
    log(f"  RMSE reduction: {corrector.correction_coverage(SA, delta_s)['rmse_reduction_pct']:.1f}%")

    # 2. Train QΔ (quick)
    log("\n[2] QΔ training (5k steps)...")
    q_delta = QDeltaModule(obs_dim, act_dim, hidden_dim=128, lr=3e-4,
                           gamma=0.99, tau=5e-3, penalty_scale=0.1, device=DEVICE)
    sim_qd = env_cls(mode="sim")
    obs_qd, _ = sim_qd.reset(seed=SEED)
    buf_s = np.zeros((10000, obs_dim), np.float32); buf_a = np.zeros((10000, act_dim), np.float32)
    buf_s2 = np.zeros((10000, obs_dim), np.float32); buf_d = np.zeros((10000, 1), np.float32)
    bp = bs = 0
    for step in range(5000):
        a = sim_qd.action_space.sample()
        obs2, _, d, tr, _ = sim_qd.step(a)
        buf_s[bp]=obs_qd; buf_a[bp]=a; buf_s2[bp]=obs2; buf_d[bp]=float(d and not tr)
        bp = (bp+1) % 10000; bs = min(bs+1, 10000)
        if bs >= 64:
            idx = np.random.randint(0, bs, 64)
            s_t = torch.FloatTensor(buf_s[idx]).to(DEVICE)
            a_t = torch.FloatTensor(buf_a[idx]).to(DEVICE)
            s2_t = torch.FloatTensor(buf_s2[idx]).to(DEVICE)
            d_t = torch.FloatTensor(buf_d[idx]).to(DEVICE)
            a2_t = torch.randn_like(a_t) * 0.1
            sa_np = np.concatenate([buf_s[idx], buf_a[idx]], axis=-1).astype(np.float32)
            dp = corrector.predict_batch(sa_np)
            gap_r = torch.FloatTensor(np.mean(dp**2, axis=-1, keepdims=True)).to(DEVICE)
            q_delta.update(s_t, a_t, s2_t, a2_t, d_t, gap_r)
        obs_qd = obs2
        if d or tr: obs_qd, _ = sim_qd.reset()
    sim_qd.close()
    log("  Done")

    # 3. Random policy trajectory
    log("\n[3] RANDOM POLICY trajectory")
    sim3 = env_cls(mode="sim"); real3 = env_cls(mode="real")
    random_data = run_trajectory(
        sim3, real3, lambda s: sim3.action_space.sample(),
        corrector, q_delta, train_center, N_STEPS, "Random", seed=99)
    sim3.close(); real3.close()
    np.savez(os.path.join(NPZ_DIR, "traj_random.npz"), **random_data)
    log(f"  Saved → {NPZ_DIR}/traj_random.npz")

    # 4. Train policy
    log("\n[4] Training policy (20k steps)...")
    env_train = env_cls(mode="sim")
    agent = RESACAgent(obs_dim, act_dim, float(env_train.action_space.high[0]),
                       hidden_dim=256, n_critics=3, beta=-2.0, lr=3e-4, device=DEVICE)
    from experiments.mvp_qdelta import ReplayBuffer
    buf = ReplayBuffer(obs_dim, act_dim, 50000)
    obs, _ = env_train.reset(seed=SEED)
    for step in range(1, 20001):
        a = env_train.action_space.sample() if step < 2000 else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env_train.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env_train.reset()
        if step >= 2000 and buf.size >= 256: agent.update(buf)
    env_train.close()
    log("  Done")

    # 5. Trained policy trajectory
    log("\n[5] TRAINED POLICY trajectory")
    sim5 = env_cls(mode="sim"); real5 = env_cls(mode="real")
    trained_data = run_trajectory(
        sim5, real5, lambda s: agent.get_action(s, deterministic=True),
        corrector, q_delta, train_center, N_STEPS, "Trained", seed=77)
    sim5.close(); real5.close()
    np.savez(os.path.join(NPZ_DIR, "traj_trained.npz"), **trained_data)
    log(f"  Saved → {NPZ_DIR}/traj_trained.npz")

    # 6. Comparison
    log(f"\n{'='*60}")
    log("COMPARISON")
    log(f"{'='*60}")
    for name, d in [("Random", random_data), ("Trained", trained_data)]:
        raw_t = sum(np.abs(e[VEL_DIMS]).mean() for e in d["raw_error"])
        corr_t = sum(np.abs(e[VEL_DIMS]).mean() for e in d["corrected_error"])
        pct = (1 - corr_t / max(raw_t, 1e-8)) * 100
        log(f"  {name:10s}: correction improvement={pct:+.1f}%  "
            f"gate_mean={d['gate'].mean():.3f}  QΔ_mean={d['q_delta'].mean():.3f}  "
            f"dist_mean={d['sa_dist'].mean():.1f}")

    log(f"\nPer-step data saved in {NPZ_DIR}/")
    log("  traj_random.npz: random policy trajectory (all 27 dims per step)")
    log("  traj_trained.npz: trained policy trajectory (all 27 dims per step)")
    log("  Fields: s_sim, s_real, s_corrected, ns_sim, ns_real, action,")
    log("          r_sim, r_real, delta_true, delta_pred, per_model_pred,")
    log("          ensemble_std, gate, gap_signal, q_delta, penalty,")
    log("          sa_dist, raw_error, corrected_error, done_sim, done_real")


if __name__ == "__main__":
    main()
