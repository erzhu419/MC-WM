"""
Trajectory-level step-by-step diagnosis.

Runs sim and real in lockstep for N steps, records EVERYTHING per step:
  - s_sim, s_real, a (raw states and action)
  - Δs_true = s_real_next - s_sim_next
  - Δs_sindy = SINDy prediction
  - s_corrected = s_sim_next + Δs_sindy
  - correction_error = |s_corrected - s_real_next|
  - raw_error = |s_sim_next - s_real_next|
  - improvement = raw_error - correction_error (positive = correction helps)
  - gap_signal = ||Δs_sindy||²
  - ensemble_std per dim
  - ensemble gate value
  - QΔ(s, a) prediction (if QΔ trained)

All written to log, step by step.

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
N_TRAJ_STEPS = 50
VEL_DIMS = list(range(13, 27))

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def main():
    log("="*80)
    log("TRAJECTORY-LEVEL STEP-BY-STEP DIAGNOSIS — CarpetAnt")
    log("="*80)

    env_cls = CarpetAntEnv
    obs_dim = 27; act_dim = 8

    # ── 1. Fit SINDy ensemble on paired data
    log("\n[1] Fitting SINDy ensemble on 3000 paired steps...")
    sim_fit = env_cls(mode="sim"); real_fit = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_s, _ = sim_fit.reset(seed=SEED); os_r, _ = real_fit.reset(seed=SEED)
    ep = 0
    for _ in range(3000):
        a = sim_fit.action_space.sample()
        ns_s, _, ds, ts, _ = sim_fit.step(a)
        ns_r, _, dr, tr, _ = real_fit.step(a)
        SA_list.append(np.concatenate([os_s, a]))
        ds_list.append(ns_r - ns_s)
        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim_fit.reset(seed=ep+SEED); os_r, _ = real_fit.reset(seed=ep+SEED)
    sim_fit.close(); real_fit.close()
    SA = np.array(SA_list, np.float32)
    delta_s = np.array(ds_list, np.float32)

    corrector = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.1)
    corrector.fit(SA, delta_s)
    log(f"  Done. RMSE reduction: {corrector.correction_coverage(SA, delta_s)['rmse_reduction_pct']:.1f}%")

    # ── 2. Train a QΔ on the gap signal (quick, 5k steps)
    log("\n[2] Training QΔ (5k steps in sim for gap signal calibration)...")
    q_delta = QDeltaModule(obs_dim, act_dim, hidden_dim=128, lr=3e-4,
                           gamma=0.99, tau=5e-3, penalty_scale=0.1, device=DEVICE)

    sim_qd = env_cls(mode="sim")
    agent_qd = RESACAgent(obs_dim, act_dim, float(sim_qd.action_space.high[0]),
                           hidden_dim=256, n_critics=3, beta=-2.0, lr=3e-4, device=DEVICE)
    obs_qd, _ = sim_qd.reset(seed=SEED)
    # Quick buffer
    buf_s = np.zeros((10000, obs_dim), np.float32)
    buf_a = np.zeros((10000, act_dim), np.float32)
    buf_s2 = np.zeros((10000, obs_dim), np.float32)
    buf_d = np.zeros((10000, 1), np.float32)
    bp = bs = 0
    for step in range(5000):
        a = sim_qd.action_space.sample() if step < 1000 else agent_qd.get_action(obs_qd, deterministic=False)
        obs2, r, d, tr, _ = sim_qd.step(a)
        buf_s[bp]=obs_qd; buf_a[bp]=a; buf_s2[bp]=obs2; buf_d[bp]=float(d and not tr)
        bp = (bp+1) % 10000; bs = min(bs+1, 10000)
        # Train QΔ on gap signal
        if bs >= 64:
            idx = np.random.randint(0, bs, 64)
            s_t = torch.FloatTensor(buf_s[idx]).to(DEVICE)
            a_t = torch.FloatTensor(buf_a[idx]).to(DEVICE)
            s2_t = torch.FloatTensor(buf_s2[idx]).to(DEVICE)
            d_t = torch.FloatTensor(buf_d[idx]).to(DEVICE)
            a2_t = torch.randn_like(a_t) * 0.1  # dummy next action
            # Gap reward from SINDy
            sa_np = np.concatenate([buf_s[idx], buf_a[idx]], axis=-1).astype(np.float32)
            delta_pred = corrector.predict_batch(sa_np)
            gap_r = torch.FloatTensor(np.mean(delta_pred**2, axis=-1, keepdims=True)).to(DEVICE)
            q_delta.update(s_t, a_t, s2_t, a2_t, d_t, gap_r)
        obs_qd = obs2
        if d or tr: obs_qd, _ = sim_qd.reset()
    sim_qd.close()
    log("  QΔ trained (5k steps)")

    # ── 3. Run trajectory diagnosis: sim + real in lockstep
    log(f"\n[3] TRAJECTORY DIAGNOSIS ({N_TRAJ_STEPS} steps)")
    log("  Using random policy (same as SINDy training distribution)")

    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=99); os_r, _ = real.reset(seed=99)

    # Header
    log(f"\n  {'step':>4s} | {'raw_err':>8s} {'corr_err':>8s} {'improv':>8s} | "
        f"{'gap_sig':>8s} {'ens_std':>8s} {'gate':>6s} | "
        f"{'QΔ':>8s} {'penalty':>8s} | "
        f"{'Δs_true':>8s} {'Δs_pred':>8s} {'pred/true':>9s}")
    log("  " + "-"*115)

    cumul_raw = cumul_corr = 0.0
    for step in range(N_TRAJ_STEPS):
        a = sim.action_space.sample()
        ns_s, _, ds, ts, _ = sim.step(a)
        ns_r, _, dr, tr, _ = real.step(a)

        # True residual
        delta_true = ns_r - ns_s

        # SINDy prediction + uncertainty
        delta_pred, stds, gate = corrector.predict_with_uncertainty(os_s, a)

        # Corrected state
        ns_corr = ns_s + gate * delta_pred

        # Errors (averaged over vel dims only — non-vel dims have zero gap)
        raw_err = np.abs(ns_r[VEL_DIMS] - ns_s[VEL_DIMS]).mean()
        corr_err = np.abs(ns_r[VEL_DIMS] - ns_corr[VEL_DIMS]).mean()
        improvement = raw_err - corr_err

        # Gap signal
        SA_pt = np.concatenate([os_s, a]).reshape(1, -1).astype(np.float32)
        gap_signal = float(np.mean(corrector.predict_batch(SA_pt)**2))

        # Ensemble std (mean over vel dims)
        ens_std_mean = float(stds[VEL_DIMS].mean())

        # QΔ value
        s_t = torch.FloatTensor(os_s).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(a).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            qd_val = float(q_delta.q_delta(s_t, a_t).item())
        qd_penalty = qd_val * 0.1

        # Δs magnitude (vel dims)
        ds_true_mag = float(np.abs(delta_true[VEL_DIMS]).mean())
        ds_pred_mag = float(np.abs(delta_pred[VEL_DIMS]).mean())
        ratio = ds_pred_mag / max(ds_true_mag, 1e-8)

        cumul_raw += raw_err
        cumul_corr += corr_err

        log(f"  {step:4d} | {raw_err:8.4f} {corr_err:8.4f} {improvement:+8.4f} | "
            f"{gap_signal:8.4f} {ens_std_mean:8.4f} {gate:6.3f} | "
            f"{qd_val:8.4f} {qd_penalty:8.4f} | "
            f"{ds_true_mag:8.4f} {ds_pred_mag:8.4f} {ratio:9.2f}")

        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            os_s, _ = sim.reset(seed=99+step); os_r, _ = real.reset(seed=99+step)
            log(f"  ---- EPISODE RESET at step {step} ----")

    sim.close(); real.close()

    # Summary
    log(f"\n  {'='*60}")
    log(f"  Cumulative raw error:       {cumul_raw:.4f}")
    log(f"  Cumulative corrected error: {cumul_corr:.4f}")
    pct = (1 - cumul_corr/max(cumul_raw, 1e-8)) * 100
    log(f"  Correction improvement:     {pct:+.1f}%")
    log(f"  {'='*60}")

    # ── 4. Same but with trained policy
    log(f"\n[4] TRAJECTORY DIAGNOSIS with TRAINED POLICY (20k steps)")
    log("  Training policy in sim...")
    env_train = env_cls(mode="sim")
    agent = RESACAgent(obs_dim, act_dim, float(env_train.action_space.high[0]),
                       hidden_dim=256, n_critics=3, beta=-2.0, lr=3e-4, device=DEVICE)
    from experiments.mvp_qdelta import ReplayBuffer
    buf = ReplayBuffer(obs_dim, act_dim, 50000)
    obs, _ = env_train.reset(seed=SEED)
    for step in range(1, 20001):
        a_t = env_train.action_space.sample() if step < 2000 else agent.get_action(obs, deterministic=False)
        obs2, r_t, d_t, tr_t, _ = env_train.step(a_t)
        buf.add(obs, a_t, r_t, obs2, float(d_t and not tr_t))
        obs = obs2
        if d_t or tr_t: obs, _ = env_train.reset()
        if step >= 2000 and buf.size >= 256: agent.update(buf)
    env_train.close()
    log("  Policy trained (20k)")

    sim2 = env_cls(mode="sim"); real2 = env_cls(mode="real")
    os_s, _ = sim2.reset(seed=77); os_r, _ = real2.reset(seed=77)

    log(f"\n  {'step':>4s} | {'raw_err':>8s} {'corr_err':>8s} {'improv':>8s} | "
        f"{'gap_sig':>8s} {'gate':>6s} | "
        f"{'QΔ':>8s} {'penalty':>8s} | "
        f"{'Δs_true':>8s} {'Δs_pred':>8s} {'pred/true':>9s}")
    log("  " + "-"*105)

    cumul_raw2 = cumul_corr2 = 0.0
    for step in range(N_TRAJ_STEPS):
        a = agent.get_action(os_s, deterministic=True)
        ns_s, _, ds, ts, _ = sim2.step(a)
        ns_r, _, dr, tr, _ = real2.step(a)

        delta_true = ns_r - ns_s
        delta_pred, stds, gate = corrector.predict_with_uncertainty(os_s, a)
        ns_corr = ns_s + gate * delta_pred

        raw_err = np.abs(ns_r[VEL_DIMS] - ns_s[VEL_DIMS]).mean()
        corr_err = np.abs(ns_r[VEL_DIMS] - ns_corr[VEL_DIMS]).mean()
        improvement = raw_err - corr_err

        SA_pt = np.concatenate([os_s, a]).reshape(1, -1).astype(np.float32)
        gap_signal = float(np.mean(corrector.predict_batch(SA_pt)**2))

        s_t = torch.FloatTensor(os_s).unsqueeze(0).to(DEVICE)
        a_t_tensor = torch.FloatTensor(a).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            qd_val = float(q_delta.q_delta(s_t, a_t_tensor).item())

        ds_true_mag = float(np.abs(delta_true[VEL_DIMS]).mean())
        ds_pred_mag = float(np.abs(delta_pred[VEL_DIMS]).mean())
        ratio = ds_pred_mag / max(ds_true_mag, 1e-8)

        cumul_raw2 += raw_err
        cumul_corr2 += corr_err

        log(f"  {step:4d} | {raw_err:8.4f} {corr_err:8.4f} {improvement:+8.4f} | "
            f"{gap_signal:8.4f} {gate:6.3f} | "
            f"{qd_val:8.4f} {qd_val*0.1:8.4f} | "
            f"{ds_true_mag:8.4f} {ds_pred_mag:8.4f} {ratio:9.2f}")

        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            os_s, _ = sim2.reset(seed=77+step); os_r, _ = real2.reset(seed=77+step)
            log(f"  ---- EPISODE RESET at step {step} ----")

    sim2.close(); real2.close()

    log(f"\n  {'='*60}")
    log(f"  [Trained] Cumul raw error:       {cumul_raw2:.4f}")
    log(f"  [Trained] Cumul corrected error: {cumul_corr2:.4f}")
    pct2 = (1 - cumul_corr2/max(cumul_raw2, 1e-8)) * 100
    log(f"  [Trained] Correction improvement: {pct2:+.1f}%")

    log(f"\n  COMPARISON:")
    log(f"    Random policy correction:  {pct:+.1f}%")
    log(f"    Trained policy correction: {pct2:+.1f}%")
    log(f"    Generalization drop:       {pct - pct2:.1f} pp")
    log(f"  {'='*60}")


if __name__ == "__main__":
    main()
