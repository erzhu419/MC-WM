"""
Step-level residual verification:
1. Reset sim and real to same seed
2. Execute same action in both
3. Record raw obs difference per step
4. Check if SINDy predicts it correctly
5. Print per-dim, per-step data

This is the ground truth test — no trajectory divergence, no accumulation.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector
from experiments.mvp_aero_cheetah import SINDyStateCorrector

SEED = 42
LOG = "/tmp/residual_stepwise.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def main():
    log("="*70)
    log("STEP-LEVEL RESIDUAL VERIFICATION — CarpetAnt")
    log("="*70)

    env_cls = CarpetAntEnv
    obs_dim = 27; act_dim = 8

    # ── Step 1: Collect paired data for SINDy fitting
    log("\n[1] Collecting paired data (synchronized reset, same actions)...")
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_sim, _ = sim.reset(seed=SEED); os_real, _ = real.reset(seed=SEED)
    ep = 0
    for i in range(3000):
        a = sim.action_space.sample()
        ns_sim, _, ds, ts, _ = sim.step(a)
        ns_real, _, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_sim, a]))
        ds_list.append(ns_real - ns_sim)
        os_sim, os_real = ns_sim, ns_real
        if ds or ts or dr or tr:
            ep += 1; os_sim, _ = sim.reset(seed=ep+SEED); os_real, _ = real.reset(seed=ep+SEED)
    sim.close(); real.close()
    SA = np.array(SA_list, np.float32)
    delta_s = np.array(ds_list, np.float32)
    log(f"  {len(SA)} steps, {ep} episodes")

    # ── Step 2: Raw residual analysis (before any model)
    log("\n[2] RAW RESIDUAL ANALYSIS (Δs = s_real - s_sim per step)")
    log(f"  {'dim':>4s} {'mean':>10s} {'std':>10s} {'|mean|':>10s} {'max|Δ|':>10s} {'description':>20s}")
    VEL_DIMS = list(range(13, 27))
    for d in range(obs_dim):
        col = delta_s[:, d]
        desc = "velocity" if d in VEL_DIMS else ("z_height" if d == 0 else "other")
        log(f"  {d:4d} {col.mean():10.5f} {col.std():10.5f} {np.abs(col).mean():10.5f} {np.abs(col).max():10.5f} {desc:>20s}")

    # ── Step 3: Check theoretical prediction
    # CarpetAnt real: obs[vel_dims] *= 0.7
    # So Δs_i = 0.7*v - v = -0.3*v for vel dims
    # Δs_i = 0 for non-vel dims
    log("\n[3] THEORETICAL CHECK: Δs should be -0.3*v for vel dims, 0 for others")
    for d in VEL_DIMS[:3]:  # first 3 vel dims as sample
        v_sim = SA[:, d]  # velocity in sim obs (part of SA = [obs, action])
        predicted_delta = -0.3 * v_sim
        actual_delta = delta_s[:, d]
        error = np.abs(actual_delta - predicted_delta)
        log(f"  dim {d}: theoretical -0.3*v | actual Δs | error: mean={error.mean():.6f} max={error.max():.6f}")

    # ── Step 4: Fit SINDy and check per-step predictions
    log("\n[4] SINDY FIT + PER-STEP PREDICTION CHECK")
    single = SINDyStateCorrector(obs_dim)
    single.fit(SA, delta_s)

    ensemble = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=0.01)
    ensemble.fit(SA, delta_s)

    # Predict on training data
    pred_single = single.predict_batch(SA)
    pred_ens = ensemble.predict_batch(SA)

    log(f"\n  Per-dim prediction error on training data:")
    log(f"  {'dim':>4s} {'|Δs|':>10s} {'single_err':>12s} {'ens_err':>12s} {'single_%':>10s} {'ens_%':>10s}")
    for d in range(obs_dim):
        true_mag = np.abs(delta_s[:, d]).mean()
        single_err = np.abs(delta_s[:, d] - pred_single[:, d]).mean()
        ens_err = np.abs(delta_s[:, d] - pred_ens[:, d]).mean()
        s_pct = single_err / max(true_mag, 1e-8) * 100
        e_pct = ens_err / max(true_mag, 1e-8) * 100
        marker = "✓" if s_pct < 30 else ("~" if s_pct < 70 else "✗")
        log(f"  {d:4d} {true_mag:10.5f} {single_err:12.5f} {ens_err:12.5f} {s_pct:9.1f}% {e_pct:9.1f}% {marker}")

    # ── Step 5: Print first 10 steps raw data for manual inspection
    log("\n[5] FIRST 10 STEPS — RAW DATA (vel dim 13 = vx)")
    d = 13  # vx
    log(f"  {'step':>4s} {'s_sim':>10s} {'s_real':>10s} {'Δs_true':>10s} {'Δs_sindy':>10s} {'Δs_ens':>10s} {'-0.3*v':>10s} {'err':>10s}")
    for i in range(min(10, len(SA))):
        v_sim = SA[i, d]
        ds_true = delta_s[i, d]
        ds_pred = pred_single[i, d]
        ds_ens = pred_ens[i, d]
        theory = -0.3 * v_sim
        err = ds_true - ds_pred
        log(f"  {i:4d} {v_sim:10.5f} {v_sim+ds_true:10.5f} {ds_true:10.5f} {ds_pred:10.5f} {ds_ens:10.5f} {theory:10.5f} {err:10.5f}")

    # ── Step 6: Ensemble disagreement analysis
    log("\n[6] ENSEMBLE DISAGREEMENT on training data")
    _, all_stds, gates = ensemble.predict_batch_with_uncertainty(SA)
    log(f"  Gate distribution: mean={gates.mean():.4f} median={np.median(gates):.4f} "
        f"p10={np.percentile(gates, 10):.4f} p90={np.percentile(gates, 90):.4f}")
    log(f"  Per-dim avg disagreement (std across 5 models):")
    for d in range(obs_dim):
        s = all_stds[:, d].mean()
        tag = "vel" if d in VEL_DIMS else ""
        log(f"    dim {d:2d}: std={s:.6f} {tag}")

    # ── Step 7: What gate value would be needed?
    log("\n[7] REQUIRED GATE FOR FULL CORRECTION")
    log("  With gate=g, corrected_obs = obs_sim + g * sindy_prediction")
    log("  For perfect correction, need g=1.0")
    log(f"  Current gate at tau=0.01: median={np.median(gates):.4f}")
    log(f"  Current gate at tau=0.05: median={1.0/(1.0+all_stds.mean(1)/0.05).mean():.4f}")
    log(f"  Current gate at tau=0.10: median={1.0/(1.0+all_stds.mean(1)/0.10).mean():.4f}")
    log(f"  Current gate at tau=0.50: median={1.0/(1.0+all_stds.mean(1)/0.50).mean():.4f}")
    log(f"  Current gate at tau=1.00: median={1.0/(1.0+all_stds.mean(1)/1.00).mean():.4f}")


if __name__ == "__main__":
    main()
