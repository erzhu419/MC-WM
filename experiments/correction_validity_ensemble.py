"""
Correction validity: single SINDy vs SINDy Ensemble on CarpetAnt.

Key question: does ensemble disagreement prevent OOD correction damage?
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from mc_wm.residual.sindy_ensemble import SINDyEnsembleCorrector
from experiments.mvp_aero_cheetah import SINDyStateCorrector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/correction_ensemble.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def collect_parallel(env_cls, policy_fn, n_steps, single_corr, ensemble_corr, seed=SEED):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_sim, _ = sim.reset(seed=seed); os_real, _ = real.reset(seed=seed)

    raw_errs, single_errs, ens_mean_errs, ens_gated_errs = [], [], [], []
    gate_values = []
    ep = 0

    for step in range(n_steps):
        a = policy_fn(os_sim)
        ns_sim, _, ds, ts, _ = sim.step(a)
        ns_real, _, dr, tr, _ = real.step(a)

        raw_err = np.abs(ns_real - ns_sim).mean()
        raw_errs.append(raw_err)

        # Single SINDy
        SA = np.concatenate([os_sim, a]).reshape(1, -1).astype(np.float32)
        delta_single = single_corr.predict_batch(SA)[0]
        single_errs.append(np.abs(ns_real - (ns_sim + delta_single)).mean())

        # Ensemble: mean correction
        delta_ens, stds, gate = ensemble_corr.predict_with_uncertainty(os_sim, a)
        ens_mean_errs.append(np.abs(ns_real - (ns_sim + delta_ens)).mean())

        # Ensemble: GATED correction (the key test)
        ns_gated = ns_sim + gate * delta_ens
        ens_gated_errs.append(np.abs(ns_real - ns_gated).mean())
        gate_values.append(gate)

        os_sim, os_real = ns_sim, ns_real
        if ds or ts or dr or tr:
            ep += 1
            os_sim, _ = sim.reset(seed=seed+ep)
            os_real, _ = real.reset(seed=seed+ep)

    sim.close(); real.close()

    def reduction(corr, raw):
        return (1 - np.mean(corr) / max(np.mean(raw), 1e-8)) * 100

    return {
        "raw_mae": np.mean(raw_errs),
        "single_mae": np.mean(single_errs),
        "single_reduction": reduction(single_errs, raw_errs),
        "ens_mean_mae": np.mean(ens_mean_errs),
        "ens_mean_reduction": reduction(ens_mean_errs, raw_errs),
        "ens_gated_mae": np.mean(ens_gated_errs),
        "ens_gated_reduction": reduction(ens_gated_errs, raw_errs),
        "gate_mean": np.mean(gate_values),
        "gate_median": np.median(gate_values),
        "gate_frac_low": np.mean(np.array(gate_values) < 0.3),
    }


def main():
    log(f"Device: {DEVICE}")
    log("="*60)
    log("SINGLE vs ENSEMBLE SINDy — CarpetAnt")
    log("="*60)

    env_cls = CarpetAntEnv; obs_dim = 27

    # Fit both correctors on same data
    log("\n[1] Collecting paired data + fitting...")
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    SA_list, ds_list = [], []
    os_, _ = sim.reset(seed=SEED); or_, _ = real.reset(seed=SEED)
    ep = 0
    for _ in range(3000):
        a = sim.action_space.sample()
        ns, _, ds, ts, _ = sim.step(a)
        nr, _, dr, tr, _ = real.step(a)
        SA_list.append(np.concatenate([os_, a]))
        ds_list.append(nr - ns)
        os_, or_ = ns, nr
        if ds or ts or dr or tr:
            ep += 1; os_, _ = sim.reset(seed=ep+SEED); or_, _ = real.reset(seed=ep+SEED)
    sim.close(); real.close()
    SA = np.array(SA_list, dtype=np.float32)
    delta_s = np.array(ds_list, dtype=np.float32)

    single = SINDyStateCorrector(obs_dim)
    single.fit(SA, delta_s)

    # Sweep gate_tau to find optimal
    log("\n  Fitting ensemble (K=5)...")
    for tau in [0.01, 0.05, 0.1, 0.5]:
        ensemble = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=tau)
        ensemble.fit(SA, delta_s)

        # Test A: Random policy
        log(f"\n--- gate_tau = {tau} ---")
        def random_policy(obs):
            return env_cls(mode="sim").action_space.sample()

        r_a = collect_parallel(env_cls, random_policy, 1000, single, ensemble)
        log(f"[A] Random policy:")
        log(f"  Raw MAE:          {r_a['raw_mae']:.4f}")
        log(f"  Single reduction: {r_a['single_reduction']:+.1f}%")
        log(f"  Ens mean reduct:  {r_a['ens_mean_reduction']:+.1f}%")
        log(f"  Ens GATED reduct: {r_a['ens_gated_reduction']:+.1f}%")
        log(f"  Gate: mean={r_a['gate_mean']:.3f} median={r_a['gate_median']:.3f} frac_low={r_a['gate_frac_low']:.2f}")

    # Train a policy for Test B
    log("\n[2] Training policy (20k steps)...")
    env = env_cls(mode="sim")
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])
    agent = RESACAgent(od, ad, al, hidden_dim=256, n_critics=3,
                       beta=-2.0, lr=3e-4, device=DEVICE)
    buf_s = np.zeros((50000, od), np.float32)
    buf_a = np.zeros((50000, ad), np.float32)
    buf_r = np.zeros((50000, 1), np.float32)
    buf_s2 = np.zeros((50000, od), np.float32)
    buf_d = np.zeros((50000, 1), np.float32)
    buf_ptr = buf_size = 0
    class SBuf:
        @property
        def size(self): return buf_size
        def sample(self, n):
            idx = np.random.randint(0, buf_size, n)
            return (torch.FloatTensor(buf_s[idx]).to(DEVICE),
                    torch.FloatTensor(buf_a[idx]).to(DEVICE),
                    torch.FloatTensor(buf_r[idx]).to(DEVICE),
                    torch.FloatTensor(buf_s2[idx]).to(DEVICE),
                    torch.FloatTensor(buf_d[idx]).to(DEVICE))
    sbuf = SBuf()
    obs, _ = env.reset(seed=SEED)
    for step in range(1, 20001):
        a = env.action_space.sample() if step < 2000 else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf_s[buf_ptr]=obs; buf_a[buf_ptr]=a; buf_r[buf_ptr]=r
        buf_s2[buf_ptr]=obs2; buf_d[buf_ptr]=float(d and not tr)
        buf_ptr = (buf_ptr+1) % 50000; buf_size = min(buf_size+1, 50000)
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= 2000 and buf_size >= 256:
            agent.update(sbuf)
    env.close()
    log("  Policy trained")

    # Test B: Trained policy — the critical test
    def trained_policy(obs):
        return agent.get_action(obs, deterministic=True)

    log("\n[B] Trained policy (OOD):")
    for tau in [0.01, 0.05, 0.1, 0.5]:
        ensemble = SINDyEnsembleCorrector(obs_dim, K=5, gate_tau=tau)
        ensemble.fit(SA, delta_s)
        r_b = collect_parallel(env_cls, trained_policy, 1000, single, ensemble)
        log(f"\n--- gate_tau = {tau} ---")
        log(f"  Raw MAE:          {r_b['raw_mae']:.4f}")
        log(f"  Single reduction: {r_b['single_reduction']:+.1f}%  ← NO GATE")
        log(f"  Ens mean reduct:  {r_b['ens_mean_reduction']:+.1f}%  ← NO GATE")
        log(f"  Ens GATED reduct: {r_b['ens_gated_reduction']:+.1f}%  ← WITH GATE")
        log(f"  Gate: mean={r_b['gate_mean']:.3f} median={r_b['gate_median']:.3f} frac_low={r_b['gate_frac_low']:.2f}")

    log(f"\n{'='*60}")
    log("If ens_gated_reduction > 0% on trained policy → ensemble gate works")
    log("If ens_gated_reduction ≈ 0% → gate correctly shuts off OOD correction")
    log("If ens_gated_reduction << 0% → ensemble doesn't help, need different approach")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
