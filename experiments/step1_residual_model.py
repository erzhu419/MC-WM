"""
Step 1: Train and validate a residual model Δ(s,a) = s_real_next - s_sim_next.

No RL, no policy, no correction, no gap detection.
Just: can an MLP ensemble learn the dynamics gap with sufficient data?

Both sim and real are MuJoCo simulators (unlimited data).
Use random policy to collect paired data, scale up from 3k to 100k.

Validate:
  1. Training loss convergence
  2. Holdout prediction accuracy (per-dim RMSE)
  3. Generalization to trained-policy distribution
  4. Ensemble disagreement calibration
  5. Prediction accuracy vs data size (3k, 10k, 30k, 100k)

GravityCheetah: sim=2x gravity, real=1x gravity.

运行: conda run -n MC-WM python3 -u experiments/step1_residual_model.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LOG = "/tmp/step1_residual.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


class ResidualMLP(nn.Module):
    """Single MLP: (s, a) → Δs."""
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class ResidualEnsemble:
    """K MLPs with bootstrap training."""

    def __init__(self, obs_dim, act_dim, K=5, hidden=256, device="cpu"):
        self.K = K
        self.device = device
        self.input_dim = obs_dim + act_dim
        self.output_dim = obs_dim
        self.models = [ResidualMLP(self.input_dim, obs_dim, hidden).to(device) for _ in range(K)]
        self.optimizers = [optim.Adam(m.parameters(), lr=1e-3) for m in self.models]

    def fit(self, SA_train, ds_train, SA_val, ds_val,
            n_epochs=200, batch_size=512, subsample_ratio=0.8):
        """Train with validation tracking."""
        N = len(SA_train)
        SA_t = torch.FloatTensor(SA_train).to(self.device)
        ds_t = torch.FloatTensor(ds_train).to(self.device)
        SA_v = torch.FloatTensor(SA_val).to(self.device)
        ds_v = torch.FloatTensor(ds_val).to(self.device)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            epoch_train_loss = 0.0
            for k, (model, opt) in enumerate(zip(self.models, self.optimizers)):
                n_sub = int(N * subsample_ratio)
                idx = np.random.choice(N, n_sub, replace=True)
                model.train()
                perm = np.random.permutation(n_sub)
                batch_loss = 0.0; n_batches = 0
                for i in range(0, n_sub, batch_size):
                    bi = idx[perm[i:i+batch_size]]
                    pred = model(SA_t[bi])
                    loss = nn.MSELoss()(pred, ds_t[bi])
                    opt.zero_grad(); loss.backward(); opt.step()
                    batch_loss += float(loss); n_batches += 1
                epoch_train_loss += batch_loss / max(n_batches, 1)
                model.eval()

            train_losses.append(epoch_train_loss / self.K)

            # Validation
            with torch.no_grad():
                preds = torch.stack([m(SA_v) for m in self.models])  # (K, N_val, obs)
                val_mse = float(((preds.mean(0) - ds_v) ** 2).mean())
                val_losses.append(val_mse)

            if (epoch + 1) % 50 == 0:
                log(f"    epoch {epoch+1:3d} | train={train_losses[-1]:.5f} val={val_losses[-1]:.5f}")

        return train_losses, val_losses

    def predict(self, SA):
        """Ensemble mean prediction."""
        SA_t = torch.FloatTensor(SA).to(self.device)
        with torch.no_grad():
            preds = torch.stack([m(SA_t) for m in self.models])  # (K, N, obs)
            return preds.mean(0).cpu().numpy(), preds.std(0).cpu().numpy()


def collect_paired(env_cls, n_steps, policy_fn=None, seed=42):
    """Collect paired (sim, real) data. policy_fn=None → random policy."""
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    SA, ds, rewards_sim, rewards_real = [], [], [], []
    ep = 0
    for _ in range(n_steps):
        a = sim.action_space.sample() if policy_fn is None else policy_fn(os_s)
        ns_s, rs, d_s, t_s, _ = sim.step(a)
        ns_r, rr, d_r, t_r, _ = real.step(a)
        SA.append(np.concatenate([os_s, a]))
        ds.append(ns_r - ns_s)
        rewards_sim.append(rs); rewards_real.append(rr)
        os_s, os_r = ns_s, ns_r
        if d_s or t_s or d_r or t_r:
            ep += 1; os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()
    return (np.array(SA, np.float32), np.array(ds, np.float32),
            np.array(rewards_sim), np.array(rewards_real), ep)


def evaluate_model(model, SA, ds_true, label):
    """Comprehensive evaluation of residual model."""
    pred_mean, pred_std = model.predict(SA)

    # Per-dim RMSE
    per_dim_rmse = np.sqrt(np.mean((ds_true - pred_mean) ** 2, axis=0))
    per_dim_true_mag = np.sqrt(np.mean(ds_true ** 2, axis=0))
    per_dim_reduction = 1 - per_dim_rmse / np.maximum(per_dim_true_mag, 1e-8)

    # Overall
    overall_rmse = np.sqrt(np.mean((ds_true - pred_mean) ** 2))
    overall_true = np.sqrt(np.mean(ds_true ** 2))
    overall_reduction = (1 - overall_rmse / max(overall_true, 1e-8)) * 100

    # Rank correlation (per sample: is predicted magnitude correlated with true magnitude?)
    true_gap_per_sample = np.mean(ds_true ** 2, axis=1)
    pred_gap_per_sample = np.mean(pred_mean ** 2, axis=1)
    rho, _ = spearmanr(true_gap_per_sample, pred_gap_per_sample)

    # Sign accuracy per dim (does prediction have correct direction?)
    sign_acc = np.mean(np.sign(ds_true) == np.sign(pred_mean), axis=0)

    # Disagreement calibration: does high disagreement = high error?
    pred_err_per_sample = np.mean((ds_true - pred_mean) ** 2, axis=1)
    disagree_per_sample = np.mean(pred_std, axis=1)
    rho_dis, _ = spearmanr(disagree_per_sample, pred_err_per_sample)

    log(f"\n  [{label}] N={len(SA)}")
    log(f"  Overall RMSE reduction: {overall_reduction:.1f}%")
    log(f"  Rank correlation (gap magnitude): {rho:.3f}")
    log(f"  Disagreement vs error correlation: {rho_dis:.3f}")
    log(f"  Per-dim results:")
    log(f"    {'dim':>4s} {'|Δtrue|':>8s} {'RMSE':>8s} {'reduct%':>8s} {'sign_acc':>8s}")
    for d in range(len(per_dim_rmse)):
        if per_dim_true_mag[d] > 0.001:  # skip zero-gap dims
            log(f"    {d:4d} {per_dim_true_mag[d]:8.4f} {per_dim_rmse[d]:8.4f} "
                f"{per_dim_reduction[d]*100:7.1f}% {sign_acc[d]:8.2f}")

    return {
        "overall_reduction": overall_reduction,
        "rank_corr": rho,
        "disagree_corr": rho_dis,
        "per_dim_reduction": per_dim_reduction,
    }


def main():
    log("="*70)
    log("STEP 1: RESIDUAL MODEL VALIDATION")
    log("Can an MLP ensemble learn Δs = s_real - s_sim?")
    log("="*70)
    log(f"Device: {DEVICE}")
    log(f"Env: GravityCheetah (sim=2x gravity, real=1x)")

    env_cls = GravityCheetahEnv

    # ── Collect data at multiple scales
    log("\n[1] Collecting paired data at multiple scales...")
    datasets = {}
    for n in [3000, 10000, 30000, 100000]:
        SA, ds, rs, rr, ep = collect_paired(env_cls, n, seed=SEED)
        datasets[n] = (SA, ds)
        log(f"  N={n:>6d}: {ep} episodes, |Δs| mean={np.abs(ds).mean():.4f}, "
            f"reward gap mean={np.abs(rr-rs).mean():.4f}")

    # ── Train and evaluate at each scale
    log("\n[2] Training residual models at each scale...")
    results = {}
    for n in [3000, 10000, 30000, 100000]:
        SA, ds = datasets[n]
        # 80/20 split
        split = int(0.8 * n)
        SA_train, SA_val = SA[:split], SA[split:]
        ds_train, ds_val = ds[:split], ds[split:]

        log(f"\n  --- N={n} (train={split}, val={n-split}) ---")
        model = ResidualEnsemble(17, 6, K=5, hidden=256, device=DEVICE)
        train_losses, val_losses = model.fit(SA_train, ds_train, SA_val, ds_val,
                                              n_epochs=200, batch_size=512)

        # Evaluate on validation set
        res = evaluate_model(model, SA_val, ds_val, f"N={n} holdout")
        results[n] = res

        # If N=100k, also evaluate on trained-policy distribution
        if n == 100000:
            log("\n  [3] Evaluating on TRAINED POLICY distribution...")
            # Quick-train a policy in sim
            from mc_wm.policy.resac_agent import RESACAgent
            env = env_cls(mode="sim")
            agent = RESACAgent(17, 6, float(env.action_space.high[0]),
                               hidden_dim=256, n_critics=3, beta=-2.0, lr=3e-4, device=DEVICE)
            # Minimal replay buffer
            buf_s = np.zeros((20000, 17), np.float32)
            buf_a = np.zeros((20000, 6), np.float32)
            bp = bs = 0
            obs, _ = env.reset(seed=SEED)
            class MiniBuf:
                @property
                def size(self): return bs
                def sample(self, n):
                    idx = np.random.randint(0, bs, n)
                    return tuple(torch.FloatTensor(x[idx]).to(DEVICE) for x in
                                 [buf_s, buf_a, np.zeros((20000,1),np.float32),
                                  buf_s, np.zeros((20000,1),np.float32)])
            mbuf = MiniBuf()
            for step in range(20000):
                a = env.action_space.sample() if step < 2000 else agent.get_action(obs, deterministic=False)
                obs2, r, d, tr, _ = env.step(a)
                buf_s[bp] = obs; buf_a[bp] = a
                bp = (bp+1) % 20000; bs = min(bs+1, 20000)
                obs = obs2
                if d or tr: obs, _ = env.reset()
                if step >= 2000 and bs >= 256: agent.update(mbuf)
            env.close()
            log("    Policy trained (20k steps)")

            # Collect paired data with trained policy
            SA_tp, ds_tp, _, _, ep_tp = collect_paired(
                env_cls, 5000, policy_fn=lambda s: agent.get_action(s, deterministic=False), seed=99)
            log(f"    Trained-policy paired data: {len(SA_tp)} steps, {ep_tp} episodes")
            evaluate_model(model, SA_tp, ds_tp, "trained-policy (N=100k model)")

            # Compare: model trained on 3k vs 100k, evaluated on trained-policy
            model_3k = ResidualEnsemble(17, 6, K=5, hidden=256, device=DEVICE)
            SA3, ds3 = datasets[3000]
            model_3k.fit(SA3[:2400], ds3[:2400], SA3[2400:], ds3[2400:], n_epochs=200)
            evaluate_model(model_3k, SA_tp, ds_tp, "trained-policy (N=3k model)")

    # ── Summary: RMSE reduction vs data size
    log(f"\n{'='*70}")
    log("SUMMARY: How much data does the residual model need?")
    log(f"{'='*70}")
    log(f"  {'N':>8s} {'RMSE_reduct':>12s} {'Rank_corr':>10s} {'Disagree_corr':>14s}")
    for n in [3000, 10000, 30000, 100000]:
        r = results[n]
        log(f"  {n:>8d} {r['overall_reduction']:11.1f}% {r['rank_corr']:10.3f} {r['disagree_corr']:14.3f}")


if __name__ == "__main__":
    main()
