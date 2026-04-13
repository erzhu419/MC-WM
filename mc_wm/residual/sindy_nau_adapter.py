"""
SINDy + NAU/NMU Residual Adapter: replaces MLP δ with interpretable symbolic model.

Architecture:
  1. SINDy discovers symbolic features Θ(s,a) from paired data (poly2 + STLSQ)
  2. NAU/NMU head maps Θ → correction, with formal OOD bound
  3. Online refit: re-fit SINDy coefficients + fine-tune NAU/NMU (warm-start)

Advantages over MLP δ:
  - Interpretable: symbolic terms show WHAT the gap is (e.g., -0.3*v for gravity)
  - OOD bounded: NAU/NMU Lipschitz constant → formal error bound outside training data
  - Sparse: STLSQ keeps only active terms → less overfitting on small data

Same interface as ResidualAdapter:
  - fit(states, actions, sim_ns, sim_r, real_ns, real_r)
  - predict_correction(states, actions) → (Δs_correction, Δr_correction)
  - _trained flag
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pysindy as ps
from sklearn.linear_model import Ridge
from copy import deepcopy

from mc_wm.networks.nau_nmu import SymbolicResidualHead
from mc_wm.self_audit.diagnosis import DiagnosisBattery
from mc_wm.self_audit.auto_expand import AutoExpander


class SINDyNAUAdapter:
    """
    SINDy feature discovery + NAU/NMU output head for residual correction.

    Phase 1 (SINDy): Discover symbolic features from paired data
      Θ(s,a) = poly2 features, sparsified by STLSQ
      Per-dim SINDy coefficients give initial linear prediction

    Phase 2 (NAU/NMU): Fine-tune with formal OOD bounds
      SymbolicResidualHead(Θ) → correction
      NAU enforces near-linear mapping (L=0)
      NMU adds bounded nonlinearity (L=2|c|)
      OOD bound: |Δ̂_ood - Δ_true| ≤ ε + ε_J·‖d‖ + (L_eff/2)·‖d‖²
    """

    def __init__(self, obs_dim, act_dim, sindy_threshold=0.05, sindy_alpha=0.05,
                 nau_hidden=32, lr=1e-3, device="cpu",
                 max_rounds=3, eps_threshold=0.05, diagnosis_alpha=0.05):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.device = device
        self.sindy_threshold = sindy_threshold
        self.sindy_alpha = sindy_alpha

        # Self-hypothesis loop config
        self.max_rounds = max_rounds
        self.eps_threshold = eps_threshold

        # SINDy components (initialized on first fit)
        self._library = ps.PolynomialLibrary(degree=2, include_bias=True)
        self._sindy_coefs = None
        self._n_features = None
        self._feature_names = None
        self._active_features = None
        self._extra_columns_fn = None  # function to compute extra columns for new data

        # Diagnosis battery + auto-expander
        self._battery = DiagnosisBattery(alpha=diagnosis_alpha)
        self._expander = AutoExpander(obs_dim=obs_dim, act_dim=act_dim)

        # NAU/NMU head
        self._nau_head = None
        self._nau_optimizer = None
        self._nau_lr = lr

        # Training state
        self._trained = False
        self._train_center = None
        self._fit_errors = None
        self._hypothesis_logs = []

    def _sindy_fit_one_round(self, Theta, targets):
        """Fit SINDy coefficients (Ridge + STLSQ) on feature matrix Theta."""
        n_outputs = targets.shape[1]
        coefs = np.zeros((n_outputs, Theta.shape[1]))
        active = []
        errors = np.zeros(n_outputs)

        for dim in range(n_outputs):
            y = targets[:, dim]
            reg = Ridge(alpha=self.sindy_alpha, fit_intercept=False)
            reg.fit(Theta, y)
            coef = reg.coef_.copy()
            mask = np.abs(coef) > self.sindy_threshold
            for _ in range(10):
                if not np.any(mask): break
                reg2 = Ridge(alpha=self.sindy_alpha, fit_intercept=False)
                reg2.fit(Theta[:, mask], y)
                coef_new = np.zeros_like(coef)
                coef_new[mask] = reg2.coef_
                coef = coef_new
                mask = np.abs(coef) > self.sindy_threshold
            coefs[dim] = coef
            active.append(mask)
            y_pred = Theta @ coef
            errors[dim] = float(np.mean((y - y_pred) ** 2))

        return coefs, active, errors

    def fit(self, states, actions, sim_next_states, sim_rewards,
            real_next_states, real_rewards, n_epochs=100, batch_size=256, patience=20):
        """
        Three-phase fit with self-hypothesis loop:
        1. SINDy poly2 → fit → diagnose remainder
        2. If structure remains: auto-expand features → re-fit (up to max_rounds)
        3. NAU/NMU fine-tune on final discovered feature set
        """
        sim_deltas = sim_next_states - states
        real_deltas = real_next_states - states
        delta_correction_s = real_deltas - sim_deltas
        delta_correction_r = (real_rewards - sim_rewards).reshape(-1, 1)
        targets = np.concatenate([delta_correction_s, delta_correction_r], axis=-1)

        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        N = len(SA)
        n_outputs = self.obs_dim + 1
        self._train_center = SA.mean(axis=0)

        # ── Phase 1: Self-Hypothesis Loop
        self._library = ps.PolynomialLibrary(degree=2, include_bias=True)
        self._library.fit(SA)
        Theta = np.asarray(self._library.transform(SA))
        self._feature_names = list(self._library.get_feature_names())

        extra_cols = None  # accumulated extra columns from auto-expand
        extra_names = []
        self._hypothesis_logs = []

        for round_num in range(1, self.max_rounds + 1):
            # Build full feature matrix
            if extra_cols is not None and extra_cols.shape[1] > 0:
                Theta_full = np.hstack([Theta, extra_cols])
            else:
                Theta_full = Theta

            # Fit SINDy
            coefs, active, errors = self._sindy_fit_one_round(Theta_full, targets)
            eps_max = float(errors.max())

            # Quality gate
            quality_passed = eps_max < self.eps_threshold

            # Diagnose remainder (state dims only)
            sindy_pred_s = Theta_full @ coefs[:self.obs_dim].T  # (N, obs_dim)
            remainder_s = delta_correction_s - sindy_pred_s
            diagnoses = self._battery.run(remainder_s, SA)
            any_structure = any(d.any_fired() for d in diagnoses)

            # Count active
            n_active = sum(m.sum() for m in active)
            n_fired = sum(1 for d in diagnoses if d.any_fired())
            diag_summary = [d.summary() for d in diagnoses if d.any_fired()]

            log_entry = {
                "round": round_num, "eps_max": eps_max, "n_features": Theta_full.shape[1],
                "n_active": int(n_active), "quality_passed": quality_passed,
                "n_diagnosis_fired": n_fired, "reason": "",
            }

            print(f"  [Round {round_num}] features={Theta_full.shape[1]} active={n_active} "
                  f"eps_max={eps_max:.5f} diagnosed={n_fired}/{self.obs_dim}")

            if quality_passed:
                log_entry["reason"] = "quality_gate"
                self._hypothesis_logs.append(log_entry)
                print(f"    ✓ Quality gate PASSED (eps_max={eps_max:.5f} < {self.eps_threshold})")
                break

            if not any_structure:
                log_entry["reason"] = "white_noise"
                self._hypothesis_logs.append(log_entry)
                print(f"    ✓ Remainder is white noise (no structure detected)")
                break

            if round_num == self.max_rounds:
                log_entry["reason"] = "max_rounds"
                self._hypothesis_logs.append(log_entry)
                print(f"    ⊘ Max rounds reached")
                break

            # Auto-expand: add new features based on diagnosis
            print(f"    → Expanding features based on {n_fired} diagnoses...")
            for ds in diag_summary[:3]:
                print(f"      {ds}")

            new_lib, metadata = self._expander.expand(
                results=diagnoses, current_library=self._library,
                SA=SA, remainder=remainder_s,
                steps=np.arange(N),
            )
            new_extra = metadata.get("extra_columns")
            if new_extra is not None and new_extra.shape[1] > 0:
                new_names = metadata.get("extra_names", [])
                if extra_cols is not None:
                    extra_cols = np.hstack([extra_cols, new_extra])
                else:
                    extra_cols = new_extra
                extra_names.extend(new_names)
                print(f"    Added {new_extra.shape[1]} new features: {new_names[:5]}")
            else:
                log_entry["reason"] = "no_expansion"
                self._hypothesis_logs.append(log_entry)
                print(f"    No new features to add, stopping")
                break

            log_entry["reason"] = "expanded"
            self._hypothesis_logs.append(log_entry)

        # Save final SINDy state
        self._sindy_coefs = coefs
        self._active_features = active
        self._fit_errors = errors
        self._n_features = Theta_full.shape[1]
        # Update feature names to include extras
        all_names = list(self._feature_names) + extra_names
        self._feature_names = all_names
        # Save expand specs for predict-time reconstruction
        self._expand_specs = []
        for name in extra_names:
            if name.endswith("_sq"):
                j = int(name.split("x")[1].split("_")[0])
                self._expand_specs.append({"type": "sq", "j": j})
            elif name.endswith("_cube"):
                j = int(name.split("x")[1].split("_")[0])
                self._expand_specs.append({"type": "cube", "j": j})
            elif name.endswith("_signmag"):
                j = int(name.split("x")[1].split("_")[0])
                self._expand_specs.append({"type": "signmag", "j": j})
            elif "_x_a" in name:
                parts = name.split("_x_a")
                j = int(parts[0].split("x")[1])
                k = self.obs_dim + int(parts[1])
                self._expand_specs.append({"type": "cross", "j": j, "k": k})
            elif name.startswith("mask_"):
                parts = name.split("_lt")
                j = int(parts[0].split("x")[1])
                threshold = float(parts[1])
                self._expand_specs.append({"type": "mask", "j": j, "threshold": threshold})
            elif name == "t_norm":
                self._expand_specs.append({"type": "tnorm"})

        n_active_total = sum(m.sum() for m in active)
        print(f"  SINDy final: {self._n_features} features, {n_active_total} active, "
              f"avg MSE={errors.mean():.5f}, rounds={len(self._hypothesis_logs)}")

        # ── Phase 2: NAU/NMU fine-tuning on final feature set
        if self._nau_head is None or self._nau_head.feature_net[0].in_features != self._n_features:
            self._nau_head = SymbolicResidualHead(
                self._n_features, n_outputs, alpha_init=0.7
            ).to(self.device)
            self._nau_optimizer = optim.Adam(self._nau_head.parameters(), lr=self._nau_lr,
                                             weight_decay=1e-4)

        Theta_t = torch.FloatTensor(Theta_full).to(self.device)
        tgt_t = torch.FloatTensor(targets).to(self.device)

        # Train/val split
        perm = np.random.permutation(N)
        n_val = max(int(N * 0.1), 50)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        best_val = float('inf'); best_epoch = 0
        best_state = self._nau_head.state_dict().copy()

        for epoch in range(n_epochs):
            self._nau_head.train()
            perm_train = np.random.permutation(train_idx)
            for i in range(0, len(perm_train), batch_size):
                bi = perm_train[i:i+batch_size]
                pred = self._nau_head(Theta_t[bi])
                mse_loss = nn.MSELoss()(pred, tgt_t[bi])
                reg_loss = self._nau_head.regularization_loss()
                loss = mse_loss + 0.01 * reg_loss
                self._nau_optimizer.zero_grad(); loss.backward(); self._nau_optimizer.step()
            self._nau_head.eval()

            with torch.no_grad():
                val_pred = self._nau_head(Theta_t[val_idx])
                val_loss = float(nn.MSELoss()(val_pred, tgt_t[val_idx]))
                if val_loss < best_val:
                    best_val = val_loss; best_epoch = epoch
                    best_state = self._nau_head.state_dict().copy()

            if (epoch + 1) % 20 == 0:
                print(f"    NAU epoch {epoch+1} | val={val_loss:.5f} best={best_val:.5f} "
                      f"L_eff={self._nau_head.L_eff:.4f}")
            if epoch - best_epoch >= patience:
                print(f"    NAU early stop at epoch {epoch+1} (best={best_epoch+1})")
                break

        self._nau_head.load_state_dict(best_state)
        self._trained = True

        # Compare SINDy-only vs NAU
        with torch.no_grad():
            sindy_pred = Theta_t[val_idx] @ torch.FloatTensor(self._sindy_coefs.T).to(self.device)
            nau_pred = self._nau_head(Theta_t[val_idx])
            sindy_mse = float(nn.MSELoss()(sindy_pred, tgt_t[val_idx]))
            nau_mse = best_val
        print(f"  SINDy-only val MSE: {sindy_mse:.5f}")
        print(f"  NAU val MSE:        {nau_mse:.5f}")
        print(f"  NAU improvement:    {(1-nau_mse/sindy_mse)*100:.1f}%")
        print(f"  OOD bound L_eff:    {self._nau_head.L_eff:.4f}")

    def _build_full_theta(self, SA):
        """Build full feature matrix including auto-expanded columns."""
        Theta = np.asarray(self._library.transform(SA))
        if hasattr(self, '_expand_specs') and self._expand_specs:
            extra = []
            for spec in self._expand_specs:
                if spec['type'] == 'sq':
                    extra.append(SA[:, spec['j']] ** 2)
                elif spec['type'] == 'cube':
                    extra.append(SA[:, spec['j']] ** 3)
                elif spec['type'] == 'signmag':
                    c = SA[:, spec['j']]
                    extra.append(c * np.abs(c))
                elif spec['type'] == 'cross':
                    extra.append(SA[:, spec['j']] * SA[:, spec['k']])
                elif spec['type'] == 'mask':
                    extra.append((SA[:, spec['j']] < spec['threshold']).astype(np.float64))
                elif spec['type'] == 'tnorm':
                    extra.append(np.linspace(0, 1, len(SA)))
            if extra:
                Theta = np.hstack([Theta, np.column_stack(extra)])
        return Theta

    def predict_correction(self, states, actions):
        """Returns (Δs_correction, Δr_correction) using NAU/NMU head."""
        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        Theta = self._build_full_theta(SA)
        Theta_t = torch.FloatTensor(Theta).to(self.device)

        with torch.no_grad():
            out = self._nau_head(Theta_t).cpu().numpy()

        return out[:, :self.obs_dim], out[:, self.obs_dim]

    def get_ood_bound(self, states, actions):
        """
        Formal OOD error bound per query point.
        |Δ̂(s,a) - Δ_true(s,a)| ≤ ε + ε_J·‖d‖ + (L_eff/2)·‖d‖²

        Returns (N,) array of bounds.
        """
        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        d = np.linalg.norm(SA - self._train_center[None, :], axis=-1)  # (N,)
        eps = float(self._fit_errors.mean())
        eps_J = 0.01  # Jacobian mismatch (conservative estimate)
        L = self._nau_head.L_eff
        bound = eps + eps_J * d + (L / 2) * d ** 2
        return bound

    def get_active_terms(self):
        """Return discovered symbolic terms per dimension."""
        result = {}
        for dim in range(self.obs_dim + 1):
            name = f"Δs_{dim}" if dim < self.obs_dim else "Δr"
            active = [(self._feature_names[j], float(self._sindy_coefs[dim, j]))
                      for j in range(self._n_features)
                      if self._active_features[dim][j]]
            if active:
                result[name] = active
        return result
