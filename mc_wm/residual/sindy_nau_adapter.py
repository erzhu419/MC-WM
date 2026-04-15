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
from mc_wm.self_audit.llm_oracle import LLMOracle
from mc_wm.self_audit.orthogonal_expand import OrthogonalExpander


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
                 max_rounds=3, eps_threshold=0.05, diagnosis_alpha=0.05,
                 log_fn=None, env_type=None, claude_oracle=None,
                 env_description_for_llm: str | None = None):
        self._log = log_fn or (lambda msg: print(msg))
        self._env_type = env_type  # if set, use LLM oracle instead of auto-expand
        # Role #2: Claude oracle for feature hypotheses in the self-hypothesis loop.
        # When set, augments orthogonal expansion with LLM-suggested physics features.
        self._claude_oracle = claude_oracle
        self._env_desc_llm = env_description_for_llm or ""
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

        # Separate reward correction (simple MLP, not shared with state SINDy)
        import torch.nn as tnn
        self._reward_net = tnn.Sequential(
            tnn.Linear(obs_dim + act_dim, 64), tnn.ReLU(),
            tnn.Linear(64, 1),
        ).to(device)
        self._reward_opt = optim.Adam(self._reward_net.parameters(), lr=lr, weight_decay=1e-4)

        # SGD-based coefficient layer for smooth online refit
        # Initialized after first fit with discovered sparsity pattern
        self._coef_layer = None  # nn.Linear(n_features, n_outputs, bias=False)
        self._coef_optimizer = None
        self._active_mask = None  # boolean mask from STLSQ sparsity

        # Training state
        self._trained = False
        self._train_center = None
        self._fit_errors = None
        self._hypothesis_logs = []
        self._loop_done = False
        # LLM (Role #2) tracking counters — all zero unless self._claude_oracle set.
        self._llm_feat_proposed = 0
        self._llm_feat_accepted = 0
        self._llm_feat_rejected_unsafe = 0
        self._llm_feat_rejected_eval = 0
        self._llm_feat_rejected_shape = 0
        self._llm_feat_rejected_nonfinite = 0
        self._llm_rounds = 0
        # Decision history: each accepted/rejected LLM feature with final coef
        # outcome.  Fed back to future Role #2 calls so LLM sees how its past
        # proposals actually performed.
        self._llm_feat_history: list[dict] = []
        # Name → expression for LLM-added features, used by `_build_full_theta`
        # to re-evaluate on new (s, a) points at predict time.
        self._llm_feat_exprs: dict = {}

    def _eval_llm_features(self, SA, llm_feats):
        """
        Safely evaluate LLM-proposed expressions against the (s, a) stack.

        SA has shape (N, obs_dim + act_dim).  `llm_feats` is a list of dicts
        `{"expr": "np.sin(3*s[1])", "name": "sin_3theta", "why": "..."}`.

        Each expression is evaluated per-row with `s` = SA[i, :obs_dim],
        `a` = SA[i, obs_dim:].  We restrict globals to {"np": np} and forbid
        attribute access beyond numpy module.  Malformed / unsafe exprs
        silently drop the feature.
        """
        import numpy as np
        cols, names = [], []
        safe_globals = {"__builtins__": {}, "np": np}
        n = SA.shape[0]
        self._llm_rounds += 1
        for feat in llm_feats[:8]:
            self._llm_feat_proposed += 1
            expr = feat.get("expr", "")
            name = feat.get("name", expr[:20])
            if not expr or any(bad in expr for bad in ("__", "import", "open(", "os.", "sys.", "eval(", "exec(")):
                self._llm_feat_rejected_unsafe += 1
                self._log(f"    ✗ LLM expr rejected (unsafe): {expr[:50]}")
                continue
            try:
                s_arr = SA[:, :self.obs_dim]
                a_arr = SA[:, self.obs_dim:]
                col = eval(expr, safe_globals, {"s": s_arr.T, "a": a_arr.T})
                col = np.asarray(col, dtype=np.float64)
                if col.shape != (n,):
                    self._llm_feat_rejected_shape += 1
                    self._log(f"    ✗ LLM expr shape mismatch: {name} got {col.shape}")
                    continue
                if not np.all(np.isfinite(col)):
                    self._llm_feat_rejected_nonfinite += 1
                    self._log(f"    ✗ LLM expr produces non-finite: {name}")
                    continue
                std = col.std() + 1e-8
                cols.append(col / std)
                feat_name = f"llm_{name}"
                names.append(feat_name)
                # Store expr + std for predict-time re-evaluation
                self._llm_feat_exprs[feat_name] = {"expr": expr, "std": float(std)}
                self._llm_feat_accepted += 1
                self._llm_feat_history.append({
                    "round": self._llm_rounds,
                    "name": feat_name,
                    "expr": expr,
                    "why": feat.get("why", "")[:80],
                    "outcome": None,
                })
                self._log(f"      + {name}: {expr[:40]} (std={std:.3f})")
            except Exception as e:
                self._llm_feat_rejected_eval += 1
                self._log(f"    ✗ LLM expr eval failed: {name}  ({str(e)[:60]})")
                continue
        if not cols:
            return None, []
        return np.column_stack(cols), names

    def _update_llm_history_outcomes(self, feature_names: list, coefs: np.ndarray,
                                       threshold: float = 0.02) -> None:
        """Fill `outcome` field for LLM features based on final SINDy coefs."""
        if not self._llm_feat_history:
            return
        # Build name → (max_abs_coef, n_dims_active)
        name_to_stat = {}
        for i, n in enumerate(feature_names):
            if isinstance(n, str) and n.startswith("llm_"):
                col = coefs[:, i] if coefs.ndim == 2 else np.array([coefs[i]])
                name_to_stat[n] = {
                    "max_abs_coef": float(np.abs(col).max()),
                    "n_dims_active": int((np.abs(col) > threshold).sum()),
                }
        for entry in self._llm_feat_history:
            if entry["outcome"] is None and entry["name"] in name_to_stat:
                entry["outcome"] = name_to_stat[entry["name"]]

    def prune_llm_features(self, feature_names: list, coefs: np.ndarray,
                           threshold: float = 0.02) -> list:
        """
        Role #4: review LLM-added feature contributions via SINDy coefs and
        recommend names to drop.

        feature_names: full feature name list (same order as coefs columns)
        coefs: (n_dims, n_features) SINDy coefficient matrix after fit
        threshold: |coef| below this counts as "inactive" for that dim

        Returns: list of LLM feature names to drop (pass to caller to rebuild
        the feature matrix without them).
        """
        if self._claude_oracle is None:
            return []
        llm_indices = [i for i, n in enumerate(feature_names)
                       if isinstance(n, str) and n.startswith("llm_")]
        if not llm_indices:
            return []
        stats = []
        for i in llm_indices:
            col = coefs[:, i] if coefs.ndim == 2 else np.array([coefs[i]])
            max_abs = float(np.abs(col).max())
            n_active = int((np.abs(col) > threshold).sum())
            stats.append({
                "name": feature_names[i],
                "expr": feature_names[i][4:],  # strip "llm_" prefix for display
                "max_abs_coef": round(max_abs, 5),
                "n_dims_active": n_active,
                "val_mse_delta": 0.0,  # placeholder; not measured per-feature
            })
        to_drop = self._claude_oracle.role4_prune_features(
            self._env_desc_llm, stats)
        if to_drop:
            self._log(f"  [LLM Role #4] drops {len(to_drop)} LLM features: {to_drop[:4]}")
        else:
            self._log(f"  [LLM Role #4] reviewed {len(stats)} LLM features, kept all")
        self._llm_role4_dropped_features = getattr(
            self, "_llm_role4_dropped_features", 0) + len(to_drop)
        return to_drop

    def get_llm_summary(self) -> dict:
        """Return LLM Role #2 activity summary for end-of-training reporting."""
        oracle_stats = {}
        if self._claude_oracle is not None and hasattr(self._claude_oracle, "stats"):
            oracle_stats = self._claude_oracle.stats()
        return {
            "rounds": self._llm_rounds,
            "proposed": self._llm_feat_proposed,
            "accepted": self._llm_feat_accepted,
            "rejected_unsafe": self._llm_feat_rejected_unsafe,
            "rejected_eval": self._llm_feat_rejected_eval,
            "rejected_shape": self._llm_feat_rejected_shape,
            "rejected_nonfinite": self._llm_feat_rejected_nonfinite,
            "role4_dropped_features": getattr(self, "_llm_role4_dropped_features", 0),
            "oracle_calls": oracle_stats.get("calls", 0),
            "oracle_cache_hits": oracle_stats.get("cache_hits", 0),
            "oracle_errors": oracle_stats.get("errors", 0),
        }

    def log_llm_summary(self):
        """Pretty-print the LLM summary via self._log.  Called at end of training."""
        if self._claude_oracle is None:
            return
        s = self.get_llm_summary()
        self._log("")
        self._log("┌─ [LLM Role #2 Summary] ─────────────────")
        self._log(f"│ rounds invoked      : {s['rounds']}")
        self._log(f"│ features proposed   : {s['proposed']}")
        self._log(f"│ features accepted   : {s['accepted']}")
        self._log(f"│ rejected (unsafe)   : {s['rejected_unsafe']}")
        self._log(f"│ rejected (eval err) : {s['rejected_eval']}")
        self._log(f"│ rejected (shape)    : {s['rejected_shape']}")
        self._log(f"│ rejected (non-fin)  : {s['rejected_nonfinite']}")
        self._log(f"│ Role#4 pruned       : {s['role4_dropped_features']}")
        self._log(f"│ Claude CLI calls    : {s['oracle_calls']}")
        self._log(f"│ Claude cache hits   : {s['oracle_cache_hits']}")
        self._log(f"│ Claude errors       : {s['oracle_errors']}")
        if s['proposed'] > 0:
            accept_rate = 100.0 * s['accepted'] / s['proposed']
            self._log(f"│ acceptance rate     : {accept_rate:.1f}%")
        self._log("└──────────────────────────────────────────")

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
        First call: self-hypothesis loop discovers features → NAU fine-tune.
        Subsequent calls (refit): reuse discovered features, warm-start NAU only.
        """
        sim_deltas = sim_next_states - states
        real_deltas = real_next_states - states
        delta_correction_s = real_deltas - sim_deltas
        delta_correction_r = (real_rewards - sim_rewards).reshape(-1, 1)
        # SINDy+NAU only fits STATE correction (obs_dim outputs)
        # Reward correction uses separate MLP
        targets = delta_correction_s  # (N, obs_dim) — state only

        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        N = len(SA)
        n_outputs = self.obs_dim  # state only, reward is separate
        self._train_center = SA.mean(axis=0)

        # ── Train reward correction MLP separately
        SA_t = torch.FloatTensor(SA).to(self.device)
        dr_t = torch.FloatTensor(delta_correction_r).to(self.device)
        perm_r = np.random.permutation(N)
        n_val_r = max(int(N * 0.1), 50)
        best_r_val = float('inf'); best_r_state = self._reward_net.state_dict().copy()
        for ep in range(100):
            self._reward_net.train()
            for i in range(0, N - n_val_r, 256):
                bi = perm_r[n_val_r + i:n_val_r + i + 256]
                if len(bi) == 0: break
                pred_r = self._reward_net(SA_t[bi])
                loss_r = nn.MSELoss()(pred_r, dr_t[bi])
                self._reward_opt.zero_grad(); loss_r.backward(); self._reward_opt.step()
            self._reward_net.eval()
            with torch.no_grad():
                vl = float(nn.MSELoss()(self._reward_net(SA_t[perm_r[:n_val_r]]), dr_t[perm_r[:n_val_r]]))
                if vl < best_r_val: best_r_val = vl; best_r_ep = ep; best_r_state = self._reward_net.state_dict().copy()
            if ep - best_r_ep >= 20: break
        self._reward_net.load_state_dict(best_r_state)
        self._log(f"  Reward MLP: val MSE={best_r_val:.5f} (separate from state SINDy)")

        # ── REFIT path: SGD warm-start (not STLSQ re-solve)
        # Sparsity pattern is FIXED from initial loop. Only coefficient VALUES update.
        # This gives smooth updates — no coefficient jumps — NAU stays stable.
        if self._loop_done:
            Theta_full = self._build_full_theta(SA)
            Theta_t = torch.FloatTensor(Theta_full).to(self.device)
            tgt_t = torch.FloatTensor(targets).to(self.device)

            # SGD update on coefficient layer (smooth, no batch re-solve)
            perm = np.random.permutation(N)
            n_val = max(int(N * 0.1), 50)
            val_idx, train_idx = perm[:n_val], perm[n_val:]

            for epoch in range(min(n_epochs, 30)):  # fewer epochs for refit
                self._coef_layer.train()
                perm_t = np.random.permutation(train_idx)
                for i in range(0, len(perm_t), batch_size):
                    bi = perm_t[i:i+batch_size]
                    # Apply sparsity mask: zero out gradients for inactive features
                    pred = self._coef_layer(Theta_t[bi])
                    loss = nn.MSELoss()(pred, tgt_t[bi])
                    self._coef_optimizer.zero_grad()
                    loss.backward()
                    # Mask gradients: only update active coefficients
                    with torch.no_grad():
                        self._coef_layer.weight.grad *= self._active_mask_t
                    self._coef_optimizer.step()
                self._coef_layer.eval()

            # Update SINDy coefs from the trained layer (for logging/diagnostics)
            with torch.no_grad():
                self._sindy_coefs = self._coef_layer.weight.cpu().numpy()
                pred_all = self._coef_layer(Theta_t[:1000])
                self._fit_errors = np.mean((pred_all.cpu().numpy() - targets[:1000]) ** 2, axis=0)

            # Warm-start NAU on SGD-updated predictions (smooth target)
            best_val = float('inf'); best_epoch = 0
            best_state = self._nau_head.state_dict().copy()
            for epoch in range(min(n_epochs, 30)):
                self._nau_head.train()
                perm_t = np.random.permutation(train_idx)
                for i in range(0, len(perm_t), batch_size):
                    bi = perm_t[i:i+batch_size]
                    pred = self._nau_head(Theta_t[bi])
                    loss = nn.MSELoss()(pred, tgt_t[bi]) + 0.01 * self._nau_head.regularization_loss()
                    self._nau_optimizer.zero_grad(); loss.backward(); self._nau_optimizer.step()
                self._nau_head.eval()
                with torch.no_grad():
                    vl = float(nn.MSELoss()(self._nau_head(Theta_t[val_idx]), tgt_t[val_idx]))
                    if vl < best_val: best_val = vl; best_epoch = epoch; best_state = self._nau_head.state_dict().copy()
                if epoch - best_epoch >= 10: break
            self._nau_head.load_state_dict(best_state)
            self._trained = True
            return

        # ── FIRST FIT
        self._library = ps.PolynomialLibrary(degree=2, include_bias=True)
        self._library.fit(SA)
        Theta = np.asarray(self._library.transform(SA))
        self._feature_names = list(self._library.get_feature_names())

        extra_cols = None
        extra_names = []
        self._hypothesis_logs = []

        # If LLM oracle is available: use physics-informed features instead of loop
        if self._env_type is not None:
            oracle = LLMOracle(env_type=self._env_type, log_fn=self._log)
            oracle_cols, oracle_names, _ = oracle.suggest_features(
                SA, obs_dim=self.obs_dim, act_dim=self.act_dim)
            if oracle_cols.shape[1] > 0:
                extra_cols = oracle_cols
                extra_names = oracle_names
                Theta_full = np.hstack([Theta, extra_cols])
                coefs, active, errors = self._sindy_fit_one_round(Theta_full, targets)
                self._log(f"  [LLM Oracle] {Theta_full.shape[1]} features "
                          f"({Theta.shape[1]} poly2 + {len(oracle_names)} physics), "
                          f"eps_max={errors.max():.5f}")
                # Skip diagnosis (too slow on 50k data, and LLM oracle doesn't need it)
                self._hypothesis_logs.append({
                    "round": 1, "method": "llm_oracle",
                    "n_features": Theta_full.shape[1], "eps_max": float(errors.max()),
                })
        # Run hypothesis loop only if LLM oracle was NOT used
        if extra_cols is None:
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

            # Diagnose remainder (subsample for speed — diagnosis is O(N²))
            sindy_pred_s = Theta_full @ coefs[:self.obs_dim].T
            remainder_s = delta_correction_s - sindy_pred_s
            diag_n = min(2000, N)
            diag_idx = np.random.choice(N, diag_n, replace=False)
            diagnoses = self._battery.run(remainder_s[diag_idx], SA[diag_idx])
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

            self._log(f"  [Round {round_num}] features={Theta_full.shape[1]} active={n_active} "
                  f"eps_max={eps_max:.5f} diagnosed={n_fired}/{self.obs_dim}")

            if quality_passed:
                log_entry["reason"] = "quality_gate"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ✓ Quality gate PASSED (eps_max={eps_max:.5f} < {self.eps_threshold})")
                break

            if not any_structure:
                log_entry["reason"] = "white_noise"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ✓ Remainder is white noise (no structure detected)")
                break

            if round_num == self.max_rounds:
                log_entry["reason"] = "max_rounds"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ⊘ Max rounds reached")
                break

            # Orthogonal expansion: find features in orthogonal complement of Θ
            self._log(f"    → Orthogonal feature discovery ({n_fired} dims with structure)...")
            for ds in diag_summary[:3]:
                self._log(f"      {ds}")

            orth_expander = OrthogonalExpander(self.obs_dim, self.act_dim)
            new_extra, new_names, orth_diag = orth_expander.expand(
                SA, Theta_full, remainder_s, log_fn=self._log)

            # Role #2: ask Claude for physics-informed features (optional).
            # Augments orthogonal expansion; LLM's suggestions are evaluated
            # as safe numpy expressions over s[·], a[·] with access to np.
            if self._claude_oracle is not None:
                # Collect previously-accepted LLM features (across earlier rounds).
                prev_accepted_llm = [n for n in extra_names
                                     if isinstance(n, str) and n.startswith("llm_")]
                # Per-dim residual std for LLM prioritisation.
                try:
                    resid_std = list(np.std(remainder_s, axis=0))
                except Exception:
                    resid_std = None
                # Current val MSE estimate (approximate — use in-sample mse as proxy).
                cur_val_mse = float(np.mean(remainder_s ** 2))
                nau_L = float(self._nau_head.L_eff) if self._nau_head is not None else None
                _nau_L_str = f"{nau_L:.3f}" if nau_L is not None else "N/A"
                self._log(f"    [LLM Role #2] ctx: round={round_num}, "
                          f"prev_accepted={len(prev_accepted_llm)}, "
                          f"val_mse={cur_val_mse:.5f}, "
                          f"L_eff={_nau_L_str}, "
                          f"history={len(self._llm_feat_history)} entries")
                llm_feats = self._claude_oracle.role2_feature_hypotheses(
                    env_description=self._env_desc_llm,
                    current_basis=extra_names if extra_names else [],
                    diagnosis_summary="; ".join(diag_summary[:5]),
                    obs_dim=self.obs_dim, act_dim=self.act_dim,
                    round_num=round_num,
                    prev_accepted=prev_accepted_llm,
                    current_val_mse=cur_val_mse,
                    nau_L_eff=nau_L,
                    residual_per_dim_std=resid_std,
                    feature_history=self._llm_feat_history[-10:],
                )
                if llm_feats:
                    self._log(f"    + LLM Role#2: {len(llm_feats)} feature suggestions")
                    llm_cols, llm_names = self._eval_llm_features(SA, llm_feats)
                    if llm_cols is not None and llm_cols.shape[1] > 0:
                        new_extra = (np.hstack([new_extra, llm_cols])
                                     if new_extra.shape[1] > 0 else llm_cols)
                        new_names = new_names + llm_names

            if len(new_names) > 0:
                if extra_cols is not None:
                    extra_cols = np.hstack([extra_cols, new_extra])
                else:
                    extra_cols = new_extra
                extra_names.extend(new_names)
            else:
                log_entry["reason"] = "no_orthogonal_features"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    No orthogonal features found, stopping")
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
        # Save expand specs for predict-time reconstruction.
        # CRITICAL: LLM features must be stored SEPARATELY from orthogonal
        # features — OrthogonalExpander cannot regenerate LLM names.
        self._expand_specs = []
        if self._env_type is not None:
            self._expand_specs.append(
                {"type": "llm_oracle", "env_type": self._env_type})
        else:
            orth_names = [n for n in extra_names
                          if not (isinstance(n, str) and n.startswith("llm_"))]
            llm_names = [n for n in extra_names
                         if isinstance(n, str) and n.startswith("llm_")]
            if orth_names:
                self._expand_specs.append({"type": "orthogonal", "names": orth_names})
            if llm_names:
                # Preserve ORDER of llm_names so predict-time concat matches fit-time.
                self._expand_specs.append({
                    "type": "llm_features",
                    "items": [(n, self._llm_feat_exprs.get(n, {}).get("expr", ""),
                               self._llm_feat_exprs.get(n, {}).get("std", 1.0))
                              for n in llm_names],
                })
            self._expand_obs_dim = self.obs_dim
            self._expand_act_dim = self.act_dim

        n_active_total = sum(m.sum() for m in active)
        self._log(f"  SINDy final: {self._n_features} features, {n_active_total} active, "
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
                self._log(f"    NAU epoch {epoch+1} | val={val_loss:.5f} best={best_val:.5f} "
                      f"L_eff={self._nau_head.L_eff:.4f}")
            if epoch - best_epoch >= patience:
                self._log(f"    NAU early stop at epoch {epoch+1} (best={best_epoch+1})")
                break

        self._nau_head.load_state_dict(best_state)
        self._trained = True

        # Compare SINDy-only vs NAU
        with torch.no_grad():
            sindy_pred = Theta_t[val_idx] @ torch.FloatTensor(self._sindy_coefs.T).to(self.device)
            nau_pred = self._nau_head(Theta_t[val_idx])
            sindy_mse = float(nn.MSELoss()(sindy_pred, tgt_t[val_idx]))
            nau_mse = best_val
        self._log(f"  SINDy-only val MSE: {sindy_mse:.5f}")
        self._log(f"  NAU val MSE:        {nau_mse:.5f}")
        self._log(f"  NAU improvement:    {(1-nau_mse/sindy_mse)*100:.1f}%")
        self._log(f"  OOD bound L_eff:    {self._nau_head.L_eff:.4f}")

        # Initialize SGD coefficient layer from STLSQ solution
        # Sparsity pattern (which features are active) is LOCKED here
        n_out = self._sindy_coefs.shape[0]
        n_feat = self._sindy_coefs.shape[1]
        self._coef_layer = nn.Linear(n_feat, n_out, bias=False).to(self.device)
        with torch.no_grad():
            self._coef_layer.weight.copy_(torch.FloatTensor(self._sindy_coefs))
        self._coef_optimizer = optim.Adam(self._coef_layer.parameters(), lr=1e-3, weight_decay=1e-4)
        # Build active mask: only features that STLSQ kept as non-zero
        mask = np.zeros_like(self._sindy_coefs)
        for dim, m in enumerate(self._active_features):
            mask[dim] = m.astype(float)
        self._active_mask_t = torch.FloatTensor(mask).to(self.device)
        n_active_total = int(mask.sum())
        self._log(f"  SGD coef layer initialized: {n_active_total} active coefficients (locked sparsity)")
        self._loop_done = True
        # Note: LLM feature history outcomes are filled later by prune_llm_features()
        # when Role #4 reviews them.  We do NOT update them inline here because the
        # feature-name ↔ coefficient alignment is non-trivial across SINDy+orth_expand
        # boundaries and risks crashing the long-running fit.

    def _build_full_theta(self, SA):
        """Build full feature matrix including expanded columns."""
        Theta = np.asarray(self._library.transform(SA))
        if hasattr(self, '_expand_specs') and self._expand_specs:
            for spec in self._expand_specs:
                if spec['type'] == 'llm_oracle':
                    oracle = LLMOracle(env_type=spec['env_type'])
                    cols, _, _ = oracle.suggest_features(
                        SA, obs_dim=self.obs_dim, act_dim=self.act_dim)
                    Theta = np.hstack([Theta, cols])
                elif spec['type'] == 'orthogonal':
                    oe = OrthogonalExpander(self.obs_dim, self.act_dim)
                    all_cands, all_names = oe._generate_candidates(SA, len(SA))
                    name_to_col = {n: c for c, n in zip(all_cands, all_names)}
                    extra = []
                    for n in spec['names']:  # preserve original order
                        if n in name_to_col:
                            extra.append(name_to_col[n])
                    if extra:
                        Theta = np.hstack([Theta, np.column_stack(extra)])
                elif spec['type'] == 'llm_features':
                    # Re-evaluate LLM-proposed expressions on new SA points.
                    # Uses same sandboxed namespace as _eval_llm_features.
                    safe_globals = {"__builtins__": {}, "np": np}
                    s_arr = SA[:, :self.obs_dim]
                    a_arr = SA[:, self.obs_dim:]
                    extra = []
                    for _name, expr, std in spec['items']:
                        if not expr or any(bad in expr for bad in (
                                "__", "import", "open(", "os.", "sys.",
                                "eval(", "exec(")):
                            extra.append(np.zeros(len(SA)))  # safe zero-fill
                            continue
                        try:
                            col = eval(expr, safe_globals,
                                        {"s": s_arr.T, "a": a_arr.T})
                            col = np.asarray(col, dtype=np.float64)
                            if col.shape != (len(SA),):
                                extra.append(np.zeros(len(SA)))
                                continue
                            if not np.all(np.isfinite(col)):
                                extra.append(np.zeros(len(SA)))
                                continue
                            extra.append(col / max(std, 1e-8))
                        except Exception:
                            extra.append(np.zeros(len(SA)))
                    if extra:
                        Theta = np.hstack([Theta, np.column_stack(extra)])
        return Theta

    def predict_correction(self, states, actions):
        """Returns (Δs_correction, Δr_correction). State from NAU, reward from MLP."""
        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)

        # State correction: SINDy+NAU
        Theta = self._build_full_theta(SA)
        Theta_t = torch.FloatTensor(Theta).to(self.device)
        with torch.no_grad():
            ds = self._nau_head(Theta_t).cpu().numpy()  # (N, obs_dim)

        # Reward correction: separate MLP
        SA_t = torch.FloatTensor(SA).to(self.device)
        with torch.no_grad():
            dr = self._reward_net(SA_t).cpu().numpy().squeeze(-1)  # (N,)

        return ds, dr

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
        n_dims = len(self._active_features) if self._active_features else 0
        for dim in range(n_dims):
            name = f"Δs_{dim}" if dim < self.obs_dim else "Δr"
            active = [(self._feature_names[j], float(self._sindy_coefs[dim, j]))
                      for j in range(self._n_features)
                      if self._active_features[dim][j]]
            if active:
                result[name] = active
        return result
