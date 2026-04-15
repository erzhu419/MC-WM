"""
Role #5: Meta-Hyperparameter Orchestrator.

Wraps the Claude oracle's `role5_tune_hyperparameters` to:
  - expose a concrete schema of tunable hyperparameters
  - validate LLM-proposed values against the schema
  - track trial history (HPs + final metrics of each trial)
  - record decisions + outcomes for the LLM's own feedback loop

Usage:
    orch = HPOrchestrator(oracle, env_description, initial_hp=dict(...))
    # training loop calls this periodically:
    new_hp = orch.propose(current_hp, training_metrics)
    # apply new_hp to the running config
    # at end of training (or at a milestone), record the outcome:
    orch.record_trial_outcome(final_hp, reward, viol, val_mse)
"""

from __future__ import annotations
import json
from typing import Any, Optional


# Default tunable HP schema (safe ranges).  Extend as needed.
DEFAULT_HP_SCHEMA: dict = {
    "qdelta_gamma": {
        "type": "float", "range": [0.0, 0.95],
        "doc": "γ for Bellman QΔ critic (0=per-step, high=long horizon)",
    },
    "rollout_batch": {
        "type": "int", "range": [100, 1000],
        "doc": "Model rollouts generated per cycle",
    },
    "rollout_freq": {
        "type": "int", "range": [100, 1000],
        "doc": "Env steps between rollout generations",
    },
    "model_train_freq": {
        "type": "int", "range": [500, 5000],
        "doc": "Env steps between δ refits",
    },
    "audit_percentile": {
        "type": "int", "range": [80, 99],
        "doc": "Role #3 suspicious-transition percentile threshold",
    },
    "max_hypothesis_rounds": {
        "type": "int", "range": [1, 6],
        "doc": "Self-hypothesis loop max rounds (only used at first fit)",
    },
    "icrl_combine": {
        "type": "enum", "choices": ["top_k", "soft"],
        "doc": "How ICRL φ combines with QΔ",
    },
    "icrl_top_k_frac": {
        "type": "float", "range": [0.3, 0.95],
        "doc": "Keep top-K fraction when icrl_combine=top_k",
    },
}


class HPOrchestrator:
    """
    LLM-driven hyperparameter tuner.  Safe: never mutates anything directly;
    always returns a validated HP dict that the training loop chooses to apply.
    """

    def __init__(
        self,
        oracle,                          # ClaudeCLIOracle instance
        env_description: str,
        initial_hp: dict,
        hp_schema: Optional[dict] = None,
        log_fn=None,
    ) -> None:
        self._oracle = oracle
        self._env_desc = env_description
        self._schema = hp_schema or DEFAULT_HP_SCHEMA
        # Restrict current_hp to keys present in schema.
        self._current_hp = {k: v for k, v in initial_hp.items() if k in self._schema}
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self._trial_history: list[dict] = []   # completed trials
        self._propose_history: list[dict] = [] # every LLM call's proposal
        self.n_proposals = 0
        self.n_applied = 0
        self.n_clamped = 0
        self.n_rejected = 0

    # ──────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────

    def _validate_value(self, key: str, val: Any) -> tuple[Any, str]:
        """Return (validated_value, note).  note="" on clean accept."""
        spec = self._schema.get(key)
        if spec is None:
            return None, f"key {key} not in schema"
        t = spec.get("type")
        if t == "enum":
            choices = spec.get("choices", [])
            if val in choices:
                return val, ""
            return None, f"{val!r} not in {choices}"
        if t in ("float", "int"):
            lo, hi = spec.get("range", [None, None])
            try:
                val_cast = int(val) if t == "int" else float(val)
            except (TypeError, ValueError):
                return None, f"type error for {val!r}"
            if lo is not None and val_cast < lo:
                return lo, f"clamped {val_cast}→{lo} (min)"
            if hi is not None and val_cast > hi:
                return hi, f"clamped {val_cast}→{hi} (max)"
            return val_cast, ""
        return None, f"unknown type {t}"

    # ──────────────────────────────────────────────────────────────────
    # Propose
    # ──────────────────────────────────────────────────────────────────

    def propose(self, training_metrics: dict) -> dict:
        """
        Ask the LLM for the next HP configuration.  Returns a dict of
        validated HPs to APPLY (only keys that differ from current_hp AND
        pass validation).  Empty dict means "no change".
        """
        self.n_proposals += 1
        # Build a trimmed history for the prompt (last 8 trials).
        trial_sub = self._trial_history[-8:]
        response = self._oracle.role5_tune_hyperparameters(
            env_description=self._env_desc,
            current_hp=dict(self._current_hp),
            hp_schema=self._schema,
            trial_history=trial_sub,
            training_metrics=training_metrics,
        )
        if not response:
            self._log("  [LLM Role #5] empty response; no HP change")
            return {}
        proposed = response.get("proposed_hp", {})
        reasons = response.get("reasons", {})
        # Validate each proposed value.
        clean: dict = {}
        for k, v in proposed.items():
            val, note = self._validate_value(k, v)
            if val is None:
                self.n_rejected += 1
                self._log(f"    ✗ Role #5 reject {k}={v}  ({note})")
                continue
            if note:
                self.n_clamped += 1
                self._log(f"    ~ Role #5 clamp {k}: {note}")
            if val != self._current_hp.get(k):
                clean[k] = val
        self._propose_history.append({
            "proposed": proposed, "applied": clean, "reasons": reasons,
            "metrics": {k: training_metrics.get(k) for k in
                        ("reward_trend", "violation_trend", "val_mse")},
        })
        if clean:
            self.n_applied += 1
            self._log(f"  [LLM Role #5] apply {len(clean)} HP changes:")
            for k, v in clean.items():
                old = self._current_hp.get(k)
                why = reasons.get(k, "(no reason provided)")[:80]
                self._log(f"    {k}: {old} → {v}    why: {why}")
            self._current_hp.update(clean)
        else:
            self._log(f"  [LLM Role #5] no actionable change proposed")
        return clean

    # ──────────────────────────────────────────────────────────────────
    # Outcome recording (for next LLM call's trial_history prompt)
    # ──────────────────────────────────────────────────────────────────

    def record_trial_outcome(self, hp_used: dict, reward: float,
                              viol: float, val_mse: float, step: int = 0) -> None:
        """Called at end of training (or at a milestone) to log a trial."""
        self._trial_history.append({
            "hp": {k: hp_used.get(k) for k in self._schema if k in hp_used},
            "reward": round(float(reward), 1),
            "viol": round(float(viol), 3),
            "val_mse": round(float(val_mse), 4),
            "step": int(step),
        })
        self._log(f"  [LLM Role #5] trial recorded: "
                  f"reward={reward:.1f} viol={viol:.2f} val_mse={val_mse:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────

    def current(self) -> dict:
        return dict(self._current_hp)

    def get_summary(self) -> dict:
        return {
            "n_proposals": self.n_proposals,
            "n_applied": self.n_applied,
            "n_clamped": self.n_clamped,
            "n_rejected": self.n_rejected,
            "n_trials_recorded": len(self._trial_history),
            "final_hp": dict(self._current_hp),
        }

    def log_summary(self):
        s = self.get_summary()
        self._log("")
        self._log("┌─ [LLM Role #5 Summary] ───────────────────")
        self._log(f"│ proposals requested : {s['n_proposals']}")
        self._log(f"│ changes applied     : {s['n_applied']}")
        self._log(f"│ values clamped      : {s['n_clamped']}")
        self._log(f"│ values rejected     : {s['n_rejected']}")
        self._log(f"│ trials recorded     : {s['n_trials_recorded']}")
        self._log(f"│ final HP            : {json.dumps(s['final_hp'])[:80]}")
        self._log("└───────────────────────────────────────────")
