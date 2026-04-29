"""
Hypothesis-Evidence System for MC-WM.

Replaces the prior "feature-expansion loop + LLM prompt log" pattern with a
single typed ``Hypothesis`` object per claim, plus a counterfactual A/B
harness that records auditable evidence.  Every new SINDy feature, LLM
constraint, or HP change goes through this pipeline:

    1. Generate a ``Hypothesis`` with a falsifiable ``expected_metric``.
    2. Run ``counterfactual_test`` (with-hypothesis vs without).
    3. Append measured ``evidence``.
    4. Set ``decision`` and ``failure_reason``.
    5. Persist to ``~/.mcwm/hypotheses/<run_id>.jsonl``.

This replaces the LLM-as-judge "this feature looks plausible" with a
proper scientific record: every claim has an outcome that can be
re-aggregated across runs and re-checked by a human.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional


HYPOTHESIS_DIR = Path.home() / ".mcwm" / "hypotheses"


@dataclass
class Hypothesis:
    """
    A single falsifiable claim about a candidate feature / constraint / HP.

    Fields:
        hid:             stable id (UUID hex prefix; assigned at creation).
        claim:           one-sentence English description ("sin(s[1]) reduces
                         residual MSE on dim 8 because gravity couples via
                         torso angle").
        source:          who proposed this — "llm_role2", "orth_expander",
                         "llm_role1_constraint", "role5_hp", "human", "pool".
        kind:            "feature" | "constraint" | "hp_change".
        expr:            the actual expression / predicate / HP-key=value.
        env:             env identifier where the claim is being tested.
        round:           SINDy refit round number when the claim was made.
        expected_metric: which metric should improve, e.g. "val_mse",
                         "rollout_reward", "violation_rate".
        expected_direction: "decrease" | "increase".
        expected_min_delta: minimum |Δmetric| considered a positive outcome.
        evidence:        dict of measured deltas (filled by ``record_outcome``).
        decision:        "accepted" | "rejected" | "deferred".
        failure_reason:  "redundant" | "insignificant" | "destabilising" |
                         "unsafe" | None.
        timestamp:       ISO-8601 string at creation.
    """

    claim: str
    source: str
    kind: str
    expr: str
    env: str
    round: int = 0
    expected_metric: str = "val_mse"
    expected_direction: str = "decrease"
    expected_min_delta: float = 0.005

    hid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: dict = field(default_factory=dict)
    decision: str = "deferred"
    failure_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def record_outcome(self, evidence: dict, decision: str,
                       failure_reason: Optional[str] = None) -> None:
        """Set the post-test fields in-place."""
        self.evidence.update(evidence)
        self.decision = decision
        self.failure_reason = failure_reason

    def passes(self) -> bool:
        """Did the measured Δmetric meet ``expected_min_delta`` in the right
        direction?  Used as the default acceptance gate by ``HypothesisLog``.
        """
        m = self.evidence.get(self.expected_metric)
        if m is None:
            return False
        if self.expected_direction == "decrease":
            return m <= -self.expected_min_delta
        else:
            return m >= self.expected_min_delta


class HypothesisLog:
    """
    Append-only persistent log of ``Hypothesis`` records.

    Each MC-WM run uses one log file:
      ``~/.mcwm/hypotheses/<run_id>.jsonl``

    The format is JSON Lines so the file is greppable and resumable; every
    new hypothesis is one line, every outcome update appends a new line
    with the same ``hid`` (so a grep can show the whole history).
    """

    def __init__(self, run_id: str, root: Path = HYPOTHESIS_DIR):
        self.run_id = run_id
        self.path = root / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, h: Hypothesis) -> None:
        """Append one record (creation or outcome)."""
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")

    def all(self) -> list[Hypothesis]:
        """Read back every record in the log (one per line)."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(Hypothesis(**d))
        return out

    def summary(self) -> dict:
        """Aggregate counts by source and decision (for end-of-run logging)."""
        recs = self.all()
        by_src: dict[str, dict[str, int]] = {}
        for h in recs:
            d = by_src.setdefault(h.source, {"accepted": 0, "rejected": 0, "deferred": 0})
            d[h.decision] = d.get(h.decision, 0) + 1
        return {
            "total": len(recs),
            "by_source": by_src,
            "log_path": str(self.path),
        }


def counterfactual_val_mse(
    *,
    fit_fn_with: Callable[[], float],
    fit_fn_without: Callable[[], float],
) -> dict:
    """
    Generic A/B harness for a feature-or-constraint hypothesis.

    Both ``fit_fn_with`` and ``fit_fn_without`` should return a held-out
    validation loss (lower is better).  They MUST share the same train/val
    split, the same RNG seed, and the same number of training iterations
    so the only difference is the hypothesis under test.

    Returns:
      {
        "val_mse_with":      float,
        "val_mse_without":   float,
        "val_mse_delta":     float    (with - without; negative ⇒ helps),
      }
    """
    mse_w = float(fit_fn_with())
    mse_o = float(fit_fn_without())
    return {
        "val_mse_with": mse_w,
        "val_mse_without": mse_o,
        "val_mse_delta": mse_w - mse_o,
    }


def leave_one_feature_out_mse(theta_full, beta_full, target, feature_index: int) -> float:
    """
    Closed-form leave-one-feature-out: reuse the existing ridge coefficients
    and zero out the j-th column.  This is a cheap proxy for ``val_mse_with``
    minus ``val_mse_without`` when the per-feature coefficient is small;
    used by Role #4 prune to fill the previously-placeholder
    ``val_mse_delta`` field on accepted features.

    Args:
        theta_full:   (N, F) feature matrix on the validation slice.
        beta_full:    (D, F) ridge coefficients, output × feature.
        target:       (N, D) per-output targets on the validation slice.
        feature_index: column to ablate.

    Returns:
        ``mse_drop = mse_without_j - mse_with_j``.  Positive ⇒ the feature
        was helping; negative ⇒ removing it improves fit.
    """
    import numpy as np
    pred_with = theta_full @ beta_full.T  # (N, D)
    mse_with = float(np.mean((pred_with - target) ** 2))

    # Ablate column `feature_index`.
    beta_drop = beta_full.copy()
    beta_drop[:, feature_index] = 0.0
    pred_without = theta_full @ beta_drop.T
    mse_without = float(np.mean((pred_without - target) ** 2))

    return mse_without - mse_with
