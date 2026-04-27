"""
Cross-environment feature pool for self-hypothesis discovery.

Persists accepted SINDy features across runs and environments so that
features that worked once can be retried as candidates in later runs
(possibly in different envs).  Each entry tracks per-env acceptance
counts and average reward gain so the candidate-injection step can rank
by global priority.

JSON storage at ~/.mcwm/feature_pool.json by default.  Format::

    {
      "schema_version": 1,
      "features": {
        "sin_3_x2": {
          "expr": "np.sin(3 * s[:, 2])",
          "envs": {
            "gravity_cheetah": {"accept": 5, "reject": 2, "avg_reward_gain": 0.012}
          },
          "accept_total": 5,
          "reject_total": 2,
          "global_avg_reward_gain": 0.012,
          "last_used_iso": "2026-04-26T..."
        }
      }
    }

The pool is intentionally global (one file per user, not per project) so
cross-project transfer (MC-WM → CS-BAPR → BAPR) is possible without any
code changes.  ``Stage A`` of RAHD only seeds candidates; ``Stage D``
(reward-aware acceptance) writes the actual reward gain back into the
pool so future ranks improve.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_POOL_PATH = Path.home() / ".mcwm" / "feature_pool.json"
SCHEMA_VERSION = 1
_LOCK = threading.Lock()  # local lock; file is per-user, not multi-host


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class FeaturePool:
    """
    Thread-safe (within one process) persistent feature pool.

    Typical lifecycle in a training run:

    1. ``pool = FeaturePool()`` — loads from disk
    2. ``cands = pool.query_candidates(env="gravity_cheetah", top_k=10)``
       — returns expressions to add to the candidate library
    3. After training, for each accepted feature with measured reward gain::

           pool.record(name, expr, env, reward_gain=0.012, was_accepted=True)
           pool.save()

    Reward gain is the optional Stage-D mini-rollout score; without it, a
    feature still counts as "accepted" but with ``reward_gain=0`` so its
    global priority only reflects acceptance frequency.
    """

    def __init__(self, path: str | os.PathLike | None = None,
                 priority_floor: float = 0.0):
        self.path = Path(path) if path is not None else DEFAULT_POOL_PATH
        # When `query_candidates` ranks features, anything with global
        # priority below this threshold is dropped from the result.
        self.priority_floor = priority_floor
        self._data = self._load()

    # ─── Persistence ─────────────────────────────────────────────────
    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        try:
            with self.path.open("r") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable — start fresh, but back up the bad file.
            backup = self.path.with_suffix(".bad")
            try:
                self.path.rename(backup)
            except OSError:
                pass
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        if d.get("schema_version") != SCHEMA_VERSION:
            # Future-proof: in v2+ migrate fields here. For now reset.
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        return d

    def save(self) -> None:
        """Atomic save (write to .tmp then rename)."""
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)

    # ─── Read API ────────────────────────────────────────────────────
    def all_features(self) -> dict:
        """Return raw dict {name: meta}.  Caller must not mutate."""
        return self._data["features"]

    def query_candidates(self, env: str | None = None,
                         top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Return up to ``top_k`` candidate features ranked by global priority.

        Priority = (accept_total / max(1, accept_total + reject_total))
                 * (1 + reward_gain_clipped),  reward_gain in [0, 1]

        Features that have never appeared in ``env`` are ranked using the
        global priority alone; features already accepted in the same env
        get a small bonus so they re-appear preferentially.

        Returns:
            list of (name, expr, priority) tuples, sorted desc by priority.
        """
        feats = self._data["features"]
        ranked: list[tuple[str, str, float]] = []
        for name, meta in feats.items():
            acc = meta.get("accept_total", 0)
            rej = meta.get("reject_total", 0)
            denom = max(1, acc + rej)
            base = acc / denom
            gain = max(0.0, min(1.0, meta.get("global_avg_reward_gain", 0.0)))
            priority = base * (1.0 + gain)
            if env is not None and env in meta.get("envs", {}):
                priority *= 1.1  # mild bonus for in-env reuse
            if priority < self.priority_floor:
                continue
            ranked.append((name, meta["expr"], priority))
        ranked.sort(key=lambda t: -t[2])
        return ranked[:top_k]

    # ─── Write API ───────────────────────────────────────────────────
    def record(self, name: str, expr: str, env: str,
               reward_gain: float | None = None,
               was_accepted: bool = True) -> None:
        """
        Update the pool with one feature outcome from a training run.

        Args:
          name: feature name (e.g. "sin_3_x2")
          expr: Python expression (e.g. "np.sin(3 * s[:, 2])")
          env: env identifier (e.g. "gravity_soft_ceiling")
          reward_gain: per-feature reward improvement from mini-rollout.
                       ``None`` is allowed if Stage D wasn't run; treated
                       as 0.0 in the priority computation.
          was_accepted: True if the feature was accepted into the basis
                        (e.g. orthogonality + correlation passed).
        """
        with _LOCK:
            feats = self._data["features"]
            entry = feats.setdefault(name, {
                "expr": expr,
                "envs": {},
                "accept_total": 0,
                "reject_total": 0,
                "global_avg_reward_gain": 0.0,
            })
            # Refresh expression text — useful when expressions evolve.
            entry["expr"] = expr
            entry["last_used_iso"] = _utc_now()

            env_stats = entry["envs"].setdefault(
                env, {"accept": 0, "reject": 0, "avg_reward_gain": 0.0}
            )
            n_prev_global = entry["accept_total"]

            if was_accepted:
                entry["accept_total"] += 1
                env_stats["accept"] += 1
            else:
                entry["reject_total"] += 1
                env_stats["reject"] += 1

            # Running mean update: only counts accepted records with reward gain.
            if was_accepted and reward_gain is not None:
                # Per-env mean
                env_acc = env_stats["accept"]
                env_stats["avg_reward_gain"] = (
                    (env_stats["avg_reward_gain"] * (env_acc - 1) + reward_gain)
                    / max(1, env_acc)
                )
                # Global mean (across all accepted records ever, all envs)
                entry["global_avg_reward_gain"] = (
                    (entry["global_avg_reward_gain"] * n_prev_global + reward_gain)
                    / max(1, entry["accept_total"])
                )

    def record_batch(self, items: Iterable[dict]) -> None:
        """Convenience wrapper to record many outcomes at once."""
        for it in items:
            self.record(**it)


def evaluate_expression(expr: str, s: "np.ndarray", a: "np.ndarray"):
    """
    Safely evaluate a feature expression against numpy state-action arrays.

    Mirrors the safe-eval used in `sindy_nau_adapter._eval_llm_features`:
    expressions can reference ``s`` (shape (N, obs_dim)) and ``a`` (shape
    (N, act_dim)) and ``np``.  Anything else raises NameError.

    Returns:
        (N,) numpy array, or None on failure / non-finite values.
    """
    import numpy as np
    safe_globals = {"__builtins__": {}, "np": np}
    safe_locals = {"s": s, "a": a}
    try:
        out = eval(expr, safe_globals, safe_locals)
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != s.shape[0]:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr
