"""
Constraint System: monotonically growing set of physical/semantic constraints.

v4 Architecture Component #5:
  C₀ ⊆ C₁ ⊆ C₂ ⊆ ...
  New constraints are added, old ones never removed.

Two sources:
  Role #1 (initial): Physics constraints from env analysis (hardcoded)
  Role #3 (runtime): Audit suspicious corrections, add new rules

Any correction that violates ANY constraint is rejected → fallback to raw sim.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Constraint:
    name: str
    description: str
    check_fn: object  # callable(s, a, s_corrected, r_corrected) → bool (True=violation)
    source: str = "role1"  # "role1" | "role3" | "llm1" | "llm3"
    added_at_step: int = 0
    n_checked: int = 0        # how many (s,a,s',r) this constraint has seen
    n_rejected: int = 0       # how many it flagged as violating


class ConstraintSystem:
    """
    Monotonically growing constraint set.

    Usage:
        cs = ConstraintSystem(env_type="gravity_cheetah")
        # Check if a corrected transition is valid
        ok, violations = cs.check(s, a, s_next_corrected, r_corrected)
        if not ok:
            use raw sim instead

        # Runtime: audit suspicious corrections and add new constraints
        cs.audit_and_expand(suspicious_transitions, step)
    """

    def __init__(self, env_type="gravity_cheetah", log_fn=None,
                 claude_oracle=None, env_description_for_llm: str = "",
                 real_buffer_for_fpr=None, fpr_threshold: float = 0.01):
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self.constraints: List[Constraint] = []
        self._n_checked = 0
        self._n_rejected = 0
        self._claude_oracle = claude_oracle
        self._env_desc_llm = env_description_for_llm
        # Role #1/#3/#4 counters.
        self._llm_role1_proposed = 0
        self._llm_role1_accepted = 0
        self._llm_role3_calls = 0
        self._llm_role3_new_constraints = 0
        self._llm_role3_valid_corrections = 0
        self._llm_role4_dropped_constraints = 0
        # Decision history: each entry records an LLM action + its eventual
        # fate, fed back to future LLM calls so it learns from past decisions.
        #   {"step": int, "role": "role1"|"role3"|"role4_drop",
        #    "name": str, "expr": str, "reason": str,
        #    "outcome": {"reject_rate": float, "pruned_at": int|None, ...}}
        self._llm_decision_history: list[dict] = []
        # Training-state metrics (reward trend, policy entropy, buffer size)
        # fed in from training loop via `update_training_metrics`.
        self._training_metrics: dict = {"reward_trend": [], "buffer_size": 0}
        # Async executor for Role #3 / Role #4 (never blocks training).
        # max_workers=1 serializes LLM calls; pending futures are drained at
        # the *next* audit step before issuing a new request.
        from concurrent.futures import ThreadPoolExecutor
        self._llm_executor = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix="mcwm_llm")
        self._pending_role3_future = None   # (future, step_submitted)
        self._pending_role4_future = None   # (future, step_submitted)

        # Role #1: initial constraints (hardcoded physics).
        if env_type == "gravity_cheetah":
            self._init_gravity_cheetah()
        self._log(f"  ConstraintSystem: {len(self.constraints)} hardcoded "
                  f"initial constraints (Role #1 physics)")

        # Role #1 LLM extension: ask Claude for additional env-specific constraints.
        if self._claude_oracle is not None and self._env_desc_llm:
            self._augment_with_llm_role1(real_buffer=real_buffer_for_fpr,
                                          fpr_threshold=fpr_threshold)

    def _init_gravity_cheetah(self):
        """
        Role #1: Physics constraints for HalfCheetah.

        HalfCheetah obs (17-dim):
          [0]=rootz (height), [1]=rooty (body angle)
          [2:8]=6 joint angles, [8]=vx, [9]=vz, [10]=va
          [11:17]=6 joint angular velocities

        Gravity is 1x in real (9.81), 2x in sim (19.62).
        Correction should make sim behave like 1x gravity.
        """
        # ── Physical possibility constraints
        self.constraints.append(Constraint(
            name="height_bounds",
            description="Cheetah height must be in [-1, 2] (can't go underground or fly)",
            check_fn=lambda s, a, sc, r: sc[0] < -1.0 or sc[0] > 2.0,
        ))
        self.constraints.append(Constraint(
            name="body_angle",
            description="Body angle |rooty| < 3.0 rad (can't do >1 full rotation per step)",
            check_fn=lambda s, a, sc, r: abs(sc[1]) > 3.0,
        ))
        self.constraints.append(Constraint(
            name="forward_velocity",
            description="|vx| < 15 m/s (physical speed limit for this robot)",
            check_fn=lambda s, a, sc, r: abs(sc[8]) > 15.0,
        ))
        self.constraints.append(Constraint(
            name="vertical_velocity",
            description="|vz| < 15 m/s (can't launch vertically)",
            check_fn=lambda s, a, sc, r: abs(sc[9]) > 15.0,
        ))
        self.constraints.append(Constraint(
            name="angular_velocity",
            description="|angular_vel| < 30 rad/s (physical rotation limit)",
            check_fn=lambda s, a, sc, r: abs(sc[10]) > 30.0,
        ))

        # ── Joint constraints
        for j in range(6):
            dim = 2 + j
            vel_dim = 11 + j
            self.constraints.append(Constraint(
                name=f"joint{j}_angle",
                description=f"Joint {j} angle in [-1.5, 1.5] rad",
                check_fn=lambda s, a, sc, r, d=dim: abs(sc[d]) > 1.5,
            ))
            self.constraints.append(Constraint(
                name=f"joint{j}_velocity",
                description=f"Joint {j} angular velocity |ω| < 40 rad/s",
                check_fn=lambda s, a, sc, r, d=vel_dim: abs(sc[d]) > 40.0,
            ))

        # ── State transition magnitude (not correction — sc is next_state, not corrected_current)
        # Real env stats: joint vels can be ±25, so per-step change can be large
        # Only flag truly extreme transitions
        self.constraints.append(Constraint(
            name="extreme_transition",
            description="Per-dim state change |s'−s| < 30 (filters catastrophic predictions)",
            check_fn=lambda s, a, sc, r: np.max(np.abs(sc - s)) > 30.0
                if len(s) == len(sc) else False,
        ))

        # ── Reward constraints
        self.constraints.append(Constraint(
            name="reward_bounds",
            description="Reward in [-10, 10] (physical reward range)",
            check_fn=lambda s, a, sc, r: r < -10.0 or r > 10.0,
        ))

        # ── Semantic: gravity correction direction
        self.constraints.append(Constraint(
            name="gravity_direction",
            description="Gravity correction should generally reduce acceleration "
                        "(real=1x < sim=2x), so corrected vz should be less negative",
            check_fn=lambda s, a, sc, r: False,  # soft constraint, logged but not rejected
        ))

    def check(self, s, a, s_corrected, r_corrected):
        """
        Check if a corrected transition violates any constraint.

        Returns: (ok: bool, violations: list of constraint names)
        Also updates per-constraint n_checked / n_rejected for Role #4 pruning.
        """
        violations = []
        for c in self.constraints:
            c.n_checked += 1
            try:
                if c.check_fn(s, a, s_corrected, r_corrected):
                    violations.append(c.name)
                    c.n_rejected += 1
            except Exception:
                pass  # constraint evaluation error → skip

        self._n_checked += 1
        if violations:
            self._n_rejected += 1

        return len(violations) == 0, violations

    def check_batch(self, states, actions, s_corrected, r_corrected):
        """
        Batch check. Returns (ok_mask, violation_counts).

        ok_mask: (N,) bool — True if transition passes all constraints
        violation_counts: (N,) int — number of violated constraints per transition
        """
        N = len(states)
        ok_mask = np.ones(N, dtype=bool)
        violation_counts = np.zeros(N, dtype=int)

        for i in range(N):
            ok, violations = self.check(
                states[i], actions[i], s_corrected[i], r_corrected[i])
            ok_mask[i] = ok
            violation_counts[i] = len(violations)

        return ok_mask, violation_counts

    # ──────────────────────────────────────────────────────────────────
    # LLM (Role #1 / #3) integration
    # ──────────────────────────────────────────────────────────────────

    def _compile_llm_check(self, expr: str):
        """
        Compile an LLM-proposed `check` expression into a safe check_fn.

        Expression is a Python boolean returning True iff the transition is
        INFEASIBLE (should be rejected).  Namespace: s, a, sc (s_corrected),
        r (r_corrected), s_next (alias for sc), abs, np.  Blocks attribute
        access, imports, dunders.
        """
        if not expr or not isinstance(expr, str):
            return None
        lowered = expr.strip()
        if any(bad in lowered for bad in ("__", "import ", "open(", "os.", "sys.",
                                           "eval(", "exec(", "subprocess", "compile(")):
            return None
        import numpy as np
        safe_globals = {"__builtins__": {"abs": abs, "max": max, "min": min,
                                           "any": any, "all": all, "range": range,
                                           "len": len, "True": True, "False": False}}
        safe_globals["np"] = np
        # Pre-compile for cheap per-call eval.
        try:
            code = compile(expr, "<llm_constraint>", "eval")
        except SyntaxError:
            return None

        def check_fn(s, a, sc, r, _code=code, _g=safe_globals):
            try:
                return bool(eval(_code, _g,
                                 {"s": s, "a": a, "sc": sc, "r": r,
                                  "s_next": sc}))
            except Exception:
                return False  # eval failure ⇒ treat as feasible (conservative)
        return check_fn

    def _augment_with_llm_role1(self, real_buffer=None, fpr_threshold: float = 0.01):
        """Ask Claude for extra env-specific constraints and register them.

        ``real_buffer``, when supplied, is a tuple ``(s, a, s_next, r)``
        of arrays from real-env transitions.  Each LLM-proposed predicate
        is evaluated on these transitions; the predicate's empirical
        false-positive rate (= fraction of real transitions it incorrectly
        flags as violations) must be below ``fpr_threshold`` for the
        constraint to be accepted.  If ``real_buffer`` is None, the FPR
        check is skipped (and we log this loudly so the paper can be
        accurate about which configurations are FPR-validated).
        """
        self._log("  [LLM Role #1] querying Claude for additional constraints ...")
        proposals = self._claude_oracle.role1_initial_constraints(self._env_desc_llm)
        if real_buffer is None:
            self._log("    ⚠ no real-buffer supplied: FPR validation SKIPPED")
        else:
            self._log(f"    FPR validation enabled (threshold={fpr_threshold:.3f}) "
                      f"on {len(real_buffer[0])} real transitions")

        rejected_fpr = 0
        for c in proposals or []:
            self._llm_role1_proposed += 1
            name = c.get("name", "llm_unnamed")
            check_expr = c.get("check", "")
            why = c.get("why", "")
            fn = self._compile_llm_check(check_expr)
            if fn is None:
                self._log(f"    ✗ rejected (unsafe/syntax): {name}  {check_expr[:60]}")
                continue

            # FPR validation: count how often this predicate would flag
            # an actual real-env transition as a constraint violation.
            # A high FPR means the constraint is too aggressive (would
            # reject good rollouts during training).
            if real_buffer is not None:
                try:
                    s_arr, a_arr, sn_arr, r_arr = real_buffer
                    n_total = len(s_arr)
                    n_flag = 0
                    # Sample at most 5000 transitions to keep validation fast.
                    import numpy as np
                    n_eval = min(5000, n_total)
                    idx = np.random.default_rng(0).choice(n_total, n_eval, replace=False)
                    for i in idx:
                        if fn(s_arr[i], a_arr[i], sn_arr[i], float(r_arr[i])):
                            n_flag += 1
                    fpr = n_flag / max(1, n_eval)
                    if fpr > fpr_threshold:
                        rejected_fpr += 1
                        self._log(f"    ✗ rejected (FPR={fpr:.3%} > {fpr_threshold:.3%}): "
                                  f"{name}  {check_expr[:60]}")
                        continue
                except Exception as e:
                    self._log(f"    ⚠ FPR check failed on {name}: {e}; accepting "
                              f"by default (less safe)")

            self.constraints.append(Constraint(
                name=f"llm1_{name}", description=f"{check_expr}  // {why[:80]}",
                check_fn=fn, source="llm1",
            ))
            self._llm_role1_accepted += 1
            self._llm_decision_history.append({
                "step": 0, "role": "role1_add",
                "name": f"llm1_{name}", "expr": check_expr,
                "reason": why[:80], "outcome": None,
            })
            self._log(f"    + {name}: {check_expr[:70]}")
        self._log(f"  [LLM Role #1] accepted {self._llm_role1_accepted}/"
                  f"{self._llm_role1_proposed}; rejected_fpr={rejected_fpr}; "
                  f"total constraints now {len(self.constraints)}")

    def _llm_role3_extend(self, suspicious_states, suspicious_actions,
                           suspicious_corr, step: int):
        """
        Role #3 (ASYNC): submit Claude audit request on a background thread;
        training returns immediately.  The response is applied at the NEXT
        `audit_suspicious` call via `_apply_completed_role3_future`.
        """
        if self._claude_oracle is None:
            return
        self._llm_role3_calls += 1
        transitions = []
        n = min(5, len(suspicious_states))
        for i in range(n):
            transitions.append({
                "s": [round(float(x), 3) for x in suspicious_states[i]],
                "a": [round(float(x), 3) for x in suspicious_actions[i]],
                "correction_magnitude": round(
                    float(np.linalg.norm(suspicious_corr[i])), 2),
            })
        # FULL constraint list (not just last 10) so LLM sees redundancy.
        existing = [f"{c.name}: {c.description}" for c in self.constraints]
        per_constraint = [
            {"name": c.name,
             "reject_count": c.n_rejected,
             "total_checks": c.n_checked,
             "reject_rate": round(c.n_rejected / max(1, c.n_checked), 4),
             "source": c.source}
            for c in self.constraints
        ]
        sys_rate = self._n_rejected / max(1, self._n_checked)
        if self._pending_role3_future is not None and not self._pending_role3_future[0].done():
            self._log(f"    [LLM Role #3] prior request still pending, skip submit")
            return
        history_sum = self._recent_history_summary()
        future = self._llm_executor.submit(
            self._claude_oracle.role3_audit,
            self._env_desc_llm, transitions, existing,
            step, sys_rate, per_constraint,
            history_sum,
            dict(self._training_metrics),
        )
        self._pending_role3_future = (future, step)
        self._log(f"    [LLM Role #3] ctx: constraints={len(existing)}, "
                  f"sys_reject={sys_rate:.1%}, history={len(history_sum)} entries, "
                  f"reward_trend_len={len(self._training_metrics.get('reward_trend',[]))}")
        self._log(f"    [LLM Role #3] async submit (step={step})")

    def _apply_completed_role3_future(self):
        """Called by audit_suspicious at start; apply pending Role #3 if done."""
        if self._pending_role3_future is None:
            return
        future, submit_step = self._pending_role3_future
        if not future.done():
            return  # still running, leave for next audit
        self._pending_role3_future = None
        try:
            verdict = future.result(timeout=0.0)
        except Exception as e:
            self._log(f"    [LLM Role #3 async] call failed: {e}")
            return
        if not verdict:
            return
        self._log(f"    [LLM Role #3] verdict (submitted step={submit_step}): "
                  f"{verdict.get('verdict','?')}")
        reasoning = verdict.get("reasoning", "")
        if reasoning:
            self._log(f"    reasoning: {reasoning[:180]}")
        if verdict.get("verdict") == "valid_large_correction":
            self._llm_role3_valid_corrections += 1
            return
        new_c = verdict.get("new_constraint")
        if not isinstance(new_c, dict):
            return
        fn = self._compile_llm_check(new_c.get("check", ""))
        if fn is None:
            self._log(f"    ✗ Role #3 check invalid: {new_c.get('check','')[:60]}")
            return
        added_name = f"llm3_{new_c.get('name','unnamed')}_step{submit_step}"
        self.constraints.append(Constraint(
            name=added_name,
            description=f"{new_c.get('check','')}  // {new_c.get('why','')[:80]}",
            check_fn=fn, source="llm3", added_at_step=submit_step,
        ))
        self._llm_role3_new_constraints += 1
        self._llm_decision_history.append({
            "step": submit_step, "role": "role3_add",
            "name": added_name, "expr": new_c.get("check", ""),
            "reason": new_c.get("why", "")[:80], "outcome": None,
        })
        self._log(f"    + Role #3 added constraint: {new_c.get('name','?')}: "
                  f"{new_c.get('check','')[:70]}")

    def prune_llm_constraints(self, step: int = 0, min_checks: int = 1000) -> int:
        """
        Role #4 (ASYNC): submit prune request in background.  The drop decision
        is applied at the next `prune_llm_constraints` call via
        `_apply_completed_role4_future`.

        Returns: number of constraints dropped *this call* (from a previously-
        completed async request), not from the one just submitted.
        """
        if self._claude_oracle is None:
            return 0
        # 1. Apply any previously-completed Role #4 decision first.
        dropped_now = self._apply_completed_role4_future()
        # 2. If a request is still pending, do NOT submit a new one.
        if self._pending_role4_future is not None and not self._pending_role4_future[0].done():
            return dropped_now
        # 3. Submit new Role #4 request (non-blocking).
        eligible = [c for c in self.constraints
                    if c.source in ("llm1", "llm3") and c.n_checked >= min_checks]
        if not eligible:
            return dropped_now
        stats_payload = [
            {
                "name": c.name,
                "expr": c.description.split("//")[0].strip(),
                "why": c.description.split("//")[-1].strip() if "//" in c.description else "",
                "source": c.source,
                "reject_count": c.n_rejected,
                "total_checks": c.n_checked,
                "reject_rate": round(c.n_rejected / max(1, c.n_checked), 4),
            }
            for c in eligible
        ]
        # Enrich context: hardcoded constraint names + system reject rate +
        # violation trend (stored by caller — see `update_violation_trend`).
        hc_names = [c.name for c in self.constraints if c.source == "role1"]
        sys_rate = self._n_rejected / max(1, self._n_checked)
        trend = getattr(self, "_violation_trend", [])
        history_sum = self._recent_history_summary()
        future = self._llm_executor.submit(
            self._claude_oracle.role4_prune_constraints,
            self._env_desc_llm, stats_payload,
            step, sys_rate, trend, hc_names,
            history_sum,
            dict(self._training_metrics),
        )
        self._pending_role4_future = (future, step)
        self._log(f"    [LLM Role #4] async submit (reviewing {len(eligible)} LLM constraints, "
                  f"sys_reject={sys_rate:.1%}, viol_trend[-3:]={trend[-3:] if trend else []}, "
                  f"history={len(history_sum)} entries)")
        return dropped_now

    def update_violation_trend(self, avg_violations_per_ep: float) -> None:
        """External hook — caller (training script) feeds each eval's viol/ep."""
        if not hasattr(self, "_violation_trend"):
            self._violation_trend = []
        self._violation_trend.append(float(avg_violations_per_ep))
        self._violation_trend = self._violation_trend[-20:]

    def update_training_metrics(self, reward: float = None, buffer_size: int = None,
                                  policy_entropy: float = None,
                                  qdelta_weight_stats: dict = None) -> None:
        """
        External hook: feed in current training-state metrics for LLM context.
        Called from training loop each eval.
        qdelta_weight_stats: {"min": ..., "mean": ..., "max": ..., "std": ...}
        — lets Role #3 detect "measurement artifact" when QΔ collapses.
        """
        if reward is not None:
            self._training_metrics["reward_trend"].append(float(reward))
            self._training_metrics["reward_trend"] = self._training_metrics["reward_trend"][-20:]
        if buffer_size is not None:
            self._training_metrics["buffer_size"] = int(buffer_size)
        if policy_entropy is not None:
            self._training_metrics["policy_entropy"] = float(policy_entropy)
        if qdelta_weight_stats is not None:
            self._training_metrics.setdefault("qdelta_weight_history", [])
            self._training_metrics["qdelta_weight_history"].append(dict(qdelta_weight_stats))
            self._training_metrics["qdelta_weight_history"] = \
                self._training_metrics["qdelta_weight_history"][-10:]

    def _recent_history_summary(self, max_entries: int = 10) -> list[dict]:
        """Summarise last N decisions (newest first) for LLM feedback prompts."""
        entries = list(reversed(self._llm_decision_history[-max_entries:]))
        return [
            {"step": e["step"], "role": e["role"], "name": e["name"],
             "reason": e.get("reason", "")[:100],
             "outcome": e.get("outcome")}
            for e in entries
        ]

    def _apply_completed_role4_future(self) -> int:
        """Drain finished Role #4 future and apply drops.  Returns #dropped."""
        if self._pending_role4_future is None:
            return 0
        future, submit_step = self._pending_role4_future
        if not future.done():
            return 0
        self._pending_role4_future = None
        try:
            to_drop = future.result(timeout=0.0)
        except Exception as e:
            self._log(f"    [LLM Role #4 async] call failed: {e}")
            return 0
        if not to_drop:
            self._log(f"    [LLM Role #4] reviewed, kept all")
            return 0
        drop_set = set(to_drop)
        kept, dropped_names = [], []
        # Snapshot reject stats of soon-to-be-dropped constraints for history.
        drop_stats = {c.name: {"reject_rate": c.n_rejected / max(1, c.n_checked),
                               "total_checks": c.n_checked}
                      for c in self.constraints if c.name in drop_set}
        for c in self.constraints:
            if c.name in drop_set and c.source in ("llm1", "llm3"):
                dropped_names.append(c.name)
            else:
                kept.append(c)
        self.constraints = kept
        self._llm_role4_dropped_constraints += len(dropped_names)
        # Update outcome of the dropped constraint's original add-decision.
        step_now = submit_step
        n_outcome_filled = 0
        for entry in self._llm_decision_history:
            if entry["name"] in drop_set and entry["outcome"] is None:
                entry["outcome"] = {
                    "pruned_at_step": step_now,
                    "final_reject_rate": drop_stats.get(entry["name"], {}).get("reject_rate", 0.0),
                    "total_checks": drop_stats.get(entry["name"], {}).get("total_checks", 0),
                }
                n_outcome_filled += 1
        if n_outcome_filled:
            self._log(f"    [LLM history] filled {n_outcome_filled} outcomes "
                      f"(feeds back into future Role #2/#3/#4 prompts)")
        # Record the prune decision itself.
        for name in dropped_names:
            self._llm_decision_history.append({
                "step": step_now, "role": "role4_drop",
                "name": name, "expr": "",
                "reason": f"dropped at step {step_now}",
                "outcome": None,
            })
        self._log(f"    [LLM Role #4] dropped {len(dropped_names)}: "
                  f"{dropped_names[:5]}")
        return len(dropped_names)

    def get_llm_summary(self) -> dict:
        s = {"role1_proposed": self._llm_role1_proposed,
             "role1_accepted": self._llm_role1_accepted,
             "role3_calls": self._llm_role3_calls,
             "role3_new_constraints": self._llm_role3_new_constraints,
             "role3_valid_corrections": self._llm_role3_valid_corrections,
             "role4_dropped_constraints": getattr(self, "_llm_role4_dropped_constraints", 0)}
        if self._claude_oracle is not None and hasattr(self._claude_oracle, "stats"):
            s.update({"oracle_" + k: v for k, v in
                      self._claude_oracle.stats().items()})
        return s

    def finalize_async_llm(self, timeout: float = 5.0):
        """
        Wait briefly for any in-flight LLM calls to finish and apply their
        results.  Called at end-of-training so final summary reflects all
        completed work.  timeout protects against hangs.
        """
        if self._claude_oracle is None:
            return
        import time
        deadline = time.time() + timeout
        for name, slot_attr, apply_fn in [
            ("role3", "_pending_role3_future", self._apply_completed_role3_future),
            ("role4", "_pending_role4_future", self._apply_completed_role4_future),
        ]:
            pending = getattr(self, slot_attr, None)
            if pending is None:
                continue
            future, _ = pending
            remaining = max(0.0, deadline - time.time())
            try:
                future.result(timeout=remaining)
            except Exception:
                pass
            apply_fn()
        # Shutdown executor (prevents zombie thread at exit).
        self._llm_executor.shutdown(wait=False, cancel_futures=True)

    def log_llm_summary(self):
        if self._claude_oracle is None:
            return
        s = self.get_llm_summary()
        self._log("")
        self._log("┌─ [ConstraintSystem LLM Summary] ────────")
        self._log(f"│ Role #1 proposed  : {s['role1_proposed']}")
        self._log(f"│ Role #1 accepted  : {s['role1_accepted']}")
        self._log(f"│ Role #3 calls     : {s['role3_calls']}")
        self._log(f"│ Role #3 new cstr  : {s['role3_new_constraints']}")
        self._log(f"│ Role #3 accepted as valid: {s['role3_valid_corrections']}")
        self._log(f"│ Role #4 dropped constraints: {s['role4_dropped_constraints']}")
        self._log(f"└─────────────────────────────────────────")

    def add_constraint(self, name, description, check_fn, step=0):
        """Role #3: Add a new constraint (monotonic growth)."""
        self.constraints.append(Constraint(
            name=name, description=description,
            check_fn=check_fn, source="role3",
            added_at_step=step,
        ))
        self._log(f"  Constraint added (Role #3): {name} — {description}")

    def audit_suspicious(self, states, actions, s_corrected, r_corrected,
                          corrections_magnitude, step=0, threshold_percentile=95):
        """
        Role #3: Find suspicious corrections and potentially add new constraints.

        Suspicious = passes all current constraints BUT correction is unusually large.

        Note: Role #3 LLM calls are ASYNC — this method returns immediately
        after submitting; the verdict is applied at the NEXT audit call.
        """
        # Drain any pending Role #3 future from previous audit (non-blocking).
        self._apply_completed_role3_future()

        N = len(states)
        ok_mask, _ = self.check_batch(states, actions, s_corrected, r_corrected)
        # Guard: if EVERYTHING is being rejected (some LLM constraint too
        # strict), skip audit to avoid crash and flag the issue loudly.
        n_ok = int(ok_mask.sum())
        if n_ok == 0:
            self._log(f"  Constraint audit skipped: 0/{N} transitions pass — "
                      f"constraint set likely over-restrictive (Role #4 prune needed)")
            return
        if n_ok < 10:
            self._log(f"  Constraint audit skipped: only {n_ok}/{N} transitions pass")
            return

        # Among passing transitions, find those with large corrections
        mag = corrections_magnitude  # (N,) — per-transition correction magnitude
        threshold = np.percentile(mag[ok_mask], threshold_percentile)
        suspicious = ok_mask & (mag > threshold)
        n_suspicious = int(suspicious.sum())

        if n_suspicious == 0:
            return

        self._log(f"  Constraint audit: {n_suspicious} suspicious corrections "
                  f"(pass constraints but |Δ| > p{threshold_percentile}={threshold:.3f})")

        # Analyze suspicious corrections for patterns
        sus_corr = s_corrected[suspicious] - states[suspicious]
        sus_states = states[suspicious]

        # Auto-detect: any dimension consistently at extreme values?
        for dim in range(min(sus_states.shape[1], 17)):
            vals = sus_states[:, dim]
            if np.std(vals) < 0.01:  # all suspicious states cluster in one value
                continue
            # Check if corrections are unreasonably large for this dim
            corr_dim = np.abs(sus_corr[:, dim])
            if corr_dim.mean() > 3.0:  # mean correction > 3 in this dim
                max_val = float(np.percentile(np.abs(sus_states[:, dim]), 99))
                self.add_constraint(
                    name=f"auto_dim{dim}_extreme_corr",
                    description=f"Dim {dim}: correction > 3.0 when |state| > {max_val:.1f}",
                    check_fn=lambda s, a, sc, r, d=dim, mv=max_val:
                        abs(s[d]) > mv and abs(sc[d] - s[d]) > 3.0,
                    step=step,
                )

        # Role #3 LLM extension: ask Claude to verify the suspicious batch and
        # propose one additional constraint if truly infeasible.  Only run
        # when oracle is set AND at least 3 suspicious samples (avoid noisy calls).
        if self._claude_oracle is not None and n_suspicious >= 3:
            self._llm_role3_extend(sus_states, actions[suspicious], sus_corr, step)

    def get_stats(self):
        return {
            "n_constraints": len(self.constraints),
            "n_checked": self._n_checked,
            "n_rejected": self._n_rejected,
            "reject_rate": self._n_rejected / max(self._n_checked, 1),
            "role1": sum(1 for c in self.constraints if c.source == "role1"),
            "role3": sum(1 for c in self.constraints if c.source == "role3"),
        }
