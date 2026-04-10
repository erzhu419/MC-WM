"""
The Self-Hypothesizing Loop (dev manual §4).

This is the core contribution of MC-WM: an automated hypothesis-falsification loop
that iteratively discovers the structure of the sim-real residual without LLM help.

Loop:
    Round 1:  Fit SINDy with poly2 library
    Test:     Quality gate (holdout error ε < ε_threshold)
    Falsify:  Statistical diagnosis on remainder
    Expand:   Add features based on diagnosis (no LLM needed)
    Round 2+: Re-fit with expanded library
    Stop:     Gate passes OR remainder is pure white noise OR round == max_rounds

Optional LLM oracle: called ONLY if all 4 mechanisms tried and gate still fails.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from mc_wm.residual.sindy_track import SINDyTrack, make_poly2_library
from mc_wm.self_audit.diagnosis import DiagnosisBattery, DiagnosisResult
from mc_wm.self_audit.auto_expand import AutoExpander


@dataclass
class RoundLog:
    """Log for one round of the hypothesis loop."""
    round_num: int
    fit_errors: Dict[str, np.ndarray]   # per-element SINDy fit errors
    quality_passed: bool
    diagnoses: List[DiagnosisResult]
    mechanisms_fired: List[str]
    accepted: bool
    reason: str   # "quality_gate" | "white_noise" | "max_rounds" | "llm_oracle"


class HypothesisLoop:
    """
    Runs the self-hypothesizing loop on a filled ResidualBuffer.

    After completion:
        loop.sindy_track  — fitted SINDy track ready for prediction
        loop.logs         — per-round diagnostic trace
        loop.accepted_round — which round was accepted
    """

    def __init__(
        self,
        sindy_track: SINDyTrack,
        obs_dim: int,
        act_dim: int,
        eps_threshold: float = 0.01,    # quality gate threshold
        max_rounds: int = 3,
        diagnosis_alpha: float = 0.05,
        llm_oracle=None,                 # optional; see llm_oracle.py
    ):
        self.sindy_track = sindy_track
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eps_threshold = eps_threshold
        self.max_rounds = max_rounds
        self.battery = DiagnosisBattery(alpha=diagnosis_alpha)
        self.expander = AutoExpander(obs_dim=obs_dim, act_dim=act_dim)
        self.llm_oracle = llm_oracle
        self.logs: List[RoundLog] = []
        self.accepted_round: Optional[int] = None
        self._current_library = make_poly2_library()

    # ------------------------------------------------------------------
    def run(self, buffer) -> List[RoundLog]:
        """
        Execute the loop.

        Args:
            buffer: ResidualBuffer (must already be filled with samples)

        Returns:
            logs: list of RoundLog objects
        """
        self.logs = []
        SA, delta_s = buffer.to_arrays("s")
        _,  delta_r = buffer.to_arrays("r")
        _,  delta_d = buffer.to_arrays("d")
        steps = buffer.get_steps()

        SA_hist = None
        use_history = False
        extra_columns = None   # extra feature columns from auto-expand

        for round_num in range(1, self.max_rounds + 1):
            # ----------------------------------------------------------
            # Step 1: HYPOTHESIZE — fit SINDy
            # ----------------------------------------------------------
            self.sindy_track.update_library(self._current_library)

            SA_fit = SA
            if use_history:
                try:
                    SA_hist_arr, delta_s_hist = buffer.to_arrays_with_history("s")
                    SA_fit = SA_hist_arr
                    SA_fit2, delta_r2 = buffer.to_arrays_with_history("r")
                    SA_fit3, delta_d2 = buffer.to_arrays_with_history("d")
                    delta_r_fit = delta_r2
                    delta_d_fit = delta_d2
                except Exception:
                    SA_fit = SA
                    delta_r_fit = delta_r
                    delta_d_fit = delta_d
            else:
                delta_r_fit = delta_r
                delta_d_fit = delta_d

            # Append auto-expanded feature columns (Mechanism 2/3/4 outputs)
            if extra_columns is not None and extra_columns.shape[1] > 0:
                N_fit = len(SA_fit)
                SA_fit = np.hstack([SA_fit, extra_columns[:N_fit]])

            N_fit = len(SA_fit)
            self.sindy_track.fit(
                SA_fit,
                delta_s[:N_fit],
                delta_r_fit[:N_fit],
                delta_d_fit[:N_fit],
            )
            fit_errors = self.sindy_track.get_fit_errors()

            # ----------------------------------------------------------
            # Step 2: TEST — quality gate
            # ----------------------------------------------------------
            eps_max = float(max(
                fit_errors["eps_s"].max(),
                fit_errors["eps_r"].max(),
                fit_errors["eps_d"].max(),
            ))
            quality_passed = eps_max < self.eps_threshold

            # ----------------------------------------------------------
            # Step 3: FALSIFY — statistical diagnosis on remainder
            # ----------------------------------------------------------
            # predict_raw must use the SAME feature matrix as fit
            # (SA_fit already includes history + extra_columns)
            pred = self.sindy_track.predict_raw(SA_fit)
            remainder_s = delta_s[:N_fit] - pred["delta_s"]
            diagnoses = self.battery.run(remainder_s, SA[:N_fit])
            any_structure = any(d.any_fired() for d in diagnoses)

            # ----------------------------------------------------------
            # Decide: accept or continue?
            # ----------------------------------------------------------
            if quality_passed:
                log = RoundLog(
                    round_num=round_num,
                    fit_errors=fit_errors,
                    quality_passed=True,
                    diagnoses=diagnoses,
                    mechanisms_fired=[],
                    accepted=True,
                    reason="quality_gate",
                )
                self.logs.append(log)
                self.accepted_round = round_num
                return self.logs

            if not any_structure:
                log = RoundLog(
                    round_num=round_num,
                    fit_errors=fit_errors,
                    quality_passed=False,
                    diagnoses=diagnoses,
                    mechanisms_fired=[],
                    accepted=True,
                    reason="white_noise",
                )
                self.logs.append(log)
                self.accepted_round = round_num
                return self.logs

            # ----------------------------------------------------------
            # Step 4: EXPAND — automated mechanisms
            # ----------------------------------------------------------
            new_library, expand_meta = self.expander.expand(
                results=diagnoses,
                current_library=self._current_library,
                SA=SA,
                remainder=remainder_s,
                steps=steps,
                SA_hist=SA_hist,
            )
            use_history = expand_meta.get("use_history", use_history)
            self._current_library = new_library
            # Store extra columns for next round's fit + predict
            new_extra = expand_meta.get("extra_columns")
            if new_extra is not None and new_extra.shape[1] > 0:
                if extra_columns is not None and extra_columns.shape[1] > 0:
                    extra_columns = np.hstack([extra_columns, new_extra])
                else:
                    extra_columns = new_extra

            log = RoundLog(
                round_num=round_num,
                fit_errors=fit_errors,
                quality_passed=False,
                diagnoses=diagnoses,
                mechanisms_fired=expand_meta.get("mechanisms_fired", []),
                accepted=False,
                reason="expanding",
            )
            self.logs.append(log)

        # ----------------------------------------------------------
        # Max rounds reached — try LLM oracle if available
        # ----------------------------------------------------------
        if self.llm_oracle is not None:
            try:
                diagnosis_report = self._build_diagnosis_report()
                new_features = self.llm_oracle.query(diagnosis_report)
                if new_features:
                    extended_lib = self.llm_oracle.build_library(new_features, self._current_library)
                    self.sindy_track.update_library(extended_lib)
                    self.sindy_track.fit(SA, delta_s, delta_r, delta_d)
                    log = RoundLog(
                        round_num=self.max_rounds + 1,
                        fit_errors=self.sindy_track.get_fit_errors(),
                        quality_passed=False,
                        diagnoses=[],
                        mechanisms_fired=["llm_oracle"],
                        accepted=True,
                        reason="llm_oracle",
                    )
                    self.logs.append(log)
                    self.accepted_round = self.max_rounds + 1
                    return self.logs
            except Exception as e:
                pass

        # Accept best model, classify remainder as aleatoric
        log = RoundLog(
            round_num=self.max_rounds,
            fit_errors=self.sindy_track.get_fit_errors(),
            quality_passed=False,
            diagnoses=[],
            mechanisms_fired=[],
            accepted=True,
            reason="max_rounds",
        )
        self.logs.append(log)
        self.accepted_round = self.max_rounds
        return self.logs

    # ------------------------------------------------------------------
    def _build_diagnosis_report(self) -> str:
        """Build a text report for the LLM oracle."""
        lines = ["Self-hypothesizing loop diagnosis report:"]
        for log in self.logs:
            lines.append(f"\nRound {log.round_num}: eps_max={log.fit_errors['eps_s'].max():.4f}")
            lines.append(f"  mechanisms_fired: {log.mechanisms_fired}")
            for d in log.diagnoses:
                lines.append(f"  {d.summary()}")
        return "\n".join(lines)

    def print_summary(self):
        print(f"=== HypothesisLoop Summary ({len(self.logs)} rounds) ===")
        for log in self.logs:
            eps_max = log.fit_errors["eps_s"].max() if len(log.fit_errors["eps_s"]) > 0 else float("nan")
            status = "ACCEPTED" if log.accepted else "expanding..."
            print(f"  Round {log.round_num}: eps_max={eps_max:.4f}  {status} ({log.reason})")
            if log.mechanisms_fired:
                print(f"    mechanisms: {log.mechanisms_fired}")
        print(f"  Accepted at round: {self.accepted_round}")
