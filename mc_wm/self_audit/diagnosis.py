"""
Statistical Diagnosis Battery (dev manual §4.1, Step 3).

Run on the remainder Δ_remainder = Δ - Δ̂ after each SINDy round.

Four tests:
  1. Autocorrelation   → temporal structure?     (Ljung-Box test)
  2. Heteroscedasticity → state-dep variance?    (Breusch-Pagan test)
  3. Normality          → heavy tails / modes?   (Shapiro-Wilk or D'Agostino)
  4. Stationarity       → drifting dynamics?     (KPSS test)

Each test fires (True) if the null hypothesis is rejected at α=0.05.
The culprit variable (for heteroscedasticity) is identified via per-feature correlation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings("ignore")   # statsmodels can be verbose


@dataclass
class DiagnosisResult:
    """Diagnosis output for one residual dimension."""
    dim: int

    # Test outcomes (True = detected, i.e., structure still present)
    autocorrelation: bool = False
    heteroscedastic: bool = False
    non_normal: bool = False
    non_stationary: bool = False

    # Metadata
    heteroscedastic_culprit: Optional[int] = None   # feature index with highest corr to |resid|
    heavy_tail_kurtosis: float = 0.0
    autocorr_lag1: float = 0.0
    mean_drift: float = 0.0

    def any_fired(self) -> bool:
        return self.autocorrelation or self.heteroscedastic or self.non_normal or self.non_stationary

    def summary(self) -> str:
        active = []
        if self.autocorrelation:
            active.append(f"autocorr(lag1={self.autocorr_lag1:.3f})")
        if self.heteroscedastic:
            active.append(f"heteroscedastic(culprit={self.heteroscedastic_culprit})")
        if self.non_normal:
            active.append(f"non_normal(kurtosis={self.heavy_tail_kurtosis:.2f})")
        if self.non_stationary:
            active.append(f"non_stationary(drift={self.mean_drift:.3f})")
        if not active:
            return f"dim {self.dim}: CLEAN (pure white noise)"
        return f"dim {self.dim}: " + ", ".join(active)


class DiagnosisBattery:
    """
    Runs the four statistical tests on residual remainders.

    Usage:
        battery = DiagnosisBattery(alpha=0.05)
        results = battery.run(remainder, SA_features)
        # results: List[DiagnosisResult], one per output dim
    """

    def __init__(self, alpha: float = 0.05, max_lag: int = 5):
        self.alpha = alpha
        self.max_lag = max_lag

    def run(
        self,
        remainder: np.ndarray,   # shape (N, out_dim) — residual after SINDy
        SA: np.ndarray,          # shape (N, input_dim) — for heteroscedasticity culprit
    ) -> List[DiagnosisResult]:
        """
        Run all four tests on each output dimension of the remainder.

        Returns list of DiagnosisResult, one per column of remainder.
        """
        if remainder.ndim == 1:
            remainder = remainder[:, None]

        results = []
        for i in range(remainder.shape[1]):
            r = remainder[:, i]
            result = DiagnosisResult(dim=i)

            result.autocorrelation, result.autocorr_lag1 = self._test_autocorr(r)
            result.heteroscedastic, result.heteroscedastic_culprit = self._test_hetero(r, SA)
            result.non_normal, result.heavy_tail_kurtosis = self._test_normality(r)
            result.non_stationary, result.mean_drift = self._test_stationarity(r)

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Test 1: Autocorrelation (Ljung-Box)
    # ------------------------------------------------------------------

    def _test_autocorr(self, r: np.ndarray):
        """
        Ljung-Box test for autocorrelation in the residual series.
        Returns (fired, lag1_autocorr).
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(r, lags=[self.max_lag], return_df=True)
            p_value = float(result["lb_pvalue"].iloc[0])
            fired = p_value < self.alpha
        except Exception:
            # Fallback: manual lag-1 autocorrelation test
            if len(r) < 10:
                return False, 0.0
            lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            fired = abs(lag1) > 0.1
            return fired, lag1

        # Compute lag-1 for metadata even if test passed
        lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0.0
        return fired, lag1

    # ------------------------------------------------------------------
    # Test 2: Heteroscedasticity (Breusch-Pagan)
    # ------------------------------------------------------------------

    def _test_hetero(self, r: np.ndarray, SA: np.ndarray):
        """
        Breusch-Pagan test: regress |residual|² on SA features.
        If significant, identify the culprit variable.

        Returns (fired, culprit_index).
        """
        fired = False
        culprit = None

        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            import statsmodels.api as sm
            exog = sm.add_constant(SA)
            _, p_value, _, _ = het_breuschpagan(r, exog)
            fired = p_value < self.alpha
        except Exception:
            # Fallback: correlation between |r| and each SA feature
            abs_r = np.abs(r)
            corrs = np.array([abs(float(np.corrcoef(abs_r, SA[:, j])[0, 1]))
                              if np.std(SA[:, j]) > 1e-8 else 0.0
                              for j in range(SA.shape[1])])
            max_corr = corrs.max()
            fired = max_corr > 0.15
            if fired:
                culprit = int(corrs.argmax())
            return fired, culprit

        if fired:
            # Identify culprit: feature most correlated with |r|²
            abs_r2 = r ** 2
            corrs = np.array([abs(float(np.corrcoef(abs_r2, SA[:, j])[0, 1]))
                              if np.std(SA[:, j]) > 1e-8 else 0.0
                              for j in range(SA.shape[1])])
            culprit = int(corrs.argmax())

        return fired, culprit

    # ------------------------------------------------------------------
    # Test 3: Normality (D'Agostino + kurtosis check)
    # ------------------------------------------------------------------

    def _test_normality(self, r: np.ndarray):
        """
        D'Agostino K² test for normality + excess kurtosis check.
        Returns (fired, kurtosis).
        """
        from scipy import stats
        kurt = float(stats.kurtosis(r))
        fired = False

        if len(r) >= 8:
            try:
                _, p_value = stats.normaltest(r)
                fired = (p_value < self.alpha) or (abs(kurt) > 2.0)
            except Exception:
                fired = abs(kurt) > 2.0

        return fired, kurt

    # ------------------------------------------------------------------
    # Test 4: Stationarity (KPSS — level stationarity)
    # ------------------------------------------------------------------

    def _test_stationarity(self, r: np.ndarray):
        """
        KPSS test for level stationarity (H0: stationary → reject means non-stationary).
        Returns (fired, mean_drift).
        """
        fired = False
        drift = 0.0

        if len(r) < 20:
            return False, 0.0

        # Mean drift: difference between first-half and second-half mean
        half = len(r) // 2
        drift = float(r[half:].mean() - r[:half].mean())

        try:
            from statsmodels.tsa.stattools import kpss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_value, _, _ = kpss(r, regression="c", nlags="auto")
            fired = p_value < self.alpha
        except Exception:
            # Fallback: Augmented Dickey-Fuller
            try:
                from statsmodels.tsa.stattools import adfuller
                stat, p_value, *_ = adfuller(r, autolag="AIC")
                # ADF H0: unit root (non-stationary). p > alpha → non-stationary
                fired = p_value > self.alpha
            except Exception:
                fired = abs(drift) > 0.05 * (np.std(r) + 1e-8)

        return fired, drift

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self, results: List[DiagnosisResult]) -> str:
        lines = ["=== Diagnosis Summary ==="]
        for r in results:
            lines.append(r.summary())
        n_clean = sum(1 for r in results if not r.any_fired())
        lines.append(f"--- {n_clean}/{len(results)} dims clean ---")
        return "\n".join(lines)
