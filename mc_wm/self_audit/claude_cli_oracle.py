"""
Claude Oracle: LLM-in-the-loop for MC-WM (md/MC-WM_Constraint_Subsystem_Report.md §6).

Primary transport: Anthropic Python SDK with ANTHROPIC_API_KEY.
Fallback transport: `claude -p` CLI subprocess (subscription plan).

Roles:
  #1 — Initial physical constraints  (called once at Phase 0)
  #2 — Feature hypotheses for the self-hypothesis loop (replaces hardcoded expansion)
  #3 — Runtime constraint audit / ICRL φ explanation

API key discovery order:
  1. ANTHROPIC_API_KEY env var (if already set)
  2. ~/.config/mcwm/api_key.env (sourced into env if present)
  3. No key → fall back to CLI subprocess

Responses cached to /tmp/mcwm_claude_cache/ by SHA256 of (model|prompt) so
repeated refits are free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from typing import Any, Optional


DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheapest tier — ~10-20× less than Sonnet
DEFAULT_CLI = "claude"
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_TOKENS = 2048  # Enriched prompts may generate ~1500-tok reasoning
                           # for Role #3; keep headroom to avoid truncation


def _load_api_key_from_config() -> Optional[str]:
    """Parse `export ANTHROPIC_API_KEY="..."` from ~/.config/mcwm/api_key.env."""
    path = os.path.expanduser("~/.config/mcwm/api_key.env")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export ANTHROPIC_API_KEY"):
                    _, _, val = line.partition("=")
                    return val.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


class ClaudeCLIOracle:
    """
    SDK-first Claude oracle with CLI fallback.

    Usage:
        oracle = ClaudeCLIOracle()          # auto-picks SDK if key available
        reply = oracle.ask("... ```json {...} ```")

    Or force CLI mode for debugging:
        oracle = ClaudeCLIOracle(force_cli=True)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cli: str = DEFAULT_CLI,
        timeout: int = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        log_fn=None,
        cache_dir: Optional[str] = "/tmp/mcwm_claude_cache",
        force_cli: bool = False,
    ) -> None:
        self.model = model
        self.cli = cli
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_errors = 0
        self.n_retries = 0

        # Pick transport: SDK if possible, else CLI.
        self._use_sdk = False
        self._sdk_client = None
        if not force_cli:
            api_key = os.environ.get("ANTHROPIC_API_KEY") or _load_api_key_from_config()
            if api_key:
                try:
                    import anthropic  # noqa: F401
                    self._sdk_client = anthropic.Anthropic(api_key=api_key)
                    self._use_sdk = True
                    self._log(f"  [Claude Oracle] SDK mode (key: ...{api_key[-4:]})")
                except ImportError:
                    self._log("  [Claude Oracle] anthropic SDK missing; falling back to CLI")
        if not self._use_sdk:
            self._log("  [Claude Oracle] CLI subprocess mode")

    # ──────────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.txt")

    def _read_cache(self, key: str) -> Optional[str]:
        p = self._cache_path(key)
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    return f.read()
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, value: str) -> None:
        p = self._cache_path(key)
        if not p:
            return
        try:
            with open(p, "w") as f:
                f.write(value)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # SDK transport (preferred)
    # ──────────────────────────────────────────────────────────────────

    def _call_sdk(self, prompt: str) -> str:
        """
        Call Anthropic SDK messages API with retry on transient errors.
        Returns concatenated text blocks, or "" on terminal failure.
        """
        import anthropic
        backoffs = [0, 2, 5, 10]  # 4 attempts
        last_err = ""
        for attempt, wait_s in enumerate(backoffs):
            if wait_s > 0:
                self.n_retries += 1
                self._log(f"  [Claude SDK] transient error, retry "
                          f"{attempt}/{len(backoffs)-1} after {wait_s}s ...")
                time.sleep(wait_s)
            try:
                resp = self._sdk_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                blocks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                return "\n".join(blocks).strip()
            except anthropic.APIStatusError as e:
                last_err = f"APIStatusError status={e.status_code} body={str(e)[:200]}"
                # Retry on 5xx / overloaded; stop on 4xx (auth/quota/bad request).
                if e.status_code < 500 and e.status_code != 429:
                    break
            except anthropic.APIConnectionError as e:
                last_err = f"APIConnectionError {e}"
            except anthropic.RateLimitError as e:
                last_err = f"RateLimitError {e}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                break  # unknown error → don't retry
        self._log(f"  [Claude SDK] all retries exhausted: {last_err}")
        return ""

    # ──────────────────────────────────────────────────────────────────
    # CLI transport (fallback)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_transient_api_error(out: str, err: str) -> bool:
        text = (out + "\n" + err).lower()
        return any(sig in text for sig in (
            "api error: 5", "internal server error", "overloaded",
            "rate limit", "gateway timeout", "service unavailable",
            "econnreset", "etimedout",
        ))

    def _call_cli_once(self, prompt: str, use_stdin: bool) -> tuple[int, str, str]:
        try:
            if use_stdin:
                result = subprocess.run(
                    [self.cli, "-p", "--model", self.model],
                    input=prompt, capture_output=True, text=True,
                    timeout=self.timeout,
                )
            else:
                result = subprocess.run(
                    [self.cli, "-p", prompt, "--model", self.model],
                    capture_output=True, text=True, timeout=self.timeout,
                )
            return result.returncode, result.stdout.strip(), result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"timeout after {self.timeout}s"
        except FileNotFoundError:
            return -2, "", f"`{self.cli}` not on PATH"

    def _call_cli(self, prompt: str) -> str:
        backoffs = [0, 2, 5, 10]
        last_rc, last_out, last_err = 0, "", ""
        for attempt, wait_s in enumerate(backoffs):
            if wait_s > 0:
                self.n_retries += 1
                self._log(f"  [Claude CLI] transient error, retry "
                          f"{attempt}/{len(backoffs)-1} after {wait_s}s ...")
                time.sleep(wait_s)
            rc, out, err = self._call_cli_once(prompt, use_stdin=True)
            last_rc, last_out, last_err = rc, out, err
            if rc == 0 and out:
                return out
            if not self._is_transient_api_error(out, err):
                break
        rc2, out2, err2 = self._call_cli_once(prompt, use_stdin=False)
        if rc2 == 0 and out2:
            return out2
        self._log("  [Claude CLI] All retries + fallback exhausted:")
        self._log(f"    stdin rc={last_rc} stderr={last_err or '(empty)'}")
        self._log(f"    arg   rc={rc2} stderr={err2 or '(empty)'}")
        return ""

    # ──────────────────────────────────────────────────────────────────
    # Top-level transport dispatcher
    # ──────────────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        cache_key = f"{self.model}|{prompt}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached
        self.n_calls += 1
        raw = self._call_sdk(prompt) if self._use_sdk else self._call_cli(prompt)
        if raw:
            self._write_cache(cache_key, raw)
        else:
            self.n_errors += 1
        return raw

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Pull the first ```json ... ``` block (or bare JSON) and parse."""
        if not text:
            return None
        m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        raw = m.group(1) if m else text
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = raw.find(opener), raw.rfind(closer)
            if 0 <= i < j:
                candidate = raw[i : j + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        return None

    def ask(self, prompt: str) -> Optional[Any]:
        return self._extract_json(self._call(prompt))

    def ask_text(self, prompt: str) -> str:
        return self._call(prompt)

    def stats(self) -> dict:
        return {
            "calls": self.n_calls,
            "cache_hits": self.n_cache_hits,
            "errors": self.n_errors,
            "retries": self.n_retries,
            "transport": "sdk" if self._use_sdk else "cli",
        }

    # ──────────────────────────────────────────────────────────────────
    # Role prompts
    # ──────────────────────────────────────────────────────────────────

    def role1_initial_constraints(self, env_description: str) -> list[dict]:
        prompt = f"""You are a physics-informed oracle for an MBRL safety system.

FRAMEWORK — You propose Type 2 OOD constraints ("shared forbidden zone"):
  Both sim and real agree: these states must NEVER occur.
  Example: cheetah below ground (z<0), torso fully inverted (|angle|>π).
  These are NOT Type 1 (extrapolation / unfamiliar-but-valid).  A
  feature-based residual model handles Type 1.  You handle physics
  impossibilities.

ENVIRONMENT DESCRIPTION:
{env_description}

TASK: Propose hard Type 2 physical constraints.  Requirements:
  - Consistent with physics in BOTH sim and real (shared forbidden)
  - Symbolic and interpretable (no neural net outputs)
  - State-dependent (not trivial global bounds)
  - NOT about extrapolation / unseen regimes (that's Type 1)

Respond with ONE json block containing a list of up to 8 constraints:

```json
[
  {{"name": "max_joint_angle", "check": "abs(s[1]) < 1.5", "why": "torso flip would make reward meaningless"}}
]
```

No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict) and "check" in c]
        return []

    def role2_feature_hypotheses(
        self,
        env_description: str,
        current_basis: list[str],
        diagnosis_summary: str,
        obs_dim: int,
        act_dim: int,
        round_num: int = 1,
        prev_accepted: list[str] | None = None,
        current_val_mse: float | None = None,
        nau_L_eff: float | None = None,
        residual_per_dim_std: list[float] | None = None,
        feature_history: list[dict] | None = None,
    ) -> list[dict]:
        prev_accepted = prev_accepted or []
        val_line = (f"Current SINDy+NAU val MSE: {current_val_mse:.5f}"
                    if current_val_mse is not None else "Val MSE: unknown")
        L_line = (f"NAU L_eff (Lipschitz bound): {nau_L_eff:.3f}"
                  if nau_L_eff is not None else "")
        per_dim_line = ""
        if residual_per_dim_std:
            top3 = sorted(enumerate(residual_per_dim_std), key=lambda x: -x[1])[:5]
            per_dim_line = ("Residual std by dim (top 5 most structured): " +
                            ", ".join(f"dim{i}={v:.3f}" for i, v in top3))
        history_line = ""
        if feature_history:
            history_line = ("\nPRIOR LLM FEATURES & THEIR FATES (learn from these):\n"
                            + json.dumps(feature_history[:10], indent=2))
        prompt = f"""You are a scientific-hypothesis oracle for a symbolic residual world model.

FRAMEWORK — Two OOD types, YOU handle only Type 1:
  Type 1 OOD ("extrapolation, no data but learnable pattern"):
    Example: we never had vz=8 m/s but we saw vz=1..5 and the pattern
    generalises.  Symptom: residual has structure (high std, oscillatory
    diagnosis, coefficient gap) that a richer basis could explain.
    ACTION: propose new symbolic FEATURES (your job).
  Type 2 OOD ("infeasible region, never-happen physics"):
    Example: cheetah levitating at z=10 m.  Symptom: correction
    magnitude is huge because the state itself is nonsensical.  A
    feature will NOT help — you cannot model "the impossible".
    ACTION: constraint system's Role #3 handles these; NOT your job.

If diagnosis_summary below hints Type 2 (e.g. "correction_magnitude is
15 when state has z=5", "extreme angular velocity states"), RETURN AN
EMPTY LIST so the constraint system picks it up.  Otherwise propose
Type 1 features.

ENVIRONMENT:
{env_description}

ROUND: {round_num} of the self-hypothesis loop.

PREVIOUSLY ACCEPTED LLM FEATURES (do NOT propose duplicates): {prev_accepted}

CURRENT SINDY BASIS ({len(current_basis)} features): {current_basis[:40]}

RESIDUAL DIAGNOSIS (what the current basis is MISSING):
{diagnosis_summary}

{val_line}
{L_line}
{per_dim_line}
{history_line}

OBS DIM = {obs_dim}, ACT DIM = {act_dim}.
State variables are s[0..{obs_dim-1}], action variables a[0..{act_dim-1}].

TASK: Propose up to 6 NEW basis features the current basis cannot express AND
that are NOT in PREVIOUSLY ACCEPTED.  Target the dims with highest residual
std (listed above).  Examples of non-poly2 structure:
  - sin/cos of an angle variable, possibly with integer multiplier
  - exp/tanh of velocity (bounded nonlinearities)
  - piecewise (indicator) functions using np.where
  - 3-way products like s[i]*s[j]*s[k]

Respond with ONE json block:
```json
[
  {{"expr": "np.sin(3*s[1])", "name": "sin_3theta", "why": "torque oscillates with 3x period on dim 1"}}
]
```
No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, list):
            return [f for f in parsed if isinstance(f, dict) and "expr" in f]
        return []

    # ──────────────────────────────────────────────────────────────────
    # Role #4 — Prune (evaluate + remove underperforming hypotheses / constraints)
    # ──────────────────────────────────────────────────────────────────

    def role4_prune_features(
        self,
        env_description: str,
        features_with_stats: list[dict],
        current_val_mse: float | None = None,
        n_base_features: int | None = None,
        n_llm_features: int | None = None,
    ) -> list[str]:
        """
        Review LLM-added SINDy features and recommend names to DROP.

        `features_with_stats[i]` = {
            "name": "llm_sin_3theta",
            "expr": "np.sin(3*s[1])",
            "max_abs_coef": 0.421,      # peak STLSQ coefficient across dims
            "n_dims_active": 2,         # in how many output dims coef > threshold
            "val_mse_delta": -0.003,    # drop in val MSE when added (negative = good)
        }

        Returns a list of NAMES to drop.  Conservative — LLM should return [] by
        default unless it has a clear physical reason to remove.
        """
        ctx_lines = []
        if current_val_mse is not None:
            ctx_lines.append(f"Current SINDy+NAU val MSE: {current_val_mse:.5f}")
        if n_base_features is not None:
            ctx_lines.append(f"Base (poly2 + NAU/NMU) features: {n_base_features}")
        if n_llm_features is not None:
            ctx_lines.append(f"LLM-added features: {n_llm_features}")
        ctx_block = "\n".join(ctx_lines)
        prompt = f"""You are an ablation oracle for a symbolic residual model.

ENVIRONMENT:
{env_description}

MODEL CONTEXT:
{ctx_block}

LLM-ADDED FEATURES (only these are eligible for pruning; base poly2 features are locked):
{json.dumps(features_with_stats, indent=2)}

TASK: For each feature, decide KEEP or DROP.  DROP when:
  - max_abs_coef is very small (< 0.02) — contributes negligibly
  - n_dims_active <= 1 AND max_abs_coef < 0.05 — likely curve fitting
  - val_mse_delta is positive (feature HURT val MSE when added)
  - the expression duplicates an already-present polynomial term
  - the physical justification ("why") does not actually apply to this
    environment

BE CONSERVATIVE: if uncertain, KEEP.  You may return an empty list.

Respond with ONE json block listing ONLY the names to drop:
```json
{{
  "drop": ["llm_name1", "llm_name2"],
  "reason_per_drop": {{"llm_name1": "coef near zero", "llm_name2": "duplicates poly2"}}
}}
```
No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, dict):
            drops = parsed.get("drop", [])
            if isinstance(drops, list):
                return [n for n in drops if isinstance(n, str)]
        return []

    def role4_prune_constraints(
        self,
        env_description: str,
        constraints_with_stats: list[dict],
        step: int = 0,
        system_reject_rate: float | None = None,
        violation_trend: list[float] | None = None,
        hardcoded_constraint_names: list[str] | None = None,
        decision_history: list[dict] | None = None,
        training_metrics: dict | None = None,
    ) -> list[str]:
        """
        Review LLM-added constraints and recommend names to DROP.

        `constraints_with_stats[i]` = {
            "name": "llm1_vertical_velocity_sanity",
            "expr": "abs(s[9]) > 20.0",
            "why": "...",
            "reject_count": 1243,
            "total_checks": 50000,
            "reject_rate": 0.0249,
        }

        Returns list of NAMES to drop.  DROP when rejection is too aggressive
        (false positives likely) or the constraint duplicates another.
        """
        sys_line = (f"SYSTEM-WIDE REJECT RATE: {system_reject_rate:.1%}"
                    if system_reject_rate is not None else "")
        trend_line = ""
        if violation_trend:
            trend_line = (f"VIOLATION/EP TREND (recent evals): "
                          f"{[f'{v:.2f}' for v in violation_trend[-5:]]}")
        hc_line = ""
        if hardcoded_constraint_names:
            hc_line = (f"HARDCODED PHYSICS CONSTRAINTS (cannot drop, listed for "
                       f"redundancy check):\n{hardcoded_constraint_names[:20]}")
        history_line = ""
        if decision_history:
            history_line = ("\nPRIOR LLM DECISIONS (your past adds/drops + outcomes):\n"
                            + json.dumps(decision_history[:10], indent=2))
        tm_line = ""
        if training_metrics:
            rt = training_metrics.get("reward_trend", [])
            tm_line = (f"\nTRAINING STATE: buffer_size="
                       f"{training_metrics.get('buffer_size','?')}, "
                       f"recent_reward={[round(r,1) for r in rt[-5:]]}")
        prompt = f"""You are an ablation oracle for a model-based RL constraint system.

TRAIN STEP: {step}
{sys_line}
{trend_line}
{tm_line}

ENVIRONMENT:
{env_description}

{hc_line}
{history_line}

LLM-ADDED CONSTRAINTS (only these are eligible for pruning):
{json.dumps(constraints_with_stats, indent=2)}

TASK: For each LLM-added constraint, decide KEEP or DROP.  DROP when:
  - reject_rate > 0.3 — filtering valid transitions (false positive)
  - reject_count == 0 after many (>500) checks — the constraint is vacuous
  - the constraint is weaker/duplicate of a HARDCODED constraint above
  - the physical justification no longer applies given current dynamics
  - violation_trend is already at zero — safety net unnecessary

BE CONSERVATIVE: if uncertain, KEEP.  You may return an empty list.

Respond with ONE json block:
```json
{{
  "drop": ["llm1_name1", "llm3_name2"],
  "reason_per_drop": {{"llm1_name1": "reject rate 0.45 too high", ...}}
}}
```
No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, dict):
            drops = parsed.get("drop", [])
            if isinstance(drops, list):
                return [n for n in drops if isinstance(n, str)]
        return []

    def role3_audit(
        self,
        env_description: str,
        suspicious_transitions: list[dict],
        current_constraints: list[str],
        step: int = 0,
        system_reject_rate: float | None = None,
        per_constraint_stats: list[dict] | None = None,
        decision_history: list[dict] | None = None,
        training_metrics: dict | None = None,
    ) -> dict:
        stats_line = ""
        if per_constraint_stats:
            # Show each constraint's reject rate to help LLM spot redundancy.
            stats_line = "\nPER-CONSTRAINT REJECT RATES (so far in training):\n"
            for s in per_constraint_stats[:30]:
                stats_line += (f"  {s.get('name','?')}: "
                               f"rate={s.get('reject_rate',0):.3f} "
                               f"({s.get('reject_count',0)}/{s.get('total_checks',0)})\n")
        sysrate_line = (f"\nSYSTEM-WIDE REJECT RATE: {system_reject_rate:.1%}"
                        if system_reject_rate is not None else "")
        history_line = ""
        if decision_history:
            history_line = ("\nPRIOR LLM DECISIONS (your past adds/drops + outcomes — "
                            "learn from these):\n"
                            + json.dumps(decision_history[:10], indent=2))
        tm_line = ""
        if training_metrics:
            rt = training_metrics.get("reward_trend", [])
            tm_line = (f"\nTRAINING STATE: buffer_size="
                       f"{training_metrics.get('buffer_size','?')}, "
                       f"recent_reward={[round(r,1) for r in rt[-5:]]}")
        prompt = f"""You are a runtime safety auditor for an MBRL system.

FRAMEWORK — Two OOD types:
  Type 1 OOD ("extrapolation, learnable pattern"):
    Example: correction is large because the model hasn't seen enough
    data in this regime, but the dynamics are valid.  Solution: add
    a feature to the residual model (NOT your job).
  Type 2 OOD ("infeasible region, shared forbidden zone"):
    Example: state violates physics (z < 0, |angle| > π).  Symptom:
    correction is large AND the state itself is impossible.  A
    constraint should prevent policy from visiting here.
    YOUR JOB.

If the suspicious transitions below are Type 1 (large correction but
states are physically plausible), set verdict="valid_large_correction"
and leave new_constraint=null.  Only propose a new constraint when the
state itself violates shared physics.

TRAIN STEP: {step}
{tm_line}

ENVIRONMENT:
{env_description}

ALL CURRENT HARD CONSTRAINTS ({len(current_constraints)} total):
{json.dumps(current_constraints, indent=2)}

{stats_line}{sysrate_line}
{history_line}

RECENT SUSPICIOUS TRANSITIONS (pass all current constraints but have unusually
large correction magnitude):
{json.dumps(suspicious_transitions[:10], indent=2)}

TASK: Decide:
  1. Are these transitions genuinely infeasible, or is the model's residual
     correction large for valid physical reasons?
  2. If infeasible: propose ONE new constraint that would catch these AND is
     NOT duplicate of an existing one listed above.
  3. If valid: set new_constraint to null and explain.

Respond with ONE json block:
```json
{{
  "verdict": "infeasible" | "valid_large_correction",
  "reasoning": "...",
  "new_constraint": {{"name": "...", "check": "...", "why": "..."}} | null
}}
```
No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, dict):
            return parsed
        return {}


    # ──────────────────────────────────────────────────────────────────
    # Role #5 — Meta-hyperparameter orchestrator (LLM-as-BO)
    # ──────────────────────────────────────────────────────────────────

    def role5_tune_hyperparameters(
        self,
        env_description: str,
        current_hp: dict,
        hp_schema: dict,
        trial_history: list[dict],
        training_metrics: dict,
    ) -> dict:
        """
        LLM acts as a Bayesian-optimization-style meta-optimizer over the
        hyperparameter space defined by `hp_schema`.

        `current_hp`: current HP values, e.g. {"qdelta_gamma": 0.5, ...}
        `hp_schema`: {"qdelta_gamma": {"type": "float", "range": [0.0, 0.95]},
                       "icrl_combine": {"type": "enum", "choices": ["top_k","soft"]}}
        `trial_history`: [{"hp": {...}, "reward": 5388, "viol": 0.03,
                            "val_mse": 0.42, "step": 50000}, ...]
        `training_metrics`: {"reward_trend": [...], "violation_trend": [...],
                              "buffer_size": N, "val_mse": ...}

        Returns: dict of proposed HP values to adopt for the next training
        phase.  ALL values MUST be within the schema's legal range.  LLM is
        encouraged to stay close to current_hp unless trial_history gives
        strong evidence to move.  Returns empty dict on failure.
        """
        prompt = f"""You are a Bayesian-optimization meta-agent for an MBRL training pipeline.

FRAMEWORK — You are the outer-loop optimizer over hyperparameters.  Your
job is to pick the next hyperparameter configuration to try, given past
trials and current training state.  Think of yourself as the acquisition
function in BO — you balance exploration (try unexplored regions) and
exploitation (stay near known-good configurations).

ENVIRONMENT:
{env_description}

CURRENT HYPERPARAMETERS:
{json.dumps(current_hp, indent=2)}

LEGAL HP SCHEMA (range / enum choices):
{json.dumps(hp_schema, indent=2)}

TRIAL HISTORY (past HPs and their final performance, newest last):
{json.dumps(trial_history[-8:], indent=2)}

CURRENT TRAINING STATE:
  reward_trend (last 5):   {training_metrics.get('reward_trend', [])[-5:]}
  violation_trend (last 5): {training_metrics.get('violation_trend', [])[-5:]}
  val_mse:                 {training_metrics.get('val_mse', 'unknown')}
  buffer_size:             {training_metrics.get('buffer_size', 'unknown')}

TASK: Propose the NEXT HP values.  Rules:
  1. ALL values MUST be within the schema's range or choices.
  2. Change AT MOST 3 hyperparameters at a time (BO convention —
     one-at-a-time perturbation gives clean attribution).
  3. Justify each change with a reason tied to observed data.
  4. If current HPs are already near-optimal (reward plateau AND
     violations low), return {{}} (empty) — no change.
  5. BE CONSERVATIVE: prefer small moves (<=20% change for floats)
     unless trial_history shows current region is bad.

Respond with ONE json block:
```json
{{
  "proposed_hp": {{"qdelta_gamma": 0.7, "rollout_batch": 500}},
  "reasons": {{
    "qdelta_gamma": "reward plateaued at γ=0.5; try deeper Bellman horizon",
    "rollout_batch": "model MSE stable, can reduce compute by larger batches"
  }},
  "expected_reward_direction": "+",
  "expected_viol_direction": "0"
}}
```
No prose outside the JSON block.
"""
        parsed = self.ask(prompt)
        if isinstance(parsed, dict) and isinstance(parsed.get("proposed_hp"), dict):
            return parsed
        return {}


_default_oracle: Optional[ClaudeCLIOracle] = None


def get_default_oracle() -> ClaudeCLIOracle:
    global _default_oracle
    if _default_oracle is None:
        _default_oracle = ClaudeCLIOracle()
    return _default_oracle
