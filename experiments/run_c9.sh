#!/usr/bin/env bash
# Launcher for c9 (RAHD + P0 + P1 + GLM) runs.
# Logs go to MC-WM/runs/<timestamp>_<tag>.log so they survive WSL2 reboots
# (unlike /tmp).
#
# Usage:
#   ./experiments/run_c9.sh <preset> <seed> [extra flags]
#
# Presets:
#   full       — RAHD + P0 + P1 + GLM   (recommended; 4919 ± 187, viol 0.82)
#   noP1       — RAHD + P0   + GLM     (no belief-cond QΔ)
#   noRAHD     — P0 only                (no RAHD modules)
#   bare       — legacy c9               (none of P0/P1/RAHD)
#
# Extra flags after the seed are forwarded as-is to step2_mbrl_residual.py
# (e.g. ``--env friction_walker_soft_ceiling --qdelta_gamma 0.3``).

set -u
PRESET=${1:-full}
SEED=${2:-42}
shift 2 || true

REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs"
mkdir -p "$LOGDIR"
STAMP=$(date '+%Y%m%d_%H%M%S')
TAG="${PRESET}_s${SEED}"
LOG="$LOGDIR/${STAMP}_${TAG}.log"

# Common flags shared by all presets.
COMMON=(
  --mode c9
  --env gravity_soft_ceiling
  --seed "$SEED"
  --qdelta_gamma 0.5
)

case "$PRESET" in
  full)
    PRESET_FLAGS=(
      --bapr_warmup_iters 100
      --qdelta_belief_sig
      --rahd_feature_pool
      --rahd_max_delta_beta 5.0
      --rahd_policy_aware
      --use_claude_llm
      --llm_backend glm
    )
    ;;
  noP1)
    PRESET_FLAGS=(
      --bapr_warmup_iters 100
      --rahd_feature_pool
      --rahd_max_delta_beta 5.0
      --rahd_policy_aware
      --use_claude_llm
      --llm_backend glm
    )
    ;;
  noRAHD)
    PRESET_FLAGS=(
      --bapr_warmup_iters 100
    )
    ;;
  bare)
    PRESET_FLAGS=(
      --bapr_warmup_iters 0
      --no_reward_norm
    )
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Valid: full | noP1 | noRAHD | bare" >&2
    exit 2
    ;;
esac

# Source persistent API keys (GLM_API_KEY, GLM_BASE_URL, etc.).
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"

cd "$REPO"
echo "[$(date -Is)] launching: preset=$PRESET seed=$SEED log=$LOG"
echo "[$(date -Is)] extra flags: $*"
exec conda run --no-capture-output -n MC-WM python3 -u \
  experiments/step2_mbrl_residual.py \
  "${COMMON[@]}" "${PRESET_FLAGS[@]}" "$@" \
  > "$LOG" 2>&1
