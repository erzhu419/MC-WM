#!/usr/bin/env bash
# E3: matched feature-library baselines on gravity_soft_ceiling × 3 seeds.
# Each baseline uses NAU on a fixed feature library WITHOUT LLM Role #2
# and WITHOUT orthogonal expansion, so the only nonlinearity comes from
# the library itself.  Compares against the existing P0+RAHD+P1+GLM
# (Table tab:p1-isolation) result of 4919 ± 187.
#
# Configs:
#   poly2_only   : poly2 + NAU only
#   poly3_only   : poly3 + NAU only (richer library)
#   trig_only    : sin/cos library + NAU only
#   random_100   : 100 random Fourier features + NAU only
#
# 4 configs × 3 seeds = 12 runs, sequential within this slot.

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs/E3"
mkdir -p "$LOGDIR"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

# Common flags: c9 pipeline minus LLM (no Role #2/#1/#3), no RAHD-A pool,
# no policy-aware fit, no QΔ-belief.  This isolates the FEATURE-LIBRARY
# contribution; all other RL/safety machinery matches.
COMMON=(
  --mode c9
  --env gravity_soft_ceiling
  --qdelta_gamma 0.5
  --bapr_warmup_iters 100
)

declare -A FLIBS=(
  ["poly2_only"]="--feature_library poly2_only"
  ["poly3_only"]="--feature_library poly3_only"
  ["trig_only"]="--feature_library trig_only"
  ["random_100"]="--feature_library random --random_feature_count 100"
)
ORDER=(poly2_only poly3_only trig_only random_100)

run_one() {
  local tag=$1 seed=$2
  local extra=${FLIBS[$tag]}
  local log="$LOGDIR/${tag}_s${seed}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[E3 $(date +%H:%M:%S)] ${tag}_s${seed} finished, skipping"
    return 0
  fi
  echo "[E3 $(date -Is)] launching ${tag}_s${seed}"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    "${COMMON[@]}" $extra --seed "$seed" \
    > "$log" 2>&1
  echo "[E3 $(date -Is)] done ${tag}_s${seed} rc=$?"
  grep "c9 last 3 avg" "$log" | tail -2
}

echo "[E3 $(date -Is)] starting 4 configs × 3 seeds = 12 runs"
for tag in "${ORDER[@]}"; do
  for seed in 42 123 456; do
    run_one "$tag" "$seed"
  done
done
echo "[E3 $(date -Is)] all 12 runs complete"
