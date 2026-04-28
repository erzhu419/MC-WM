#!/usr/bin/env bash
# E4: multi-seed full ablation matrix on gravity_soft_ceiling.
# Re-runs the legacy single-seed ablations (Pareto gate / Bellman QΔ /
# Role #5 / LLM Roles) with three seeds each, plus a full c9-base
# control with the same seeds.  All flags below come from the legacy
# ablation script that produced the s42 numbers in tab:ablation.
#
# 5 configs × 3 seeds = 15 runs, sequential within this slot.
# Logs to runs/E4/.
#
# This is the "c9 (legacy)" pipeline (no RAHD module).  RAHD ablation
# is a separate sweep; we keep them separate so the table comparisons
# are not confounded.

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs/E4"
mkdir -p "$LOGDIR"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

# Legacy c9 baseline flags (matching the qdg50 results in tab:ablation).
COMMON=(
  --mode c9
  --env gravity_soft_ceiling
  --qdelta_gamma 0.5
  --use_claude_llm
  --llm_backend glm
  --use_role5_hp
  --role5_pareto_gate
  --role5_viol_hard_cap 20.0
)

# Ablation removals (tag, extra-flags-or-removals).  Each removal is a
# minimal change from the full c9 config above.
declare -A FLAGS=(
  ["full_c9"]=""
  ["no_pareto_gate"]="--role5_pareto_gate=false"
  ["no_qdelta"]="--qdelta_gamma 0"
  ["no_role5"]="--use_role5_hp=false"
  ["no_llm"]="--use_claude_llm=false"
)
ORDER=(full_c9 no_pareto_gate no_qdelta no_role5 no_llm)

run_one() {
  local tag=$1 seed=$2
  local extra=${FLAGS[$tag]}
  local log="$LOGDIR/${tag}_s${seed}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[E4 $(date +%H:%M:%S)] ${tag}_s${seed} finished, skipping"
    return 0
  fi
  echo "[E4 $(date -Is)] launching ${tag}_s${seed}"
  # Build full flag list, filtering out '<flag>=false' tokens that mean
  # "remove this flag from COMMON".  Argparse store_true treats absence
  # as false, so we just drop them.
  local cmd_flags=()
  local removals=()
  for tok in $extra; do
    if [[ "$tok" == --*=false ]]; then
      removals+=("${tok%=false}")
    else
      cmd_flags+=("$tok")
    fi
  done
  # Filter COMMON to drop removed flags (and their next argument if value-bearing).
  local filtered=()
  local skip_next=0
  for tok in "${COMMON[@]}"; do
    if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
    local drop=0
    for r in "${removals[@]}"; do
      if [[ "$tok" == "$r" ]]; then drop=1; break; fi
    done
    if [[ $drop -eq 1 ]]; then continue; fi
    filtered+=("$tok")
  done
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    "${filtered[@]}" "${cmd_flags[@]}" --seed "$seed" \
    > "$log" 2>&1
  echo "[E4 $(date -Is)] done ${tag}_s${seed} rc=$?"
  grep "c9 last 3 avg" "$log" | tail -2
}

echo "[E4 $(date -Is)] starting 5 configs × 3 seeds = 15 runs"
for tag in "${ORDER[@]}"; do
  for seed in 42 123 456; do
    run_one "$tag" "$seed"
  done
done
echo "[E4 $(date -Is)] all 15 runs complete"
