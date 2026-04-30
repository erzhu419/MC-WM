#!/usr/bin/env bash
# E5: single-factor gap-taxonomy sweep (HalfCheetah morphology only).
#
# Three factor sweeps:
#   gravity   ∈ {1.5, 2.0, 3.0, 4.0}     — physics-dynamics gap (works-side)
#   obs_noise ∈ {0.5, 1.0, 2.0, 5.0}     — observation gap (fails-side)
#   actuator  ∈ {0.5, 1.0, 2.0}          — actuator gap (fails-side)
#
# 11 envs × 3 seeds = 33 runs.  Each ~25-50 min serial; ~3-slot parallel
# total ≈ 9 h.  Logs to runs/E5/.

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs/E5"
mkdir -p "$LOGDIR"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

run_one() {
  local env=$1 seed=$2
  local log="$LOGDIR/${env}_s${seed}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[E5 $(date +%H:%M:%S)] ${env}_s${seed} finished, skipping"
    return 0
  fi
  echo "[E5 $(date -Is)] launching ${env}_s${seed}"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    --mode c9 --env "$env" --seed "$seed" \
    --qdelta_gamma 0.5 --bapr_warmup_iters 100 --qdelta_belief_sig \
    --rahd_feature_pool --rahd_max_delta_beta 5.0 --rahd_policy_aware \
    --use_claude_llm --llm_backend glm \
    > "$log" 2>&1
  echo "[E5 $(date -Is)] done ${env}_s${seed} rc=$?"
  grep "c9 last 3 avg" "$log" | tail -2
}

ENVS=(
  gravity_sweep_15 gravity_sweep_20 gravity_sweep_30 gravity_sweep_40
  obs_noise_05 obs_noise_10 obs_noise_20 obs_noise_50
  actuator_scale_05 actuator_scale_10 actuator_scale_20
)

# Three slots: split envs across slots.  Each slot processes its envs
# sequentially and retries up to 3 times.
slot_runner() {
  local slot_id=$1 ; shift
  local envs=("$@")
  for i in 1 2 3; do
    for env in "${envs[@]}"; do
      for seed in 42 123 456; do
        run_one "$env" "$seed"
      done
    done
    # exit early if all slot envs/seeds done
    local rem=0
    for env in "${envs[@]}"; do
      for seed in 42 123 456; do
        grep -q "c9 last 3 avg: real=" "$LOGDIR/${env}_s${seed}.log" 2>/dev/null || rem=$((rem+1))
      done
    done
    if [ "$rem" -eq 0 ]; then break; fi
    echo "[slot $slot_id] retry pass $i, rem=$rem"
  done
  echo "[slot $slot_id] DONE"
}

# 11 envs split into 3 slots: 4 / 4 / 3
slot_runner A "${ENVS[@]:0:4}"  > "$LOGDIR/_slot_A.log" 2>&1 &
PID_A=$!
slot_runner B "${ENVS[@]:4:4}"  > "$LOGDIR/_slot_B.log" 2>&1 &
PID_B=$!
slot_runner C "${ENVS[@]:8:3}"  > "$LOGDIR/_slot_C.log" 2>&1 &
PID_C=$!
echo "[E5 $(date -Is)] slots A=$PID_A B=$PID_B C=$PID_C"
wait $PID_A $PID_B $PID_C
echo "[E5 $(date -Is)] all slots complete"
