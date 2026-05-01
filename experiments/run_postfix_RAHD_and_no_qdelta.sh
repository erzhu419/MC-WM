#!/usr/bin/env bash
# Two parallel slots, both on gravity_soft_ceiling × 3 seeds:
#   slot R — TRUE post-fix RAHD c9 (the headline number paper claims)
#   slot Q — TRUE no-QΔ ablation via --disable_qdelta_filter
# Logs to runs/RAHD_postfix/ and runs/no_qdelta_true/.
#
# These together resolve reviewer issues #2 (E4 was legacy not RAHD) and
# #3 (the "no QΔ" ablation didn't actually disable the filter).

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR_R="$REPO/runs/RAHD_postfix"
LOGDIR_Q="$REPO/runs/no_qdelta_true"
mkdir -p "$LOGDIR_R" "$LOGDIR_Q"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

run_RAHD() {
  local seed=$1
  local log="$LOGDIR_R/gravity_soft_ceiling_s${seed}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[RAHD-postfix $(date +%H:%M:%S)] s$seed finished, skipping"
    return 0
  fi
  echo "[RAHD-postfix $(date -Is)] launching s$seed"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_soft_ceiling --seed "$seed" \
    --qdelta_gamma 0.5 --bapr_warmup_iters 100 --qdelta_belief_sig \
    --rahd_feature_pool --rahd_max_delta_beta 5.0 --rahd_policy_aware \
    --use_claude_llm --llm_backend glm \
    > "$log" 2>&1
  echo "[RAHD-postfix $(date -Is)] done s$seed rc=$?"
}

run_NoQD() {
  local seed=$1
  local log="$LOGDIR_Q/gravity_soft_ceiling_s${seed}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[no-QΔ $(date +%H:%M:%S)] s$seed finished, skipping"
    return 0
  fi
  echo "[no-QΔ $(date -Is)] launching s$seed"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_soft_ceiling --seed "$seed" \
    --qdelta_gamma 0.5 --bapr_warmup_iters 100 --qdelta_belief_sig \
    --rahd_feature_pool --rahd_max_delta_beta 5.0 --rahd_policy_aware \
    --use_claude_llm --llm_backend glm \
    --disable_qdelta_filter \
    > "$log" 2>&1
  echo "[no-QΔ $(date -Is)] done s$seed rc=$?"
}

slot_R() {
  for i in 1 2 3; do
    for s in 42 123 456; do
      run_RAHD "$s"
    done
    local rem=0
    for s in 42 123 456; do
      grep -q "c9 last 3 avg: real=" "$LOGDIR_R/gravity_soft_ceiling_s${s}.log" 2>/dev/null || rem=$((rem+1))
    done
    [ $rem -eq 0 ] && break
    echo "[slot_R] retry pass $i, rem=$rem"
  done
  echo "[slot_R] DONE"
}

slot_Q() {
  for i in 1 2 3; do
    for s in 42 123 456; do
      run_NoQD "$s"
    done
    local rem=0
    for s in 42 123 456; do
      grep -q "c9 last 3 avg: real=" "$LOGDIR_Q/gravity_soft_ceiling_s${s}.log" 2>/dev/null || rem=$((rem+1))
    done
    [ $rem -eq 0 ] && break
    echo "[slot_Q] retry pass $i, rem=$rem"
  done
  echo "[slot_Q] DONE"
}

slot_R > "$LOGDIR_R/_slot.log" 2>&1 &
PID_R=$!
slot_Q > "$LOGDIR_Q/_slot.log" 2>&1 &
PID_Q=$!
echo "[master $(date -Is)] slots R=$PID_R Q=$PID_Q"
wait $PID_R $PID_Q
echo "[master $(date -Is)] all slots complete"
