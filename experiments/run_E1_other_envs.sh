#!/usr/bin/env bash
# E1: RAHD c9 on the two environments where we don't have it yet
# (gravity_ceiling, friction_walker_soft_ceiling), 3 seeds each.
# Logs to MC-WM/runs/E1/, persistent.
#
# Each run ~50 min on a single A-class GPU. Total ~5 h serial.

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs/E1"
mkdir -p "$LOGDIR"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

run_one() {
  local env=$1 seed=$2
  local tag="${env}_s${seed}"
  local log="$LOGDIR/${tag}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[E1 $(date +%H:%M:%S)] $tag already finished (resume), skipping"
    return 0
  fi
  echo "[E1 $(date -Is)] launching $tag"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    --mode c9 --env "$env" --seed "$seed" \
    --qdelta_gamma 0.5 \
    --bapr_warmup_iters 100 \
    --qdelta_belief_sig \
    --rahd_feature_pool \
    --rahd_max_delta_beta 5.0 \
    --rahd_policy_aware \
    --use_claude_llm \
    --llm_backend glm \
    > "$log" 2>&1
  local rc=$?
  echo "[E1 $(date -Is)] done $tag rc=${rc}"
  grep "c9 last 3 avg" "$log" | tail -2
}

echo "[E1 $(date -Is)] starting RAHD c9 sweep on 2 envs × 3 seeds"
for env in gravity_ceiling friction_walker_soft_ceiling; do
  for seed in 42 123 456; do
    run_one "$env" "$seed"
  done
done
echo "[E1 $(date -Is)] all 6 runs complete"
