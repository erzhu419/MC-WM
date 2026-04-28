#!/usr/bin/env bash
# E1 slot A: RAHD c9 on gravity_ceiling × 3 seeds (sequential within slot).
set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs/E1"
mkdir -p "$LOGDIR"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$REPO"

run_one() {
  local seed=$1
  local tag="gravity_ceiling_s${seed}"
  local log="$LOGDIR/${tag}.log"
  if grep -q "c9 last 3 avg: real=" "$log" 2>/dev/null; then
    echo "[E1-gc $(date +%H:%M:%S)] $tag finished, skipping"
    return 0
  fi
  echo "[E1-gc $(date -Is)] launching $tag"
  conda run --no-capture-output -n MC-WM python3 -u \
    experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_ceiling --seed "$seed" \
    --qdelta_gamma 0.5 --bapr_warmup_iters 100 --qdelta_belief_sig \
    --rahd_feature_pool --rahd_max_delta_beta 5.0 --rahd_policy_aware \
    --use_claude_llm --llm_backend glm \
    > "$log" 2>&1
  echo "[E1-gc $(date -Is)] done $tag rc=$?"
  grep "c9 last 3 avg" "$log" | tail -2
}

echo "[E1-gc $(date -Is)] starting gravity_ceiling × 3 seeds"
for seed in 42 123 456; do
  run_one "$seed"
done
echo "[E1-gc $(date -Is)] all 3 runs complete"
