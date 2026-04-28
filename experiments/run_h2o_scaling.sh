#!/usr/bin/env bash
# H2O+ data-scaling queue: 9 runs = ratio ∈ {0.1, 0.25, 0.5} × seed ∈ {42, 123, 456}.
#
# Logs and checkpoints go to persistent paths (not /tmp) to survive
# WSL2 reboots:
#   logs:  MC-WM/runs/h2o_scaling/<ratio>_s<seed>.log
#   ckpts: ~/mc_wm_h2o_ckpts/<ratio>_s<seed>/ckpt.pt
#
# Each run takes ~4h on a single A-class GPU. Serial total ≈ 36h.
# Resume on interruption: just re-run this script — h2o+_main.py auto-loads
# ckpt.pt and continues from saved epoch.
#
# Usage: ./experiments/run_h2o_scaling.sh

set -u
H2O_DIR=/home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC
LOG_DIR=/home/erzhu419/mine_code/MC-WM/runs/h2o_scaling
CKPT_BASE=/home/erzhu419/mc_wm_h2o_ckpts
PY=/home/erzhu419/anaconda3/envs/PBRL/bin/python

mkdir -p "$LOG_DIR" "$CKPT_BASE"
[ -f "$HOME/.api_keys" ] && . "$HOME/.api_keys"
cd "$H2O_DIR"

run_one() {
  local ratio=$1 seed=$2
  local tag="ratio${ratio//./p}_s${seed}"
  local log="$LOG_DIR/${tag}.log"
  local ckpt_dir="$CKPT_BASE/${tag}"

  if [ -f "$ckpt_dir/ckpt.pt.done" ]; then
    echo "[scaling $(date +%H:%M:%S)] $tag already complete (.done marker), skipping"
    return 0
  fi

  echo "[scaling $(date -Is)] launching $tag"
  "$PY" -u h2o+_main.py \
    --env_list=HalfCheetah-v2 \
    --data_source=medium_replay \
    --unreal_dynamics=gravity \
    --variety_list=2.0 \
    --seed="${seed}" \
    --n_epochs=50 \
    --n_rollout_steps_per_epoch=1000 \
    --n_train_step_per_epoch=1000 \
    --real_residual_ratio="${ratio}" \
    --eval_period=10 \
    --logging.online=False \
    --logging.output_dir="${ckpt_dir}/wandb" \
    --ckpt_dir="${ckpt_dir}" \
    --ckpt_every=5 \
    >> "${log}" 2>&1
  local rc=$?
  echo "[scaling $(date -Is)] done $tag rc=${rc}"
  grep "average_return" "${log}" | tail -3
}

echo "[scaling $(date -Is)] starting 9-run queue"
for ratio in 0.1 0.25 0.5; do
  for seed in 42 123 456; do
    run_one "$ratio" "$seed"
  done
done
echo "[scaling $(date -Is)] all 9 runs complete"
