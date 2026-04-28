#!/usr/bin/env bash
# Master 3-slot parallel runner with retry-on-OOM-kill semantics.
# Each orchestrator script self-skips runs whose log already contains
# the success marker, so repeated invocations retry only failed runs.
#
# Slots:
#   A: E1 gravity_ceiling × 3 + friction_walker × 3 (sequential within)
#   B: E3 feature-library baselines × 12 (sequential within)
#   C: E4 ablation matrix × 15 (sequential within)
#
# 3 parallel python processes use ~9-12 GB RAM, comfortably under
# the 13 GB available on this 29 GB box; 4 was OOM-killed.

set -u
REPO=/home/erzhu419/mine_code/MC-WM
LOGDIR="$REPO/runs"
MASTER_LOG="$LOGDIR/master_orchestrator.log"
mkdir -p "$LOGDIR"

# Slot A: chain E1-gc and E1-fw sequentially
slot_A() {
  for i in 1 2 3; do  # retry up to 3 times
    "$REPO/experiments/run_E1_gravity_ceiling.sh"
    "$REPO/experiments/run_E1_friction_walker.sh"
    # break early if both fully done
    local rem_gc=0 rem_fw=0
    for s in 42 123 456; do
      grep -q "c9 last 3 avg: real=" "$LOGDIR/E1/gravity_ceiling_s${s}.log" 2>/dev/null || rem_gc=$((rem_gc+1))
      grep -q "c9 last 3 avg: real=" "$LOGDIR/E1/friction_walker_soft_ceiling_s${s}.log" 2>/dev/null || rem_fw=$((rem_fw+1))
    done
    if [ $((rem_gc + rem_fw)) -eq 0 ]; then break; fi
    echo "[slot_A] retry pass $i complete, remaining gc=$rem_gc fw=$rem_fw"
  done
  echo "[slot_A] DONE"
}

slot_B() {
  for i in 1 2 3; do
    "$REPO/experiments/run_E3_feature_baselines.sh"
    local rem=0
    for cfg in poly2_only poly3_only trig_only random_100; do
      for s in 42 123 456; do
        grep -q "c9 last 3 avg: real=" "$LOGDIR/E3/${cfg}_s${s}.log" 2>/dev/null || rem=$((rem+1))
      done
    done
    if [ $rem -eq 0 ]; then break; fi
    echo "[slot_B] retry pass $i complete, remaining=$rem"
  done
  echo "[slot_B] DONE"
}

slot_C() {
  for i in 1 2 3; do
    "$REPO/experiments/run_E4_ablation_matrix.sh"
    local rem=0
    for cfg in full_c9 no_pareto_gate no_qdelta no_role5 no_llm; do
      for s in 42 123 456; do
        grep -q "c9 last 3 avg: real=" "$LOGDIR/E4/${cfg}_s${s}.log" 2>/dev/null || rem=$((rem+1))
      done
    done
    if [ $rem -eq 0 ]; then break; fi
    echo "[slot_C] retry pass $i complete, remaining=$rem"
  done
  echo "[slot_C] DONE"
}

echo "[master $(date -Is)] launching 3 slots in parallel" | tee -a "$MASTER_LOG"
slot_A > "$LOGDIR/E1/_slot_A.log" 2>&1 &
PID_A=$!
slot_B > "$LOGDIR/E3/_slot_B.log" 2>&1 &
PID_B=$!
slot_C > "$LOGDIR/E4/_slot_C.log" 2>&1 &
PID_C=$!
echo "[master $(date -Is)] PIDs: A=$PID_A B=$PID_B C=$PID_C" | tee -a "$MASTER_LOG"
wait $PID_A $PID_B $PID_C
echo "[master $(date -Is)] all slots complete" | tee -a "$MASTER_LOG"
