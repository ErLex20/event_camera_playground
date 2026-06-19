#!/usr/bin/env bash
# Warp-compensation experiment on DSEC thun_00_a.
# Runs config E (prior+reg+EMA off, order=2, 6 scales = 32x32) for three warp
# modes {off, coarse, gt}, saving each mode's dense flow under
# logs/moment_flow/warp_exp/<mode>. Restores the original config on exit.
set -u
WS=/home/neo/workspace
CFG=$WS/src/event_detector_cpp/config/event_detector_cpp.yaml
GT_DIR=$WS/logs/dsec/thun_00_a/optical_flow_forward
PATCH="python3 $WS/tools/dsec_flow_benchmark/warp_config_patch.py $CFG"

BACKUP=$(mktemp)
cp "$CFG" "$BACKUP"
restore() { cp "$BACKUP" "$CFG"; rm -f "$BACKUP"; echo ">>> config restored"; }
trap restore EXIT

# Config E (isolate the data solve): prior tiny, no spatial reg, no temporal EMA.
E_OVERRIDES=(
  flow_num_scales=6
  flow_time_aware_order=2
  flow_max_window_ms=20.0
  flow_prior_lambda=0.05
  flow_reg_lambda=0.0
  flow_reg_sweeps=0
  flow_track_gamma=0.0
)

run_mode() {
  local mode="$1"; shift
  local out="$WS/logs/moment_flow/warp_exp/$mode"
  echo "======== MODE: $mode -> $out ========"
  cp "$BACKUP" "$CFG"                          # start from pristine each time
  $PATCH "${E_OVERRIDES[@]}" \
    "flow_warp_compensation=$mode" \
    "flow_save_output_dir=$out" \
    "$@"
  mkdir -p "$out"
  REPLAY_TIMEOUT=${REPLAY_TIMEOUT:-150} bash "$WS/tools/dsec_flow_benchmark/run_flow_test.sh" \
    2>&1 | grep -E 'RUN_DIR|saved:|EPE=|DONE|FATAL|Error|Traceback' || true
}

run_mode off
run_mode coarse
run_mode gt "flow_warp_gt_dir=$GT_DIR"

echo ">>> ALL WARP RUNS DONE"
