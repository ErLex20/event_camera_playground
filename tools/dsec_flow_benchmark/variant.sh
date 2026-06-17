#!/usr/bin/env bash
# variant.sh NAME [flow_key=value ...]
# Resets all tunable flow_* params to the iter00 baseline, applies overrides,
# points output at smooth_flow_search/NAME, and runs one benchmark iteration.
# Constraints held fixed: flow_num_scales=6, flow_max_window_ms=20.
set -u
WS=/home/neo/workspace
CFG=$WS/src/event_detector_cpp/config/event_detector_cpp.yaml
NAME=$1; shift

# baseline (iter00) tunable values
declare -A P=(
  [flow_max_window_ms]=20.0
  [flow_num_scales]=6
  [flow_cell_size_px]=4
  [flow_decay_enabled]=true
  [flow_decay_tau_us]=30000
  [flow_cell_min_mass]=3.0
  [flow_cell_min_lambda]=0.0005
  [flow_cell_max_residual_ratio]=0.95
  [flow_tile_min_mass]=15.0
  [flow_tile_min_cells]=2
  [flow_tile_min_lambda]=0.00001
  [flow_time_aware_order]=2
  [flow_aperture_ratio]=0.04
  [flow_tikhonov_eps]=0.0005
  [flow_prior_lambda]=3.0
  [flow_prior_conf_ratio]=0.0
  [flow_reg_lambda]=5.0
  [flow_reg_sweeps]=12
  [flow_reg_sigma]=60.0
  [flow_max_speed_px_s]=3000.0
)
for kv in "$@"; do P[${kv%%=*}]=${kv#*=}; done

for k in "${!P[@]}"; do
  v=${P[$k]}
  # replace the entire value region between 'key:' and the trailing '#comment'
  sed -i -E "s|^(\s*${k}:)[^#]*(#.*)?$|\1 ${v}      \2|" "$CFG"
done
sed -i -E "s|^(\s*flow_save_output_dir:)[^#]*(#.*)?$|\1 ${WS}/logs/moment_flow/smooth_flow_search/${NAME}      \2|" "$CFG"

echo "=== variant $NAME ==="; for kv in "$@"; do echo "  override $kv"; done
bash "$WS/tools/dsec_flow_benchmark/run_flow_test.sh" 2>&1 | grep -E 'denseEPE|saved:|FATAL|Error|Traceback'
