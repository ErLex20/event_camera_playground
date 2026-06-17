#!/usr/bin/env bash
# One MomentFlow tuning iteration: replay thun_00_a through event_detector_cpp,
# sample the flow debug topics, then benchmark dense+sparse and score tile
# coherence. RUN_DIR is taken from flow_save_output_dir in the config (single
# source of truth) so you only edit the yaml between iterations.
set -u
WS=/home/neo/workspace
CFG=$WS/src/event_detector_cpp/config/event_detector_cpp.yaml
GT_DIR=$WS/logs/dsec/thun_00_a/optical_flow_forward
H5=$WS/logs/dsec/thun_00_a/events_left/events.h5
REPLAY_TIMEOUT=${REPLAY_TIMEOUT:-120}

RUN_DIR=$(grep -E '^\s*flow_save_output_dir:' "$CFG" | sed -E 's/^[^:]*:\s*([^ #]+).*/\1/')
[ -z "$RUN_DIR" ] && { echo "FATAL: no flow_save_output_dir in config"; exit 1; }
echo ">>> RUN_DIR=$RUN_DIR"
mkdir -p "$RUN_DIR/debug_samples"
rm -f "$RUN_DIR/debug_samples/"*.png "$RUN_DIR"/*.json "$RUN_DIR"/*.txt "$RUN_DIR"/*.csv 2>/dev/null

set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u

pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 2
  pkill -INT -f dua_component_container_mt 2>/dev/null
  pkill -f dsec_publisher 2>/dev/null
  pkill -f sample_ros_flow_debug 2>/dev/null
  sleep 1
  pkill -KILL -f dua_component_container_mt 2>/dev/null
}
trap cleanup EXIT

# 1) detector
setsid ros2 launch event_detector_cpp event_detector_cpp.launch.py \
  > "$RUN_DIR/detector.log" 2>&1 &
pids+=($!)
echo ">>> detector launching (pid $!), waiting for node up..."
sleep 14

# 2) sampler (captures /flow_*_debug into debug_samples/)
setsid python3 -m tools.dsec_flow_benchmark.sample_ros_flow_debug \
  --output-dir "$RUN_DIR/debug_samples" --seconds "$REPLAY_TIMEOUT" \
  --target-debug-frames 60 > "$RUN_DIR/sampler.log" 2>&1 &
pids+=($!)
SAMPLER_PID=$!
sleep 1

# 3) publisher (loop=false -> exits when replay done)
setsid ros2 launch dsec_publisher dsec_publisher.launch.py \
  events_h5:="$H5" topic:=/event_camera/events realtime_factor:="${RF:-0.5}" \
  > "$RUN_DIR/dsec.log" 2>&1 &
pids+=($!)
PUB_PID=$!
echo ">>> replay started (pid $PUB_PID); waiting up to ${REPLAY_TIMEOUT}s..."

# wait for publisher group to finish
t=0
while kill -0 "$PUB_PID" 2>/dev/null && [ "$t" -lt "$REPLAY_TIMEOUT" ]; do
  sleep 2; t=$((t+2))
done
echo ">>> replay finished after ~${t}s; grace for final saves..."
sleep 4
cleanup
trap - EXIT

# ---- evaluate ----
echo ">>> dense saved: $(ls "$RUN_DIR/dense/"*.png 2>/dev/null | wc -l)  sparse: $(ls "$RUN_DIR/sparse/"*.png 2>/dev/null | wc -l)  debug: $(ls "$RUN_DIR/debug_samples/"*_debug_*.png 2>/dev/null | wc -l)"

cd "$WS"
python3 -m tools.dsec_flow_benchmark --gt-dir "$GT_DIR" --pred-dir "$RUN_DIR/dense" \
  --mask-mode gt --output-json "$RUN_DIR/dense_benchmark_gt.json" \
  --output-csv "$RUN_DIR/dense_benchmark_gt_frames.csv" 2>&1 | tee "$RUN_DIR/dense_benchmark_gt.txt"
python3 -m tools.dsec_flow_benchmark --gt-dir "$GT_DIR" --pred-dir "$RUN_DIR/sparse" \
  --mask-mode intersection --output-json "$RUN_DIR/sparse_benchmark_intersection.json" \
  --output-csv "$RUN_DIR/sparse_benchmark_intersection_frames.csv" 2>&1 | tee "$RUN_DIR/sparse_benchmark_intersection.txt"
echo ">>> tile coherence (dense):"
python3 -m tools.dsec_flow_benchmark.coherence --pred-dir "$RUN_DIR/dense" \
  --output-json "$RUN_DIR/coherence.json"
echo ">>> magnitude correlation (dense vs GT):"
python3 -m tools.dsec_flow_benchmark.magstats --gt-dir "$GT_DIR" --pred-dir "$RUN_DIR/dense" \
  --output-json "$RUN_DIR/magstats.json"
# mean count of data-solved final tiles from the detector log (full+aperture of 1024)
grep -oE "tiles_final\(full/aperture/fallback\)=[0-9]+/[0-9]+/[0-9]+" "$RUN_DIR/detector.log" 2>/dev/null \
  | sed -E 's/.*=([0-9]+)\/([0-9]+)\/([0-9]+)/\1 \2 \3/' \
  | awk '{f+=$1;a+=$2;b+=$3;n++} END{if(n)printf "%.0f %.0f %.0f\n",f/n,a/n,b/n}' \
  > "$RUN_DIR/tiles_final_avg.txt"

# append a one-line comparison row
python3 - "$RUN_DIR" "$CFG" <<'PY'
import json, sys, re
run, cfg = sys.argv[1], sys.argv[2]
def jget(p, *ks):
    try:
        d=json.load(open(p))
        for k in ks: d=d[k]
        return d
    except Exception: return float('nan')
de=jget(run+"/dense_benchmark_gt.json","summary","epe") if False else None
def epe(path):
    try:
        d=json.load(open(path)); s=d.get("summary",d)
        return s.get("epe", s.get("EPE", float('nan')))
    except Exception: return float('nan')
def ae(path):
    try:
        d=json.load(open(path)); s=d.get("summary",d)
        return s.get("ae", s.get("AE", float('nan')))
    except Exception: return float('nan')
dE=epe(run+"/dense_benchmark_gt.json"); sE=epe(run+"/sparse_benchmark_intersection.json")
dA=ae(run+"/dense_benchmark_gt.json")
dis=jget(run+"/coherence.json","tile_disorder_deg")
ratio=jget(run+"/magstats.json","ratio_med"); slope=jget(run+"/magstats.json","slope")
pr=jget(run+"/magstats.json","pearson_r"); cosm=jget(run+"/magstats.json","cos_med")
try: dtiles=open(run+"/tiles_final_avg.txt").read().split(); dtiles=int(dtiles[0])+int(dtiles[1])
except Exception: dtiles=-1
txt=open(cfg).read()
def g(k):
    m=re.search(r'^\s*%s:\s*([0-9.]+)'%k, txt, re.M); return m.group(1) if m else "?"
name=run.split("/")[-1]
row=("%-26s EPE=%6.3f/%6.3f AE=%5.1f disord=%5.2f ratio=%4.2f slope=%4.2f r=%4.2f cos=%4.2f dtiles=%4d "
     "| prior=%s conf=%s reg=%s/%s/%s order=%s decay=%s resid=%s cell=%s tmin_c=%s tmin_m=%s clambda=%s")%(
    name,dE,sE,dA,dis,ratio,slope,pr,cosm,dtiles,
    g("flow_prior_lambda"),g("flow_prior_conf_ratio"),g("flow_reg_lambda"),g("flow_reg_sweeps"),g("flow_reg_sigma"),
    g("flow_time_aware_order"),re.search(r'flow_decay_enabled:\s*(\w+)',txt).group(1),
    g("flow_cell_max_residual_ratio"),g("flow_cell_size_px"),g("flow_tile_min_cells"),g("flow_tile_min_mass"),
    g("flow_cell_min_lambda"))
print(row)
open("/home/neo/workspace/logs/moment_flow/smooth_flow_search/RESULTS.tsv","a").write(row+"\n")
PY
echo ">>> DONE: $RUN_DIR"
