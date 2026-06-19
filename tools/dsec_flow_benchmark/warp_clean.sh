#!/usr/bin/env bash
# Kill leftover ROS/replay processes from interrupted warp runs. Run via
# `bash warp_clean.sh` so this script's own command line does not contain the
# match patterns (avoids pkill self-match).
for pat in dua_component_container_mt dsec_publisher sample_ros_flow_debug; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 2
left=$(pgrep -c -f dua_component_container_mt 2>/dev/null || true)
echo "containers_left=$left"
