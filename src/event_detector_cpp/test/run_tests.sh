#!/usr/bin/env bash
# Fast standalone build+run of the ROS-independent flow-core unit tests.
# (The same tests are also wired into ament via ament_add_gtest in CMakeLists.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
OUT="$HERE/.build"
mkdir -p "$OUT"

SRCS=(
  "$PKG/src/event_detector_cpp/flow_propagation.cpp"
  "$PKG/src/event_detector_cpp/flow_objective.cpp"
)
TESTS=(
  "$HERE/test_propagation.cpp"
  "$HERE/test_objective.cpp"
  "$HERE/test_solver.cpp"
)

g++ -std=c++17 -O2 -Wall -Wextra -fopenmp \
  -I"$PKG/include" -I/usr/include/eigen3 \
  "${SRCS[@]}" "${TESTS[@]}" \
  -lgtest -lgtest_main -lpthread \
  -o "$OUT/flow_tests"

"$OUT/flow_tests" "$@"
