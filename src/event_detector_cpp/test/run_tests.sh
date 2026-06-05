#!/usr/bin/env bash
# Fast standalone build+run of the ROS-independent flow-core unit tests.
# (The same tests are also wired into ament via ament_add_gtest in CMakeLists.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
OUT="$HERE/.build"
mkdir -p "$OUT"

TESTS=(
  "$HERE/test_moment_flow.cpp"
)

g++ -std=c++17 -O2 -Wall -Wextra \
  -I"$PKG/include" -I/usr/include/eigen3 \
  "${TESTS[@]}" \
  -lgtest -lgtest_main -lpthread \
  -o "$OUT/flow_tests"

"$OUT/flow_tests" "$@"

g++ -std=c++17 -O2 -Wall -Wextra \
  -I"$PKG/include" -I/usr/include/eigen3 \
  "$HERE/bench_moment_flow.cpp" \
  -o "$OUT/moment_flow_bench"
