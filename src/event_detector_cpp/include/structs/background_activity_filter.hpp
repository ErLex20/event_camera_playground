/**
 * Background Activity (BA) filter struct.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 25, 2026
 */

/**
 * Copyright 2024 dotX Automation s.r.l.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace event_detector_cpp
{

/**
 * Background Activity (BA) filter for event cameras.
 *
 * Rejects isolated, uncorrelated events (hot pixels, shot noise) by requiring
 * that at least one 8-connected neighbour fired within a time window dt_ns
 * before accepting an event.
 *
 * The per-pixel last-timestamp table is persistent across packets — it must
 * not be reset between callbacks. Polarity is ignored: a neighbour of any
 * polarity counts as correlated.
 *
 * Timestamps are always recorded (even for rejected events) so that the first
 * real event in a newly active region can establish context for its neighbours.
 */
struct BackgroundActivityFilter
{
  uint32_t width{0};
  uint32_t height{0};
  float dt_ns{0.0f};  // correlation time window [ns]

  /** Per-pixel last-event timestamp [ns]. Initialised to -infinity (never fired). */
  std::vector<float> last_ts;

  void resize(uint32_t w, uint32_t h, float dt_ns_val)
  {
    width = w;
    height = h;
    dt_ns = dt_ns_val;
    last_ts.assign(
      static_cast<std::size_t>(w) * h,
      -std::numeric_limits<float>::infinity());
  }

  void reset()
  {
    std::fill(
      last_ts.begin(), last_ts.end(),
      -std::numeric_limits<float>::infinity());
  }

  /**
   * Records the event timestamp and returns true if the event is correlated
   * (i.e., at least one 8-connected neighbour fired within dt_ns).
   *
   * Always updates last_ts for the current pixel so future neighbours can use
   * it for their own correlation checks.
   */
  inline bool check_and_update(uint16_t x, uint16_t y, float ts_ns) noexcept
  {
    last_ts[static_cast<std::size_t>(y) * width + x] = ts_ns;

    const int32_t x0 = std::max(0, static_cast<int32_t>(x) - 1);
    const int32_t x1 = std::min(static_cast<int32_t>(width)  - 1, static_cast<int32_t>(x) + 1);
    const int32_t y0 = std::max(0, static_cast<int32_t>(y) - 1);
    const int32_t y1 = std::min(static_cast<int32_t>(height) - 1, static_cast<int32_t>(y) + 1);

    for (int32_t row = y0; row <= y1; ++row) {
      for (int32_t col = x0; col <= x1; ++col) {
        if (row == static_cast<int32_t>(y) && col == static_cast<int32_t>(x)) {
          continue;
        }
        if (ts_ns - last_ts[static_cast<std::size_t>(row) * width + col] <= dt_ns) {
          return true;
        }
      }
    }
    return false;
  }
};

}  // namespace event_detector_cpp
