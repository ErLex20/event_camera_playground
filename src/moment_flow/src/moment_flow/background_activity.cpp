/**
 * Moment Flow background-activity filtering.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 28, 2026
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

#include "moment_flow/moment_flow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace moment_flow
{

// A backward time step larger than any plausible packet overlap means a stream
// discontinuity (e.g. camera reconnect), not a boundary overlap.
namespace
{

constexpr int64_t kTimeResetThresholdUs = 100'000;  // 100 ms

}  // namespace

EventStore EventDetector::compute_ba(const EventStore & raw_in)
{
  const EventStore * raw = &raw_in;
  EventStore clipped;

  // EVK4 packets can overlap slightly at boundaries. Keep the in-order tail
  // for small overlaps and reset state on real stream discontinuities.
  if (raw->getLowestTime() < filter_high_us_) {
    if (filter_high_us_ - raw->getLowestTime() > kTimeResetThresholdUs) {
      // Stream discontinuity: rebuild stateful processors and re-baseline.
      RCLCPP_WARN(get_logger(), "Event timestamps jumped back; resetting state");
      reset_state();
    } else {
      clipped = raw->sliceTime(filter_high_us_);
      raw = &clipped;
      if (raw->isEmpty()) {
        return EventStore();
      }
    }
  }

  if (res_.width <= 0 || res_.height <= 0 || ba_last_us_.empty()) {
    filter_high_us_ = raw->getHighestTime();
    return *raw;
  }

  const int w = res_.width;
  const int h = res_.height;
  const int64_t dt_us = std::max<int64_t>(
    1,
    static_cast<int64_t>(std::llround(ba_filter_dt_ms_ * 1000.0)));
  EventStore filtered;
  filtered.reserve(raw->size());

  for (const auto & event : *raw) {
    const int x = event.x();
    const int y = event.y();
    const int64_t t_us = event.timestamp();
    if (x < 0 || y < 0 || x >= w || y >= h) {
      continue;
    }

    bool supported = false;
    for (int dy = -1; dy <= 1 && !supported; ++dy) {
      const int ny = y + dy;
      if (ny < 0 || ny >= h) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        if (nx < 0 || nx >= w || (dx == 0 && dy == 0)) {
          continue;
        }
        const int64_t last_us = ba_last_us_[static_cast<std::size_t>(ny * w + nx)];
        if (last_us != std::numeric_limits<int64_t>::lowest() &&
          t_us >= last_us &&
          t_us - last_us <= dt_us)
        {
          supported = true;
          break;
        }
      }
    }

    ba_last_us_[static_cast<std::size_t>(y * w + x)] = t_us;
    if (supported) {
      filtered.push_back(event);
    }
  }

  filter_high_us_ = raw->getHighestTime();
  return filtered;
}

}  // namespace moment_flow
