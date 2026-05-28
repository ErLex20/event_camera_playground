/**
 * Event Detector background-activity filtering.
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

#include "event_detector_cpp/event_detector_cpp.hpp"

namespace event_detector_cpp
{

// A backward time step larger than any plausible packet overlap means a stream
// discontinuity (e.g. camera reconnect), not a boundary overlap.
constexpr int64_t kTimeResetThresholdUs = 100'000;  // 100 ms

dv::EventStore EventDetector::compute_ba(const dv::EventStore & raw_in)
{
  dv::EventStore raw = raw_in;

  // dv::EventFilterBase::accept() requires incoming lowest time >= previously
  // processed highest time. EVK4 packets can overlap slightly at boundaries.
  if (raw.getLowestTime() < filter_high_us_) {
    if (filter_high_us_ - raw.getLowestTime() > kTimeResetThresholdUs) {
      // Stream discontinuity: rebuild stateful processors and re-baseline.
      RCLCPP_WARN(get_logger(), "Event timestamps jumped back; resetting state");
      reset_state();
    } else {
      raw = raw.sliceTime(filter_high_us_);  // shallow, zero-copy; keep in-order tail
      if (raw.isEmpty()) {
        return dv::EventStore();
      }
    }
  }

  // Call accept() directly to avoid a bug in operator<< in dv-processing.
  ba_filter_.value().accept(raw);
  dv::EventStore events = ba_filter_.value().generateEvents();
  filter_high_us_ = raw.getHighestTime();
  return events;
}

}  // namespace event_detector_cpp
