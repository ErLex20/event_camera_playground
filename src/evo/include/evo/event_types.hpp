/**
 * Internal event types for the merged EVO node.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
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

#include <cstdint>

namespace evo
{

/**
 * ros::Time-compatible timestamp used by the ported EVO stages.
 *
 * The original ROS1 code stored event timestamps as ros::Time and relied on
 * .toSec()/.toNSec() plus subtraction and ordering. This lightweight wrapper
 * (nanoseconds) reproduces exactly that subset of the API so the stage code
 * can be ported verbatim. Stored as signed nanoseconds so that durations
 * (end - start) behave correctly.
 */
struct EventTime
{
  int64_t nsec_{0};

  EventTime() = default;
  explicit EventTime(int64_t nsec) : nsec_(nsec) {}

  double toSec() const { return static_cast<double>(nsec_) * 1e-9; }
  int64_t toNSec() const { return nsec_; }

  EventTime operator-(const EventTime & o) const { return EventTime(nsec_ - o.nsec_); }
  EventTime operator+(const EventTime & o) const { return EventTime(nsec_ + o.nsec_); }

  bool operator<(const EventTime & o) const { return nsec_ < o.nsec_; }
  bool operator>(const EventTime & o) const { return nsec_ > o.nsec_; }
  bool operator<=(const EventTime & o) const { return nsec_ <= o.nsec_; }
  bool operator>=(const EventTime & o) const { return nsec_ >= o.nsec_; }
  bool operator==(const EventTime & o) const { return nsec_ == o.nsec_; }
};

/**
 * Decoded event, mirroring the fields of dvs_msgs::Event (x, y, ts, polarity).
 */
struct Event
{
  uint16_t x{0};
  uint16_t y{0};
  EventTime ts;
  bool polarity{false};

  Event() = default;
  Event(uint16_t x_, uint16_t y_, EventTime ts_, bool polarity_)
  : x(x_), y(y_), ts(ts_), polarity(polarity_) {}
};

}  // namespace evo
