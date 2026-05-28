/**
 * LIO-SAM utils implementation.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * June 26, 2025
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

void EventDetector::activate()
{
  // Set running flag
  running_.store(true, std::memory_order_release);

  RCLCPP_WARN(this->get_logger(), "Event Detector ACTIVATED");
}

void EventDetector::deactivate()
{
  // Set running flag
  running_.store(false, std::memory_order_release);

  RCLCPP_WARN(this->get_logger(), "Event Detector DEACTIVATED");
}

} // namespace event_detector_cpp
