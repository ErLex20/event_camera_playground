/**
 * Event Detector workers implementation.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 28, 2025
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

#include <mutex>
#include <utility>

namespace event_detector_cpp
{

void EventDetector::worker_thread_routine()
{
  RCLCPP_WARN(this->get_logger(), "Worker START");

  while (true) {
    EventChunk chunk;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(
        lock,
        [this] {
          return !queue_.empty() || !running_.load(std::memory_order_acquire);
        });

      // The only way out of wait() with an empty queue is a shutdown request;
      // a non-empty queue is drained first even after deactivation.
      if (queue_.empty()) {
        break;
      }
      chunk = std::move(queue_.front());
      queue_.pop_front();
    }

    // Allocate resolution-dependent processors on the first packet.
    if (!ts_on_.has_value()) {
      lazy_init(chunk.width, chunk.height);
    }

    // Background-activity filtering (per packet).
    dv::EventStore events = ba_filter_enabled_ ? compute_ba(chunk.events) : chunk.events;
    if (events.isEmpty()) {
      continue;
    }

    // Surface of Active Events (per packet).
    if (sae_enabled_) {
      publish_image(
        pub_sae_, compute_sae(events), sensor_msgs::image_encodings::BGR8, chunk.header);
    }

    // IWE and optical flow run once per flow window.
    if (iwe_enabled_ || flow_enabled_) {
      if (flow_window_start_us_ == std::numeric_limits<int64_t>::lowest()) {
        flow_window_start_us_ = events.getLowestTime();
      }
      flow_accum_.add(events);
      if (events.getHighestTime() - flow_window_start_us_ >=
        static_cast<int64_t>(flow_window_ms_ * 1000.0))
      {
        dv::EventStore window = flow_accum_;
        flow_accum_ = dv::EventStore();
        flow_window_start_us_ = events.getHighestTime();

        // The global IWE velocity seeds the per-patch flow when IWE is enabled;
        // otherwise the flow falls back to a zero seed and its own search.
        cv::Vec2f v_global(0.0f, 0.0f);
        if (iwe_enabled_) {
          publish_image(
            pub_iwe_, compute_iwe(window, v_global),
            sensor_msgs::image_encodings::MONO8, chunk.header);
        }
        if (flow_enabled_) {
          publish_image(
            pub_flow_, compute_optical_flow(window, v_global),
            sensor_msgs::image_encodings::BGR8, chunk.header);
        }
      }
    }
  }

  RCLCPP_WARN(this->get_logger(), "Worker STOP");
}

} // namespace event_detector_cpp
