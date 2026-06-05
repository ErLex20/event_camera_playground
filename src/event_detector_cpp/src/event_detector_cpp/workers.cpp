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

#include <algorithm>
#include <limits>
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
    // Evaluate time to calculate everything
    auto t_ba = std::chrono::high_resolution_clock::now();
    dv::EventStore sae_events = ba_filter_enabled_ ? compute_ba(chunk.events) : chunk.events;
    if (sae_events.isEmpty() && !(iwe_enabled_ || flow_enabled_)) {
      continue;
    }
    auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - t_ba).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "BA filter: %ld ms", dt1);

    // Surface of Active Events (per packet).
    if (sae_enabled_) {
      auto t_sae = std::chrono::high_resolution_clock::now();
      if (!sae_events.isEmpty()) {
        publish_image(
          pub_sae_, compute_sae(sae_events), sensor_msgs::image_encodings::BGR8, chunk.header);
      }
      auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_sae).count();
      RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "SAE: %ld ms", dt2);
    }

    // IWE and optical flow run once per fixed-duration flow window.
    if (iwe_enabled_ || flow_enabled_) {
      int64_t chunk_first_us = std::numeric_limits<int64_t>::max();
      int64_t chunk_last_us = std::numeric_limits<int64_t>::lowest();
      for (const auto & e : chunk.events) {
        chunk_first_us = std::min(chunk_first_us, e.timestamp());
        chunk_last_us = std::max(chunk_last_us, e.timestamp());
      }
      if (!chunk.events.isEmpty()) {
        flow_accum_.add(chunk.events);
        flow_accum_first_us_ = std::min(flow_accum_first_us_, chunk_first_us);
        flow_accum_last_us_ = std::max(flow_accum_last_us_, chunk_last_us);
      }

      const int64_t flow_accum_events = static_cast<int64_t>(flow_accum_.size());
      const bool have_flow_time =
        flow_accum_first_us_ != std::numeric_limits<int64_t>::max() &&
        flow_accum_last_us_ != std::numeric_limits<int64_t>::lowest() &&
        flow_accum_last_us_ >= flow_accum_first_us_;
      const double flow_span_ms = have_flow_time
        ? static_cast<double>(flow_accum_last_us_ - flow_accum_first_us_) * 1e-3
        : 0.0;
      const bool time_ready =
        flow_accum_events >= 2 &&
        flow_span_ms >= flow_max_window_ms_;

      if (time_ready) {
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), *get_clock(), 1000,
          "Flow window close: events=%ld span=%.3f ms target=%.3f ms",
          flow_accum_events, flow_span_ms, flow_max_window_ms_);
        dv::EventStore window = flow_accum_;
        flow_accum_ = dv::EventStore();
        flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
        flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();

        auto t_flow = std::chrono::high_resolution_clock::now();
        FlowResult res = solve_flow_moment(window);
        auto dt_flow = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - t_flow).count();
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), *get_clock(), 1000, "Flow moment: %ld ms", dt_flow);

        if (iwe_enabled_) {
          publish_image(
            pub_iwe_, res.iwe, sensor_msgs::image_encodings::MONO8, chunk.header);
        }
        if (flow_enabled_) {
          publish_image(
            pub_flow_, res.flow, sensor_msgs::image_encodings::BGR8, chunk.header);
        }
      }
    }
  }

  RCLCPP_WARN(this->get_logger(), "Worker STOP");
}

} // namespace event_detector_cpp
