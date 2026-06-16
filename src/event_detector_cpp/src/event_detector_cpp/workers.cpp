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
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace event_detector_cpp
{

namespace
{

std::shared_ptr<dv::EventPacket> make_event_packet()
{
  return std::make_shared<dv::EventPacket>();
}

dv::EventStore make_event_store(const std::shared_ptr<dv::EventPacket> & packet)
{
  return dv::EventStore(std::const_pointer_cast<const dv::EventPacket>(packet));
}

}  // namespace

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
    if (sae_events.isEmpty() && !(iwe_enabled_ || flow_enabled_ || flow_save_enabled_)) {
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

    // IWE and optical flow run once per fixed-duration flow window. When a
    // benchmark timestamp schedule is configured, it only controls which
    // windows are saved and how they are named.
    if (iwe_enabled_ || flow_enabled_ || flow_save_enabled_) {
      dv::EventStore flow_events = chunk.events;
      const bool scheduled_flow_active =
        flow_save_enabled_ &&
        flow_save_prepared_ &&
        !flow_save_windows_.empty() &&
        flow_save_next_window_ < flow_save_windows_.size();
      if (scheduled_flow_active) {
        flow_events = accumulate_scheduled_flow(chunk.events, chunk.header);
        if (flow_events.isEmpty()) {
          continue;
        }
      }

      int64_t chunk_first_us = std::numeric_limits<int64_t>::max();
      int64_t chunk_last_us = std::numeric_limits<int64_t>::lowest();
      for (const auto & e : flow_events) {
        chunk_first_us = std::min(chunk_first_us, e.timestamp());
        chunk_last_us = std::max(chunk_last_us, e.timestamp());
      }
      if (!flow_events.isEmpty()) {
        flow_accum_.add(flow_events);
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
        const int64_t save_from_us = flow_accum_first_us_;
        const int64_t save_to_us = flow_accum_last_us_;
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

        if (flow_save_enabled_ && flow_save_prepared_ && flow_save_windows_.empty()) {
          const int64_t file_index =
            flow_save_first_index_ + flow_save_sequence_index_ * flow_save_index_step_;
          save_flow_png(res.flow_dense, file_index, save_from_us, save_to_us);
          flow_save_sequence_index_ += 1;
        }

        if (iwe_enabled_) {
          publish_image(
            pub_iwe_, res.iwe, sensor_msgs::image_encodings::MONO8, chunk.header);
        }
        if (flow_enabled_) {
          publish_image(
            pub_flow_dense_debug_, res.flow_dense_debug,
            sensor_msgs::image_encodings::BGR8, chunk.header);
          publish_image(
            pub_flow_dense_, res.flow_dense,
            sensor_msgs::image_encodings::TYPE_32FC2, chunk.header);
          publish_image(
            pub_flow_events_debug_, res.flow_events_debug,
            sensor_msgs::image_encodings::BGR8, chunk.header);
          publish_image(
            pub_flow_events_, res.flow_events,
            sensor_msgs::image_encodings::TYPE_32FC2, chunk.header);
        }
      }
    }
  }

  RCLCPP_WARN(this->get_logger(), "Worker STOP");
}

dv::EventStore EventDetector::accumulate_scheduled_flow(
  const dv::EventStore & events,
  const std_msgs::msg::Header & header)
{
  if (events.isEmpty() || flow_save_next_window_ >= flow_save_windows_.size()) {
    return events;
  }

  auto packet = make_event_packet();
  auto unscheduled_packet = make_event_packet();

  auto estimate_to_us = [this](const FlowSaveWindow & window) -> int64_t {
    const int64_t requested_us = std::max<int64_t>(
      1,
      static_cast<int64_t>(std::llround(flow_max_window_ms_ * 1000.0)));
    return std::min(window.to_us, window.from_us + requested_us);
  };

  auto flush_packet = [&]() {
    if (packet->elements.empty()) {
      return;
    }
    flow_accum_.add(make_event_store(packet));
    packet = make_event_packet();
  };

  auto close_current_window = [&]() {
    flush_packet();
    if (flow_save_next_window_ >= flow_save_windows_.size()) {
      return;
    }

    const FlowSaveWindow window = flow_save_windows_[flow_save_next_window_];
    const int64_t estimate_to = estimate_to_us(window);
    RCLCPP_INFO(
      get_logger(),
      "Scheduled flow window close: index=%" PRId64
      " estimate=[%" PRId64 ", %" PRId64 ") encode=[%" PRId64 ", %" PRId64 ") events=%zu",
      window.file_index, window.from_us, estimate_to, window.from_us, window.to_us,
      static_cast<std::size_t>(flow_accum_.size()));

    FlowResult res;
    if (flow_accum_.size() >= 2) {
      res = solve_flow_moment(
        flow_accum_,
        window.from_us + (estimate_to - window.from_us) / 2);
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Scheduled flow estimate interval [%" PRId64 ", %" PRId64
        ") has fewer than 2 events; saving zero flow",
        window.from_us, estimate_to);
    }

    save_flow_png(res.flow_dense, window.file_index, window.from_us, window.to_us);

    if (iwe_enabled_) {
      publish_image(pub_iwe_, res.iwe, sensor_msgs::image_encodings::MONO8, header);
    }
    if (flow_enabled_) {
      publish_image(
        pub_flow_dense_debug_, res.flow_dense_debug, sensor_msgs::image_encodings::BGR8, header);
      publish_image(
        pub_flow_dense_, res.flow_dense, sensor_msgs::image_encodings::TYPE_32FC2, header);
      publish_image(
        pub_flow_events_debug_, res.flow_events_debug, sensor_msgs::image_encodings::BGR8, header);
      publish_image(
        pub_flow_events_, res.flow_events, sensor_msgs::image_encodings::TYPE_32FC2, header);
    }

    flow_accum_ = dv::EventStore();
    flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
    flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();
    flow_save_next_window_ += 1;
  };

  for (const auto & event : events) {
    const int64_t t_us = event.timestamp();

    while (flow_save_next_window_ < flow_save_windows_.size() &&
      t_us >= estimate_to_us(flow_save_windows_[flow_save_next_window_]))
    {
      close_current_window();
    }
    if (flow_save_next_window_ >= flow_save_windows_.size()) {
      unscheduled_packet->elements.push_back(event);
      continue;
    }

    const FlowSaveWindow & window = flow_save_windows_[flow_save_next_window_];
    const int64_t estimate_to = estimate_to_us(window);
    if (t_us < window.from_us) {
      continue;
    }
    if (t_us >= estimate_to) {
      continue;
    }

    packet->elements.push_back(event);
    flow_accum_first_us_ = std::min(flow_accum_first_us_, t_us);
    flow_accum_last_us_ = std::max(flow_accum_last_us_, t_us);
  }

  flush_packet();
  if (unscheduled_packet->elements.empty()) {
    return dv::EventStore();
  }
  return make_event_store(unscheduled_packet);
}

} // namespace event_detector_cpp
