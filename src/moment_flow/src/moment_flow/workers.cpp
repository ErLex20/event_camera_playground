/**
 * Moment Flow workers implementation.
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

#include "moment_flow/moment_flow.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>

namespace moment_flow
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
    if (res_.width <= 0) {
      lazy_init(chunk.width, chunk.height);
    }

    // Background-activity filtering (per packet).
    const auto t_ba = std::chrono::high_resolution_clock::now();
    bool filtered_empty = chunk.events.isEmpty();
    if (ba_filter_enabled_) {
      EventStore filtered = compute_ba(chunk.events);
      filtered_empty = filtered.isEmpty();
    }
    if (filtered_empty && !(iwe_enabled_ || flow_enabled_ || flow_save_enabled_)) {
      continue;
    }
    if (debug_) {
      const auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_ba).count();
      RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "BA filter: %ld ms", dt1);
    }

    // IWE and optical flow run once per fixed-duration flow window. When a
    // benchmark timestamp schedule is configured, it only controls which
    // windows are saved and how they are named.
    if (iwe_enabled_ || flow_enabled_ || flow_save_enabled_) {
      EventStore flow_events;
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
      } else {
        flow_events = std::move(chunk.events);
      }

      int64_t chunk_first_us = std::numeric_limits<int64_t>::max();
      int64_t chunk_last_us = std::numeric_limits<int64_t>::lowest();
      for (const auto & e : flow_events) {
        chunk_first_us = std::min(chunk_first_us, e.timestamp());
        chunk_last_us = std::max(chunk_last_us, e.timestamp());
      }
      if (!flow_events.isEmpty()) {
        // Some RMW implementations (e.g. rmw_zenoh) do not guarantee in-order
        // delivery of consecutive messages from the same publisher. Drop the
        // offending chunk rather than crash the whole component.
        try {
          flow_accum_.add(std::move(flow_events));
          flow_accum_first_us_ = std::min(flow_accum_first_us_, chunk_first_us);
          flow_accum_last_us_ = std::max(flow_accum_last_us_, chunk_last_us);
        } catch (const std::out_of_range &) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *get_clock(), 1000,
            "Dropping event chunk delivered out of temporal order (span [%" PRId64
            ", %" PRId64 "] us, accum high %" PRId64 " us)",
            chunk_first_us, chunk_last_us, flow_accum_last_us_);
        }
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
        if (debug_) {
          RCLCPP_INFO_THROTTLE(
            this->get_logger(), *get_clock(), 1000,
            "Flow window close: events=%ld span=%.3f ms target=%.3f ms",
            flow_accum_events, flow_span_ms, flow_max_window_ms_);
        }
        const int64_t save_from_us = flow_accum_first_us_;
        const int64_t save_to_us = flow_accum_last_us_;
        EventStore window = flow_accum_;
        flow_accum_ = EventStore();
        flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
        flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();

        FlowResult res = solve_flow_moment(window);

        if (flow_save_enabled_ && flow_save_prepared_ && flow_save_windows_.empty()) {
          const int64_t file_index =
            flow_save_first_index_ + flow_save_sequence_index_ * flow_save_index_step_;
          save_flow_results(res, file_index, save_from_us, save_to_us);
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
            pub_flow_tiles_, res.flow_tiles,
            sensor_msgs::image_encodings::TYPE_32FC2, chunk.header);
          publish_image(
            pub_flow_tile_debug_, res.flow_tile_debug,
            sensor_msgs::image_encodings::BGR8, chunk.header);
          if (flow_events_enabled_) {
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
  }

  RCLCPP_WARN(this->get_logger(), "Worker STOP");
}

EventStore EventDetector::accumulate_scheduled_flow(
  const EventStore & events,
  const std_msgs::msg::Header & header)
{
  if (events.isEmpty() || flow_save_next_window_ >= flow_save_windows_.size()) {
    return events;
  }

  EventStore packet;
  EventStore unscheduled_packet;

  auto estimate_to_us = [this](const FlowSaveWindow & window) -> int64_t {
    const int64_t requested_us = std::max<int64_t>(
      1,
      static_cast<int64_t>(std::llround(flow_max_window_ms_ * 1000.0)));
    return std::min(window.to_us, window.from_us + requested_us);
  };

  auto flush_packet = [&]() {
    if (packet.isEmpty()) {
      return;
    }
    // See the worker_thread_routine() note above: guard against out-of-order
    // delivery across messages (observed with rmw_zenoh).
    try {
      flow_accum_.add(std::move(packet));
    } catch (const std::out_of_range &) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Dropping scheduled-flow packet delivered out of temporal order (span [%" PRId64
        ", %" PRId64 "] us)",
        packet.front().timestamp(), packet.back().timestamp());
    }
    packet = EventStore();
  };

  // Solve one window and throw the result away. A non-contiguous export schedule
  // (the DSEC-Flow test set samples at 2 Hz, leaving ~400 ms between the end of
  // one scheduled interval and the start of the next) would otherwise leave the
  // tracked field, and with it the prior and aperture fallbacks, several hundred
  // milliseconds stale by the time the next export is solved. Running the skipped
  // stream keeps the warm start exactly as fresh as it is on the contiguous
  // training schedule, so the export cadence no longer influences the estimate.
  auto close_filler_window = [&]() {
    flush_packet();
    if (flow_accum_.size() >= 2) {
      const int64_t mid_us =
        flow_accum_first_us_ + (flow_accum_last_us_ - flow_accum_first_us_) / 2;
      solve_flow_moment(flow_accum_, mid_us);
    }
    flow_accum_ = EventStore();
    flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
    flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();
  };

  auto close_current_window = [&]() {
    flush_packet();
    if (flow_save_next_window_ >= flow_save_windows_.size()) {
      return;
    }

    const FlowSaveWindow window = flow_save_windows_[flow_save_next_window_];
    const int64_t estimate_to = estimate_to_us(window);
    if (debug_) {
      RCLCPP_INFO(
        get_logger(),
        "Scheduled flow window close: index=%" PRId64
        " estimate=[%" PRId64 ", %" PRId64 ") encode=[%" PRId64 ", %" PRId64 ") events=%zu",
        window.file_index, window.from_us, estimate_to, window.from_us, window.to_us,
        static_cast<std::size_t>(flow_accum_.size()));
    }

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

    save_flow_results(res, window.file_index, window.from_us, window.to_us);

    if (iwe_enabled_) {
      publish_image(pub_iwe_, res.iwe, sensor_msgs::image_encodings::MONO8, header);
    }
    if (flow_enabled_) {
      publish_image(
        pub_flow_dense_debug_, res.flow_dense_debug, sensor_msgs::image_encodings::BGR8, header);
      publish_image(
        pub_flow_dense_, res.flow_dense, sensor_msgs::image_encodings::TYPE_32FC2, header);
      publish_image(
        pub_flow_tiles_, res.flow_tiles, sensor_msgs::image_encodings::TYPE_32FC2, header);
      publish_image(
        pub_flow_tile_debug_, res.flow_tile_debug, sensor_msgs::image_encodings::BGR8, header);
      if (flow_events_enabled_) {
        publish_image(
          pub_flow_events_debug_, res.flow_events_debug,
          sensor_msgs::image_encodings::BGR8, header);
        publish_image(
          pub_flow_events_, res.flow_events, sensor_msgs::image_encodings::TYPE_32FC2, header);
      }
    }

    flow_accum_ = EventStore();
    flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
    flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();
    flow_save_prev_estimate_end_us_ = estimate_to;
    flow_save_next_window_ += 1;
  };

  // Whether the accumulators currently hold gap events is derived from their own
  // timestamps rather than tracked in a flag: a gap spans many packets, so a
  // per-call flag would be lost across chunk boundaries and pre-window events
  // would leak into the next export.
  auto holds_gap_events = [this](int64_t from_us) {
      return flow_accum_first_us_ != std::numeric_limits<int64_t>::max() &&
             flow_accum_first_us_ < from_us;
    };

  const int64_t window_us = std::max<int64_t>(
    1,
    static_cast<int64_t>(std::llround(flow_max_window_ms_ * 1000.0)));

  for (const auto & event : events) {
    const int64_t t_us = event.timestamp();

    while (flow_save_next_window_ < flow_save_windows_.size() &&
      t_us >= estimate_to_us(flow_save_windows_[flow_save_next_window_]))
    {
      close_current_window();
    }
    if (flow_save_next_window_ >= flow_save_windows_.size()) {
      unscheduled_packet.push_back(event);
      continue;
    }

    const FlowSaveWindow & window = flow_save_windows_[flow_save_next_window_];
    const int64_t estimate_to = estimate_to_us(window);
    if (t_us < window.from_us) {
      // Consecutive scheduled intervals on the DSEC training schedule abut to
      // within a microsecond, so only a skipped span worth a fair fraction of a
      // window is worth solving: filling those would add near-empty solves and
      // change the contiguous-schedule result.
      // A negative previous end means nothing has been solved yet, which is the
      // stream preceding the very first scheduled interval. That is the longest
      // skipped span of the run and the one where a warm start is worth most, so
      // it is filled like any other gap: the length test is simply skipped,
      // since there is no previous window to measure against. Treating it as
      // "not worth filling" made the first export a cold start no matter how
      // much stream preceded it -- invisible on the training schedule, whose
      // ground truth begins 0.1 s in, but 41 s on interlaken_00_b and 54 s on
      // zurich_city_14_c of the test split.
      const bool no_previous_estimate = flow_save_prev_estimate_end_us_ < 0;
      if (!flow_save_gap_fill_ ||
        (!no_previous_estimate &&
        window.from_us - flow_save_prev_estimate_end_us_ < window_us / 2))
      {
        continue;
      }
      packet.push_back(event);
      flow_accum_first_us_ = std::min(flow_accum_first_us_, t_us);
      flow_accum_last_us_ = std::max(flow_accum_last_us_, t_us);
      if (flow_accum_last_us_ - flow_accum_first_us_ >= window_us) {
        close_filler_window();
      }
      continue;
    }
    if (t_us >= estimate_to) {
      continue;
    }
    // The remainder of the skipped span must not survive into the scheduled
    // accumulation. Solving it is only worthwhile if it spans a fair fraction of
    // a window: the moment model needs temporal baseline, and a sliver of a few
    // milliseconds has almost none, so Var(tau) collapses and v = d/s explodes.
    // Such a window would then poison the tracked field of the very export it
    // precedes, so a too-short remainder is discarded instead of solved.
    if (holds_gap_events(window.from_us)) {
      const int64_t span = flow_accum_last_us_ - flow_accum_first_us_;
      if (span >= window_us / 2) {
        close_filler_window();
      } else {
        packet = EventStore();
        flow_accum_ = EventStore();
        flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
        flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();
      }
    }

    packet.push_back(event);
    flow_accum_first_us_ = std::min(flow_accum_first_us_, t_us);
    flow_accum_last_us_ = std::max(flow_accum_last_us_, t_us);
  }

  flush_packet();
  if (unscheduled_packet.isEmpty()) {
    return EventStore();
  }
  return unscheduled_packet;
}

} // namespace moment_flow
