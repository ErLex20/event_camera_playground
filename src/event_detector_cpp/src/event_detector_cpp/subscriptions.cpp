/**
 * Event Detector subscriptions implementation.
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

#include "event_detector_cpp/event_detector_cpp.hpp"

#include <dua_cv_bridge/dua_cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/core.hpp>

namespace event_detector_cpp
{

// A backward time step larger than any plausible packet overlap means a stream
// discontinuity (e.g. camera reconnect), not a boundary overlap.
constexpr int64_t kTimeResetThresholdUs = 100'000;  // 100 ms

void EventDetector::callback_event_packet(EventPacket::ConstSharedPtr msg)
{
  if (!running_.load(std::memory_order_acquire) || msg->events.empty()) {
    return;
  }

  auto * decoder = decoder_factory_.getInstance(*msg);
  if (!decoder) {
    RCLCPP_WARN_ONCE(get_logger(), "No decoder for encoding '%s'", msg->encoding.c_str());
    return;
  }

  // Lazy init on first packet.
  if (!ts_on_.has_value()) {
    res_ = cv::Size(static_cast<int>(msg->width), static_cast<int>(msg->height));
    ts_on_.emplace(res_);
    ts_off_.emplace(res_);

    if (ba_filter_enabled_) {
      ba_filter_.emplace(res_, dv::Duration(static_cast<int64_t>(ba_filter_dt_ms_ * 1000.0)));
    }
  }

  // Decode into EventStore.
  EventStoreBuilder builder;
  decoder->decode(*msg, &builder);
  dv::EventStore raw = builder.takeStore();

  if (raw.isEmpty()) {
    return;
  }

  // Optional BA filter pass (call accept() directly to avoid a bug in operator<< in dv-processing).
  dv::EventStore events;
  if (ba_filter_enabled_ && ba_filter_.has_value()) {
    // dv::EventFilterBase::accept() requires incoming lowest time >= previously
    // processed highest time. EVK4 packets can overlap slightly at boundaries.
    if (raw.getLowestTime() < filter_high_us_) {
      if (filter_high_us_ - raw.getLowestTime() > kTimeResetThresholdUs) {
        // Stream discontinuity: rebuild stateful processors and re-baseline.
        RCLCPP_WARN(get_logger(), "Event timestamps jumped back; resetting filter/surfaces");
        ba_filter_.emplace(res_, dv::Duration(static_cast<int64_t>(ba_filter_dt_ms_ * 1000.0)));
        ts_on_->reset();
        ts_off_->reset();
        filter_high_us_ = std::numeric_limits<int64_t>::lowest();
        flow_accum_ = dv::EventStore();
        flow_window_start_us_ = std::numeric_limits<int64_t>::lowest();
        prev_flow_.release();
      } else {
        raw = raw.sliceTime(filter_high_us_);  // shallow, zero-copy; keep in-order tail
        if (raw.isEmpty()) {
          return;
        }
      }
    }
    ba_filter_.value().accept(raw);
    events = ba_filter_.value().generateEvents();
    filter_high_us_ = raw.getHighestTime();
  } else {
    events = raw;
  }

  if (events.isEmpty()) {
    return;
  }

  // ── Optical flow ────────────────────────────────────────────────────────────
  // Accumulate filtered events and run a dense flow estimate once per window,
  // decoupling the flow rate from the (faster, variable) packet rate.
  if (flow_enabled_) {
    if (flow_window_start_us_ == std::numeric_limits<int64_t>::lowest()) {
      flow_window_start_us_ = events.getLowestTime();
    }
    flow_accum_.add(events);
    if (events.getHighestTime() - flow_window_start_us_ >=
      static_cast<int64_t>(flow_window_ms_ * 1000.0))
    {
      estimate_and_publish_flow(flow_accum_, msg->header);
      flow_accum_ = dv::EventStore();
      flow_window_start_us_ = events.getHighestTime();
    }
  }

  // Update time surfaces.
  ts_t_max_ = events.getHighestTime();

  for (const auto & ev : events) {
    if (ev.polarity()) {
      ts_on_->at(static_cast<uint32_t>(ev.y()), static_cast<uint32_t>(ev.x())) = ev.timestamp();
    } else {
      ts_off_->at(static_cast<uint32_t>(ev.y()), static_cast<uint32_t>(ev.x())) = ev.timestamp();
    }
  }

  // ── SAE ───────────────────────────────────────────────────────────────────

  const uint32_t w        = static_cast<uint32_t>(msg->width);
  const uint32_t h        = static_cast<uint32_t>(msg->height);
  const uint32_t n        = w * h;
  const int64_t window_us = static_cast<int64_t>(time_window_ms_ * 1000.0);
  const double inv_window = 1.0 / static_cast<double>(window_us);
  const int64_t t_max     = ts_t_max_;

  cv::Mat sae_img(
    static_cast<int>(h),
    static_cast<int>(w),
    CV_8UC3,
    cv::Scalar(0, 0, 0));

  #pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t row = i / w;
    const uint32_t col = i % w;
    uint8_t * px = sae_img.data + static_cast<std::ptrdiff_t>(i) * 3;

    const int64_t on_ts = ts_on_->at(row, col);
    if (on_ts > 0) {
      const double v = static_cast<double>(on_ts - t_max + window_us) * inv_window;
      px[1] = v > 0.0 ? static_cast<uint8_t>(v * 255.0) : uint8_t{0};
    }

    const int64_t off_ts = ts_off_->at(row, col);
    if (off_ts > 0) {
      const double v = static_cast<double>(off_ts - t_max + window_us) * inv_window;
      px[2] = v > 0.0 ? static_cast<uint8_t>(v * 255.0) : uint8_t{0};
    }
  }

  auto sae_msg = dua_cv_bridge::frame_to_msg(sae_img, sensor_msgs::image_encodings::BGR8);
  sae_msg->header = msg->header;
  pub_sae_->publish(*sae_msg);
}

}  // namespace event_detector_cpp
