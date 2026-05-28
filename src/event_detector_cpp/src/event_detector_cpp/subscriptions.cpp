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

  // Lazy init — all three surfaces use the same sensor geometry.
  if (sae_.width == 0) {
    sae_.resize(msg->width, msg->height);
    eros_.resize(msg->width, msg->height, static_cast<uint32_t>(eros_k_));
    ba_filter_.resize(
      msg->width, msg->height,
      static_cast<float>(ba_filter_dt_ms_ * 1.0e6));
  }

  // SAE is a per-packet snapshot — reset before each decode.
  sae_.reset();

  // Single decode pass: BA filter check (if enabled) feeds both SAE and EROS.
  BackgroundActivityFilter * ba_ptr = ba_filter_enabled_ ? &ba_filter_ : nullptr;
  CombinedProcessor proc(sae_, eros_, ba_ptr);
  decoder->decode(*msg, &proc);

  // ── SAE ───────────────────────────────────────────────────────────────────

  if (sae_.t_max > 0.0f) {
    const uint32_t n       = sae_.width * sae_.height;
    const float window_ns  = static_cast<float>(time_window_ms_ * 1.0e6);
    const float inv_window = 1.0f / window_ns;
    const float t_max      = sae_.t_max;

    cv::Mat sae_img(
      static_cast<int>(sae_.height),
      static_cast<int>(sae_.width),
      CV_8UC3,
      cv::Scalar(0, 0, 0));

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n; ++i) {
      uint8_t * px = sae_img.data + static_cast<std::ptrdiff_t>(i) * 3;

      if (sae_.on[i] > 0.0f) {
        const float v = (sae_.on[i] - t_max + window_ns) * inv_window;
        px[1] = v > 0.0f ? static_cast<uint8_t>(v * 255.0f) : uint8_t{0};
      }

      if (sae_.off[i] > 0.0f) {
        const float v = (sae_.off[i] - t_max + window_ns) * inv_window;
        px[2] = v > 0.0f ? static_cast<uint8_t>(v * 255.0f) : uint8_t{0};
      }
    }

    auto sae_msg = dua_cv_bridge::frame_to_msg(sae_img, sensor_msgs::image_encodings::BGR8);
    sae_msg->header = msg->header;
    pub_sae_->publish(*sae_msg);
  }

  // ── EROS ──────────────────────────────────────────────────────────────────

  const uint32_t ne = eros_.width * eros_.height;

  cv::Mat eros_img(
    static_cast<int>(eros_.height),
    static_cast<int>(eros_.width),
    CV_8UC3,
    cv::Scalar(0, 0, 0));

  // Values are in [0.0, 1.0] — scale directly; no time normalization needed.
  #pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < ne; ++i) {
    uint8_t * px = eros_img.data + static_cast<std::ptrdiff_t>(i) * 3;
    px[1] = static_cast<uint8_t>(eros_.on[i]  * 255.0f);  // green = ON
    px[2] = static_cast<uint8_t>(eros_.off[i] * 255.0f);  // red   = OFF
  }

  auto eros_msg = dua_cv_bridge::frame_to_msg(eros_img, sensor_msgs::image_encodings::BGR8);
  eros_msg->header = msg->header;
  pub_eros_->publish(*eros_msg);
}

}  // namespace event_detector_cpp
