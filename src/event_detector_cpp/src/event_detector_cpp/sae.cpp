/**
 * Event Detector Surface of Active Events rendering.
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

#include <opencv2/core.hpp>

namespace event_detector_cpp
{

cv::Mat EventDetector::compute_sae(const dv::EventStore & events)
{
  if (events.isEmpty() || !ts_on_.has_value() || !ts_off_.has_value()) {
    return cv::Mat();
  }

  // Update time surfaces with the latest event timestamps per pixel/polarity.
  ts_t_max_ = events.getHighestTime();
  for (const auto & ev : events) {
    if (ev.polarity()) {
      ts_on_->at(static_cast<uint32_t>(ev.y()), static_cast<uint32_t>(ev.x())) = ev.timestamp();
    } else {
      ts_off_->at(static_cast<uint32_t>(ev.y()), static_cast<uint32_t>(ev.x())) = ev.timestamp();
    }
  }

  const uint32_t w        = static_cast<uint32_t>(res_.width);
  const uint32_t h        = static_cast<uint32_t>(res_.height);
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

  return sae_img;
}

}  // namespace event_detector_cpp
