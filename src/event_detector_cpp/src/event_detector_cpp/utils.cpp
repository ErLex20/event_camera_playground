/**
 * Event Detector utils implementation.
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

namespace event_detector_cpp
{

void EventDetector::activate()
{
  // Set running flag
  running_.store(true, std::memory_order_release);
  thread_worker_ = std::thread(
    &EventDetector::worker_thread_routine,
    this);

  RCLCPP_WARN(this->get_logger(), "Event Detector ACTIVATED");
}

void EventDetector::deactivate()
{
  // Clear running flag and wake the worker so it can drain and exit.
  running_.store(false, std::memory_order_release);
  queue_cv_.notify_all();
  if (thread_worker_.joinable()) {
    thread_worker_.join();
  }

  RCLCPP_WARN(this->get_logger(), "Event Detector DEACTIVATED");
}

void EventDetector::lazy_init(int width, int height)
{
  res_ = cv::Size(width, height);
  ts_on_.emplace(res_);
  ts_off_.emplace(res_);
  if (ba_filter_enabled_) {
    ba_filter_.emplace(res_, dv::Duration(static_cast<int64_t>(ba_filter_dt_ms_ * 1000.0)));
  }
}

void EventDetector::reset_state()
{
  if (ba_filter_enabled_) {
    ba_filter_.emplace(res_, dv::Duration(static_cast<int64_t>(ba_filter_dt_ms_ * 1000.0)));
  }
  if (ts_on_.has_value()) {
    ts_on_->reset();
  }
  if (ts_off_.has_value()) {
    ts_off_->reset();
  }
  filter_high_us_ = std::numeric_limits<int64_t>::lowest();
  flow_accum_ = dv::EventStore();
  prev_flow_field_ = Eigen::VectorXf();
  prev_flow_tiles_ = 0;
}

void EventDetector::publish_image(
  const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & pub,
  const cv::Mat & img,
  const std::string & encoding,
  const std_msgs::msg::Header & header)
{
  if (img.empty()) {
    return;
  }
  auto msg = dua_cv_bridge::frame_to_msg(img, encoding);
  msg->header = header;
  pub->publish(*msg);
}

} // namespace event_detector_cpp
