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

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace event_detector_cpp
{

namespace
{

std::string trim(const std::string & in)
{
  auto begin = in.begin();
  while (begin != in.end() && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }
  auto end = in.end();
  while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
    --end;
  }
  return std::string(begin, end);
}

std::vector<int64_t> parse_int64_fields(std::string line)
{
  for (char & c : line) {
    if (c == ',') {
      c = ' ';
    }
  }
  std::istringstream stream(line);
  std::vector<int64_t> fields;
  std::string token;
  while (stream >> token) {
    fields.push_back(std::stoll(token));
  }
  return fields;
}

}  // namespace

void EventDetector::activate()
{
  prepare_flow_saving();

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
  flow_accum_first_us_ = std::numeric_limits<int64_t>::max();
  flow_accum_last_us_ = std::numeric_limits<int64_t>::lowest();
  prev_flow_field_ = Eigen::VectorXf();
  prev_flow_tiles_ = 0;
  moment_flow_.reset();
  flow_save_next_window_ = 0;
  flow_save_sequence_index_ = 0;
}

void EventDetector::prepare_flow_saving()
{
  flow_save_windows_.clear();
  flow_save_next_window_ = 0;
  flow_save_sequence_index_ = 0;
  flow_save_prepared_ = false;

  if (!flow_save_enabled_) {
    return;
  }

  const std::filesystem::path output_dir(flow_save_output_dir_);
  std::error_code ec;
  // Dense (flow_dense) and sparse (flow_events) predictions go into separate
  // subdirectories so both benchmarks can run from a single node run.
  for (const auto & dir :
    {output_dir, output_dir / "dense", output_dir / "sparse"})
  {
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      RCLCPP_ERROR(
        get_logger(), "Could not create flow save directory '%s': %s",
        dir.string().c_str(), ec.message().c_str());
      return;
    }
  }

  if (flow_save_clear_output_) {
    for (const auto & dir : {output_dir / "dense", output_dir / "sparse"}) {
      for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
          RCLCPP_WARN(
            get_logger(), "Could not scan flow save directory '%s': %s",
            dir.string().c_str(), ec.message().c_str());
          break;
        }
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
          std::filesystem::remove(entry.path(), ec);
          if (ec) {
            RCLCPP_WARN(
              get_logger(), "Could not remove stale flow PNG '%s': %s",
              entry.path().string().c_str(), ec.message().c_str());
            ec.clear();
          }
        }
      }
      ec.clear();
    }
  }

  if (!flow_save_timestamp_file_.empty()) {
    std::ifstream file(flow_save_timestamp_file_);
    if (!file.is_open()) {
      RCLCPP_ERROR(
        get_logger(), "Could not open flow save timestamp file '%s'",
        flow_save_timestamp_file_.c_str());
      return;
    }

    std::string line;
    int64_t implicit_index = flow_save_first_index_;
    while (std::getline(file, line)) {
      line = trim(line);
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::vector<int64_t> fields;
      try {
        fields = parse_int64_fields(line);
      } catch (const std::exception &) {
        // Header row, e.g. "from_timestamp_us,to_timestamp_us,file_index".
        continue;
      }

      if (fields.size() < 2) {
        continue;
      }

      const int64_t from_us = fields[0];
      const int64_t to_us = fields[1];
      if (to_us <= from_us) {
        RCLCPP_WARN(
          get_logger(), "Skipping invalid flow save interval [%" PRId64 ", %" PRId64 ")",
          from_us, to_us);
        continue;
      }

      const int64_t file_index = (fields.size() >= 3) ? fields[2] : implicit_index;
      flow_save_windows_.push_back({from_us, to_us, file_index});
      implicit_index += flow_save_index_step_;
    }

    std::sort(
      flow_save_windows_.begin(), flow_save_windows_.end(),
      [](const FlowSaveWindow & a, const FlowSaveWindow & b) {
        return a.from_us < b.from_us;
      });

    std::ofstream schedule_copy(output_dir / "moment_flow_timestamps.txt");
    if (schedule_copy.is_open()) {
      schedule_copy << "# from_timestamp_us, to_timestamp_us, file_index\n";
      for (const auto & window : flow_save_windows_) {
        schedule_copy << window.from_us << ", " << window.to_us << ", "
                      << window.file_index << '\n';
      }
    }
  }

  flow_save_prepared_ = true;
  if (flow_save_windows_.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "Raw flow saving enabled without a timestamp schedule; saving ordinary "
      "%.3f ms flow windows to '%s'",
      flow_max_window_ms_, flow_save_output_dir_.c_str());
  } else {
    RCLCPP_INFO(
      get_logger(), "Raw flow saving enabled: %zu scheduled windows -> '%s'",
      flow_save_windows_.size(), flow_save_output_dir_.c_str());
  }
}

void EventDetector::save_flow_results(
  const FlowResult & res,
  int64_t file_index,
  int64_t from_us,
  int64_t to_us)
{
  // Dense keeps its original behavior on degenerate windows (all-valid zeros);
  // only the sparse field treats a missing estimate as invalid.
  save_flow_png(res.flow_dense, "dense", /*empty_is_invalid=*/false, file_index, from_us, to_us);
  save_flow_png(res.flow_events, "sparse", /*empty_is_invalid=*/true, file_index, from_us, to_us);
}

void EventDetector::save_flow_png(
  const cv::Mat & flow_velocity,
  const std::string & subdir,
  bool empty_is_invalid,
  int64_t file_index,
  int64_t from_us,
  int64_t to_us)
{
  if (!flow_save_enabled_ || !flow_save_prepared_) {
    return;
  }
  if (to_us <= from_us) {
    RCLCPP_WARN(
      get_logger(), "Not saving flow for invalid interval [%" PRId64 ", %" PRId64 ")",
      from_us, to_us);
    return;
  }
  if (res_.width <= 0 || res_.height <= 0) {
    return;
  }

  cv::Mat velocity = flow_velocity;
  if (velocity.empty()) {
    if (empty_is_invalid) {
      // Sparse field with no estimate: write a fully-invalid frame (preserves
      // file-index alignment) rather than a fake all-valid zero field.
      const float nan = std::numeric_limits<float>::quiet_NaN();
      velocity = cv::Mat(res_.height, res_.width, CV_32FC2, cv::Scalar(nan, nan));
    } else {
      // Dense field: original behavior, an all-valid zero field.
      velocity = cv::Mat::zeros(res_.height, res_.width, CV_32FC2);
    }
  }
  if (velocity.type() != CV_32FC2) {
    RCLCPP_ERROR(
      get_logger(), "Cannot save flow PNG: expected CV_32FC2 velocity, got type %d",
      velocity.type());
    return;
  }

  const double dt_s = static_cast<double>(to_us - from_us) * 1e-6;
  cv::Mat encoded(velocity.rows, velocity.cols, CV_16UC3, cv::Scalar(1, 32768, 32768));
  for (int y = 0; y < velocity.rows; ++y) {
    const cv::Vec2f * vrow = velocity.ptr<cv::Vec2f>(y);
    cv::Vec<uint16_t, 3> * erow = encoded.ptr<cv::Vec<uint16_t, 3>>(y);
    for (int x = 0; x < velocity.cols; ++x) {
      // DSEC's on-disk channels are RGB=(flow_x, flow_y, valid). OpenCV stores
      // and writes this matrix as BGR, so memory is (valid, flow_y, flow_x).
      // flow_events leaves unsupported pixels as NaN; mark those invalid so the
      // benchmark scores only event-supported pixels (use --mask-mode intersection).
      if (!std::isfinite(vrow[x][0]) || !std::isfinite(vrow[x][1])) {
        erow[x][0] = 0;
        erow[x][1] = 32768;
        erow[x][2] = 32768;
        continue;
      }
      const double flow_x = static_cast<double>(vrow[x][0]) * dt_s;
      const double flow_y = static_cast<double>(vrow[x][1]) * dt_s;
      const double enc_x = std::round(flow_x * 128.0 + 32768.0);
      const double enc_y = std::round(flow_y * 128.0 + 32768.0);
      erow[x][0] = 1;
      erow[x][1] = static_cast<uint16_t>(std::clamp(enc_y, 0.0, 65535.0));
      erow[x][2] = static_cast<uint16_t>(std::clamp(enc_x, 0.0, 65535.0));
    }
  }

  std::ostringstream name;
  name << std::setw(6) << std::setfill('0') << file_index << ".png";
  const std::filesystem::path output_path =
    std::filesystem::path(flow_save_output_dir_) / subdir / name.str();

  if (!cv::imwrite(output_path.string(), encoded)) {
    RCLCPP_ERROR(get_logger(), "Could not write raw flow PNG '%s'", output_path.string().c_str());
    return;
  }

  RCLCPP_INFO(
    get_logger(), "Saved DSEC raw flow '%s' for [%" PRId64 ", %" PRId64 ") dt=%.6f s",
    output_path.string().c_str(), from_us, to_us, dt_s);
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
