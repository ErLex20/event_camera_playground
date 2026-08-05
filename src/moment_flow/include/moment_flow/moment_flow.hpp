/**
 * Moment Flow node definition.
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

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <dua_cv_bridge/dua_cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_msgs/msg/event_packet.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <moment_flow/flow/moment_flow.hpp>

using namespace event_camera_msgs::msg;

namespace moment_flow
{

struct Event
{
  int64_t t_us;
  int16_t x_px;
  int16_t y_px;
  bool polarity;

  Event() = default;

  Event(int64_t timestamp_us, int16_t x, int16_t y, bool p)
  : t_us(timestamp_us),
    x_px(x),
    y_px(y),
    polarity(p)
  {}

  int64_t timestamp() const { return t_us; }
  int16_t x() const { return x_px; }
  int16_t y() const { return y_px; }
};

class EventStore
{
public:
  using container_type = std::vector<Event>;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;

  EventStore() = default;

  explicit EventStore(container_type events)
  : events_(std::move(events))
  {}

  bool isEmpty() const { return events_.empty(); }
  std::size_t size() const { return events_.size(); }
  void clear() { events_.clear(); }
  void reserve(std::size_t n) { events_.reserve(n); }

  iterator begin() { return events_.begin(); }
  iterator end() { return events_.end(); }
  const_iterator begin() const { return events_.begin(); }
  const_iterator end() const { return events_.end(); }

  const Event & front() const { return events_.front(); }
  const Event & back() const { return events_.back(); }

  void push_back(const Event & event) { events_.push_back(event); }
  void push_back(Event && event) { events_.push_back(std::move(event)); }
  void push_back(int64_t timestamp_us, int16_t x, int16_t y, bool polarity)
  {
    events_.emplace_back(timestamp_us, x, y, polarity);
  }

  int64_t getLowestTime() const
  {
    return events_.empty() ? std::numeric_limits<int64_t>::max() : events_.front().timestamp();
  }

  int64_t getHighestTime() const
  {
    return events_.empty() ? std::numeric_limits<int64_t>::lowest() : events_.back().timestamp();
  }

  EventStore sliceTime(int64_t from_us) const
  {
    const auto first = std::lower_bound(
      events_.begin(), events_.end(), from_us,
      [](const Event & event, int64_t t_us) {
        return event.timestamp() < t_us;
      });
    return EventStore(container_type(first, events_.end()));
  }

  void add(const EventStore & other)
  {
    if (other.isEmpty()) {
      return;
    }
    if (!events_.empty() && other.getLowestTime() < getHighestTime()) {
      throw std::out_of_range("EventStore::add received out-of-order events");
    }
    events_.insert(events_.end(), other.begin(), other.end());
  }

  void add(EventStore && other)
  {
    if (other.isEmpty()) {
      return;
    }
    if (!events_.empty() && other.getLowestTime() < getHighestTime()) {
      throw std::out_of_range("EventStore::add received out-of-order events");
    }
    if (events_.empty()) {
      events_ = std::move(other.events_);
      return;
    }
    events_.insert(
      events_.end(),
      std::make_move_iterator(other.events_.begin()),
      std::make_move_iterator(other.events_.end()));
    other.events_.clear();
  }

private:
  container_type events_;
};

class EventDetector : public dua_node::NodeBase
{
public:
  /**
   * @brief Constructor.
   */
  EventDetector(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());

  /**
   * @brief Destructor.
   */
  ~EventDetector();

private:
  /* Init functions. */
  void init_parameters() override;
  void init_cgroups() override;
  void init_publishers() override;
  void init_subscribers() override;

  /**
   * @brief Activates the node.
   */
  void activate();

  /**
   * @brief Deactivates the node.
   */
  void deactivate();

  /**
   * @brief Event-packet subscription callback (executor thread).
   *
   * Decodes the packet into an EventStore and hands it to the worker thread
   * via the bounded queue; performs no feature processing itself.
   *
   * @param msg The incoming event packet.
   */
  void callback_event_packet(EventPacket::ConstSharedPtr msg);

  /**
   * @brief Worker thread main loop.
   *
   * Drains the event queue and runs the enabled features (BA, IWE, optical
   * flow) on each chunk, publishing their results. Owns all stateful processors,
   * so they need no locking. Exits when the node is deactivated and the queue
   * has been drained.
   */
  void worker_thread_routine();

  /**
   * @brief Applies the background-activity noise filter to a chunk of events.
   *
   * Enforces the monotonic ordering dv's filter requires and resets all
   * stateful processors on a stream discontinuity. Runs on the worker thread.
   *
   * @param raw Decoded events for one packet.
   * @return The filtered events (possibly empty).
   */
  EventStore compute_ba(const EventStore & raw);

  /* Outputs of one dense optical-flow solve over a window. */
  struct FlowResult
  {
    cv::Mat flow_dense_debug;   // dense flow visualization, BGR8
    cv::Mat flow_dense;         // dense image-flow velocity, CV_32FC2 [px/s]
    cv::Mat flow_tiles;         // native tile-grid image-flow velocity, CV_32FC2 [px/s]
    cv::Mat flow_tile_debug;    // native tile-grid flow visualization, BGR8
    cv::Mat flow_events_debug;  // event-supported flow visualization, BGR8
    cv::Mat flow_events;        // flow_dense only where event support exists, CV_32FC2 [px/s]
    cv::Mat iwe;                // Image of Warped Events at the window midpoint, MONO8
  };

  struct FlowTiming
  {
    double total_ms = 0.0;
    double select_events_ms = 0.0;
    double pack_events_ms = 0.0;
    double solve_moments_ms = 0.0;
    double refine_ms = 0.0;
    double render_events_ms = 0.0;
    double iwe_focus_ms = 0.0;
    double tile_flow_ms = 0.0;
    double tile_debug_ms = 0.0;
    double dense_flow_ms = 0.0;
    double flow_debug_ms = 0.0;
    double support_mask_ms = 0.0;
    double events_mask_ms = 0.0;
    double iwe_debug_ms = 0.0;
  };

  struct FlowSaveWindow
  {
    int64_t from_us;
    int64_t to_us;
    int64_t file_index;
  };

  /**
   * @brief Estimates dense optical flow from spatio-temporal moments.
   *
   * Solves a coarse-to-fine tile-field surrogate of CMax from per-cell plane fits
   * and renders both the dense flow field (HSV) and the Image of Warped Events at
   * the window midpoint.
   *
   * Runs on the worker thread.
   *
   * @param window The window of events to estimate the flow from.
   * @return The rendered debug flow (BGR8), dense flow (CV_32FC2),
   * event-supported flow (CV_32FC2), and IWE (MONO8) images.
   */
  FlowResult solve_flow_moment(
    const EventStore & window,
    std::optional<int64_t> t_ref_override_us = std::nullopt);

  /**
   * @brief Opens the end-to-end timing CSV log, if `timing_log_path_` is set.
   *
   * Truncates any pre-existing file and writes the header row. No-op (log
   * stays disabled) if the path is empty or cannot be opened.
   */
  void prepare_timing_log();

  /**
   * @brief Appends one row to the timing CSV log.
   *
   * No-op if the log is disabled. Always called regardless of `debug_`: the
   * end-to-end solve time and coarse timing breakdown are useful data, not
   * debug diagnostics.
   */
  void log_timing(
    int64_t from_us,
    int64_t to_us,
    std::size_t num_events,
    const FlowTiming & timing);

  /**
   * @brief Loads the optional benchmark-save timestamp schedule.
   *
   * If a DSEC-style timestamp file is configured, saves are produced at the
   * ground-truth cadence and named by the DSEC image index. Each estimate still
   * uses the ordinary flow_max_window_ms_ algorithm window at the start of the
   * corresponding ground-truth interval. If no timestamp file is configured,
   * saving falls back to every ordinary flow_max_window_ms_ window.
   */
  void prepare_flow_saving();

  /**
   * @brief Processes one chunk using the DSEC benchmark-save schedule.
   *
   * Splits incoming events so the estimator runs over flow_max_window_ms_ from
   * the start of each ground-truth interval, then skips ahead to the next
   * scheduled save. This keeps the algorithm window independent from the
   * ground-truth cadence.
   *
   * @return Events after the final scheduled save, so ordinary publishing can
   * continue after benchmark output is complete.
   */
  EventStore accumulate_scheduled_flow(
    const EventStore & events,
    const std_msgs::msg::Header & header);

  /**
   * @brief Saves both the dense and event-sparse flow fields as DSEC PNGs.
   *
   * Writes flow_dense to the "dense" subdirectory. When flow_events_enabled_
   * is true, also writes flow_events to the "sparse" subdirectory.
   */
  void save_flow_results(
    const FlowResult & res,
    int64_t file_index,
    int64_t from_us,
    int64_t to_us);

  /**
   * @brief Saves a per-pixel image-flow velocity field in DSEC 16-bit PNG format.
   *
   * The velocity is converted to displacement using the exact interval
   * duration, then encoded as RGB: flow_x, flow_y, valid. OpenCV writes BGR, so
   * the in-memory matrix uses B=valid, G=flow_y, R=flow_x. Non-finite pixels
   * (e.g. unsupported cells in flow_events) are written with valid=0. Files are
   * written under flow_save_output_dir_/<subdir>.
   */
  void save_flow_png(
    const cv::Mat & flow_velocity,
    const std::string & subdir,
    bool empty_is_invalid,
    int64_t file_index,
    int64_t from_us,
    int64_t to_us);

  /**
   * @brief Allocates the resolution-dependent processors on the first packet.
   *
   * @param width Sensor width [px].
   * @param height Sensor height [px].
   */
  void lazy_init(int width, int height);

  /**
   * @brief Rebuilds every stateful processor after a stream discontinuity.
   */
  void reset_state();

  /**
   * @brief Publishes an image if it is non-empty, stamping it with a header.
   *
   * @param pub Target image publisher.
   * @param img Image to publish; ignored if empty.
   * @param encoding sensor_msgs image encoding (e.g. BGR8, MONO8).
   * @param header Header to stamp onto the message.
   */
  void publish_image(
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & pub,
    const cv::Mat & img,
    const std::string & encoding,
    const std_msgs::msg::Header & header);

  /* Builds an EventStore from event_camera_codecs decoded events (ns -> us).
   * Accumulates into a vector, then sorts before handing it to the worker. */
  class EventStoreBuilder : public event_camera_codecs::EventProcessor
  {
  public:
    EventStoreBuilder() = default;

    void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
    {
      events_.emplace_back(
        static_cast<int64_t>(sensor_time / 1000),
        static_cast<int16_t>(ex),
        static_cast<int16_t>(ey),
        polarity != 0);
    }

    void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
    void finished() override {}
    void rawData(const char *, size_t) override {}

    EventStore takeStore()
    {
      std::stable_sort(
        events_.begin(), events_.end(),
        [](const Event & a, const Event & b) {
          return a.timestamp() < b.timestamp();
        });
      return EventStore(std::move(events_));
    }

  private:
    std::vector<Event> events_;
  };

  /* One unit of work transferred from the subscription callback to the worker
   * thread: the decoded events plus the metadata needed to process and stamp
   * the outputs. */
  struct EventChunk
  {
    EventStore events;
    std_msgs::msg::Header header;
    int width;
    int height;
  };

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_dense_debug_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_dense_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_tiles_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_tile_debug_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_events_debug_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_events_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_iwe_;

  /* Subscribers. */
  rclcpp::Subscription<EventPacket>::SharedPtr sub_event_packet_;

  /* Decoder. */
  event_camera_codecs::DecoderFactory<EventPacket, EventStoreBuilder> decoder_factory_;

  /* Filter / resolution state. */
  cv::Size res_;
  std::vector<int64_t> ba_last_us_;
  /* Highest BA-filter input time, used to keep packet overlaps monotonic and
   * detect stream discontinuities. */
  int64_t filter_high_us_{std::numeric_limits<int64_t>::lowest()};

  /* Optical-flow state. Events accumulate here until the time span reaches
   * flow_max_window_ms_, at which point the moment-flow estimator solves the
   * whole batch. */
  EventStore flow_accum_;
  int64_t flow_accum_first_us_{std::numeric_limits<int64_t>::max()};
  int64_t flow_accum_last_us_{std::numeric_limits<int64_t>::lowest()};
  /* Previous window's finest-scale tile flow field and its per-side tile count,
   * used to warm-start the next window. Empty (prev_flow_tiles_ == 0) until the
   * first window has been solved. */
  Eigen::VectorXf prev_flow_field_;
  int prev_flow_tiles_{0};
  std::optional<flow::MomentFlow> moment_flow_;
  std::vector<FlowSaveWindow> flow_save_windows_;
  std::size_t flow_save_next_window_{0};
  int64_t flow_save_sequence_index_{0};
  bool flow_save_prepared_{false};

  /* Node parameters. */
  bool    autostart_;
  bool    ba_filter_enabled_;
  double  ba_filter_dt_ms_;
  bool    iwe_enabled_;
  bool    flow_enabled_;
  bool    flow_events_enabled_;
  bool    debug_;
  double  flow_max_window_ms_;
  int64_t flow_max_solve_events_;
  int64_t flow_num_scales_;
  int64_t flow_cell_size_px_;
  double  flow_cell_min_mass_;
  double  flow_cell_min_lambda_;
  double  flow_cell_max_residual_ratio_;
  double  flow_tile_min_mass_;
  int64_t flow_tile_min_cells_;
  double  flow_tile_min_lambda_;
  double  flow_aperture_ratio_;
  double  flow_tikhonov_eps_;
  double  flow_prior_lambda_;
  double  flow_reg_lambda_;
  int64_t flow_reg_sweeps_;
  double  flow_reg_sigma_;
  bool    flow_refine_enabled_;
  int64_t flow_refine_iters_;
  int64_t flow_iwe_scale_;
  double  flow_max_speed_px_s_;
  bool flow_save_enabled_;
  std::string flow_save_output_dir_;
  std::string flow_save_timestamp_file_;
  bool flow_save_clear_output_;
  int64_t flow_save_first_index_;
  int64_t flow_save_index_step_;
  std::string timing_log_path_;

  /* End-to-end timing CSV log (open for the lifetime of one activation). */
  std::ofstream timing_log_stream_;
  bool timing_log_ready_{false};

  /* Threads. */
  std::thread thread_worker_;

  /* Event transfer queue (producer: callback, consumer: worker). */
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<EventChunk> queue_;

  /* Synchronization primitives. */
  std::atomic<bool> running_{false};
};

} // namespace moment_flow
