/**
 * Event Detector node definition.
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
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <dua_cv_bridge/dua_cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <dv-processing/core/core.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>

#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_msgs/msg/event_packet.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <event_detector_cpp/flow/moment_flow.hpp>

using namespace event_camera_msgs::msg;

namespace event_detector_cpp
{

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
   * Decodes the packet into a dv::EventStore and hands it to the worker thread
   * via the bounded queue; performs no feature processing itself.
   *
   * @param msg The incoming event packet.
   */
  void callback_event_packet(EventPacket::ConstSharedPtr msg);

  /**
   * @brief Worker thread main loop.
   *
   * Drains the event queue and runs the enabled features (BA, SAE, IWE, optical
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
  dv::EventStore compute_ba(const dv::EventStore & raw);

  /**
   * @brief Renders the Surface of Active Events for a chunk of events.
   *
   * Updates the per-polarity time surfaces and renders them, normalised over
   * time_window_ms. Runs on the worker thread.
   *
   * @param events The events to fold into the time surfaces.
   * @return A BGR8 SAE image.
   */
  cv::Mat compute_sae(const dv::EventStore & events);

  /* Outputs of one dense optical-flow solve over a window. */
  struct FlowResult
  {
    cv::Mat flow;  // dense flow field, BGR8 (hue = direction, value = speed)
    cv::Mat iwe;   // Image of Warped Events at the window midpoint, MONO8
  };

  /**
   * @brief Estimates dense optical flow from decayed spatio-temporal moments.
   *
   * Solves a coarse-to-fine tile-field surrogate of CMax from per-cell plane fits
   * and renders both the dense flow field (HSV) and the Image of Warped Events at
   * the window midpoint.
   *
   * Runs on the worker thread.
   *
   * @param window The window of events to estimate the flow from.
   * @return The rendered flow (BGR8) and IWE (MONO8) images.
   */
  FlowResult solve_flow_moment(const dv::EventStore & window);

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

  /* Builds a dv::EventStore from event_camera_codecs decoded events (ns → µs).
   * Accumulates into a dv::EventPacket to avoid the strict ordering requirement
   * of dv::EventStore::push_back(), then sorts before constructing the store. */
  class EventStoreBuilder : public event_camera_codecs::EventProcessor
  {
  public:
    EventStoreBuilder()
    : packet_(std::make_shared<dv::EventPacket>()) {}

    void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
    {
      packet_->elements.emplace_back(
        static_cast<int64_t>(sensor_time / 1000),
        static_cast<int16_t>(ex),
        static_cast<int16_t>(ey),
        polarity != 0);
    }

    void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
    void finished() override {}
    void rawData(const char *, size_t) override {}

    dv::EventStore takeStore()
    {
      std::stable_sort(
        packet_->elements.begin(), packet_->elements.end(),
        [](const dv::Event & a, const dv::Event & b) {
          return a.timestamp() < b.timestamp();
        });
      return dv::EventStore(
        std::const_pointer_cast<const dv::EventPacket>(packet_));
    }

  private:
    std::shared_ptr<dv::EventPacket> packet_;
  };

  /* One unit of work transferred from the subscription callback to the worker
   * thread: the decoded events plus the metadata needed to process and stamp
   * the outputs. */
  struct EventChunk
  {
    dv::EventStore events;
    std_msgs::msg::Header header;
    int width;
    int height;
  };

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_sae_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_flow_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_iwe_;

  /* Subscribers. */
  rclcpp::Subscription<EventPacket>::SharedPtr sub_event_packet_;

  /* Decoder. */
  event_camera_codecs::DecoderFactory<EventPacket, EventStoreBuilder> decoder_factory_;

  /* Surface state. */
  std::optional<dv::noise::BackgroundActivityNoiseFilter<>> ba_filter_;
  std::optional<dv::TimeSurface> ts_on_;
  std::optional<dv::TimeSurface> ts_off_;
  int64_t ts_t_max_{0};
  cv::Size res_;
  /* Mirrors the BA filter's private highest-processed time so we can keep the
   * packet stream monotonic before accept() (which throws on out-of-order input). */
  int64_t filter_high_us_{std::numeric_limits<int64_t>::lowest()};

  /* Optical-flow state. Events accumulate here until the window reaches
   * flow_num_events_, at which point the moment-flow estimator solves the whole
   * batch. */
  dv::EventStore flow_accum_;
  /* Previous window's finest-scale tile flow field and its per-side tile count,
   * used to warm-start the next window. Empty (prev_flow_tiles_ == 0) until the
   * first window has been solved. */
  Eigen::VectorXf prev_flow_field_;
  int prev_flow_tiles_{0};
  std::optional<flow::MomentFlow> moment_flow_;

  /* Node parameters. */
  bool    autostart_;
  double  time_window_ms_;
  bool    ba_filter_enabled_;
  double  ba_filter_dt_ms_;
  bool    sae_enabled_;
  bool    iwe_enabled_;
  bool    flow_enabled_;
  int64_t flow_num_events_;
  int64_t flow_num_scales_;
  int64_t flow_cell_size_px_;
  int64_t flow_decay_tau_us_;
  bool    flow_tau_adaptive_;
  double  flow_cell_min_mass_;
  double  flow_cell_min_lambda_;
  double  flow_aperture_ratio_;
  double  flow_tikhonov_eps_;
  int64_t flow_smooth_iters_;
  double  flow_smooth_alpha_;
  bool    flow_refine_enabled_;
  int64_t flow_refine_iters_;
  double  flow_refine_huber_delta_;
  int64_t flow_iwe_scale_;
  double  flow_max_speed_px_s_;

  /* Threads. */
  std::thread thread_worker_;

  /* Event transfer queue (producer: callback, consumer: worker). */
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<EventChunk> queue_;

  /* Synchronization primitives. */
  std::atomic<bool> running_{false};
};

} // namespace event_detector_cpp
