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


#include <atomic>

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_msgs/msg/event_packet.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <structs/background_activity_filter.hpp>
#include <structs/combined_processor.hpp>
#include <structs/surface_active_events.hpp>
#include <structs/surface_eros.hpp>

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

  /* Subscription callbacks. */
  void callback_event_packet(EventPacket::ConstSharedPtr msg);

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_sae_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_eros_;

  /* Subscribers. */
  rclcpp::Subscription<EventPacket>::SharedPtr sub_event_packet_;

  /* Surface state. */
  SurfaceActiveEvents sae_;
  SurfaceEros eros_;
  BackgroundActivityFilter ba_filter_;
  event_camera_codecs::DecoderFactory<EventPacket, CombinedProcessor> decoder_factory_;

  /* Node parameters. */
  bool    autostart_;
  double  time_window_ms_;
  int64_t eros_k_;
  bool    ba_filter_enabled_;
  double  ba_filter_dt_ms_;

  /* Synchronization primitives. */
  std::atomic<bool> running_{false};
};

} // namespace event_detector_cpp
