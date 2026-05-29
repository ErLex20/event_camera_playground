/**
 * EVO node definition.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 29, 2026
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <std_msgs/msg/header.hpp>

#include <event_camera_msgs/msg/event_packet.hpp>
#include <event_camera_codecs/decoder.h>
#include <event_camera_codecs/decoder_factory.h>

#include <tf2/buffer_core.h>

#include <rpg_common_ros/params_helper.hpp>

#include "evo/event_decoder.hpp"
#include "evo/event_types.hpp"

namespace evo
{

using EventPacket = event_camera_msgs::msg::EventPacket;

class EVO : public dua_node::NodeBase
{
public:
  /**
   * @brief Constructor.
   */
  EVO(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());

  /**
   * @brief Destructor.
   */
  ~EVO();

private:
  /* Init functions. */
  void init_parameters() override;
  void init_cgroups() override;
  void init_publishers() override;
  void init_subscribers() override;

  /**
   * @brief Activates the node (spawns worker threads).
   */
  void activate();

  /**
   * @brief Deactivates the node (joins worker threads).
   */
  void deactivate();

  /* Parameter helpers. */
  /// Shared (global) parameter namespace, mirroring the original ROS1 nh_.
  rpg_common_ros::ParamProvider shared_params() { return {this, ""}; }
  /// Stage-private parameter namespace, mirroring the original ROS1 nhp_.
  rpg_common_ros::ParamProvider stage_params(const std::string & stage)
  {
    return {this, stage + "."};
  }

  /* Event input. */
  /**
   * @brief Decodes an incoming event packet and fans the events out to the
   *        per-stage buffers.
   */
  void event_packet_callback(const EventPacket::ConstSharedPtr msg);

  /**
   * @brief Dispatches a freshly decoded batch of events to every active stage.
   */
  void dispatch_events(const std::vector<Event> & events);

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */

  /* Subscribers. */
  rclcpp::Subscription<EventPacket>::SharedPtr sub_event_packet_;

  /* Event decoding. */
  event_camera_codecs::DecoderFactory<EventPacket, EventCollector> decoder_factory_;
  std::vector<Event> decoded_events_;  ///< scratch buffer reused per packet

  /* Internal pose buffer (replaces cross-node TF). The tracking stage inserts
   * stamped transforms here; the mapper queries it by timestamp. */
  std::shared_ptr<tf2::BufferCore> tf_buffer_;

  /* Shared parameters (read by several stages). */
  std::string world_frame_id_;
  std::string dvs_frame_id_;
  std::string bootstrap_frame_id_;
  std::string camera_name_;
  std::string calib_file_;
  double min_depth_{0.0};
  double max_depth_{0.0};
  int num_depth_cells_{0};
  double fov_virtual_camera_deg_{0.0};
  int virtual_width_{0};
  int virtual_height_{0};

  /* Node parameters (DUA codegen). */
  bool autostart_;

  /* Threads. */
  std::thread thread_worker_;

  /* Synchronization primitives. */
  std::mutex data_mutex_;
  std::atomic<bool> running_{false};
};

} // namespace evo
