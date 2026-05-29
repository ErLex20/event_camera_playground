/**
 * EVO node implementation.
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

#include "evo/evo.hpp"

namespace evo
{

EVO::EVO(const rclcpp::NodeOptions & node_options)
: NodeBase("evo", node_options, true)
{
  dua_init_node();

  // Internal pose buffer that replaces the cross-node TF system of the
  // original four-node pipeline.
  tf_buffer_ = std::make_shared<tf2::BufferCore>();

  // Load parameters shared by several stages (mirror of the original ROS1
  // global node handle parameters).
  auto sp = shared_params();
  world_frame_id_ = rpg_common_ros::param<std::string>(sp, "world_frame_id", "world");
  dvs_frame_id_ = rpg_common_ros::param<std::string>(sp, "dvs_frame_id", "dvs_evo");
  bootstrap_frame_id_ =
    rpg_common_ros::param<std::string>(sp, "dvs_bootstrap_frame_id", "camera_0");
  camera_name_ = rpg_common_ros::param<std::string>(sp, "camera_name", "DAVIS-ijrr");
  calib_file_ = rpg_common_ros::param<std::string>(sp, "calib_file", "");
  min_depth_ = rpg_common_ros::param<double>(sp, "min_depth", 0.4);
  max_depth_ = rpg_common_ros::param<double>(sp, "max_depth", 5.0);
  num_depth_cells_ = rpg_common_ros::param<int>(sp, "num_depth_cells", 100);
  fov_virtual_camera_deg_ =
    rpg_common_ros::param<double>(sp, "fov_virtual_camera_deg", 80.0);
  virtual_width_ = rpg_common_ros::param<int>(sp, "virtual_width", 240);
  virtual_height_ = rpg_common_ros::param<int>(sp, "virtual_height", 180);

  RCLCPP_INFO(this->get_logger(), "Node initialized");

  if (autostart_) {
    activate();
  }
}

EVO::~EVO()
{
  deactivate();
}

void EVO::init_cgroups()
{
  // The event-packet subscription runs at high rate; allow it to be processed
  // concurrently (reentrant), as the original event topic was handled in its
  // own spinning node.
  cgroup_event_packet_ = dua_create_reentrant_cgroup();
}

void EVO::init_publishers()
{}

void EVO::init_subscribers()
{
  rclcpp::SubscriptionOptions opts;
  opts.callback_group = cgroup_event_packet_;

  sub_event_packet_ = this->create_subscription<EventPacket>(
    "~/events",
    dua_qos::BestEffort::get_datum_qos(),
    std::bind(&EVO::event_packet_callback, this, std::placeholders::_1),
    opts);
}

} // namespace evo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(evo::EVO)
