/**
 * Moment Flow node implementation.
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

#include "moment_flow/moment_flow.hpp"

namespace moment_flow
{

EventDetector::EventDetector(const rclcpp::NodeOptions & node_options)
: NodeBase("moment_flow", node_options, true)
{
  dua_init_node();

  RCLCPP_INFO(this->get_logger(), "Node initialized");

  if (autostart_) {
    activate();
  }
}

EventDetector::~EventDetector()
{
  deactivate();
}

void EventDetector::init_cgroups()
{
  cgroup_event_packet_ = dua_create_exclusive_cgroup();
}

void EventDetector::init_subscribers()
{
  sub_event_packet_ = dua_create_subscription<EventPacket>(
    "/event_packet",
    std::bind(
      &EventDetector::callback_event_packet,
      this,
      std::placeholders::_1),
    dua_qos::Reliable::get_datum_qos(),
    cgroup_event_packet_);
}

void EventDetector::init_publishers()
{
  pub_flow_dense_debug_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_dense_debug",
    dua_qos::BestEffort::get_image_qos(1));

  pub_flow_dense_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_dense",
    dua_qos::BestEffort::get_image_qos(1));

  pub_flow_tiles_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_tiles",
    dua_qos::BestEffort::get_image_qos(1));

  pub_flow_tile_debug_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_tile_debug",
    dua_qos::BestEffort::get_image_qos(1));

  pub_flow_events_debug_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_events_debug",
    dua_qos::BestEffort::get_image_qos(1));

  pub_flow_events_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/flow_events",
    dua_qos::BestEffort::get_image_qos(1));

  pub_iwe_ = dua_create_publisher<sensor_msgs::msg::Image>(
    "~/iwe_image",
    dua_qos::BestEffort::get_image_qos(1));
}

} // namespace moment_flow

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(moment_flow::EventDetector)
