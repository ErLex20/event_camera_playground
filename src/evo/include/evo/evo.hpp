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

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <std_msgs/msg/header.hpp>

namespace evo
{

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
   * @brief Activates the node.
   */
  void activate();

  /**
   * @brief Deactivates the node.
   */
  void deactivate();

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */

  /* Subscribers. */

  /* Node parameters. */
  bool    autostart_;

  /* Threads. */
  std::thread thread_worker_;

  /* Synchronization primitives. */
  std::atomic<bool> running_{false};
};

} // namespace evo