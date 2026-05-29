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
  cgroup_event_packet_ = dua_create_exclusive_cgroup();
}

void EVO::init_subscribers()
{}

void EVO::init_publishers()
{}

} // namespace evo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(evo::EVO)
