/**
 * EVO utils implementation.
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

void EVO::activate()
{
  // Set running flag
  running_.store(true, std::memory_order_release);

  RCLCPP_WARN(this->get_logger(), "EVO ACTIVATED");
}

void EVO::deactivate()
{
  // Clear running flag
  running_.store(false, std::memory_order_release);

  RCLCPP_WARN(this->get_logger(), "EVO DEACTIVATED");
}

} // namespace evo
