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
  if (running_.exchange(true)) {
    // Already running, nothing to do.
    return;
  }

  RCLCPP_INFO(this->get_logger(), "EVO ACTIVATED – spawning stage threads");

  // Spawn stage threads. Each stage will poll the shared running_ flag.
  thread_integrate_ = std::thread(
      [this]() { bootstrapper_->integratingThread(running_); });
  thread_bootstrap_ = std::thread(
      [this]() { bootstrapper_->bootstrappingThread(running_); });
  thread_tracking_ =
      std::thread([this]() { tracker_->trackingThread(running_); });
  thread_overlap_ =
      std::thread([this]() { tracker_->publishMapOverlapThread(running_); });
}

void EVO::deactivate()
{
  if (!running_.exchange(false)) {
    // Already stopped, nothing to do.
    return;
  }

  RCLCPP_INFO(this->get_logger(), "EVO DEACTIVATED – joining stage threads");

  if (thread_integrate_.joinable()) thread_integrate_.join();
  if (thread_bootstrap_.joinable()) thread_bootstrap_.join();
  if (thread_tracking_.joinable()) thread_tracking_.join();
  if (thread_overlap_.joinable()) thread_overlap_.join();

  RCLCPP_INFO(this->get_logger(), "EVO DEACTIVATED – all threads joined");
}

} // namespace evo
