/**
 * EVO subscriptions implementation.
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

void EVO::event_packet_callback(const EventPacket::ConstSharedPtr msg)
{
  // Pick (or create) the decoder matching this packet's encoding/geometry.
  auto * decoder = decoder_factory_.getInstance(*msg);
  if (decoder == nullptr) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "No decoder available for encoding '%s'", msg->encoding.c_str());
    return;
  }

  // Decode the whole packet into the scratch buffer. In this codecs version
  // decode(msg, processor) consumes the entire packet in a single call (it
  // sets the time base internally).
  decoded_events_.clear();
  EventCollector collector(&decoded_events_);
  decoder->decode(*msg, &collector);

  if (decoded_events_.empty()) {
    return;
  }

  // [DIAG] Track packet receipt vs. publisher seq to quantify drops.
  {
    static uint64_t recv_pkts = 0, recv_evs = 0, dropped = 0, last_seq = 0;
    static bool have_last = false;
    if (have_last && msg->seq > last_seq + 1) dropped += msg->seq - last_seq - 1;
    last_seq = msg->seq;
    have_last = true;
    recv_pkts += 1;
    recv_evs += decoded_events_.size();
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "[DIAG] recv_pkts=%lu recv_evs=%lu dropped_pkts=%lu (seq=%lu)",
      recv_pkts, recv_evs, dropped, msg->seq);
  }

  // Fan the decoded events out to every active stage.
  dispatch_events(decoded_events_);
}

void EVO::dispatch_events(const std::vector<Event> & events)
{
  // Each stage owns its own buffer and windowing logic.
  bootstrapper_->addEvents(events);
  mapper_->addEvents(events);
  tracker_->addEvents(events);
  reconstruction_->addEvents(events);
}

void EVO::on_remote_key(const std_msgs::msg::String::ConstSharedPtr msg)
{
  const std::string cmd = msg->data;

  // Dispatch the command to every stage that understands it.
  bootstrapper_->onRemoteKey(cmd);
  tracker_->onRemoteKey(cmd);
  mapper_->onRemoteKey(cmd);
}

void EVO::on_tracked_pose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
{
  // Forward the tracked pose timestamp to the mapper so it can advance its
  // event cursor and auto-trigger DSI map updates.
  rclcpp::Time stamp(msg->header.stamp);
  mapper_->onTrackedPose(stamp);
}

}  // namespace evo
