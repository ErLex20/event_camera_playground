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

  // Fan the decoded events out to every active stage.
  dispatch_events(decoded_events_);
}

void EVO::dispatch_events(const std::vector<Event> & events)
{
  // TODO(evo-port): push the decoded events into each stage's buffer,
  // preserving each buffer's windowing/decimation semantics:
  //   - bootstrapper.eventQueue_
  //   - mapper.event_queue_
  //   - tracker.events_
  //   - reconstruction.events_
  // The stages are internalized in Phase 3; until then this is a no-op other
  // than reporting throughput for bring-up.
  (void)events;
}

}  // namespace evo
