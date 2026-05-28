/**
 * Event Detector subscriptions implementation.
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

#include "event_detector_cpp/event_detector_cpp.hpp"

#include <cstddef>
#include <mutex>
#include <utility>

namespace event_detector_cpp
{

// Bound the transfer queue: when the worker falls behind, drop the oldest chunk
// rather than let the queue grow without limit.
constexpr std::size_t kMaxQueueChunks = 32;

void EventDetector::callback_event_packet(EventPacket::ConstSharedPtr msg)
{
  if (!running_.load(std::memory_order_acquire) || msg->events.empty()) {
    return;
  }

  auto * decoder = decoder_factory_.getInstance(*msg);
  if (!decoder) {
    RCLCPP_WARN_ONCE(get_logger(), "No decoder for encoding '%s'", msg->encoding.c_str());
    return;
  }

  // Decode into an EventStore; the worker thread does all feature processing.
  EventStoreBuilder builder;
  decoder->decode(*msg, &builder);
  dv::EventStore events = builder.takeStore();
  if (events.isEmpty()) {
    return;
  }

  EventChunk chunk{
    std::move(events),
    msg->header,
    static_cast<int>(msg->width),
    static_cast<int>(msg->height)};

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queue_.size() >= kMaxQueueChunks) {
      queue_.pop_front();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Worker behind; dropping oldest event chunk");
    }
    queue_.push_back(std::move(chunk));
  }
  queue_cv_.notify_one();
}

}  // namespace event_detector_cpp
