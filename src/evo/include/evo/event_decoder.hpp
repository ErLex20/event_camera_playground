/**
 * Event-packet decoder processor for the merged EVO node.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
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

#include <event_camera_codecs/event_processor.h>

#include <cstddef>
#include <vector>

#include "evo/event_types.hpp"

namespace evo
{

/**
 * Collects decoded change-detection events into a flat vector.
 *
 * Fed to event_camera_codecs::Decoder::decode(); on every eventCD callback it
 * appends an Event to out_. Trigger events are ignored (EVO does not use them).
 */
class EventCollector : public event_camera_codecs::EventProcessor
{
public:
  explicit EventCollector(std::vector<Event> * out)
  : out_(out) {}

  void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    out_->emplace_back(ex, ey, EventTime(static_cast<int64_t>(sensor_time)), polarity != 0);
  }

  void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}

  void finished() override {}

  void rawData(const char *, size_t) override {}

private:
  std::vector<Event> * out_;  ///< destination buffer (not owned)
};

}  // namespace evo
