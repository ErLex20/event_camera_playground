/**
 * Combined SAE + EROS event processor with optional BA filter.
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

#pragma once

#include <event_camera_codecs/event_processor.h>

#include <structs/background_activity_filter.hpp>
#include <structs/surface_active_events.hpp>
#include <structs/surface_eros.hpp>

namespace event_detector_cpp
{

/**
 * Feeds each decoded event through a single BA filter check, then updates
 * both the SAE and EROS surfaces in one pass.
 *
 * Using a single decoder pass (rather than separate passes for SAE and EROS)
 * ensures the BA filter's per-pixel timestamp table is updated exactly once
 * per event: two independent passes would cause the second surface to see a
 * pre-warmed filter state and filter differently from the first.
 *
 * Passing ba = nullptr disables filtering entirely.
 */
class CombinedProcessor : public event_camera_codecs::EventProcessor
{
public:
  CombinedProcessor(
    SurfaceActiveEvents & sae,
    SurfaceEros & eros,
    BackgroundActivityFilter * ba = nullptr)
  : sae_(sae), eros_(eros), ba_(ba) {}

  void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    const float ts = static_cast<float>(sensor_time);
    if (ba_ && !ba_->check_and_update(ex, ey, ts)) {
      return;
    }
    sae_.update(ex, ey, polarity, ts);
    eros_.update(ex, ey, polarity);
  }

  void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
  void finished() override {}
  void rawData(const char *, size_t) override {}

private:
  SurfaceActiveEvents & sae_;
  SurfaceEros & eros_;
  BackgroundActivityFilter * ba_;
};

}  // namespace event_detector_cpp
