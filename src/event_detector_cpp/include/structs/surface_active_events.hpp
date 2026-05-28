/**
 * Surface of Active Events (SAE) struct and event processor.
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

#include <algorithm>
#include <cstdint>
#include <vector>

namespace event_detector_cpp
{

/**
 * Dual-polarity Surface of Active Events (Benosman et al., IEEE T-NNLS 2014).
 *
 * Stores the most recent event timestamp for each pixel, separately for ON
 * (polarity=1) and OFF (polarity=0) events.
 *
 * ARM optimisation notes:
 *  - float32 allows NEON to process 4 elements/instruction vs 2 for double.
 *  - Flat row-major layout ensures sequential memory access and cache
 *    friendliness during the normalization sweep.
 *  - The normalization loop (in the subscriber callback) annotated with
 *    #pragma omp parallel for auto-vectorises to NEON on aarch64 targets.
 *  - Both planes are kept as separate contiguous arrays to avoid interleaving
 *    and allow independent SIMD sweeps without gather/scatter penalties.
 */
struct SurfaceActiveEvents
{
  uint32_t width{0};
  uint32_t height{0};

  /**
   * Most recent timestamp seen across all events in the current packet [ns].
   * Updated inline during decoding; used as the reference for normalization.
   */
  float t_max{0.0f};

  /** Last ON-event (polarity=1) timestamp per pixel [ns], flat row-major. */
  std::vector<float> on;

  /** Last OFF-event (polarity=0) timestamp per pixel [ns], flat row-major. */
  std::vector<float> off;

  void resize(uint32_t w, uint32_t h)
  {
    width = w;
    height = h;
    on.assign(static_cast<std::size_t>(w) * h, 0.0f);
    off.assign(static_cast<std::size_t>(w) * h, 0.0f);
  }

  void reset()
  {
    std::fill(on.begin(), on.end(), 0.0f);
    std::fill(off.begin(), off.end(), 0.0f);
    t_max = 0.0f;
  }

  inline void update(uint16_t x, uint16_t y, uint8_t polarity, float ts_ns) noexcept
  {
    const uint32_t idx = static_cast<uint32_t>(y) * width + x;
    (polarity ? on : off)[idx] = ts_ns;
    if (ts_ns > t_max) {
      t_max = ts_ns;
    }
  }
};

/**
 * EventProcessor adapter that feeds decoded events into a SurfaceActiveEvents.
 *
 * Templated-processor style: instantiated as DecoderFactory<EventPacket, SAEProcessor>
 * so the hot eventCD() path is inlined by the decoder with no vtable indirection.
 */
class SAEProcessor : public event_camera_codecs::EventProcessor
{
public:
  explicit SAEProcessor(SurfaceActiveEvents & sae)
  : sae_(sae) {}

  void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    sae_.update(ex, ey, polarity, static_cast<float>(sensor_time));
  }

  void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
  void finished() override {}
  void rawData(const char *, size_t) override {}

private:
  SurfaceActiveEvents & sae_;
};

}  // namespace event_detector_cpp
