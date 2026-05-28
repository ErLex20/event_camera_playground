/**
 * EROS (Event-based Reactive Optimized Surface) struct and event processor.
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
#include <cmath>
#include <cstdint>
#include <vector>

namespace event_detector_cpp
{

/**
 * EROS — Event-based Reactive Optimized Surface (Glover et al., IIT, 2022).
 * Reference: arXiv:2205.07657 "PUCK: Parallel Surface and Convolution-kernel Tracking".
 *
 * Unlike the SAE (which records per-pixel timestamps), EROS stores a scalar
 * intensity value in [0.0, 1.0] per pixel. On each incoming event at (vx, vy):
 *
 *   1. Every pixel in the square neighborhood [vx±k, vy±k] is multiplied by
 *      the decay factor d = 0.3^(1/k_eros).
 *   2. The event pixel itself is set to 1.0.
 *
 * The surface is never cleared between packets; it is a persistent memory that
 * accumulates across the full event stream.
 *
 * ARM optimisation notes:
 *  - float32 enables NEON to process 4 elements/instruction vs 2 for double.
 *  - The inner decay loop is a scalar-broadcast multiply:
 *    `row[col] *= decay_factor` → auto-vectorises to vmulq_f32 on aarch64.
 *  - The kernel (default 21×21 = 441 pixels) fits entirely in L1 cache,
 *    keeping memory bandwidth out of the critical path.
 *  - Dual polarity planes avoid interleaving and allow independent SIMD sweeps.
 */
struct SurfaceEros
{
  uint32_t width{0};
  uint32_t height{0};
  uint32_t k_eros{10};        // neighborhood half-radius [pixels]
  float    decay_factor{0.0f};  // d = 0.3^(1/k_eros), cached at resize()

  /** Last ON-event (polarity=1) intensity per pixel [0.0, 1.0], flat row-major. */
  std::vector<float> on;

  /** Last OFF-event (polarity=0) intensity per pixel [0.0, 1.0], flat row-major. */
  std::vector<float> off;

  void resize(uint32_t w, uint32_t h, uint32_t k = 10)
  {
    width = w;
    height = h;
    k_eros = k;
    decay_factor = std::pow(0.3f, 1.0f / static_cast<float>(k));
    on.assign(static_cast<std::size_t>(w) * h, 0.0f);
    off.assign(static_cast<std::size_t>(w) * h, 0.0f);
  }

  void reset()
  {
    std::fill(on.begin(), on.end(), 0.0f);
    std::fill(off.begin(), off.end(), 0.0f);
  }

  inline void update(uint16_t x, uint16_t y, uint8_t polarity) noexcept
  {
    auto & surface = polarity ? on : off;

    // Clamp neighborhood to sensor bounds.
    const int32_t x0 = std::max(0, static_cast<int32_t>(x) - static_cast<int32_t>(k_eros));
    const int32_t x1 = std::min(static_cast<int32_t>(width) - 1,
      static_cast<int32_t>(x) + static_cast<int32_t>(k_eros));
    const int32_t y0 = std::max(0, static_cast<int32_t>(y) - static_cast<int32_t>(k_eros));
    const int32_t y1 = std::min(static_cast<int32_t>(height) - 1,
      static_cast<int32_t>(y) + static_cast<int32_t>(k_eros));

    // Decay neighborhood — inner loop auto-vectorises to vmulq_f32 on aarch64.
    for (int32_t row = y0; row <= y1; ++row) {
      float * row_ptr = surface.data() + static_cast<std::size_t>(row) * width;
      for (int32_t col = x0; col <= x1; ++col) {
        row_ptr[col] *= decay_factor;
      }
    }

    // Event location snaps to 1.0.
    surface[static_cast<std::size_t>(y) * width + x] = 1.0f;
  }
};

/**
 * EventProcessor adapter that feeds decoded events into a SurfaceEros.
 * Timestamps are not used — EROS is purely event-count driven.
 */
class EROSProcessor : public event_camera_codecs::EventProcessor
{
public:
  explicit EROSProcessor(SurfaceEros & eros)
  : eros_(eros) {}

  void eventCD(uint64_t /*sensor_time*/, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    eros_.update(ex, ey, polarity);
  }

  void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
  void finished() override {}
  void rawData(const char *, size_t) override {}

private:
  SurfaceEros & eros_;
};

}  // namespace event_detector_cpp
