/**
 * Micro-benchmark for the moment-based optical-flow estimator.
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

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include "event_detector_cpp/flow/moment_flow.hpp"

using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::MomentFlow;
using event_detector_cpp::flow::MomentFlowParams;

namespace
{

Events make_events(int w, int h, int cell_size, int events_per_cell)
{
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> ut(-0.004f, 0.004f);
  std::uniform_real_distribution<float> uq(-4.0f, 4.0f);
  std::uniform_real_distribution<float> uvx(-600.0f, 600.0f);
  std::uniform_real_distribution<float> uvy(-350.0f, 350.0f);

  Events ev;
  ev.t_ref_us = 1000000;
  for (int cy = 0; cy < h / cell_size; ++cy) {
    for (int cx = 0; cx < w / cell_size; ++cx) {
      const float center_x = (static_cast<float>(cx) + 0.5f) * cell_size;
      const float center_y = (static_cast<float>(cy) + 0.5f) * cell_size;
      if (center_x < 12.0f || center_y < 12.0f ||
          center_x > w - 12.0f || center_y > h - 12.0f)
      {
        continue;
      }
      const float fx = uvx(rng);
      const float fy = uvy(rng);
      const float n = std::max(1e-6f, std::hypot(fx, fy));
      const float tx = -fy / n;
      const float ty = fx / n;
      for (int k = 0; k < events_per_cell; ++k) {
        const float t = ut(rng);
        const float q = uq(rng);
        const float x = center_x + q * tx - t * fx;
        const float y = center_y + q * ty - t * fy;
        if (x >= 1.0f && y >= 1.0f && x < w - 1.0f && y < h - 1.0f) {
          ev.x.push_back(x);
          ev.y.push_back(y);
          ev.t.push_back(t);
        }
      }
    }
  }
  return ev;
}

double elapsed_us(
  const std::chrono::steady_clock::time_point & start,
  const std::chrono::steady_clock::time_point & end)
{
  return std::chrono::duration<double, std::micro>(end - start).count();
}

}  // namespace

int main()
{
  const int W = 1280;
  const int H = 720;
  const int cell = 16;
  const Events ev = make_events(W, H, cell, 8);

  std::cout << "Tile,Variables,Events,Ingest_us,Decay_us,StageA_us,StageB_us,"
               "Smooth_us,TotalSolve_us,Total_us,Hz,ActiveCells,ValidCells,FullRankTiles,"
               "ApertureTiles,FallbackTiles\n";

  for (int scales = 1; scales <= 5; ++scales) {
    MomentFlowParams p;
    p.num_scales = scales;
    p.cell_size_px = cell;
    p.decay_tau_us = 30000;
    p.cell_min_mass = 3.0f;
    p.cell_min_lambda = 1e-4f;
    p.aperture_ratio = 0.05f;
    p.tikhonov_eps = 1e-6f;
    p.smooth_iters = 2;
    p.smooth_alpha = 0.35f;
    p.max_speed_px_s = 5000.0f;

    MomentFlow flow(W, H, p);
    Eigen::VectorXf warm;
    Eigen::VectorXf F(flow.num_vars());

    const auto t0 = std::chrono::steady_clock::now();
    flow.ingest(ev);
    flow.solve(warm, F);
    const auto t1 = std::chrono::steady_clock::now();

    const auto & prof = flow.profile();
    const double total_us = elapsed_us(t0, t1);
    const double hz = (total_us > 0.0) ? 1e6 / total_us : 0.0;
    const int tiles = 1 << (scales - 1);
    std::cout << tiles << "x" << tiles << ","
              << flow.num_vars() << ","
              << ev.size() << ","
              << prof.ingest_ms * 1000.0 << ","
              << prof.decay_ms * 1000.0 << ","
              << prof.stage_a_ms * 1000.0 << ","
              << prof.stage_b_ms * 1000.0 << ","
              << prof.smooth_ms * 1000.0 << ","
              << prof.total_solve_ms * 1000.0 << ","
              << total_us << ","
              << hz << ","
              << prof.active_cells << ","
              << prof.valid_cells << ","
              << prof.full_rank_tiles << ","
              << prof.aperture_tiles << ","
              << prof.fallback_tiles << "\n";
  }

  return 0;
}
