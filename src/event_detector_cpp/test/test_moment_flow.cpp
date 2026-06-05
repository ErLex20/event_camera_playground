/**
 * Tests for the moment-based incremental optical-flow estimator.
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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>

#include <Eigen/Core>

#include "event_detector_cpp/flow/moment_flow.hpp"

using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::MomentFlow;
using event_detector_cpp::flow::MomentFlowParams;

namespace
{

Events make_plane_events(
  int w, int h, int cell_size, float fx_warp, float fy_warp,
  int events_per_cell, unsigned seed, float x_lo, float x_hi)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ut(-0.004f, 0.004f);
  std::uniform_real_distribution<float> uq(-4.0f, 4.0f);

  const float f_norm = std::hypot(fx_warp, fy_warp);
  const float tx = (f_norm > 0.0f) ? -fy_warp / f_norm : 0.0f;
  const float ty = (f_norm > 0.0f) ? fx_warp / f_norm : 1.0f;

  Events ev;
  ev.t_ref_us = 1000000;
  for (int cy = 0; cy < h / cell_size; ++cy) {
    for (int cx = 0; cx < w / cell_size; ++cx) {
      const float center_x = (static_cast<float>(cx) + 0.5f) * cell_size;
      const float center_y = (static_cast<float>(cy) + 0.5f) * cell_size;
      if (center_x < x_lo || center_x >= x_hi ||
          center_x < 12.0f || center_y < 12.0f ||
          center_x > w - 12.0f || center_y > h - 12.0f)
      {
        continue;
      }
      for (int k = 0; k < events_per_cell; ++k) {
        const float t = ut(rng);
        const float q = uq(rng);
        const float x = center_x + q * tx - t * fx_warp;
        const float y = center_y + q * ty - t * fy_warp;
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

void append_events(Events & dst, const Events & src)
{
  if (dst.t_ref_us == 0) {
    dst.t_ref_us = src.t_ref_us;
  }
  dst.x.insert(dst.x.end(), src.x.begin(), src.x.end());
  dst.y.insert(dst.y.end(), src.y.begin(), src.y.end());
  dst.t.insert(dst.t.end(), src.t.begin(), src.t.end());
}

MomentFlowParams test_params(int scales)
{
  MomentFlowParams p;
  p.num_scales = scales;
  p.cell_size_px = 16;
  p.decay_tau_us = 30000;
  p.cell_min_mass = 3.0f;
  p.cell_min_lambda = 1e-4f;
  p.aperture_ratio = 0.05f;
  p.tikhonov_eps = 1e-6f;
  p.smooth_iters = 0;
  p.smooth_alpha = 0.0f;
  p.max_speed_px_s = 5000.0f;
  return p;
}

double neighbor_variation(const Eigen::VectorXf & F, int tiles)
{
  double acc = 0.0;
  int count = 0;
  for (int y = 0; y < tiles; ++y) {
    for (int x = 0; x < tiles; ++x) {
      const int k = y * tiles + x;
      auto add = [&](int nx, int ny) {
        if (nx >= tiles || ny >= tiles) {
          return;
        }
        const int nk = ny * tiles + nx;
        const double dx = F[2 * k] - F[2 * nk];
        const double dy = F[2 * k + 1] - F[2 * nk + 1];
        acc += dx * dx + dy * dy;
        count += 1;
      };
      add(x + 1, y);
      add(x, y + 1);
    }
  }
  return (count > 0) ? acc / static_cast<double>(count) : 0.0;
}

}  // namespace

TEST(MomentFlow, RecoversWarpSignConvention)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 420.0f;
  const float Fy = -210.0f;

  MomentFlowParams p = test_params(1);
  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, Fy, 30, 7, 0.0f, W));

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 2);
  EXPECT_NEAR(F[0], Fx, 35.0f);
  EXPECT_NEAR(F[1], Fy, 35.0f);
  EXPECT_GT(flow.profile().valid_cells, 0);
}

TEST(MomentFlow, EmptyWindowKeepsWarmStart)
{
  const int W = 128;
  const int H = 96;
  MomentFlowParams p = test_params(2);
  MomentFlow flow(W, H, p);

  Eigen::VectorXf warm(flow.num_vars());
  warm << 10.0f, -3.0f, 20.0f, -6.0f, 30.0f, -9.0f, 40.0f, -12.0f;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), warm.size());
  for (int i = 0; i < F.size(); ++i) {
    EXPECT_FLOAT_EQ(F[i], warm[i]);
  }
  EXPECT_EQ(flow.profile().valid_cells, 0);
}

TEST(MomentFlow, ApertureTilesKeepFallbackTangentialComponent)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 400.0f;
  const float FallbackTangential = 123.0f;

  MomentFlowParams p = test_params(1);
  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, 0.0f, 30, 11, 0.0f, W));

  Eigen::VectorXf warm(flow.num_vars());
  warm << 0.0f, FallbackTangential;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  EXPECT_NEAR(F[0], Fx, 35.0f);
  EXPECT_NEAR(F[1], FallbackTangential, 1e-3f);
  EXPECT_GT(flow.profile().aperture_tiles, 0);
}

TEST(MomentFlow, ProducesNonUniformTileField)
{
  const int W = 128;
  const int H = 96;
  MomentFlowParams p = test_params(2);
  MomentFlow flow(W, H, p);

  Events ev;
  append_events(ev, make_plane_events(W, H, p.cell_size_px, 350.0f, 0.0f, 30, 13, 0.0f, W / 2.0f));
  append_events(ev, make_plane_events(W, H, p.cell_size_px, -250.0f, 0.0f, 30, 17, W / 2.0f, W));
  flow.ingest(ev);

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 8);
  double mean = 0.0;
  for (int k = 0; k < F.size() / 2; ++k) {
    mean += F[2 * k];
  }
  mean /= static_cast<double>(F.size() / 2);
  double var = 0.0;
  for (int k = 0; k < F.size() / 2; ++k) {
    const double d = F[2 * k] - mean;
    var += d * d;
  }
  var /= static_cast<double>(F.size() / 2);

  EXPECT_GT(var, 10000.0);
  EXPECT_GT(F[0], 100.0f);
  EXPECT_LT(F[2], -100.0f);
}

TEST(MomentFlow, SmoothingReducesTileJumps)
{
  const int W = 128;
  const int H = 96;
  MomentFlowParams raw_params = test_params(3);
  MomentFlowParams smooth_params = raw_params;
  smooth_params.smooth_iters = 2;
  smooth_params.smooth_alpha = 0.35f;

  Events ev;
  append_events(ev, make_plane_events(W, H, raw_params.cell_size_px, 400.0f, 0.0f, 30, 21, 0.0f, W / 2.0f));
  append_events(ev, make_plane_events(W, H, raw_params.cell_size_px, -300.0f, 0.0f, 30, 23, W / 2.0f, W));

  MomentFlow raw_flow(W, H, raw_params);
  raw_flow.ingest(ev);
  Eigen::VectorXf warm;
  Eigen::VectorXf raw_F(raw_flow.num_vars());
  raw_flow.solve(warm, raw_F);

  MomentFlow smooth_flow(W, H, smooth_params);
  smooth_flow.ingest(ev);
  Eigen::VectorXf smooth_F(smooth_flow.num_vars());
  smooth_flow.solve(warm, smooth_F);

  const int tiles = smooth_flow.final_tiles();
  EXPECT_LT(neighbor_variation(smooth_F, tiles), neighbor_variation(raw_F, tiles));
  EXPECT_GT(neighbor_variation(smooth_F, tiles), 1000.0);
}
