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
#include <algorithm>
#include <random>

#include <Eigen/Core>

#include "moment_flow/flow/moment_flow.hpp"

using moment_flow::flow::Events;
using moment_flow::flow::MomentFlow;
using moment_flow::flow::MomentFlowParams;

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

Events make_incoherent_events(int w, int h, int n_events, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(4.0f, static_cast<float>(w) - 4.0f);
  std::uniform_real_distribution<float> uy(4.0f, static_cast<float>(h) - 4.0f);
  std::uniform_real_distribution<float> ut(-0.005f, 0.005f);

  Events ev;
  ev.t_ref_us = 1000000;
  for (int k = 0; k < n_events; ++k) {
    ev.x.push_back(ux(rng));
    ev.y.push_back(uy(rng));
    ev.t.push_back(ut(rng));
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
  p.cell_min_mass = 3.0f;
  p.cell_min_lambda = 1e-4f;
  p.aperture_ratio = 0.05f;
  p.tikhonov_eps = 1e-6f;
  p.flow_reg_lambda = 0.0f;
  p.flow_reg_sweeps = 0;
  p.flow_reg_sigma = 1e9f;
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

TEST(MomentFlow, AnisotropicRunsWithoutCellFits)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 420.0f;
  const float Fy = 0.0f;

  MomentFlowParams p = test_params(1);
  p.tile_min_mass = 0.0f;
  p.tile_min_cells = 1;

  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, Fy, 30, 7, 0.0f, W));

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 2);
  EXPECT_GT(flow.profile().active_cells, 0);
  EXPECT_NEAR(F[0], Fx, 80.0f);
  EXPECT_NEAR(F[1], Fy, 35.0f);
}

TEST(MomentFlow, FinalConfidenceUsesAnisotropicGeometry)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 480.0f;
  const float Fy = -160.0f;

  MomentFlowParams p = test_params(1);
  p.cell_max_residual_ratio = 1.0f;
  p.tile_min_mass = 0.0f;
  p.tile_min_cells = 1;
  p.tile_min_lambda = 0.0f;
  p.prior_lambda = 0.0f;
  p.tikhonov_eps = 1e-10f;

  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, Fy, 40, 61, 0.0f, W));

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 2);
  EXPECT_GT(flow.profile().final_full_rank_tiles + flow.profile().final_aperture_tiles, 0);

  std::vector<float> conf;
  flow.final_tile_confidence(conf);
  ASSERT_EQ(conf.size(), 1U);
  EXPECT_GT(conf[0], 1e-4f);
}

TEST(MomentFlow, TileTikhonovIsScaleAwareForFastMotion)
{
  const int W = 256;
  const int H = 256;
  const float Fx = 2600.0f;
  const float Fy = -700.0f;

  MomentFlowParams p = test_params(1);
  p.cell_size_px = 256;
  p.tile_min_mass = 0.0f;
  p.tile_min_cells = 1;
  p.tile_min_lambda = 0.0f;
  p.tikhonov_eps = 0.01f;
  p.max_speed_px_s = 5000.0f;

  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, Fy, 1000, 29, 0.0f, W));

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 2);
  EXPECT_NEAR(F[0], Fx, 250.0f);
  EXPECT_NEAR(F[1], Fy, 100.0f);
}

TEST(MomentFlow, ReversedEventOrderKeepsPlaneFit)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 500.0f;
  const float Fy = -120.0f;

  MomentFlowParams p = test_params(1);
  Events ev = make_plane_events(W, H, p.cell_size_px, Fx, Fy, 40, 31, 0.0f, W);
  std::reverse(ev.x.begin(), ev.x.end());
  std::reverse(ev.y.begin(), ev.y.end());
  std::reverse(ev.t.begin(), ev.t.end());

  MomentFlow flow(W, H, p);
  flow.ingest(ev);

  Eigen::VectorXf warm;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  ASSERT_EQ(F.size(), 2);
  EXPECT_NEAR(F[0], Fx, 60.0f);
  EXPECT_NEAR(F[1], Fy, 35.0f);
}

TEST(MomentFlow, RejectsIncoherentCellMoments)
{
  const int W = 64;
  const int H = 64;

  MomentFlowParams p = test_params(1);
  p.cell_size_px = 64;
  p.cell_min_mass = 20.0f;
  p.cell_min_lambda = 0.01f;
  p.cell_max_residual_ratio = 0.6f;
  p.tile_min_mass = 0.0f;
  p.tile_min_cells = 1;
  p.tile_min_lambda = 0.0f;

  MomentFlow flow(W, H, p);
  flow.ingest(make_incoherent_events(W, H, 200, 37));

  Eigen::VectorXf warm(flow.num_vars());
  warm << 12.0f, -8.0f;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  EXPECT_EQ(flow.profile().valid_cells, 0);
  EXPECT_GT(flow.profile().residual_reject_cells + flow.profile().speed_reject_cells, 0);
  ASSERT_EQ(F.size(), warm.size());
  EXPECT_FLOAT_EQ(F[0], warm[0]);
  EXPECT_FLOAT_EQ(F[1], warm[1]);
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

TEST(MomentFlow, AnisotropicUsesFallbackAsPrior)
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
  EXPECT_GT(F[1], 0.0f);
  EXPECT_LT(F[1], FallbackTangential);
  EXPECT_GT(flow.profile().full_rank_tiles + flow.profile().aperture_tiles, 0);
}

TEST(MomentFlow, EdgeCellsSurviveStageA)
{
  const int W = 128;
  const int H = 96;
  const float Fx = 360.0f;

  MomentFlowParams p = test_params(1);
  p.cell_min_lambda = 0.01f;

  MomentFlow flow(W, H, p);
  flow.ingest(make_plane_events(W, H, p.cell_size_px, Fx, 0.0f, 30, 41, 0.0f, W));

  Eigen::VectorXf warm(flow.num_vars());
  warm << 0.0f, 80.0f;
  Eigen::VectorXf F(flow.num_vars());
  flow.solve(warm, F);

  EXPECT_GT(flow.profile().valid_cells, 10);
  EXPECT_NEAR(F[0], Fx, 40.0f);
  EXPECT_GT(flow.profile().full_rank_tiles + flow.profile().aperture_tiles, 0);
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

TEST(MomentFlow, SpatialRegularizationNoopWhenDisabled)
{
  const int W = 128;
  const int H = 96;
  MomentFlowParams raw_params = test_params(3);
  MomentFlowParams disabled_params = raw_params;
  disabled_params.flow_reg_lambda = 10.0f;
  disabled_params.flow_reg_sweeps = 0;

  Events ev;
  append_events(
    ev,
    make_plane_events(W, H, raw_params.cell_size_px, 400.0f, 0.0f, 30, 21, 0.0f, W / 2.0f));
  append_events(
    ev,
    make_plane_events(W, H, raw_params.cell_size_px, -300.0f, 0.0f, 30, 23, W / 2.0f, W));

  MomentFlow raw_flow(W, H, raw_params);
  raw_flow.ingest(ev);
  Eigen::VectorXf warm;
  Eigen::VectorXf raw_F(raw_flow.num_vars());
  raw_flow.solve(warm, raw_F);

  MomentFlow disabled_flow(W, H, disabled_params);
  disabled_flow.ingest(ev);
  Eigen::VectorXf disabled_F(disabled_flow.num_vars());
  disabled_flow.solve(warm, disabled_F);

  ASSERT_EQ(disabled_F.size(), raw_F.size());
  for (int i = 0; i < raw_F.size(); ++i) {
    EXPECT_FLOAT_EQ(disabled_F[i], raw_F[i]);
  }
  EXPECT_EQ(disabled_flow.profile().reg_total_tiles, 0);
  EXPECT_EQ(disabled_flow.profile().reg_modified_tiles, 0);
}

TEST(MomentFlow, CoupledRegularizationReducesTileJumps)
{
  const int W = 128;
  const int H = 96;
  MomentFlowParams raw_params = test_params(3);
  MomentFlowParams reg_params = raw_params;
  reg_params.flow_reg_lambda = 5.0f;
  reg_params.flow_reg_sweeps = 4;
  reg_params.flow_reg_sigma = 1e9f;
  reg_params.prior_lambda = 1.0f;

  Events ev;
  append_events(
    ev,
    make_plane_events(W, H, raw_params.cell_size_px, 400.0f, 0.0f, 30, 21, 0.0f, W / 2.0f));
  append_events(
    ev,
    make_plane_events(W, H, raw_params.cell_size_px, -300.0f, 0.0f, 30, 23, W / 2.0f, W));

  MomentFlow raw_flow(W, H, raw_params);
  raw_flow.ingest(ev);
  Eigen::VectorXf warm;
  Eigen::VectorXf raw_F(raw_flow.num_vars());
  raw_flow.solve(warm, raw_F);

  MomentFlow reg_flow(W, H, reg_params);
  reg_flow.ingest(ev);
  Eigen::VectorXf reg_F(reg_flow.num_vars());
  reg_flow.solve(warm, reg_F);

  const int tiles = reg_flow.final_tiles();
  EXPECT_LT(neighbor_variation(reg_F, tiles), neighbor_variation(raw_F, tiles));
  EXPECT_GT(neighbor_variation(reg_F, tiles), 1000.0);
  EXPECT_GT(reg_flow.profile().reg_modified_tiles, 0);
  EXPECT_GT(reg_flow.profile().reg_mean_delta_speed, 0.0);
}
