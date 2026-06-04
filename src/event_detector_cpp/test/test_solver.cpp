/**
 * Convergence tests for the Newton-CG contrast-maximization solver: given a
 * window of events produced by a known motion, the optimizer must recover a
 * non-trivial flow close to the ground truth (not collapse to zero).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "event_detector_cpp/flow/newton_cg.hpp"
#include "event_detector_cpp/flow/objective.hpp"

using event_detector_cpp::flow::ContrastNorm;
using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::NewtonCgParams;
using event_detector_cpp::flow::Objective;
using event_detector_cpp::flow::ObjectiveParams;
using event_detector_cpp::flow::newton_cg_minimize;

namespace
{
// Events from a set of point features translating rigidly at velocity (Vx, Vy).
// A feature at reference position p (at t_mid) emits events along its trajectory
// at x = p - t*V, so warping with v = V (x' = x + t*v) collapses them onto p.
Events translating_events(int n_feat, int n_per, int w, int h,
  float t_lo, float t_hi, float vx, float vy, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> up_x(8.0f, w - 8.0f);
  std::uniform_real_distribution<float> up_y(8.0f, h - 8.0f);
  std::uniform_real_distribution<float> ut(t_lo, t_hi);
  Events e;
  for (int i = 0; i < n_feat; ++i) {
    const float px = up_x(rng), py = up_y(rng);
    for (int j = 0; j < n_per; ++j) {
      const float t = ut(rng);
      e.x.push_back(px - t * vx);
      e.y.push_back(py - t * vy);
      e.t.push_back(t);
    }
  }
  return e;
}
}  // namespace

TEST(Solver, RecoversGlobalTranslation)
{
  const int W = 64, H = 64;
  const float t_lo = -0.01f, t_hi = 0.01f;
  const float Vx = 120.0f, Vy = -60.0f;  // px/s (~1.2 px over the window)

  ObjectiveParams p;
  p.img_w = W; p.img_h = H;
  p.tiles_x = 1; p.tiles_y = 1;
  p.t_lo = t_lo; p.t_hi = t_hi;
  p.norm = ContrastNorm::L1;
  Objective obj(translating_events(200, 30, W, H, t_lo, t_hi, Vx, Vy, 7), p);

  NewtonCgParams ncg;
  ncg.newton_max_iter = 50;
  ncg.cg_max_iter = 10;
  ncg.cg_tol = 0.1f;
  ncg.fd_step = 0.5f / std::max(std::abs(t_lo), std::abs(t_hi));  // ~0.5 px

  // Paper's strategy: start from zero and let Newton-CG converge.
  Eigen::VectorXf F = Eigen::VectorXf::Zero(obj.num_vars());
  newton_cg_minimize(obj, F, ncg);

  EXPECT_NEAR(F[0], Vx, 40.0f) << "recovered vx = " << F[0];
  EXPECT_NEAR(F[1], Vy, 40.0f) << "recovered vy = " << F[1];
}

// Same recovery, but through the time-aware objective (Sec. III-C). A spatially
// constant flow is a steady state of the self-advection PDE, so the propagated
// field equals the boundary field and the global translation must still be
// recovered — exercising the full boundary->transport->sample->warp chain end
// to end in the optimizer.
TEST(Solver, RecoversGlobalTranslationTimeAware)
{
  const int W = 64, H = 64;
  const float t_lo = -0.01f, t_hi = 0.01f;
  const float Vx = 110.0f, Vy = 70.0f;  // px/s

  ObjectiveParams p;
  p.img_w = W; p.img_h = H;
  p.tiles_x = 1; p.tiles_y = 1;
  p.t_lo = t_lo; p.t_hi = t_hi;
  p.norm = ContrastNorm::L1;
  p.time_aware = true;
  p.scheme = event_detector_cpp::flow::Scheme::Upwind;
  p.time_bins = 5;
  p.prop_w = 8; p.prop_h = 8;
  p.cfl = 0.5f;
  Objective obj(translating_events(200, 30, W, H, t_lo, t_hi, Vx, Vy, 19), p);

  NewtonCgParams ncg;
  ncg.newton_max_iter = 50;
  ncg.cg_max_iter = 10;
  ncg.cg_tol = 0.1f;
  ncg.fd_step = 0.5f / std::max(std::abs(t_lo), std::abs(t_hi));

  Eigen::VectorXf F = Eigen::VectorXf::Zero(obj.num_vars());
  newton_cg_minimize(obj, F, ncg);

  EXPECT_NEAR(F[0], Vx, 40.0f) << "recovered vx = " << F[0];
  EXPECT_NEAR(F[1], Vy, 40.0f) << "recovered vy = " << F[1];
}
