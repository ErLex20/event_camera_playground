/**
 * Tests for the contrast-maximization flow objective (Sec. III-B).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "event_detector_cpp/flow/objective.hpp"

using event_detector_cpp::flow::ContrastNorm;
using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::Objective;
using event_detector_cpp::flow::ObjectiveParams;
using event_detector_cpp::flow::Scheme;

namespace
{
// A reproducible random window of events over the image.
Events make_events(int n, int w, int h, float t_lo, float t_hi, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(1.0f, w - 2.0f);
  std::uniform_real_distribution<float> uy(1.0f, h - 2.0f);
  std::uniform_real_distribution<float> ut(t_lo, t_hi);
  Events e;
  for (int i = 0; i < n; ++i) {
    e.x.push_back(ux(rng));
    e.y.push_back(uy(rng));
    e.t.push_back(ut(rng));
  }
  return e;
}
}  // namespace

// Eq. (5) is normalized by the zero-flow focus G0, so the identity warp (F = 0)
// must give exactly f = 1 (the FWL baseline).
TEST(Objective, ZeroFlowFocusIsOne)
{
  ObjectiveParams p;
  p.img_w = 24; p.img_h = 18;
  p.tiles_x = 2; p.tiles_y = 2;
  p.t_lo = -0.005f; p.t_hi = 0.005f;
  p.norm = ContrastNorm::L1;
  Objective obj(make_events(400, p.img_w, p.img_h, p.t_lo, p.t_hi, 1), p);

  Eigen::VectorXf F = Eigen::VectorXf::Zero(obj.num_vars());
  EXPECT_NEAR(obj.focus(F), 1.0f, 1e-5f);
}

// The analytic gradient must match a central finite difference of value(). Uses
// the smooth L2 contrast so there are no |.| kinks to confuse the check.
TEST(Objective, GradientMatchesFiniteDifferenceL2)
{
  ObjectiveParams p;
  p.img_w = 28; p.img_h = 22;
  p.tiles_x = 3; p.tiles_y = 3;
  p.t_lo = -0.006f; p.t_hi = 0.006f;
  p.norm = ContrastNorm::L2;
  p.tv_weight = 0.02f;
  p.tv_eps = 1e-2f;
  Objective obj(make_events(600, p.img_w, p.img_h, p.t_lo, p.t_hi, 7), p);

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> uf(-150.0f, 150.0f);  // px/s
  Eigen::VectorXf F(obj.num_vars());
  for (int i = 0; i < F.size(); ++i) F[i] = uf(rng);

  Eigen::VectorXf grad;
  obj.value_and_grad(F, grad);
  ASSERT_EQ(grad.size(), F.size());

  const float eps = 2.0f;
  for (int i = 0; i < F.size(); ++i) {
    Eigen::VectorXf fp = F, fm = F;
    fp[i] += eps; fm[i] -= eps;
    const double fd = (static_cast<double>(obj.value(fp)) - obj.value(fm)) / (2.0 * eps);
    EXPECT_NEAR(grad[i], fd, 1e-2 * (1.0 + std::abs(fd)))
      << "gradient mismatch at index " << i;
  }
}

// L1 path (the paper default) must also be gradient-consistent. Looser tol
// because |.| introduces kinks where IWE gradients cross zero.
TEST(Objective, GradientMatchesFiniteDifferenceL1)
{
  ObjectiveParams p;
  p.img_w = 28; p.img_h = 22;
  p.tiles_x = 2; p.tiles_y = 2;
  p.t_lo = -0.006f; p.t_hi = 0.006f;
  p.norm = ContrastNorm::L1;
  p.tv_weight = 0.0f;
  Objective obj(make_events(800, p.img_w, p.img_h, p.t_lo, p.t_hi, 3), p);

  std::mt19937 rng(5);
  std::uniform_real_distribution<float> uf(-120.0f, 120.0f);
  Eigen::VectorXf F(obj.num_vars());
  for (int i = 0; i < F.size(); ++i) F[i] = uf(rng);

  Eigen::VectorXf grad;
  obj.value_and_grad(F, grad);

  const float eps = 4.0f;
  int close = 0;
  for (int i = 0; i < F.size(); ++i) {
    Eigen::VectorXf fp = F, fm = F;
    fp[i] += eps; fm[i] -= eps;
    const double fd = (static_cast<double>(obj.value(fp)) - obj.value(fm)) / (2.0 * eps);
    if (std::abs(grad[i] - fd) <= 5e-2 * (1.0 + std::abs(fd))) ++close;
  }
  EXPECT_GE(close, static_cast<int>(0.75 * F.size()));  // most components agree
}

// Time-aware warp (Sec. III-C): even after transporting the boundary field
// through the self-advection PDE, the identity warp (F = 0) keeps the IWE
// unwarped, so the normalized focus is still exactly 1.
TEST(ObjectiveTimeAware, ZeroFlowFocusIsOne)
{
  ObjectiveParams p;
  p.img_w = 24; p.img_h = 24;
  p.tiles_x = 2; p.tiles_y = 2;
  p.t_lo = -0.01f; p.t_hi = 0.01f;
  p.norm = ContrastNorm::L1;
  p.time_aware = true;
  p.scheme = Scheme::Upwind;
  p.time_bins = 4;
  p.prop_w = 8; p.prop_h = 8;
  p.cfl = 0.5f;
  Objective obj(make_events(500, p.img_w, p.img_h, p.t_lo, p.t_hi, 2), p);

  Eigen::VectorXf F = Eigen::VectorXf::Zero(obj.num_vars());
  EXPECT_NEAR(obj.focus(F), 1.0f, 1e-5f);
}

// The analytic gradient of the time-aware objective (whose chain runs
// F -> boundary field -> PDE transport -> per-event flow sample -> warp ->
// contrast) must match a central finite difference. The upwind adjoint is the
// exact transpose given the frozen forward stencil directions; we use the
// "most components agree" criterion to tolerate the upwind sign kinks and the
// occasional event that warps across the image border under perturbation.
TEST(ObjectiveTimeAware, GradientMatchesFiniteDifferenceUpwind)
{
  ObjectiveParams p;
  p.img_w = 24; p.img_h = 24;
  p.tiles_x = 2; p.tiles_y = 2;
  p.t_lo = -0.02f; p.t_hi = 0.02f;
  p.norm = ContrastNorm::L2;   // smooth contrast, no |.| kinks
  p.tv_weight = 0.0f;
  p.time_aware = true;
  p.scheme = Scheme::Upwind;
  p.time_bins = 4;
  p.prop_w = 8; p.prop_h = 8;
  p.cfl = 0.5f;
  Objective obj(make_events(700, p.img_w, p.img_h, p.t_lo, p.t_hi, 11), p);

  std::mt19937 rng(23);
  std::uniform_real_distribution<float> uf(-100.0f, 100.0f);  // px/s
  Eigen::VectorXf F(obj.num_vars());
  for (int i = 0; i < F.size(); ++i) F[i] = uf(rng);

  Eigen::VectorXf grad;
  obj.value_and_grad(F, grad);
  ASSERT_EQ(grad.size(), F.size());

  const float eps = 2.0f;
  int close = 0;
  for (int i = 0; i < F.size(); ++i) {
    Eigen::VectorXf fp = F, fm = F;
    fp[i] += eps; fm[i] -= eps;
    const double fd = (static_cast<double>(obj.value(fp)) - obj.value(fm)) / (2.0 * eps);
    if (std::abs(grad[i] - fd) <= 5e-2 * (1.0 + std::abs(fd))) ++close;
  }
  EXPECT_GE(close, static_cast<int>(0.75 * F.size()));
}
