/**
 * Tests for time-aware flow propagation (Sec. III-C).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "event_detector_cpp/flow/propagation.hpp"

using event_detector_cpp::flow::Grid;
using event_detector_cpp::flow::Scheme;
using event_detector_cpp::flow::propagate;
using event_detector_cpp::flow::propagate_vjp;

// A spatially-constant flow field has zero spatial gradient, so (v.grad)v = 0
// and the field is a steady-state solution: it must be unchanged at every
// output time, for both schemes.
TEST(Propagation, ConstantFieldIsSteadyState)
{
  const int w = 8, h = 6;
  Grid v0(w, h,
    std::vector<float>(static_cast<size_t>(w) * h, 2.0f),
    std::vector<float>(static_cast<size_t>(w) * h, -1.0f));
  const std::vector<float> t_out = {-0.01f, 0.0f, 0.01f};

  for (Scheme scheme : {Scheme::Upwind, Scheme::Burgers}) {
    auto stack = propagate(v0, t_out, scheme, /*ds=*/1.0f, /*cfl=*/0.5f);
    ASSERT_EQ(stack.size(), t_out.size());
    for (const auto & g : stack) {
      ASSERT_EQ(g.w, w);
      ASSERT_EQ(g.h, h);
      for (int i = 0; i < w * h; ++i) {
        EXPECT_NEAR(g.vx[i], 2.0f, 1e-4f);
        EXPECT_NEAR(g.vy[i], -1.0f, 1e-4f);
      }
    }
  }
}

// A field with uniform vx=1 and vy linear in the column index x is advected in
// the +x direction at unit speed: vy(x,t) = vy0(x - t). Linear data is advected
// exactly (no numerical diffusion) by both schemes, so interior nodes must match
// the analytic shift. Checks the propagator does correct, non-trivial work.
TEST(Propagation, LinearRampAdvectsExactly)
{
  // Gentle slope + short time keep the sub-step count low (~3); the Neumann
  // boundary error spreads one column per sub-step, so an 8-column interior
  // margin is comfortably clean. d(vy)/dt = -vx*slope = -0.02, so over t=1 the
  // analytic result is vy = 0.02*(c - 1).
  const int w = 32, h = 6;
  const float slope = 0.02f;
  const float t = 1.0f;
  for (Scheme scheme : {Scheme::Upwind, Scheme::Burgers}) {
    Grid v0(w, h);
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        const size_t i = static_cast<size_t>(r) * w + c;
        v0.vx[i] = 1.0f;
        v0.vy[i] = slope * static_cast<float>(c);
      }
    }
    auto stack = propagate(v0, {t}, scheme, /*ds=*/1.0f, /*cfl=*/0.5f);
    ASSERT_EQ(stack.size(), 1u);
    for (int r = 0; r < h; ++r) {
      for (int c = 8; c < w - 8; ++c) {
        const size_t i = static_cast<size_t>(r) * w + c;
        EXPECT_NEAR(stack[0].vx[i], 1.0f, 1e-4f);
        EXPECT_NEAR(stack[0].vy[i], slope * (static_cast<float>(c) - t), 1e-4f);
      }
    }
  }
}

// The boundary time t = 0 must return v0 exactly (identity).
TEST(Propagation, BoundaryTimeIsIdentity)
{
  const int w = 5, h = 4;
  Grid v0(w, h);
  for (int i = 0; i < w * h; ++i) {
    v0.vx[i] = 0.1f * static_cast<float>(i);
    v0.vy[i] = -0.2f * static_cast<float>(i);
  }
  auto stack = propagate(v0, {0.0f}, Scheme::Upwind, 1.0f, 0.5f);
  ASSERT_EQ(stack.size(), 1u);
  for (int i = 0; i < w * h; ++i) {
    EXPECT_FLOAT_EQ(stack[0].vx[i], v0.vx[i]);
    EXPECT_FLOAT_EQ(stack[0].vy[i], v0.vy[i]);
  }
}

namespace
{
// <a, b> over both components of a stack of grids.
double dot(const std::vector<Grid> & a, const std::vector<Grid> & b)
{
  double s = 0.0;
  for (size_t n = 0; n < a.size(); ++n) {
    for (size_t i = 0; i < a[n].size(); ++i) {
      s += static_cast<double>(a[n].vx[i]) * b[n].vx[i];
      s += static_cast<double>(a[n].vy[i]) * b[n].vy[i];
    }
  }
  return s;
}
double dot(const Grid & a, const Grid & b)
{
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    s += static_cast<double>(a.vx[i]) * b.vx[i];
    s += static_cast<double>(a.vy[i]) * b.vy[i];
  }
  return s;
}
}  // namespace

// Adjoint consistency: for any cotangent g on the outputs and any direction d on
// the input, <g, J*d> (forward directional derivative, via finite difference)
// must equal <vjp(g), d>. Velocities are kept away from zero so the upwind
// directions don't flip under the small perturbation.
TEST(Propagation, UpwindAdjointMatchesFiniteDifference)
{
  const int w = 7, h = 5;
  const std::vector<float> t_out = {-0.02f, 0.01f};
  const float ds = 1.0f, cfl = 0.5f;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> vdist(0.5f, 1.5f);   // away from 0
  std::uniform_real_distribution<float> gdist(-1.0f, 1.0f);

  Grid v0(w, h), d(w, h);
  std::vector<Grid> g;
  for (int i = 0; i < w * h; ++i) {
    v0.vx[i] = vdist(rng); v0.vy[i] = vdist(rng);
    d.vx[i] = gdist(rng);  d.vy[i] = gdist(rng);
  }
  for (size_t n = 0; n < t_out.size(); ++n) {
    Grid gn(w, h);
    for (int i = 0; i < w * h; ++i) { gn.vx[i] = gdist(rng); gn.vy[i] = gdist(rng); }
    g.push_back(std::move(gn));
  }

  Grid a = propagate_vjp(v0, t_out, Scheme::Upwind, ds, cfl, g);
  const double rhs = dot(a, d);

  const float eps = 1e-3f;
  Grid v_plus = v0, v_minus = v0;
  for (int i = 0; i < w * h; ++i) {
    v_plus.vx[i] += eps * d.vx[i]; v_plus.vy[i] += eps * d.vy[i];
    v_minus.vx[i] -= eps * d.vx[i]; v_minus.vy[i] -= eps * d.vy[i];
  }
  auto fp = propagate(v_plus, t_out, Scheme::Upwind, ds, cfl);
  auto fm = propagate(v_minus, t_out, Scheme::Upwind, ds, cfl);
  std::vector<Grid> deriv;
  for (size_t n = 0; n < t_out.size(); ++n) {
    Grid dn(w, h);
    for (int i = 0; i < w * h; ++i) {
      dn.vx[i] = (fp[n].vx[i] - fm[n].vx[i]) / (2 * eps);
      dn.vy[i] = (fp[n].vy[i] - fm[n].vy[i]) / (2 * eps);
    }
    deriv.push_back(std::move(dn));
  }
  const double lhs = dot(g, deriv);

  EXPECT_NEAR(lhs, rhs, 1e-2 * (1.0 + std::abs(rhs)));
}
