/**
 * Time-aware flow propagation implementation (Sec. III-C).
 *
 * Integrates the self-advection PDE  v_t + (v.grad) v = 0  forward and backward
 * in time from the boundary field at t_mid, with explicit sub-steps bounded by a
 * CFL condition. Two discretizations are offered:
 *   - Upwind: non-conservative upwind differences of (v.grad)v.
 *   - Burgers: conservative (divergence-form) Lax-Friedrichs flux for the
 *     Burgers-like self-transport, which handles shocks more gracefully.
 * Both reduce to "no change" for a spatially-constant field (steady state).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#include "event_detector_cpp/flow/propagation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace event_detector_cpp::flow
{

namespace
{

// Neumann (replicate) edge sampling.
inline float at(const std::vector<float> & f, int w, int h, int c, int r)
{
  c = std::clamp(c, 0, w - 1);
  r = std::clamp(r, 0, h - 1);
  return f[static_cast<size_t>(r) * w + c];
}

// One explicit Euler sub-step of size dt (signed) producing `out` from `in`.
void step(const Grid & in, Grid & out, Scheme scheme, float ds, float dt)
{
  const int w = in.w;
  const int h = in.h;
  const float inv_ds = 1.0f / ds;

  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const size_t idx = static_cast<size_t>(r) * w + c;
      const float vx = in.vx[idx];
      const float vy = in.vy[idx];

      float rate_x;  // d/dt of vx
      float rate_y;  // d/dt of vy

      if (scheme == Scheme::Upwind) {
        // Upwind first differences chosen by the sign of the local velocity.
        auto ddx = [&](const std::vector<float> & f) {
          return (vx > 0.0f)
            ? (at(f, w, h, c, r) - at(f, w, h, c - 1, r)) * inv_ds
            : (at(f, w, h, c + 1, r) - at(f, w, h, c, r)) * inv_ds;
        };
        auto ddy = [&](const std::vector<float> & f) {
          return (vy > 0.0f)
            ? (at(f, w, h, c, r) - at(f, w, h, c, r - 1)) * inv_ds
            : (at(f, w, h, c, r + 1) - at(f, w, h, c, r)) * inv_ds;
        };
        rate_x = -(vx * ddx(in.vx) + vy * ddy(in.vx));
        rate_y = -(vx * ddx(in.vy) + vy * ddy(in.vy));
      } else {
        // Conservative Lax-Friedrichs flux for div.(v * v^a).
        auto fluxX = [&](const std::vector<float> & va, int cc, int rr) {  // face cc | cc+1
          const float aL = at(in.vx, w, h, cc, rr);
          const float aR = at(in.vx, w, h, cc + 1, rr);
          const float alpha = std::max(std::fabs(aL), std::fabs(aR));
          const float gL = aL * at(va, w, h, cc, rr);
          const float gR = aR * at(va, w, h, cc + 1, rr);
          return 0.5f * (gL + gR) - 0.5f * alpha * (at(va, w, h, cc + 1, rr) - at(va, w, h, cc, rr));
        };
        auto fluxY = [&](const std::vector<float> & va, int cc, int rr) {  // face rr | rr+1
          const float aL = at(in.vy, w, h, cc, rr);
          const float aR = at(in.vy, w, h, cc, rr + 1);
          const float alpha = std::max(std::fabs(aL), std::fabs(aR));
          const float gL = aL * at(va, w, h, cc, rr);
          const float gR = aR * at(va, w, h, cc, rr + 1);
          return 0.5f * (gL + gR) - 0.5f * alpha * (at(va, w, h, cc, rr + 1) - at(va, w, h, cc, rr));
        };
        auto divergence = [&](const std::vector<float> & va) {
          return (fluxX(va, c, r) - fluxX(va, c - 1, r)) * inv_ds +
                 (fluxY(va, c, r) - fluxY(va, c, r - 1)) * inv_ds;
        };
        rate_x = -divergence(in.vx);
        rate_y = -divergence(in.vy);
      }

      out.vx[idx] = vx + dt * rate_x;
      out.vy[idx] = vy + dt * rate_y;
    }
  }
}

// Number of explicit sub-steps to march from t=0 to t_target under the CFL
// bound. Must be identical in the forward and adjoint passes.
int num_substeps(const Grid & v0, float t_target, float ds, float cfl)
{
  if (t_target == 0.0f) {
    return 0;
  }
  float vmax = 0.0f;
  for (size_t i = 0; i < v0.size(); ++i) {
    vmax = std::max(vmax, std::hypot(v0.vx[i], v0.vy[i]));
  }
  if (vmax <= 0.0f) {
    return 1;
  }
  return std::max(1, static_cast<int>(std::ceil(std::fabs(t_target) * vmax / (cfl * ds))));
}

// Integrate v0 from t = 0 to t = t_target (signed) and return the result.
Grid integrate_to(const Grid & v0, float t_target, Scheme scheme, float ds, float cfl)
{
  Grid cur = v0;
  const int n_sub = num_substeps(v0, t_target, ds, cfl);
  if (n_sub == 0) {
    return cur;
  }
  const float dt = t_target / static_cast<float>(n_sub);

  Grid nxt(cur.w, cur.h);
  for (int s = 0; s < n_sub; ++s) {
    step(cur, nxt, scheme, ds, dt);
    std::swap(cur.vx, nxt.vx);
    std::swap(cur.vy, nxt.vy);
  }
  return cur;
}

// Accumulate val into target[clamp(c), clamp(r)] (transpose of Neumann read).
inline void scatter(std::vector<float> & t, int w, int h, int c, int r, float val)
{
  c = std::clamp(c, 0, w - 1);
  r = std::clamp(r, 0, h - 1);
  t[static_cast<size_t>(r) * w + c] += val;
}

// Adjoint of the upwind difference D_x f (direction backward when vx>0).
inline void scatter_ddx(std::vector<float> & t, int w, int h, int c, int r,
  bool backward, float bar, float inv_ds)
{
  if (backward) {  // (f[c]-f[c-1])/ds
    scatter(t, w, h, c, r, bar * inv_ds);
    scatter(t, w, h, c - 1, r, -bar * inv_ds);
  } else {         // (f[c+1]-f[c])/ds
    scatter(t, w, h, c + 1, r, bar * inv_ds);
    scatter(t, w, h, c, r, -bar * inv_ds);
  }
}
inline void scatter_ddy(std::vector<float> & t, int w, int h, int c, int r,
  bool backward, float bar, float inv_ds)
{
  if (backward) {
    scatter(t, w, h, c, r, bar * inv_ds);
    scatter(t, w, h, c, r - 1, -bar * inv_ds);
  } else {
    scatter(t, w, h, c, r + 1, bar * inv_ds);
    scatter(t, w, h, c, r, -bar * inv_ds);
  }
}

// Adjoint of one upwind step: given dL/d(out), return dL/d(in). Directions are
// frozen at `in` (matching the forward pass).
Grid step_adjoint_upwind(const Grid & in, const Grid & g_out, float ds, float dt)
{
  const int w = in.w, h = in.h;
  const float inv_ds = 1.0f / ds;
  Grid g_in(w, h);

  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const size_t idx = static_cast<size_t>(r) * w + c;
      const float vx = in.vx[idx];
      const float vy = in.vy[idx];
      const bool bx = vx > 0.0f;   // backward difference in x?
      const bool by = vy > 0.0f;   // backward difference in y?

      auto ddx = [&](const std::vector<float> & f) {
        return bx ? (at(f, w, h, c, r) - at(f, w, h, c - 1, r)) * inv_ds
                  : (at(f, w, h, c + 1, r) - at(f, w, h, c, r)) * inv_ds;
      };
      auto ddy = [&](const std::vector<float> & f) {
        return by ? (at(f, w, h, c, r) - at(f, w, h, c, r - 1)) * inv_ds
                  : (at(f, w, h, c, r + 1) - at(f, w, h, c, r)) * inv_ds;
      };
      const float gx_x = ddx(in.vx);
      const float gy_x = ddy(in.vx);
      const float gx_y = ddx(in.vy);
      const float gy_y = ddy(in.vy);

      const float avx = g_out.vx[idx];
      const float avy = g_out.vy[idx];

      // out = in + dt*rate, rate = -(S);  identity part:
      g_in.vx[idx] += avx;
      g_in.vy[idx] += avy;

      const float sx_bar = -dt * avx;   // adjoint of S_x = vx*gx_x + vy*gy_x
      const float sy_bar = -dt * avy;   // adjoint of S_y = vx*gx_y + vy*gy_y

      // Local multipliers.
      g_in.vx[idx] += sx_bar * gx_x + sy_bar * gx_y;
      g_in.vy[idx] += sx_bar * gy_x + sy_bar * gy_y;

      // Stencil adjoints (gx_x,gx_y use x-direction bx; gy_x,gy_y use y-dir by).
      scatter_ddx(g_in.vx, w, h, c, r, bx, sx_bar * vx, inv_ds);  // gx_x = ddx(vx)
      scatter_ddy(g_in.vx, w, h, c, r, by, sx_bar * vy, inv_ds);  // gy_x = ddy(vx)
      scatter_ddx(g_in.vy, w, h, c, r, bx, sy_bar * vx, inv_ds);  // gx_y = ddx(vy)
      scatter_ddy(g_in.vy, w, h, c, r, by, sy_bar * vy, inv_ds);  // gy_y = ddy(vy)
    }
  }
  return g_in;
}

// Adjoint of one conservative Lax-Friedrichs (Burgers) step. The advecting
// velocities (aL/aR, bL/bR) and the LF dissipation coefficient alpha are frozen
// at the forward state `in`, so each flux is linear in the transported quantity
// and each component's transport decouples (frozen-coefficient adjoint, as
// documented for `propagate_vjp`). This is the transpose of the linearization
// used by the optimizer; it omits the dependence of the dynamics on the
// advecting field itself, hence is an approximate gradient for the Burgers
// scheme (the upwind scheme provides the exact, FD-validated adjoint).
Grid step_adjoint_burgers(const Grid & in, const Grid & g_out, float ds, float dt)
{
  const int w = in.w, h = in.h;
  const float inv_ds = 1.0f / ds;
  Grid g_in(w, h);

  // Scatter the cotangent `fbar` of an x-/y-face flux back onto the (frozen-
  // coefficient) transported entries, with Neumann index clamping.
  auto scatter_fluxX = [&](std::vector<float> & gin, int cc, int rr, float fbar) {
    const float aL = at(in.vx, w, h, cc, rr);
    const float aR = at(in.vx, w, h, cc + 1, rr);
    const float alpha = std::max(std::fabs(aL), std::fabs(aR));
    scatter(gin, w, h, cc, rr, fbar * 0.5f * (aL + alpha));
    scatter(gin, w, h, cc + 1, rr, fbar * 0.5f * (aR - alpha));
  };
  auto scatter_fluxY = [&](std::vector<float> & gin, int cc, int rr, float fbar) {
    const float bL = at(in.vy, w, h, cc, rr);
    const float bR = at(in.vy, w, h, cc, rr + 1);
    const float alpha = std::max(std::fabs(bL), std::fabs(bR));
    scatter(gin, w, h, cc, rr, fbar * 0.5f * (bL + alpha));
    scatter(gin, w, h, cc, rr + 1, fbar * 0.5f * (bR - alpha));
  };

  // out.comp = comp + dt*(-div(comp)),  div = (Fx(c,r)-Fx(c-1,r)+Fy(c,r)-Fy(c,r-1))/ds
  auto backprop = [&](const std::vector<float> & gout, std::vector<float> & gin) {
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        const size_t idx = static_cast<size_t>(r) * w + c;
        gin[idx] += gout[idx];                    // identity part
        const float base = -dt * inv_ds * gout[idx];
        scatter_fluxX(gin, c, r, base);
        scatter_fluxX(gin, c - 1, r, -base);
        scatter_fluxY(gin, c, r, base);
        scatter_fluxY(gin, c, r - 1, -base);
      }
    }
  };

  backprop(g_out.vx, g_in.vx);
  backprop(g_out.vy, g_in.vy);
  return g_in;
}

void add_inplace(Grid & a, const Grid & b)
{
  for (size_t i = 0; i < a.size(); ++i) {
    a.vx[i] += b.vx[i];
    a.vy[i] += b.vy[i];
  }
}

}  // namespace

std::vector<Grid> propagate(
  const Grid & v0,
  const std::vector<float> & t_out_s,
  Scheme scheme,
  float ds,
  float cfl)
{
  std::vector<Grid> out;
  out.reserve(t_out_s.size());
  for (float t : t_out_s) {
    out.push_back(integrate_to(v0, t, scheme, ds, cfl));
  }
  return out;
}

Grid propagate_vjp(
  const Grid & v0,
  const std::vector<float> & t_out_s,
  Scheme scheme,
  float ds,
  float cfl,
  const std::vector<Grid> & grad_out)
{
  Grid acc(v0.w, v0.h);
  for (size_t n = 0; n < t_out_s.size(); ++n) {
    const float t = t_out_s[n];
    if (t == 0.0f) {
      add_inplace(acc, grad_out[n]);  // identity branch
      continue;
    }
    const int n_sub = num_substeps(v0, t, ds, cfl);
    const float dt = t / static_cast<float>(n_sub);

    // Forward replay, storing every intermediate state.
    std::vector<Grid> states;
    states.reserve(static_cast<size_t>(n_sub) + 1);
    states.push_back(v0);
    for (int s = 0; s < n_sub; ++s) {
      Grid nx(v0.w, v0.h);
      step(states[static_cast<size_t>(s)], nx, scheme, ds, dt);
      states.push_back(std::move(nx));
    }

    // Reverse sweep.
    Grid g = grad_out[n];
    for (int s = n_sub - 1; s >= 0; --s) {
      if (scheme == Scheme::Upwind) {
        g = step_adjoint_upwind(states[static_cast<size_t>(s)], g, ds, dt);
      } else {
        g = step_adjoint_burgers(states[static_cast<size_t>(s)], g, ds, dt);
      }
    }
    add_inplace(acc, g);
  }
  return acc;
}

}  // namespace event_detector_cpp::flow
