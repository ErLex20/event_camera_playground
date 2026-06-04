/**
 * Hessian-free truncated-Newton (Newton-CG) minimizer.
 *
 * The paper minimizes the composite energy with a Newton-CG optimizer
 * (Sec. III-D). This is a standard truncated-Newton method: each outer step
 * solves the Newton system H d = -g with linear conjugate gradient, where the
 * Hessian-vector products are formed matrix-free by finite-differencing the
 * analytic gradient,
 *
 *     H(x) p  ~=  ( grad(x + eps p) - grad(x) ) / eps,
 *
 * so no explicit Hessian is ever assembled. CG is truncated by an inexact
 * (forcing-sequence) tolerance and on encountering non-positive curvature; an
 * Armijo backtracking line search guards the step. The objective need only
 * provide value() and value_and_grad().
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace event_detector_cpp::flow
{

struct NewtonCgParams
{
  int newton_max_iter = 30;  // outer Newton iterations
  int cg_max_iter = 20;      // inner CG iterations per Newton step
  float cg_tol = 0.1f;       // forcing-tolerance factor for the inner CG solve
  // The flow objective depends on the variables F (px/s) only through the warp
  // displacement F * t, with t the (small) event time. Optimizing F directly is
  // badly scaled: gradients are tiny and a fixed finite-difference step cannot
  // resolve curvature, especially once F has many tile components. We therefore
  // optimize in displacement units z, with F = var_scale * z (var_scale ~ 1/t),
  // which is well-conditioned and tile-count independent.
  float var_scale = 1.0f;
  // Finite-difference perturbation [displacement units] for Hessian-vector
  // products: large enough to exceed float splat noise, small enough to stay
  // local. In pixel-displacement coordinates a fraction of a pixel works well.
  float fd_step = 0.5f;
};

/**
 * @brief Minimize `obj` over `x` in place with Newton-CG.
 *
 * @tparam Obj Type exposing `float value(const VectorXf&) const` and
 *   `float value_and_grad(const VectorXf&, VectorXf&) const`.
 * @param obj Objective to minimize.
 * @param x   Initial guess; overwritten with the optimized variables.
 * @param p   Optimizer settings.
 */
template <class Obj>
void newton_cg_minimize(const Obj & obj, Eigen::VectorXf & x, const NewtonCgParams & p)
{
  const float s = (p.var_scale > 0.0f) ? p.var_scale : 1.0f;

  // Optimize in scaled coordinates z, with the objective variable x = s * z.
  // The gradient w.r.t. z is s * (gradient w.r.t. x) (s is an isotropic scale).
  auto val = [&](const Eigen::VectorXf & z) { return obj.value(s * z); };
  auto vag = [&](const Eigen::VectorXf & z, Eigen::VectorXf & g) {
    const float v = obj.value_and_grad(s * z, g);
    g *= s;
    return v;
  };

  Eigen::VectorXf z = x / s;
  Eigen::VectorXf g, g_probe;
  for (int outer = 0; outer < p.newton_max_iter; ++outer) {
    const float fz = vag(z, g);
    const float gnorm = g.norm();
    if (!std::isfinite(gnorm) || gnorm < 1e-6f) {
      break;
    }

    // Inner CG: solve H d = -g without forming H. Hessian-vector products come
    // from a forward finite difference of the analytic gradient.
    Eigen::VectorXf d = Eigen::VectorXf::Zero(z.size());
    Eigen::VectorXf r = -g;            // residual b - H d, with d = 0
    Eigen::VectorXf pdir = r;
    double rs_old = static_cast<double>(r.dot(r));
    const float cg_thresh = p.cg_tol * gnorm;   // inexact (forcing) tolerance
    // Norm of the FD perturbation in scaled space: a fixed absolute scale so it
    // stays meaningful even at z = 0 (where a relative step would vanish).
    const float fd_norm = (p.fd_step > 0.0f)
      ? p.fd_step
      : 1e-3f * std::max(1.0f, z.norm());

    for (int cg = 0; cg < p.cg_max_iter; ++cg) {
      const float pn = pdir.norm();
      if (pn < 1e-12f) {
        break;
      }
      // Scale the unit search direction by the target perturbation norm.
      const float eps = fd_norm / pn;
      vag(z + eps * pdir, g_probe);
      const Eigen::VectorXf Hp = (g_probe - g) / eps;

      const double pHp = static_cast<double>(pdir.dot(Hp));
      if (pHp <= 1e-12) {
        // Non-positive curvature: take what CG has, or steepest descent if none.
        if (cg == 0) {
          d = r;
        }
        break;
      }
      const double alpha = rs_old / pHp;
      d += static_cast<float>(alpha) * pdir;
      r -= static_cast<float>(alpha) * Hp;
      const double rs_new = static_cast<double>(r.dot(r));
      if (std::sqrt(rs_new) < cg_thresh) {
        break;
      }
      pdir = r + static_cast<float>(rs_new / rs_old) * pdir;
      rs_old = rs_new;
    }

    // Ensure d is a descent direction; fall back to steepest descent otherwise.
    float gd = g.dot(d);
    if (!(gd < 0.0f) || !d.allFinite()) {
      d = -g;
      gd = g.dot(d);
    }

    // Armijo backtracking line search.
    const float c = 1e-4f;
    float t = 1.0f;
    bool improved = false;
    for (int ls = 0; ls < 25; ++ls) {
      const float f_try = val(z + t * d);
      if (std::isfinite(f_try) && f_try <= fz + c * t * gd) {
        z += t * d;
        improved = true;
        break;
      }
      t *= 0.5f;
    }
    if (!improved) {
      break;  // line search failed to make progress
    }
  }
  x = s * z;
}

}  // namespace event_detector_cpp::flow
