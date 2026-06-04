/**
 * Contrast-maximization dense-flow objective (Shiba-Gallego "Secrets").
 *
 * Implements the composite energy minimized at each scale (Eq. 9):
 *
 *     E(F) = 1 / f(F) + lambda * TV(F)
 *     f(F) = [ G(t1) + 2 G(tmid) + G(tNe) ] / (4 G0)      (Eqs. 5, 6)
 *
 * where F is the tile-grid flow field at t_mid (bilinearly interpolated to a
 * dense per-event velocity), G is the IWE gradient-magnitude focus (L1 or L2),
 * and G0 is the zero-flow focus (normalization giving the FWL interpretation
 * f>1 <=> sharper than the identity warp). The objective exposes its analytic
 * gradient w.r.t. F. Kept free of OpenCV/ROS for isolated unit testing.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#pragma once

#include <vector>

#include <Eigen/Core>

#include "event_detector_cpp/flow/propagation.hpp"

namespace event_detector_cpp::flow
{

enum class ContrastNorm
{
  L1,  // mean |grad I|        (paper default, Sec. III-B1)
  L2   // mean |grad I|^2 (squared L2)
};

/// Events for one window: pixel coords (x, y) and time t relative to t_mid [s].
struct Events
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> t;
  size_t size() const { return x.size(); }
};

struct ObjectiveProfile
{
  int focus_calls = 0;
  int value_calls = 0;
  int value_and_grad_calls = 0;
  int event_flow_calls = 0;
  double build_stencils_ms = 0.0;
  double build_time_aware_ms = 0.0;
  double g0_render_ms = 0.0;
  double g0_contrast_ms = 0.0;
  double focus_ms = 0.0;
  double value_ms = 0.0;
  double value_and_grad_ms = 0.0;
  double boundary_ms = 0.0;
  double propagate_ms = 0.0;
  double event_sample_ms = 0.0;
  double render_iwe_ms = 0.0;
  double contrast_ms = 0.0;
  double backprop_ms = 0.0;
  double scatter_ms = 0.0;
  double propagate_vjp_ms = 0.0;
  double tv_ms = 0.0;
  double event_flow_ms = 0.0;
};

struct ObjectiveParams
{
  int img_w = 0;            // IWE render width  [px]
  int img_h = 0;            // IWE render height [px]
  int tiles_x = 1;          // tile grid for this scale
  int tiles_y = 1;
  float t_lo = 0.0f;        // (t1   - t_mid) [s], <= 0
  float t_hi = 0.0f;        // (tNe  - t_mid) [s], >= 0
  ContrastNorm norm = ContrastNorm::L1;
  float tv_weight = 0.0f;   // lambda (Eq. 9)
  float tv_eps = 1e-3f;     // Charbonnier epsilon
  // Time-aware propagation (Sec. III-C); when false, warp Eq. (4) is used.
  bool time_aware = false;
  Scheme scheme = Scheme::Upwind;
  int time_bins = 5;
  int prop_w = 0;           // propagation grid (defaults to a coarse grid)
  int prop_h = 0;
  float cfl = 0.5f;
  // Zero-flow normalization G(0). It depends only on the events and image (not
  // on the tiles), so across a multi-scale solve it can be computed once and
  // injected here to skip recomputation. <= 0 means "compute it".
  float g0_override = 0.0f;
  // Optional profiling sink. It is never owned by Objective and may be null.
  ObjectiveProfile * profile = nullptr;
};

/**
 * @brief Contrast-maximization objective for one event window at one scale.
 *
 * Per-event interpolation stencils that depend only on geometry (not on F) are
 * precomputed once. The variable vector F is laid out tile-major with the two
 * components interleaved: F[2*k] = vx, F[2*k+1] = vy, k = ty*tiles_x + tx.
 */
class Objective
{
public:
  Objective(Events events, ObjectiveParams params);

  /// Number of optimization variables (2 * tiles_x * tiles_y).
  int num_vars() const { return 2 * params_.tiles_x * params_.tiles_y; }

  /// Composite energy E(F).
  float value(const Eigen::VectorXf & F) const;

  /// E(F) and its analytic gradient dE/dF.
  float value_and_grad(const Eigen::VectorXf & F, Eigen::VectorXf & grad) const;

  /// Multi-reference focus f(F) (without the 1/f or TV terms).
  float focus(const Eigen::VectorXf & F) const;

  /// Per-event velocity used by the objective warp. With time-aware flow this
  /// is \hat{v}(x_k, t_k) from Eq. (8); otherwise it is v(x_k) from Eq. (4).
  void event_flow(
    const Eigen::VectorXf & F,
    std::vector<float> & vx,
    std::vector<float> & vy) const;

  /// Zero-flow normalization constant G0.
  float g0() const { return g0_; }

private:
  Events events_;
  ObjectiveParams params_;
  float g0_ = 1.0f;

  // Bilinear stencil: 4 source-node indices + weights.
  struct Stencil { int idx[4]; float w[4]; };
  std::vector<Stencil> stencils_;   // per event -> tile field F

  void build_stencils();

  // --- Time-aware flow state (Sec. III-C), built only when params_.time_aware.
  // The flow at t_mid (the field F, sampled onto a regular propagation grid) is
  // transported by the self-advection PDE to a set of time bins; each event is
  // warped by the propagated flow sampled at its own (x, t) bin (Eq. 8).
  int pw_ = 0;                          // propagation grid width  [nodes]
  int ph_ = 0;                          // propagation grid height [nodes]
  float ds_ = 1.0f;                     // grid spacing [px]
  std::vector<Stencil> bnd_stencils_;   // per prop node -> tile field F
  std::vector<Stencil> ev_prop_stencils_;  // per event -> propagation grid
  std::vector<int> ev_bin_;             // per event -> time-bin index
  std::vector<float> bin_times_;        // representative time per bin [s]

  void build_time_aware();

  // Boundary flow field v(x, t_mid) on the propagation grid, sampled from F.
  Grid boundary_field(const Eigen::VectorXf & F) const;

  // Composite focus computed with the time-aware warp; if `grad` is non-null,
  // also accumulates the focus-term gradient dE/dF into it.
  float focus_time_aware(const Eigen::VectorXf & F, Eigen::VectorXf * grad) const;
};

}  // namespace event_detector_cpp::flow
