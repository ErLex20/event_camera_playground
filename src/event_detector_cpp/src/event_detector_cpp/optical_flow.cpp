/**
 * Event Detector dense optical-flow estimation via contrast maximization.
 *
 * Implements the multi-scale contrast-maximization flow of Shiba-Gallego,
 * "Secrets of Event-Based Optical Flow, Depth and Ego-Motion Estimation by
 * Contrast Maximization" (TPAMI 2024). A tile-grid flow field is optimized
 * coarse-to-fine over a tile pyramid (2^(l-1) tiles per side, Sec. III-D),
 * each scale minimizing the composite energy E(F) = 1/f(F) + lambda*TV(F)
 * (Eq. 9) with a Hessian-free Newton-CG solver and warm-started from the
 * previous, coarser scale. The dense flow and the Image of Warped Events at
 * the window midpoint are rendered for publishing.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 28, 2026
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

#include "event_detector_cpp/event_detector_cpp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "event_detector_cpp/flow/newton_cg.hpp"
#include "event_detector_cpp/flow/objective.hpp"

namespace event_detector_cpp
{

namespace
{

using event_detector_cpp::flow::ContrastNorm;
using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::Objective;
using event_detector_cpp::flow::ObjectiveParams;
using event_detector_cpp::flow::Scheme;

// Hard cap on events fed to the objective; busier windows are strided down so
// the per-evaluation cost stays bounded without biasing the estimate. Sized to
// accommodate the paper's per-window event counts (up to 1.5M for DSEC); reduce
// flow_num_events for faster, lower-fidelity runs.
constexpr std::size_t kMaxSolveEvents = 2000000;

struct EventSample
{
  float x;
  float y;
  int64_t t_us;
};

/**
 * Bilinearly sample the tile-grid flow field at a pixel, matching the stencil
 * convention used to build the objective (tile centers at (i+0.5)*spacing,
 * replicate at the borders).
 */
inline void sample_tile_velocity(
  const Eigen::VectorXf & F, int tx, int ty, int img_w, int img_h,
  float px, float py, float & vx, float & vy)
{
  const float sx = static_cast<float>(img_w) / tx;
  const float sy = static_cast<float>(img_h) / ty;
  const float gx = px / sx - 0.5f;
  const float gy = py / sy - 0.5f;
  int i0 = static_cast<int>(std::floor(gx));
  int j0 = static_cast<int>(std::floor(gy));
  const float fx = gx - i0;
  const float fy = gy - j0;
  auto cl = [](int v, int n) { return std::clamp(v, 0, n - 1); };
  const int i0c = cl(i0, tx), i1c = cl(i0 + 1, tx);
  const int j0c = cl(j0, ty), j1c = cl(j0 + 1, ty);
  const int k00 = j0c * tx + i0c, k10 = j0c * tx + i1c;
  const int k01 = j1c * tx + i0c, k11 = j1c * tx + i1c;
  const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
  const float w01 = (1 - fx) * fy,       w11 = fx * fy;
  vx = w00 * F[2 * k00] + w10 * F[2 * k10] + w01 * F[2 * k01] + w11 * F[2 * k11];
  vy = w00 * F[2 * k00 + 1] + w10 * F[2 * k10 + 1] +
       w01 * F[2 * k01 + 1] + w11 * F[2 * k11 + 1];
}

// Resample a tile field from one tile resolution to another (warm-start across
// scales): each new tile takes the old field's value at its center pixel.
Eigen::VectorXf upsample_field(
  const Eigen::VectorXf & F_old, int tx_old, int ty_old,
  int tx_new, int ty_new, int img_w, int img_h)
{
  Eigen::VectorXf F_new(2 * tx_new * ty_new);
  for (int j = 0; j < ty_new; ++j) {
    for (int i = 0; i < tx_new; ++i) {
      const float px = (i + 0.5f) / tx_new * img_w;
      const float py = (j + 0.5f) / ty_new * img_h;
      float vx, vy;
      sample_tile_velocity(F_old, tx_old, ty_old, img_w, img_h, px, py, vx, vy);
      const int k = j * tx_new + i;
      F_new[2 * k] = vx;
      F_new[2 * k + 1] = vy;
    }
  }
  return F_new;
}

// The paper's optimization is unconstrained; only sanitize non-finite solver
// output before carrying the field to the next scale/window.
void sanitize_field(Eigen::VectorXf & F)
{
  for (int k = 0; k < F.size() / 2; ++k) {
    float vx = F[2 * k], vy = F[2 * k + 1];
    if (!std::isfinite(vx) || !std::isfinite(vy)) {
      vx = vy = 0.0f;
    }
    F[2 * k] = vx;
    F[2 * k + 1] = vy;
  }
}

}  // namespace

EventDetector::FlowResult EventDetector::solve_flow_cmax(const dv::EventStore & window)
{
  FlowResult result;
  if (window.isEmpty() || res_.width <= 0 || res_.height <= 0) {
    return result;
  }

  const int w = res_.width;
  const int h = res_.height;
  const float vis_speed_cap = static_cast<float>(flow_max_speed_px_s_);

  // Build the event set for the objective, strided down if over the cap.
  const std::size_t total = static_cast<std::size_t>(window.size());
  const std::size_t stride = (total + kMaxSolveEvents - 1) / kMaxSolveEvents;
  std::vector<EventSample> solve_samples;
  solve_samples.reserve(total / stride + 1);
  std::size_t idx = 0;
  for (const auto & e : window) {
    if (idx++ % stride != 0) {
      continue;
    }
    if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
      continue;
    }
    solve_samples.push_back({
      static_cast<float>(e.x()),
      static_cast<float>(e.y()),
      e.timestamp()});
  }
  if (solve_samples.size() < 2) {
    return result;
  }

  auto by_time = [](const EventSample & a, const EventSample & b) {
    return a.t_us < b.t_us;
  };
  const auto t_minmax = std::minmax_element(
    solve_samples.begin(), solve_samples.end(), by_time);
  const int64_t t_lo_us = t_minmax.first->t_us;
  const int64_t t_hi_us = t_minmax.second->t_us;
  const int64_t t_ref_us = t_lo_us + (t_hi_us - t_lo_us) / 2;
  const float t_lo = static_cast<float>(t_lo_us - t_ref_us) * 1e-6f;
  const float t_hi = static_cast<float>(t_hi_us - t_ref_us) * 1e-6f;

  // Warp reference is the selected event set midpoint; event times are stored
  // relative to it [s], so F estimates the flow at t_mid (Sec. III-B).
  Events ev;
  ev.x.reserve(solve_samples.size());
  ev.y.reserve(solve_samples.size());
  ev.t.reserve(solve_samples.size());
  for (const EventSample & e : solve_samples) {
    ev.x.push_back(e.x);
    ev.y.push_back(e.y);
    ev.t.push_back(static_cast<float>(e.t_us - t_ref_us) * 1e-6f);
  }

  flow::NewtonCgParams ncg;
  ncg.newton_max_iter = static_cast<int>(flow_newton_max_iter_);
  ncg.cg_max_iter = static_cast<int>(flow_cg_max_iter_);
  ncg.cg_tol = static_cast<float>(flow_cg_tol_);
  // Optimize in pixel-displacement units (F = var_scale * z, var_scale ~ 1/t)
  // so the problem is well-conditioned regardless of the number of tiles.
  const float t_span = std::max({std::abs(t_lo), std::abs(t_hi), 1e-4f});
  ncg.var_scale = 1.0f / t_span;
  ncg.fd_step = 0.5f;  // ~0.5 px finite-difference perturbation

  // Coarse-to-fine over the tile pyramid: scale l uses 2^(l-1) tiles per side.
  const int n_scales = std::max<int>(1, static_cast<int>(flow_num_scales_));
  const bool have_prev = (prev_flow_tiles_ > 0 &&
    prev_flow_field_.size() == 2 * prev_flow_tiles_ * prev_flow_tiles_);
  Eigen::VectorXf F;
  int tx_prev = 0, ty_prev = 0;
  ObjectiveParams final_params;
  float cached_g0 = 0.0f;  // G(0) is tile-independent: compute once, reuse.
  for (int l = 1; l <= n_scales; ++l) {
    const int tiles = 1 << (l - 1);

    ObjectiveParams p;
    p.img_w = w;
    p.img_h = h;
    p.tiles_x = tiles;
    p.tiles_y = tiles;
    p.t_lo = t_lo;
    p.t_hi = t_hi;
    p.norm = flow_contrast_l2_ ? ContrastNorm::L2 : ContrastNorm::L1;
    p.tv_weight = static_cast<float>(flow_tv_weight_);
    p.tv_eps = static_cast<float>(flow_tv_charbonnier_eps_);
    p.time_aware = flow_time_aware_;
    p.scheme = flow_pde_burgers_ ? Scheme::Burgers : Scheme::Upwind;
    p.time_bins = static_cast<int>(flow_time_bins_);
    // Square-cell propagation grid covering the image (ds = w / prop_w).
    p.prop_w = static_cast<int>(flow_prop_grid_);
    p.prop_h = std::max<int>(2, std::lround(
      static_cast<double>(flow_prop_grid_) * h / w));
    p.cfl = 0.5f;
    p.g0_override = cached_g0;  // 0 at the coarsest scale -> computed there

    Objective obj(ev, p);
    if (cached_g0 <= 0.0f) {
      cached_g0 = obj.g0();  // reuse for all finer scales
    }
    p.g0_override = cached_g0;
    final_params = p;

    // Within-pyramid initialization: zero at the coarsest scale, otherwise the
    // bilinearly upscaled coarser-scale flow of this window.
    Eigen::VectorXf init = (tx_prev == 0)
      ? Eigen::VectorXf::Zero(obj.num_vars())
      : upsample_field(F, tx_prev, ty_prev, tiles, tiles, w, h);

    // Cross-window initialization (Sec. III-D): the previous window's finest
    // flow, resampled to this scale. The coarsest scale is initialized from it
    // directly; finer scales average it with the within-pyramid initialization.
    if (have_prev) {
      Eigen::VectorXf prev_at =
        upsample_field(prev_flow_field_, prev_flow_tiles_, prev_flow_tiles_,
                       tiles, tiles, w, h);
      init = (tx_prev == 0) ? prev_at : (0.5f * (init + prev_at)).eval();
    }

    F = std::move(init);
    flow::newton_cg_minimize(obj, F, ncg);
    sanitize_field(F);

    tx_prev = tiles;
    ty_prev = tiles;
  }

  // Persist the finest-scale field to warm-start the next window.
  prev_flow_field_ = F;
  prev_flow_tiles_ = tx_prev;

  Events render_ev;
  render_ev.x.reserve(total);
  render_ev.y.reserve(total);
  render_ev.t.reserve(total);
  float render_t_lo = 0.0f;
  float render_t_hi = 0.0f;
  for (const auto & e : window) {
    if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
      continue;
    }
    const float tau = static_cast<float>(e.timestamp() - t_ref_us) * 1e-6f;
    render_ev.x.push_back(static_cast<float>(e.x()));
    render_ev.y.push_back(static_cast<float>(e.y()));
    render_ev.t.push_back(tau);
    render_t_lo = std::min(render_t_lo, tau);
    render_t_hi = std::max(render_t_hi, tau);
  }

  // ── Render the dense flow field as HSV (hue = direction, value = speed) ─────
  cv::Mat angle(h, w, CV_32F);
  cv::Mat magnitude(h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    float * arow = angle.ptr<float>(y);
    float * mrow = magnitude.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      float vx, vy;
      sample_tile_velocity(
        F, tx_prev, ty_prev, w, h,
        static_cast<float>(x), static_cast<float>(y), vx, vy);
      // F is the paper's warp parameter in x' = x + dt * v. Physical optical
      // flow has the opposite sign, as in the authors' dense-flow renderer.
      vx = -vx;
      vy = -vy;
      float a = std::atan2(vy, vx);                       // [-pi, pi]
      if (a < 0.0f) a += 2.0f * static_cast<float>(CV_PI);  // [0, 2pi)
      arow[x] = a;
      mrow[x] = std::hypot(vx, vy);
    }
  }

  std::vector<float> support_speeds;
  support_speeds.reserve(static_cast<size_t>(w) * h);
  double support_speed_sum = 0.0;
  double support_max_speed = 0.0;
  for (int y = 0; y < h; ++y) {
    const float * mrow = magnitude.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const float spd = mrow[x];
      if (!std::isfinite(spd)) {
        continue;
      }
      support_speeds.push_back(spd);
      support_speed_sum += spd;
      support_max_speed = std::max(support_max_speed, static_cast<double>(spd));
    }
  }

  cv::Mat hsv_parts[3];
  // Map angle [0, 2pi) to OpenCV's 8-bit hue range [0, 180).
  cv::Mat hue = angle * (180.0f / (2.0f * static_cast<float>(CV_PI)));
  hue.convertTo(hsv_parts[0], CV_8U);
  hsv_parts[1] = cv::Mat(h, w, CV_8U, cv::Scalar(255));
  double observed_max_speed = 0.0;
  cv::minMaxLoc(magnitude, nullptr, &observed_max_speed);
  double robust_max_speed = support_max_speed;
  if (!support_speeds.empty()) {
    const size_t nth = std::min(
      support_speeds.size() - 1,
      static_cast<size_t>(0.95 * static_cast<double>(support_speeds.size())));
    std::nth_element(support_speeds.begin(), support_speeds.begin() + nth, support_speeds.end());
    robust_max_speed = std::max<double>(support_speeds[nth], 0.05 * support_max_speed);
  }
  const double display_max_speed = (robust_max_speed > 1e-6)
    ? std::min(static_cast<double>(vis_speed_cap), robust_max_speed)
    : ((observed_max_speed > 1e-6)
      ? std::min(static_cast<double>(vis_speed_cap), observed_max_speed)
      : static_cast<double>(vis_speed_cap));
  const double support_mean_speed = support_speeds.empty()
    ? 0.0
    : support_speed_sum / static_cast<double>(support_speeds.size());
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "Flow speed [px/s]: mean=%.3f max=%.3f display_max=%.3f pixels=%zu",
    support_mean_speed, support_max_speed, display_max_speed, support_speeds.size());

  magnitude *= (255.0f / static_cast<float>(display_max_speed));
  cv::threshold(magnitude, magnitude, 255.0, 255.0, cv::THRESH_TRUNC);
  magnitude.convertTo(hsv_parts[2], CV_8U);
  cv::Mat hsv;
  cv::merge(hsv_parts, 3, hsv);
  cv::cvtColor(hsv, result.flow, cv::COLOR_HSV2BGR);

  // ── Render the IWE: warp every event by the objective's local flow to t_mid ─
  const int scale = std::max<int>(1, static_cast<int>(flow_iwe_scale_));
  const int iw = (w + scale - 1) / scale;
  const int ih = (h + scale - 1) / scale;
  const float inv_scale = 1.0f / static_cast<float>(scale);

  final_params.tiles_x = tx_prev;
  final_params.tiles_y = ty_prev;
  final_params.t_lo = render_t_lo;
  final_params.t_hi = render_t_hi;
  final_params.g0_override = cached_g0;
  Objective render_obj(render_ev, final_params);
  std::vector<float> render_vx;
  std::vector<float> render_vy;
  render_obj.event_flow(F, render_vx, render_vy);

  cv::Mat iwe = cv::Mat::zeros(ih, iw, CV_32F);
  for (size_t k = 0; k < render_ev.size(); ++k) {
    const float wx = (render_ev.x[k] + render_ev.t[k] * render_vx[k]) * inv_scale;
    const float wy = (render_ev.y[k] + render_ev.t[k] * render_vy[k]) * inv_scale;
    const int x0 = static_cast<int>(std::floor(wx));
    const int y0 = static_cast<int>(std::floor(wy));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= iw || y0 + 1 >= ih) {
      continue;
    }
    const float fx = wx - x0, fy = wy - y0;
    float * r0 = iwe.ptr<float>(y0);
    float * r1 = iwe.ptr<float>(y0 + 1);
    r0[x0]     += (1.0f - fx) * (1.0f - fy);
    r0[x0 + 1] += fx * (1.0f - fy);
    r1[x0]     += (1.0f - fx) * fy;
    r1[x0 + 1] += fx * fy;
  }
  cv::normalize(iwe, result.iwe, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

  return result;
}

}  // namespace event_detector_cpp
