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

// Clamp each tile's speed to max_spd and zero out any non-finite component.
void clamp_field(Eigen::VectorXf & F, float max_spd)
{
  for (int k = 0; k < F.size() / 2; ++k) {
    float vx = F[2 * k], vy = F[2 * k + 1];
    if (!std::isfinite(vx) || !std::isfinite(vy)) {
      vx = vy = 0.0f;
    }
    const float mag = std::hypot(vx, vy);
    if (mag > max_spd) {
      const float s = max_spd / mag;
      vx *= s;
      vy *= s;
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
  const float max_spd = static_cast<float>(flow_max_speed_px_s_);

  // Warp reference is the window midpoint; event times are stored relative to
  // it [s], so the field F estimates the flow at t_mid (Sec. III-B).
  const int64_t t_lo_us = window.getLowestTime();
  const int64_t t_hi_us = window.getHighestTime();
  const int64_t t_mid_us = t_lo_us + (t_hi_us - t_lo_us) / 2;

  // Build the event set for the objective, strided down if over the cap.
  const std::size_t total = static_cast<std::size_t>(window.size());
  const std::size_t stride = (total + kMaxSolveEvents - 1) / kMaxSolveEvents;
  Events ev;
  ev.x.reserve(total / stride + 1);
  ev.y.reserve(total / stride + 1);
  ev.t.reserve(total / stride + 1);
  std::size_t idx = 0;
  for (const auto & e : window) {
    if (idx++ % stride != 0) {
      continue;
    }
    if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
      continue;
    }
    ev.x.push_back(static_cast<float>(e.x()));
    ev.y.push_back(static_cast<float>(e.y()));
    ev.t.push_back(static_cast<float>(e.timestamp() - t_mid_us) * 1e-6f);
  }
  if (ev.size() < 2) {
    return result;
  }

  float t_lo = 0.0f, t_hi = 0.0f;
  for (float t : ev.t) {
    t_lo = std::min(t_lo, t);
    t_hi = std::max(t_hi, t);
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
    clamp_field(F, max_spd);

    tx_prev = tiles;
    ty_prev = tiles;
  }

  // Persist the finest-scale field to warm-start the next window.
  prev_flow_field_ = F;
  prev_flow_tiles_ = tx_prev;

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
      float a = std::atan2(vy, vx);                       // [-pi, pi]
      if (a < 0.0f) a += 2.0f * static_cast<float>(CV_PI);  // [0, 2pi)
      arow[x] = a;
      mrow[x] = std::hypot(vx, vy);
    }
  }
  cv::Mat hsv_parts[3];
  // Map angle [0, 2pi) to OpenCV's 8-bit hue range [0, 180).
  cv::Mat hue = angle * (180.0f / (2.0f * static_cast<float>(CV_PI)));
  hue.convertTo(hsv_parts[0], CV_8U);
  hsv_parts[1] = cv::Mat(h, w, CV_8U, cv::Scalar(255));
  magnitude *= (255.0f / max_spd);
  cv::threshold(magnitude, magnitude, 255.0, 255.0, cv::THRESH_TRUNC);
  magnitude.convertTo(hsv_parts[2], CV_8U);
  cv::Mat hsv;
  cv::merge(hsv_parts, 3, hsv);
  cv::cvtColor(hsv, result.flow, cv::COLOR_HSV2BGR);

  // ── Render the IWE: warp every event by its local flow to t_mid ─────────────
  const int scale = std::max<int>(1, static_cast<int>(flow_iwe_scale_));
  const int iw = (w + scale - 1) / scale;
  const int ih = (h + scale - 1) / scale;
  const float inv_scale = 1.0f / static_cast<float>(scale);
  cv::Mat iwe = cv::Mat::zeros(ih, iw, CV_32F);
  for (const auto & e : window) {
    const float px = static_cast<float>(e.x());
    const float py = static_cast<float>(e.y());
    if (px < 0 || py < 0 || px >= w || py >= h) {
      continue;
    }
    float vx, vy;
    sample_tile_velocity(F, tx_prev, ty_prev, w, h, px, py, vx, vy);
    const float tau = static_cast<float>(e.timestamp() - t_mid_us) * 1e-6f;
    // Time-aware warp Eq. (8): x' = x + (t_k - t_ref) * v.
    const float wx = (px + tau * vx) * inv_scale;
    const float wy = (py + tau * vy) * inv_scale;
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
