/**
 * Event Detector dense optical-flow estimation via moment-flow alignment.
 *
 * Maintains spatio-temporal moments on a cell grid and solves a coarse-to-fine
 * closed-form moment-domain surrogate of contrast maximization based on
 * minimizing warped event-cloud dispersion. The dense flow and the Image of
 * Warped Events at the window midpoint are rendered for publishing.
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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace event_detector_cpp
{

namespace
{

using event_detector_cpp::flow::Events;
using event_detector_cpp::flow::MomentFlowParams;

// Busier windows are strided down so the per-window moment update stays bounded.
constexpr std::size_t kMaxSolveEvents = 2000000;

struct EventSample
{
  float x;
  float y;
  int64_t t_us;
};

struct IweRenderStats
{
  size_t input = 0;
  size_t accepted = 0;
  size_t dropped = 0;
};

using ProfileClock = std::chrono::steady_clock;

double elapsed_ms(
  const ProfileClock::time_point & start,
  const ProfileClock::time_point & end = ProfileClock::now())
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * Bilinearly sample the tile-grid flow field at a pixel, matching the legacy
 * stencil convention: tile centers at (i+0.5)*spacing, replicated at borders.
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

double iwe_contrast(const cv::Mat & iwe)
{
  if (iwe.rows < 2 || iwe.cols < 2 || iwe.type() != CV_32F) {
    return 0.0;
  }
  double acc = 0.0;
  for (int y = 0; y < iwe.rows - 1; ++y) {
    const float * row = iwe.ptr<float>(y);
    const float * next = iwe.ptr<float>(y + 1);
    for (int x = 0; x < iwe.cols - 1; ++x) {
      acc += std::abs(static_cast<double>(row[x + 1] - row[x]));
      acc += std::abs(static_cast<double>(next[x] - row[x]));
    }
  }
  return acc / static_cast<double>((iwe.rows - 1) * (iwe.cols - 1));
}

IweRenderStats render_iwe_bilinear(
  const Events & events,
  const Eigen::VectorXf & F,
  const Eigen::VectorXf * A,
  int tiles,
  int img_w,
  int img_h,
  int scale,
  bool warp,
  float t_ref_target_s,
  cv::Mat & iwe)
{
  IweRenderStats stats;
  stats.input = events.size();
  const int iw = (img_w + scale - 1) / scale;
  const int ih = (img_h + scale - 1) / scale;
  const float inv_scale = 1.0f / static_cast<float>(scale);
  iwe = cv::Mat::zeros(ih, iw, CV_32F);

  for (size_t k = 0; k < events.size(); ++k) {
    float wx = events.x[k];
    float wy = events.y[k];
    if (warp) {
      const float t = events.t[k];
      const float ox = wx;
      const float oy = wy;
      float vx, vy;
      sample_tile_velocity(F, tiles, tiles, img_w, img_h, ox, oy, vx, vy);
      wx += (t - t_ref_target_s) * vx;
      wy += (t - t_ref_target_s) * vy;
      if (A != nullptr) {
        float ax, ay;
        sample_tile_velocity(*A, tiles, tiles, img_w, img_h, ox, oy, ax, ay);
        const float dq = t * t - t_ref_target_s * t_ref_target_s;
        wx += 0.5f * dq * ax;
        wy += 0.5f * dq * ay;
      }
    }
    wx *= inv_scale;
    wy *= inv_scale;

    const int x0 = static_cast<int>(std::floor(wx));
    const int y0 = static_cast<int>(std::floor(wy));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= iw || y0 + 1 >= ih) {
      stats.dropped += 1;
      continue;
    }
    const float fx = wx - x0;
    const float fy = wy - y0;
    float * r0 = iwe.ptr<float>(y0);
    float * r1 = iwe.ptr<float>(y0 + 1);
    r0[x0]     += (1.0f - fx) * (1.0f - fy);
    r0[x0 + 1] += fx * (1.0f - fy);
    r1[x0]     += (1.0f - fx) * fy;
    r1[x0 + 1] += fx * fy;
    stats.accepted += 1;
  }
  return stats;
}

cv::Mat make_iwe_support_mask(const cv::Mat & support_iwe)
{
  if (support_iwe.empty() || support_iwe.type() != CV_32F) {
    return cv::Mat();
  }

  cv::Mat support_mask(support_iwe.rows, support_iwe.cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < support_iwe.rows; ++y) {
    const float * support_row = support_iwe.ptr<float>(y);
    uint8_t * mask_row = support_mask.ptr<uint8_t>(y);
    for (int x = 0; x < support_iwe.cols; ++x) {
      if (std::isfinite(support_row[x]) && support_row[x] > 0.0f) {
        mask_row[x] = 255;
      }
    }
  }
  return support_mask;
}

cv::Mat mask_flow_by_support(const cv::Mat & flow_dense, const cv::Mat & support_mask)
{
  if (flow_dense.empty() || support_mask.empty() ||
    flow_dense.type() != CV_32FC2 || support_mask.type() != CV_8U ||
    flow_dense.size() != support_mask.size())
  {
    return cv::Mat();
  }

  const float nan = std::numeric_limits<float>::quiet_NaN();
  cv::Mat flow_events(flow_dense.rows, flow_dense.cols, CV_32FC2, cv::Scalar(nan, nan));
  for (int y = 0; y < flow_dense.rows; ++y) {
    const cv::Vec2f * dense_row = flow_dense.ptr<cv::Vec2f>(y);
    const uint8_t * mask_row = support_mask.ptr<uint8_t>(y);
    cv::Vec2f * events_row = flow_events.ptr<cv::Vec2f>(y);
    for (int x = 0; x < flow_dense.cols; ++x) {
      if (mask_row[x] != 0) {
        events_row[x] = dense_row[x];
      }
    }
  }
  return flow_events;
}

cv::Mat mask_debug_by_support(const cv::Mat & flow_debug, const cv::Mat & support_mask)
{
  if (flow_debug.empty() || support_mask.empty() ||
    flow_debug.type() != CV_8UC3 || support_mask.type() != CV_8U ||
    flow_debug.size() != support_mask.size())
  {
    return cv::Mat();
  }

  cv::Mat flow_events_debug(flow_debug.rows, flow_debug.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  flow_debug.copyTo(flow_events_debug, support_mask);
  return flow_events_debug;
}

}  // namespace

EventDetector::FlowResult EventDetector::solve_flow_moment(
  const dv::EventStore & window,
  std::optional<int64_t> t_ref_override_us)
{
  const auto t_total = ProfileClock::now();
  FlowResult result;
  if (window.isEmpty() || res_.width <= 0 || res_.height <= 0) {
    return result;
  }

  const int w = res_.width;
  const int h = res_.height;
  const int n_scales = std::max<int>(1, static_cast<int>(flow_num_scales_));
  const int final_tiles = 1 << (n_scales - 1);
  const float vis_speed_cap = static_cast<float>(flow_max_speed_px_s_);

  const auto t_select = ProfileClock::now();
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
  const double select_ms = elapsed_ms(t_select);
  if (solve_samples.size() < 2) {
    RCLCPP_INFO(
      get_logger(),
      "Flow profile: skipped moment flow, raw_events=%zu solve_events=%zu "
      "stride=%zu select=%.3f ms",
      total, solve_samples.size(), stride, select_ms);
    return result;
  }

  const auto t_pack = ProfileClock::now();
  auto by_time = [](const EventSample & a, const EventSample & b) {
    return a.t_us < b.t_us;
  };
  const auto t_minmax = std::minmax_element(
    solve_samples.begin(), solve_samples.end(), by_time);
  const int64_t t_lo_us = t_minmax.first->t_us;
  const int64_t t_hi_us = t_minmax.second->t_us;
  const int64_t t_ref_us = t_ref_override_us.value_or(t_lo_us + (t_hi_us - t_lo_us) / 2);

  Events ev;
  ev.t_ref_us = t_ref_us;
  ev.x.reserve(solve_samples.size());
  ev.y.reserve(solve_samples.size());
  ev.t.reserve(solve_samples.size());
  for (const EventSample & e : solve_samples) {
    ev.x.push_back(e.x);
    ev.y.push_back(e.y);
    ev.t.push_back(static_cast<float>(e.t_us - t_ref_us) * 1e-6f);
  }
  const double pack_ms = elapsed_ms(t_pack);

  MomentFlowParams params;
  params.num_scales = n_scales;
  params.cell_size_px = static_cast<int>(flow_cell_size_px_);
  params.decay_enabled = flow_decay_enabled_;
  params.decay_tau_us = static_cast<int>(flow_decay_tau_us_);
  params.tau_adaptive = false;
  params.cell_min_mass = static_cast<float>(flow_cell_min_mass_);
  params.cell_min_lambda = static_cast<float>(flow_cell_min_lambda_);
  params.cell_max_residual_ratio = static_cast<float>(flow_cell_max_residual_ratio_);
  params.tile_min_mass = static_cast<float>(flow_tile_min_mass_);
  params.tile_min_cells = static_cast<int>(flow_tile_min_cells_);
  params.tile_min_lambda = static_cast<float>(flow_tile_min_lambda_);
  params.aperture_ratio = static_cast<float>(flow_aperture_ratio_);
  params.tikhonov_eps = static_cast<float>(flow_tikhonov_eps_);
  params.prior_lambda = static_cast<float>(flow_prior_lambda_);
  params.flow_reg_lambda = static_cast<float>(flow_reg_lambda_);
  params.flow_reg_sweeps = static_cast<int>(flow_reg_sweeps_);
  params.flow_reg_sigma = static_cast<float>(flow_reg_sigma_);
  params.smooth_iters = static_cast<int>(flow_smooth_iters_);
  params.smooth_alpha = static_cast<float>(flow_smooth_alpha_);
  params.refine_enabled = false;
  params.refine_iters = static_cast<int>(flow_refine_iters_);
  params.refine_huber_delta = static_cast<float>(flow_refine_huber_delta_);
  params.max_speed_px_s = static_cast<float>(flow_max_speed_px_s_);
  params.time_aware_order = static_cast<int>(flow_time_aware_order_);

  if (flow_tau_adaptive_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "flow_tau_adaptive is reserved and currently behaves as disabled");
  }
  if (flow_refine_enabled_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "flow_refine_enabled is reserved for Stage C and currently behaves as disabled");
  }

  if (!moment_flow_.has_value() || !moment_flow_->compatible(w, h, params)) {
    moment_flow_.emplace(w, h, params);
  }

  Eigen::VectorXf warm_start;
  const bool have_prev = (prev_flow_tiles_ > 0 &&
    prev_flow_field_.size() == 2 * prev_flow_tiles_ * prev_flow_tiles_);
  if (have_prev) {
    warm_start = (prev_flow_tiles_ == final_tiles)
      ? prev_flow_field_
      : upsample_field(
          prev_flow_field_, prev_flow_tiles_, prev_flow_tiles_,
          final_tiles, final_tiles, w, h);
  }

  const auto t_moment = ProfileClock::now();
  moment_flow_->reset();
  moment_flow_->ingest(ev);
  Eigen::VectorXf F(moment_flow_->num_vars());
  moment_flow_->solve(warm_start, F);
  sanitize_field(F);
  {
    const int n_tiles = final_tiles * final_tiles;
    std::vector<float> conf;
    moment_flow_->final_tile_confidence(conf);
    if (static_cast<int>(flow_track_w_.size()) != n_tiles) {
      flow_track_vx_.assign(n_tiles, 0.0f);
      flow_track_vy_.assign(n_tiles, 0.0f);
      flow_track_w_.assign(n_tiles, 0.0f);
    }
    const float gamma = std::clamp(static_cast<float>(flow_track_gamma_), 0.0f, 1.0f);
    const float w_floor = 1e-3f;
    for (int k = 0; k < n_tiles; ++k) {
      const float w_meas = std::max(w_floor, conf[k]);
      const float vx_m   = -F[2 * k];
      const float vy_m   = -F[2 * k + 1];
      const float W_prev = gamma * flow_track_w_[k];
      const float Wsum   = W_prev + w_meas;
      flow_track_vx_[k]  = (W_prev * flow_track_vx_[k] + w_meas * vx_m) / Wsum;
      flow_track_vy_[k]  = (W_prev * flow_track_vy_[k] + w_meas * vy_m) / Wsum;
      flow_track_w_[k]   = Wsum;
      F[2 * k]     = -flow_track_vx_[k];
      F[2 * k + 1] = -flow_track_vy_[k];
    }
  }
  const double moment_ms = elapsed_ms(t_moment);
  const auto profile = moment_flow_->profile();

  const auto t_render_events = ProfileClock::now();
  Events render_ev;
  render_ev.t_ref_us = t_ref_us;
  render_ev.x.reserve(total);
  render_ev.y.reserve(total);
  render_ev.t.reserve(total);
  for (const auto & e : window) {
    if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
      continue;
    }
    render_ev.x.push_back(static_cast<float>(e.x()));
    render_ev.y.push_back(static_cast<float>(e.y()));
    render_ev.t.push_back(static_cast<float>(e.timestamp() - t_ref_us) * 1e-6f);
  }
  const double render_events_ms = elapsed_ms(t_render_events);

  // ---- Field-level multi-reference focus diagnostic ----
  // Warp to explicit target times in the event window coordinate system,
  // combine gradient-magnitude contrasts: f = (G_lo + 2 G_mid + G_hi)/(4 G_id).
  // f <= 1 means the field did not improve this particular focus metric. Keep
  // the candidate visible anyway; otherwise a strict gate can black out the
  // first frame and keep every later warm start at zero.
  const Eigen::VectorXf * accel_ptr =
    (params.time_aware_order >= 2) ? &moment_flow_->acceleration() : nullptr;
  const int iwe_scale = std::max<int>(1, static_cast<int>(flow_iwe_scale_));
  const float t_lo_ref_s = static_cast<float>(t_lo_us - t_ref_us) * 1e-6f;
  const float t_hi_ref_s = static_cast<float>(t_hi_us - t_ref_us) * 1e-6f;

  cv::Mat focus_id, focus_mid, focus_lo, focus_hi;
  render_iwe_bilinear(
    render_ev, F, nullptr, final_tiles, w, h, iwe_scale, false, 0.0f, focus_id);
  render_iwe_bilinear(
    render_ev, F, accel_ptr, final_tiles, w, h, iwe_scale, true, 0.0f, focus_mid);
  render_iwe_bilinear(
    render_ev, F, accel_ptr, final_tiles, w, h, iwe_scale, true, t_lo_ref_s, focus_lo);
  render_iwe_bilinear(
    render_ev, F, accel_ptr, final_tiles, w, h, iwe_scale, true, t_hi_ref_s, focus_hi);
  // Integer event coords keep the identity IWE on single pixels (no sub-pixel
  // spread) -> artificially high L1-gradient, while any warp produces fractional
  // coords that bilinear splatting smooths. Equalize with the same sigma=1px blur
  // (delta ~= Gaussian(1px)) so focus_f measures alignment, not warp.
  auto focus_contrast = [](const cv::Mat & iwe) {
    cv::Mat b;
    cv::GaussianBlur(iwe, b, cv::Size(0, 0), 1.0);
    return iwe_contrast(b);
  };
  const double g_id  = std::max(focus_contrast(focus_id), 1e-12);
  const double g_mid = focus_contrast(focus_mid);
  const double g_lo  = focus_contrast(focus_lo);
  const double g_hi  = focus_contrast(focus_hi);
  const double focus_f = (g_lo + 2.0 * g_mid + g_hi) / (4.0 * g_id);

  const bool flow_rejected = !(focus_f > 1.0);

  prev_flow_field_ = F;
  prev_flow_tiles_ = final_tiles;

  const auto t_flow_image = ProfileClock::now();
  cv::Mat angle(h, w, CV_32F);
  cv::Mat magnitude(h, w, CV_32F);
  result.flow_dense = cv::Mat(h, w, CV_32FC2);
  double vx_sum = 0.0;
  double vy_sum = 0.0;
  double vx2_sum = 0.0;
  double vy2_sum = 0.0;
  for (int y = 0; y < h; ++y) {
    float * arow = angle.ptr<float>(y);
    float * mrow = magnitude.ptr<float>(y);
    cv::Vec2f * vrow = result.flow_dense.ptr<cv::Vec2f>(y);
    for (int x = 0; x < w; ++x) {
      float vx, vy;
      sample_tile_velocity(
        F, final_tiles, final_tiles, w, h,
        static_cast<float>(x), static_cast<float>(y), vx, vy);
      vx = -vx;
      vy = -vy;
      vrow[x] = cv::Vec2f(vx, vy);
      vx_sum += vx;
      vy_sum += vy;
      vx2_sum += static_cast<double>(vx) * vx;
      vy2_sum += static_cast<double>(vy) * vy;
      float a = std::atan2(vy, vx);
      if (a < 0.0f) {
        a += 2.0f * static_cast<float>(CV_PI);
      }
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
  const double display_floor_speed = std::max(25.0, 0.10 * static_cast<double>(vis_speed_cap));
  const double raw_display_max = (robust_max_speed > 1e-6)
    ? robust_max_speed
    : ((observed_max_speed > 1e-6) ? observed_max_speed : static_cast<double>(vis_speed_cap));
  const double display_max_speed = std::clamp(
    raw_display_max, display_floor_speed, static_cast<double>(vis_speed_cap));
  const double support_mean_speed = support_speeds.empty()
    ? 0.0
    : support_speed_sum / static_cast<double>(support_speeds.size());
  const double n_pix = static_cast<double>(std::max(1, w * h));
  const double vx_mean = vx_sum / n_pix;
  const double vy_mean = vy_sum / n_pix;
  const double flow_var =
    (vx2_sum + vy2_sum) / n_pix - (vx_mean * vx_mean + vy_mean * vy_mean);

  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "Flow speed [px/s]: mean=%.3f max=%.3f display_max=%.3f var=%.3f pixels=%zu",
    support_mean_speed, support_max_speed, display_max_speed, flow_var, support_speeds.size());

  magnitude *= (255.0f / static_cast<float>(display_max_speed));
  cv::threshold(magnitude, magnitude, 255.0, 255.0, cv::THRESH_TRUNC);
  magnitude.convertTo(hsv_parts[2], CV_8U);
  cv::Mat hsv;
  cv::merge(hsv_parts, 3, hsv);
  cv::cvtColor(hsv, result.flow_dense_debug, cv::COLOR_HSV2BGR);
  const double flow_image_ms = elapsed_ms(t_flow_image);

  const auto t_iwe = ProfileClock::now();
  cv::Mat result_iwe_f;
  if (!flow_rejected) {
    result_iwe_f = focus_mid;   // reuse the t_mid warped IWE
  } else {
    result_iwe_f = focus_id;    // keep the published IWE readable when focus rejects
  }
  cv::normalize(result_iwe_f, result.iwe, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
  cv::Mat support_iwe_f;
  if (iwe_scale == 1) {
    support_iwe_f = result_iwe_f;
  } else {
    const bool support_warp = !flow_rejected;
    const Eigen::VectorXf * support_accel = support_warp ? accel_ptr : nullptr;
    render_iwe_bilinear(
      render_ev, F, support_accel, final_tiles, w, h,
      1, support_warp, 0.0f, support_iwe_f);
  }
  const cv::Mat support_mask = make_iwe_support_mask(support_iwe_f);
  result.flow_events = mask_flow_by_support(result.flow_dense, support_mask);
  result.flow_events_debug = mask_debug_by_support(result.flow_dense_debug, support_mask);
  const double iwe_ms = elapsed_ms(t_iwe);
  const double reg_modified_fraction =
    (profile.reg_total_tiles > 0)
      ? static_cast<double>(profile.reg_modified_tiles) /
        static_cast<double>(profile.reg_total_tiles)
      : 0.0;

  RCLCPP_INFO(
    get_logger(),
    "Flow profile MomentFlow: ingest=%.3f ms decay=%.3f ms stage_a=%.3f ms "
    "stage_b=%.3f ms spatial=%.3f ms solve_total=%.3f ms total=%.3f ms events=%d "
    "active_cells=%d valid_cells=%d reject(residual/speed)=%d/%d "
    "tiles_total(full/aperture/fallback/prior)=%d/%d/%d/%d "
    "tiles_final(full/aperture/fallback)=%d/%d/%d "
    "timeaware_fallback(support/reject)=%d/%d final=%d/%d "
    "reg(modified_frac/mean_delta legacy_geom/warped_geom) %.3f/%.3f %d/%d",
    profile.ingest_ms, profile.decay_ms, profile.stage_a_ms,
    profile.stage_b_ms, profile.smooth_ms, profile.total_solve_ms, moment_ms,
    profile.events_ingested, profile.active_cells, profile.valid_cells,
    profile.residual_reject_cells, profile.speed_reject_cells,
    profile.full_rank_tiles, profile.aperture_tiles, profile.fallback_tiles,
    profile.prior_tiles,
    profile.final_full_rank_tiles, profile.final_aperture_tiles, profile.final_fallback_tiles,
    profile.timeaware_support_fallback_tiles, profile.timeaware_reject_fallback_tiles,
    profile.final_timeaware_support_fallback_tiles,
    profile.final_timeaware_reject_fallback_tiles,
    reg_modified_fraction, profile.reg_mean_delta_speed,
    profile.reg_legacy_geometry_tiles, profile.reg_warped_geometry_tiles);

  RCLCPP_INFO(
    get_logger(),
    "Flow profile fallback detail: "
    "support_causes(all cells/mass/time/lambda/solve)=%d/%d/%d/%d/%d "
    "final=%d/%d/%d/%d/%d "
    "reject_causes(all no_improve/no_focus)=%d/%d final=%d/%d",
    profile.timeaware_low_cells_fallback_tiles,
    profile.timeaware_low_mass_fallback_tiles,
    profile.timeaware_no_time_fallback_tiles,
    profile.timeaware_low_lambda_fallback_tiles,
    profile.timeaware_solve_fail_fallback_tiles,
    profile.final_timeaware_low_cells_fallback_tiles,
    profile.final_timeaware_low_mass_fallback_tiles,
    profile.final_timeaware_no_time_fallback_tiles,
    profile.final_timeaware_low_lambda_fallback_tiles,
    profile.final_timeaware_solve_fail_fallback_tiles,
    profile.timeaware_no_improve_fallback_tiles,
    profile.timeaware_no_focus_fallback_tiles,
    profile.final_timeaware_no_improve_fallback_tiles,
    profile.final_timeaware_no_focus_fallback_tiles);

  RCLCPP_INFO(
    get_logger(),
    "Flow profile IWE: render_events=%zu event_pack=%.3f ms flow_image=%.3f ms "
    "iwe_render_norm=%.3f ms order=%d focus_f=%.4f (lo=%.6f mid=%.6f hi=%.6f id=%.6f) "
    "rejected=%d",
    render_ev.size(), render_events_ms, flow_image_ms, iwe_ms,
    params.time_aware_order, focus_f, g_lo, g_mid, g_hi, g_id,
    flow_rejected ? 1 : 0);

  RCLCPP_INFO(
    get_logger(),
    "Flow profile summary: raw_events=%zu solve_events=%zu stride=%zu span=%.3f ms "
    "decay_enabled=%d decay_tau=%.3f ms setup_select=%.3f ms setup_pack=%.3f ms "
    "moment=%.3f ms render=%.3f ms total=%.3f ms achieved_hz=%.3f",
    total, solve_samples.size(), stride, static_cast<double>(t_hi_us - t_lo_us) * 1e-3,
    flow_decay_enabled_ ? 1 : 0, static_cast<double>(flow_decay_tau_us_) * 1e-3,
    select_ms, pack_ms, moment_ms, render_events_ms + flow_image_ms + iwe_ms,
    elapsed_ms(t_total), 1000.0 / std::max(1e-9, elapsed_ms(t_total)));

  return result;
}

}  // namespace event_detector_cpp
