/**
 * Event Detector dense optical-flow estimation via contrast maximization.
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
#include <memory>
#include <utility>
#include <vector>

#include <dua_cv_bridge/dua_cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <dv-processing/optimization/contrast_maximization_wrapper.hpp>
#include <dv-processing/optimization/optimization_functor.hpp>

namespace event_detector_cpp
{

namespace
{

constexpr float kTwoPi = 6.283185307179586f;
// Upper bound on events fed to one patch's objective; busier patches are
// strided down to keep per-window cost bounded without changing the estimate.
constexpr std::size_t kMaxEventsPerPatch = 2000;
// The global-velocity objective accumulates into a downscaled image and over a
// strided event subset, since a single global estimate needs neither full
// spatial resolution nor every event.
constexpr int kGlobalIweScale = 4;
constexpr std::size_t kGlobalMaxEvents = 20000;

// Mean gradient magnitude (L1) of the IWE: the contrast objective of
// Shiba-Gallego (Sec. III-B1). Unlike the variance it is sensitive to the
// spatial arrangement of warped events. Higher = sharper.
inline float iwe_grad_l1(const cv::Mat & iwe)
{
  double g = 0.0;
  const int H = iwe.rows;
  const int W = iwe.cols;
  for (int y = 0; y < H - 1; ++y) {
    const float * r0 = iwe.ptr<float>(y);
    const float * r1 = iwe.ptr<float>(y + 1);
    for (int x = 0; x < W - 1; ++x) {
      const float gx = r0[x + 1] - r0[x];
      const float gy = r1[x]     - r0[x];
      g += std::abs(gx) + std::abs(gy);
    }
  }
  return static_cast<float>(g / static_cast<double>((H - 1) * (W - 1)));
}

/**
 * Per-patch contrast-maximization objective for a 2D image-plane velocity.
 *
 * Holds a patch's events as local coordinates (relative to the patch origin)
 * and times relative to the window midpoint. operator() warps them by a
 * candidate velocity (x' = x - v * tau), splats them bilinearly into an Image
 * of Warped Events, and returns 1 / contrast (inverse standard deviation) so
 * that the Levenberg-Marquardt solver, which minimizes, maximizes sharpness.
 * Mirrors the residual convention of dv::optimization::RotationLossFunctor but
 * needs no camera geometry, since the warp is purely in the image plane.
 */
class Flow2DLossFunctor : public dv::optimization::OptimizationFunctor<float>
{
public:
  Flow2DLossFunctor(
    std::vector<float> lx,
    std::vector<float> ly,
    std::vector<float> tau,
    int side,
    int margin,
    float half_window_s)
  : OptimizationFunctor<float>(2, 2),
    lx_(std::move(lx)),
    ly_(std::move(ly)),
    tau_(std::move(tau)),
    extent_(side + 2 * margin),
    margin_(static_cast<float>(margin)),
    half_window_s_(half_window_s),
    iwe_lo_(cv::Mat::zeros(extent_, extent_, CV_32F)),
    iwe_mid_(cv::Mat::zeros(extent_, extent_, CV_32F)),
    iwe_hi_(cv::Mat::zeros(extent_, extent_, CV_32F))
  {
  }

  int operator()(const Eigen::VectorXf & v, Eigen::VectorXf & cost) const override
  {
    iwe_lo_.setTo(0.0f);
    iwe_mid_.setTo(0.0f);
    iwe_hi_.setTo(0.0f);
    const float vx = v(0);
    const float vy = v(1);

    // Warp the same events to three reference times: t_mid, t1 and t_Ne. tau_ is
    // measured from t_mid, so the per-reference offsets are tau, tau+half, tau-half.
    for (std::size_t i = 0; i < lx_.size(); ++i) {
      splat(iwe_mid_, lx_[i] - vx * tau_[i],
                      ly_[i] - vy * tau_[i]);
      splat(iwe_lo_,  lx_[i] - vx * (tau_[i] + half_window_s_),
                      ly_[i] - vy * (tau_[i] + half_window_s_));
      splat(iwe_hi_,  lx_[i] - vx * (tau_[i] - half_window_s_),
                      ly_[i] - vy * (tau_[i] - half_window_s_));
    }

    // Multi-reference focus loss: average the contrasts with 1:2:1 weights
    // (Sec. III-B). Invert once at the end so LM (which minimizes) maximizes it.
    const float f = (iwe_grad_l1(iwe_lo_) +
                     2.0f * iwe_grad_l1(iwe_mid_) +
                     iwe_grad_l1(iwe_hi_)) * 0.25f;
    cost(0) = 1.0f / (f + 1e-6f);
    cost(1) = 0.0f;
    return 0;
  }

private:
  void splat(cv::Mat & iwe, float lwx, float lwy) const
  {
    const float wx = lwx + margin_;
    const float wy = lwy + margin_;
    const int x0 = static_cast<int>(std::floor(wx));
    const int y0 = static_cast<int>(std::floor(wy));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= extent_ || y0 + 1 >= extent_) {
      return;
    }
    const float fx = wx - static_cast<float>(x0);
    const float fy = wy - static_cast<float>(y0);
    float * r0 = iwe.ptr<float>(y0);
    float * r1 = iwe.ptr<float>(y0 + 1);
    r0[x0]     += (1.0f - fx) * (1.0f - fy);
    r0[x0 + 1] += fx * (1.0f - fy);
    r1[x0]     += (1.0f - fx) * fy;
    r1[x0 + 1] += fx * fy;
  }

  std::vector<float> lx_;
  std::vector<float> ly_;
  std::vector<float> tau_;
  int extent_;
  float margin_;
  float half_window_s_;
  mutable cv::Mat iwe_lo_;
  mutable cv::Mat iwe_mid_;
  mutable cv::Mat iwe_hi_;
};

/**
 * Whole-frame contrast-maximization objective for a single global velocity.
 *
 * Same warp + bilinear splat + inverse-contrast objective as Flow2DLossFunctor,
 * but over absolute image coordinates accumulated into a downscaled rectangular
 * Image of Warped Events, so one global estimate stays cheap.
 */
class GlobalWarpFunctor : public dv::optimization::OptimizationFunctor<float>
{
public:
  GlobalWarpFunctor(
    std::vector<float> ax,
    std::vector<float> ay,
    std::vector<float> tau,
    int iwe_w,
    int iwe_h,
    float inv_scale,
    float margin)
  : OptimizationFunctor<float>(2, 2),
    ax_(std::move(ax)),
    ay_(std::move(ay)),
    tau_(std::move(tau)),
    iwe_w_(iwe_w),
    iwe_h_(iwe_h),
    inv_scale_(inv_scale),
    margin_(margin)
  {
  }

  int operator()(const Eigen::VectorXf & v, Eigen::VectorXf & cost) const override
  {
    cv::Mat iwe = cv::Mat::zeros(iwe_h_, iwe_w_, CV_32F);
    const float vx = v(0);
    const float vy = v(1);

    for (std::size_t i = 0; i < ax_.size(); ++i) {
      const float wx = (ax_[i] - vx * tau_[i]) * inv_scale_ + margin_;
      const float wy = (ay_[i] - vy * tau_[i]) * inv_scale_ + margin_;
      const int x0 = static_cast<int>(std::floor(wx));
      const int y0 = static_cast<int>(std::floor(wy));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= iwe_w_ || y0 + 1 >= iwe_h_) {
        continue;
      }
      const float fx = wx - static_cast<float>(x0);
      const float fy = wy - static_cast<float>(y0);
      float * r0 = iwe.ptr<float>(y0);
      float * r1 = iwe.ptr<float>(y0 + 1);
      r0[x0]     += (1.0f - fx) * (1.0f - fy);
      r0[x0 + 1] += fx * (1.0f - fy);
      r1[x0]     += (1.0f - fx) * fy;
      r1[x0 + 1] += fx * fy;
    }

    cost(0) = 1.0f / (iwe_grad_l1(iwe) + 1e-6f);
    cost(1) = 0.0f;
    return 0;
  }

private:
  std::vector<float> ax_;
  std::vector<float> ay_;
  std::vector<float> tau_;
  int iwe_w_;
  int iwe_h_;
  float inv_scale_;
  float margin_;
};

}  // namespace

cv::Mat EventDetector::compute_optical_flow(
  const dv::EventStore & window,
  const cv::Vec2f & seed)
{
  if (window.isEmpty() || res_.width <= 0 || res_.height <= 0) {
    return cv::Mat();
  }

  const int patch     = static_cast<int>(flow_patch_size_);
  const int w         = res_.width;
  const int h         = res_.height;
  const int gw        = (w + patch - 1) / patch;
  const int gh        = (h + patch - 1) / patch;
  const float max_spd = static_cast<float>(flow_max_speed_px_s_);

  // Warp reference is the window midpoint, halving the maximum warp distance vs
  // warping to an endpoint; the IWE margin is sized to contain a max-speed warp.
  const int64_t t_lo  = window.getLowestTime();
  const int64_t t_hi  = window.getHighestTime();
  const int64_t t_ref = t_lo + (t_hi - t_lo) / 2;
  const float half_window_s = static_cast<float>(t_hi - t_lo) * 0.5e-6f;

  // Multi-reference warps reach t1 and t_Ne, spanning the full window, so size
  // the margin for a full-window max-speed warp (twice the half-window distance).
  // Estimate a flow grid at a given patch size, optionally seeded by a coarser
  // grid (bilinearly upsampled to this scale) and by the previous-window field.
  auto estimate_scale =
    [&](int patch_px, const cv::Mat & seed_field) -> cv::Mat
  {
    const int g_w = (w + patch_px - 1) / patch_px;
    const int g_h = (h + patch_px - 1) / patch_px;

    int mrg = static_cast<int>(std::ceil(max_spd * 2.0f * half_window_s));
    mrg = std::clamp(mrg, 4, 4 * patch_px);

    const int n_patches = g_w * g_h;
    std::vector<std::vector<float>> bx(n_patches), by(n_patches), bt(n_patches);
    for (const auto & ev : window) {
      const int ex = ev.x();
      const int ey = ev.y();
      if (ex < 0 || ey < 0 || ex >= w || ey >= h) {
        continue;
      }
      const int gx = ex / patch_px;
      const int gy = ey / patch_px;
      const int bi = gy * g_w + gx;
      bx[bi].push_back(static_cast<float>(ex - gx * patch_px));
      by[bi].push_back(static_cast<float>(ey - gy * patch_px));
      bt[bi].push_back(static_cast<float>(ev.timestamp() - t_ref) * 1e-6f);
    }

    cv::Mat up_seed;
    if (!seed_field.empty()) {
      cv::resize(seed_field, up_seed, cv::Size(g_w, g_h), 0.0, 0.0, cv::INTER_LINEAR);
    }
    const bool have_prev = (prev_flow_.rows == g_h && prev_flow_.cols == g_w);
    const int min_ev = static_cast<int>(flow_min_events_);

    cv::Mat field(g_h, g_w, CV_32FC2, cv::Scalar(0.0f, 0.0f));

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < n_patches; ++bi) {
      if (static_cast<int>(bx[bi].size()) < min_ev) {
        continue;
      }
      const int gx = bi % g_w;
      const int gy = bi / g_w;

      std::vector<float> lx, ly, tau;
      const std::size_t count = bx[bi].size();
      const std::size_t stride = (count + kMaxEventsPerPatch - 1) / kMaxEventsPerPatch;
      lx.reserve(count / stride + 1);
      ly.reserve(count / stride + 1);
      tau.reserve(count / stride + 1);
      for (std::size_t i = 0; i < count; i += stride) {
        lx.push_back(bx[bi][i]);
        ly.push_back(by[bi][i]);
        tau.push_back(bt[bi][i]);
      }

      auto functor = std::make_unique<Flow2DLossFunctor>(
        std::move(lx), std::move(ly), std::move(tau), patch_px, mrg, half_window_s);

      Eigen::VectorXf cand(2), cost(2), best(2);
      best << 0.0f, 0.0f;
      (*functor)(best, cost);
      float best_cost = cost(0);

      auto try_cand = [&](float cx, float cy) {
        cand << cx, cy;
        (*functor)(cand, cost);
        if (cost(0) < best_cost) {
          best_cost = cost(0);
          best = cand;
        }
      };

      // Seeds: upsampled coarser-scale estimate, previous-window field, global seed.
      if (!up_seed.empty()) {
        const cv::Vec2f sv = up_seed.at<cv::Vec2f>(gy, gx);
        try_cand(sv[0], sv[1]);
      }
      if (have_prev) {
        const cv::Vec2f pv = prev_flow_.at<cv::Vec2f>(gy, gx);
        try_cand(pv[0], pv[1]);
      }
      try_cand(seed[0], seed[1]);
      // Coarse velocity grid only at the coarsest scale, where there is nothing
      // better to start from; finer scales are seeded from the coarser result.
      if (seed_field.empty()) {
        const float speeds[] = {0.34f, 0.67f, 1.0f};
        for (float frac : speeds) {
          const float s = frac * max_spd;
          for (int d = 0; d < 8; ++d) {
            const float ang = static_cast<float>(d) * kTwoPi / 8.0f;
            try_cand(s * std::cos(ang), s * std::sin(ang));
          }
        }
      }

      dv::optimization::ContrastMaximizationWrapper<Flow2DLossFunctor> cmax(
        std::move(functor),
        static_cast<float>(flow_cmax_learning_rate_),
        0.0f, 0.000345267f, 0.0f, 0.000345267f,
        static_cast<int>(flow_cmax_max_iter_));

      Eigen::VectorXf v = cmax.optimize(best).optimizedVariable;
      float vx = v(0);
      float vy = v(1);
      const float mag = std::hypot(vx, vy);
      if (!std::isfinite(mag)) {
        vx = best(0);
        vy = best(1);
      } else if (mag > max_spd) {
        const float sc = max_spd / mag;
        vx *= sc;
        vy *= sc;
      }
      field.at<cv::Vec2f>(gy, gx) = cv::Vec2f(vx, vy);
    }
    return field;
  };

  // Coarse-to-fine: start a few scales above the target patch and refine down.
  std::vector<int> patch_scales;
  for (int p = patch * 4; p > patch; p /= 2) {
    patch_scales.push_back(p);
  }
  patch_scales.push_back(patch);

  cv::Mat flow_field;
  for (int p : patch_scales) {
    flow_field = estimate_scale(p, flow_field);
  }

  // Light surrogate for the TV regularizer (Sec. III-E): a 3x3 median per
  // channel suppresses isolated outlier patches. cv::medianBlur supports CV_32F
  // only for aperture 3/5 and not for 2-channel mats, hence the split/merge.
  {
    cv::Mat ch[2];
    cv::split(flow_field, ch);
    cv::medianBlur(ch[0], ch[0], 3);
    cv::medianBlur(ch[1], ch[1], 3);
    cv::merge(ch, 2, flow_field);
  }

  // Render the dense field as an HSV image: hue = direction, value = speed.
  cv::Mat flow_parts[2];
  cv::split(flow_field, flow_parts);
  cv::Mat magnitude, angle;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  cv::Mat hsv_parts[3];
  angle *= 0.5;
  angle.convertTo(hsv_parts[0], CV_8U);
  hsv_parts[1] = cv::Mat(gh, gw, CV_8U, cv::Scalar(255));
  magnitude *= (255.0f / max_spd);
  cv::threshold(magnitude, magnitude, 255.0, 255.0, cv::THRESH_TRUNC);
  magnitude.convertTo(hsv_parts[2], CV_8U);
  cv::Mat hsv;
  cv::merge(hsv_parts, 3, hsv);

  cv::Mat bgr_grid;
  cv::cvtColor(hsv, bgr_grid, cv::COLOR_HSV2BGR);
  cv::Mat flow_img;
  cv::resize(bgr_grid, flow_img, cv::Size(w, h), 0.0, 0.0, cv::INTER_NEAREST);

  // Persist this field to warm-start the next window.
  prev_flow_ = std::move(flow_field);

  return flow_img;
}

cv::Mat EventDetector::compute_iwe(
  const dv::EventStore & window,
  cv::Vec2f & v_global_out)
{
  v_global_out = cv::Vec2f(0.0f, 0.0f);
  if (window.isEmpty() || res_.width <= 0 || res_.height <= 0) {
    return cv::Mat();
  }

  const int w         = res_.width;
  const int h         = res_.height;
  const float max_spd = static_cast<float>(flow_max_speed_px_s_);

  const int64_t t_lo  = window.getLowestTime();
  const int64_t t_hi  = window.getHighestTime();
  const int64_t t_ref = t_lo + (t_hi - t_lo) / 2;
  const float half_window_s = static_cast<float>(t_hi - t_lo) * 0.5e-6f;

  // ── Global velocity via contrast maximization (downscaled, subsampled) ──────
  const float inv_scale = 1.0f / static_cast<float>(kGlobalIweScale);
  // Margin (in downscaled pixels) sized to contain a max-speed warp.
  int gmargin = static_cast<int>(std::ceil(max_spd * half_window_s * inv_scale));
  gmargin = std::clamp(gmargin, 4, 256);
  const int iwe_w = (w + kGlobalIweScale - 1) / kGlobalIweScale + 2 * gmargin;
  const int iwe_h = (h + kGlobalIweScale - 1) / kGlobalIweScale + 2 * gmargin;

  // Subsample events to bound the global objective's cost.
  std::vector<float> ax, ay, tau;
  const std::size_t total = static_cast<std::size_t>(window.size());
  const std::size_t stride = (total + kGlobalMaxEvents - 1) / kGlobalMaxEvents;
  ax.reserve(total / stride + 1);
  ay.reserve(total / stride + 1);
  tau.reserve(total / stride + 1);
  std::size_t k = 0;
  for (const auto & ev : window) {
    if (k++ % stride != 0) {
      continue;
    }
    ax.push_back(static_cast<float>(ev.x()));
    ay.push_back(static_cast<float>(ev.y()));
    tau.push_back(static_cast<float>(ev.timestamp() - t_ref) * 1e-6f);
  }

  cv::Vec2f v_global(0.0f, 0.0f);
  if (ax.size() >= 2) {
    auto functor = std::make_unique<GlobalWarpFunctor>(
      std::move(ax), std::move(ay), std::move(tau),
      iwe_w, iwe_h, inv_scale, static_cast<float>(gmargin));

    // Coarse seed search (as in estimate_flow), warm-started by the previous
    // global velocity, then Levenberg-Marquardt refinement.
    Eigen::VectorXf cand(2), cost(2), best(2);
    best << 0.0f, 0.0f;
    (*functor)(best, cost);
    float best_cost = cost(0);

    auto try_cand = [&](float cx, float cy) {
      cand << cx, cy;
      (*functor)(cand, cost);
      if (cost(0) < best_cost) {
        best_cost = cost(0);
        best = cand;
      }
    };

    try_cand(prev_global_v_[0], prev_global_v_[1]);
    const float speeds[] = {0.34f, 0.67f, 1.0f};
    for (float frac : speeds) {
      const float s = frac * max_spd;
      for (int d = 0; d < 8; ++d) {
        const float ang = static_cast<float>(d) * kTwoPi / 8.0f;
        try_cand(s * std::cos(ang), s * std::sin(ang));
      }
    }

    dv::optimization::ContrastMaximizationWrapper<GlobalWarpFunctor> cmax(
      std::move(functor),
      static_cast<float>(flow_cmax_learning_rate_),
      0.0f,
      0.000345267f,
      0.0f,
      0.000345267f,
      static_cast<int>(flow_cmax_max_iter_));

    Eigen::VectorXf v = cmax.optimize(best).optimizedVariable;
    float vx = v(0);
    float vy = v(1);
    const float mag = std::hypot(vx, vy);
    if (!std::isfinite(mag)) {
      vx = best(0);
      vy = best(1);
    } else if (mag > max_spd) {
      const float scale = max_spd / mag;
      vx *= scale;
      vy *= scale;
    }
    v_global = cv::Vec2f(vx, vy);
  }

  prev_global_v_ = v_global;
  v_global_out = v_global;

  // ── Full-resolution IWE: warp every event by the global velocity ────────────
  cv::Mat iwe = cv::Mat::zeros(h, w, CV_32F);
  const float vx = v_global[0];
  const float vy = v_global[1];
  for (const auto & ev : window) {
    const float tau_s = static_cast<float>(ev.timestamp() - t_ref) * 1e-6f;
    const float wx = static_cast<float>(ev.x()) - vx * tau_s;
    const float wy = static_cast<float>(ev.y()) - vy * tau_s;
    const int x0 = static_cast<int>(std::floor(wx));
    const int y0 = static_cast<int>(std::floor(wy));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) {
      continue;
    }
    const float fx = wx - static_cast<float>(x0);
    const float fy = wy - static_cast<float>(y0);
    float * r0 = iwe.ptr<float>(y0);
    float * r1 = iwe.ptr<float>(y0 + 1);
    r0[x0]     += (1.0f - fx) * (1.0f - fy);
    r0[x0 + 1] += fx * (1.0f - fy);
    r1[x0]     += (1.0f - fx) * fy;
    r1[x0 + 1] += fx * fy;
  }

  cv::Mat iwe_img;
  cv::normalize(iwe, iwe_img, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
  return iwe_img;
}

}  // namespace event_detector_cpp
