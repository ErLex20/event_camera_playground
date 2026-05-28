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
    int margin)
  : OptimizationFunctor<float>(2, 2),
    lx_(std::move(lx)),
    ly_(std::move(ly)),
    tau_(std::move(tau)),
    extent_(side + 2 * margin),
    margin_(static_cast<float>(margin))
  {
  }

  int operator()(const Eigen::VectorXf & v, Eigen::VectorXf & cost) const override
  {
    cv::Mat iwe = cv::Mat::zeros(extent_, extent_, CV_32F);
    const float vx = v(0);
    const float vy = v(1);

    for (std::size_t i = 0; i < lx_.size(); ++i) {
      const float wx = lx_[i] - vx * tau_[i] + margin_;
      const float wy = ly_[i] - vy * tau_[i] + margin_;
      const int x0 = static_cast<int>(std::floor(wx));
      const int y0 = static_cast<int>(std::floor(wy));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= extent_ || y0 + 1 >= extent_) {
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

    cv::Scalar mean, stddev;
    cv::meanStdDev(iwe, mean, stddev);
    cost(0) = 1.0f / (static_cast<float>(stddev[0]) + 1e-6f);
    cost(1) = 0.0f;
    return 0;
  }

private:
  std::vector<float> lx_;
  std::vector<float> ly_;
  std::vector<float> tau_;
  int extent_;
  float margin_;
};

}  // namespace

cv::Mat EventDetector::estimate_flow(
  const dv::EventStore & window)
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
  int margin = static_cast<int>(std::ceil(max_spd * half_window_s));
  margin = std::clamp(margin, 4, 4 * patch);

  // Bin events into patches as local coordinates and midpoint-relative times.
  const int num_patches = gw * gh;
  std::vector<std::vector<float>> px(num_patches), py(num_patches), pt(num_patches);
  for (const auto & ev : window) {
    const int ex = ev.x();
    const int ey = ev.y();
    if (ex < 0 || ey < 0 || ex >= w || ey >= h) {
      continue;
    }
    const int gx = ex / patch;
    const int gy = ey / patch;
    const int idx = gy * gw + gx;
    px[idx].push_back(static_cast<float>(ex - gx * patch));
    py[idx].push_back(static_cast<float>(ey - gy * patch));
    pt[idx].push_back(static_cast<float>(ev.timestamp() - t_ref) * 1e-6f);
  }

  cv::Mat flow_field(gh, gw, CV_32FC2, cv::Scalar(0.0f, 0.0f));
  const bool have_prev = (prev_flow_.rows == gh && prev_flow_.cols == gw);
  const int min_events = static_cast<int>(flow_min_events_);

  #pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < num_patches; ++idx) {
    if (static_cast<int>(px[idx].size()) < min_events) {
      continue;
    }
    const int gx = idx % gw;
    const int gy = idx / gw;

    // Stride down very busy patches to bound cost.
    std::vector<float> lx, ly, tau;
    const std::size_t count = px[idx].size();
    const std::size_t stride = (count + kMaxEventsPerPatch - 1) / kMaxEventsPerPatch;
    lx.reserve(count / stride + 1);
    ly.reserve(count / stride + 1);
    tau.reserve(count / stride + 1);
    for (std::size_t i = 0; i < count; i += stride) {
      lx.push_back(px[idx][i]);
      ly.push_back(py[idx][i]);
      tau.push_back(pt[idx][i]);
    }

    auto functor = std::make_unique<Flow2DLossFunctor>(
      std::move(lx), std::move(ly), std::move(tau), patch, margin);

    // Coarse seed search: the numerical-diff step is proportional to |v|, so it
    // vanishes at exactly zero velocity and the solver cannot escape. Seed it
    // with the previous estimate and a small velocity grid, keeping the most
    // contrastful candidate as the LM starting point.
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

    if (have_prev) {
      const cv::Vec2f pv = prev_flow_.at<cv::Vec2f>(gy, gx);
      try_cand(pv[0], pv[1]);
    }
    const float speeds[] = {0.34f, 0.67f, 1.0f};
    for (float frac : speeds) {
      const float s = frac * max_spd;
      for (int d = 0; d < 8; ++d) {
        const float ang = static_cast<float>(d) * kTwoPi / 8.0f;
        try_cand(s * std::cos(ang), s * std::sin(ang));
      }
    }

    dv::optimization::ContrastMaximizationWrapper<Flow2DLossFunctor> cmax(
      std::move(functor),
      static_cast<float>(flow_cmax_learning_rate_),
      0.0f,
      0.000345267f,
      0.0f,
      0.000345267f,
      static_cast<int>(flow_cmax_max_iter_));

    Eigen::VectorXf v = cmax.optimize(best).optimizedVariable;

    // Reject runaway solutions; clamp magnitude to the configured ceiling.
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
    flow_field.at<cv::Vec2f>(gy, gx) = cv::Vec2f(vx, vy);
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

}  // namespace event_detector_cpp
