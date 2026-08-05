/**
 * Moment Flow dense optical-flow estimation via moment-flow alignment.
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

#include "moment_flow/moment_flow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <omp.h>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace moment_flow
{

namespace
{

using moment_flow::flow::Events;
using moment_flow::flow::MomentFlowParams;

// Tile grids at or above this size re-warp the events by the composed field
// before solving (per-scale coarse-to-fine alignment).
constexpr int kRewarpMinTiles = 8;
constexpr float kFocusSplatOffsetIwePx = 0.21f;

struct IweRenderStats
{
  size_t input = 0;
  size_t accepted = 0;
  size_t dropped = 0;
};

using ProfileClock = std::chrono::steady_clock;

/// Resolve the configured thread budget: 0 means "whatever OpenMP offers".
int resolve_threads(int64_t configured)
{
  if (configured > 0) {
    return static_cast<int>(configured);
  }
  return std::max(1, omp_get_max_threads());
}

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

void clamp_field_speed(Eigen::VectorXf & F, float max_speed)
{
  for (int k = 0; k < F.size() / 2; ++k) {
    const float speed = std::hypot(F[2 * k], F[2 * k + 1]);
    if (speed > max_speed) {
      const float s = max_speed / speed;
      F[2 * k] *= s;
      F[2 * k + 1] *= s;
    }
  }
}

/**
 * Confidence-weighted diffusion of the composed tile field. The in-solve
 * regularizer only sees the per-pass residual field; the composed field
 * accumulates speckle across passes/frames and tiles without data keep stale
 * values. Each sweep pulls every tile toward the confidence-weighted mean of
 * its 4-neighborhood: low-confidence tiles adopt their neighbors, data-rich
 * tiles stay put. Confidences are normalized by their positive mean, so the
 * scheme is invariant to the arbitrary scale of the solver confidence.
 */
void smooth_field_confidence(
  Eigen::VectorXf & F,
  const std::vector<float> & conf,
  int tiles,
  int sweeps,
  float beta)
{
  const int n_tiles = tiles * tiles;
  if (static_cast<int>(conf.size()) < n_tiles || tiles <= 1 || sweeps <= 0) {
    return;
  }
  double conf_sum = 0.0;
  int conf_n = 0;
  for (int k = 0; k < n_tiles; ++k) {
    if (std::isfinite(conf[k]) && conf[k] > 0.0f) {
      conf_sum += conf[k];
      conf_n += 1;
    }
  }
  if (conf_n == 0) {
    return;
  }
  const float conf_scale = static_cast<float>(conf_n / conf_sum);
  constexpr float kNeighborFloor = 0.05f;  // lets flow diffuse into empty regions
  constexpr float kConfCap = 10.0f;        // one hot tile must not freeze its patch

  std::vector<float> c(static_cast<size_t>(n_tiles));
  for (int k = 0; k < n_tiles; ++k) {
    const float ck = (std::isfinite(conf[k]) && conf[k] > 0.0f)
      ? conf[k] * conf_scale : 0.0f;
    // sqrt compresses the confidence dynamic range: slow scene regions emit
    // fewer events (low mass -> low confidence) but their estimate is valid;
    // without compression fast neighbors diffuse over them.
    c[static_cast<size_t>(k)] = std::min(std::sqrt(ck), kConfCap);
  }

  Eigen::VectorXf F_prev = F;
  for (int s = 0; s < sweeps; ++s) {
    for (int ty = 0; ty < tiles; ++ty) {
      for (int tx = 0; tx < tiles; ++tx) {
        const int k = ty * tiles + tx;
        float wsum = 0.0f, ax = 0.0f, ay = 0.0f;
        auto add = [&](int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= tiles || ny >= tiles) {
              return;
            }
            const int nk = ny * tiles + nx;
            const float wn = c[static_cast<size_t>(nk)] + kNeighborFloor;
            wsum += wn;
            ax += wn * F_prev[2 * nk];
            ay += wn * F_prev[2 * nk + 1];
          };
        add(tx - 1, ty);
        add(tx + 1, ty);
        add(tx, ty - 1);
        add(tx, ty + 1);
        if (!(wsum > 0.0f)) {
          continue;
        }
        const float wk = c[static_cast<size_t>(k)];
        const float denom = wk + beta * wsum;
        F[2 * k] = (wk * F_prev[2 * k] + beta * ax) / denom;
        F[2 * k + 1] = (wk * F_prev[2 * k + 1] + beta * ay) / denom;
      }
    }
    F_prev = F;
  }
}

/**
 * Warp events to the reference time with the current tile field (and optional
 * acceleration field): x' = x + t*v(x) + 0.5*t^2*a(x), t relative to t_ref.
 * Out-of-bounds warps are dropped, mirroring MomentFlow::ingest.
 */
void warp_events_by_field(
  const Events & src,
  const Eigen::VectorXf & F,
  const Eigen::VectorXf * A,
  int tiles,
  int img_w,
  int img_h,
  int max_threads,
  Events & dst)
{
  const size_t n_src = src.size();
  dst.t_ref_us = src.t_ref_us;
  dst.x.resize(n_src);
  dst.y.resize(n_src);
  dst.t.resize(n_src);

  // Chunked warp: each chunk compacts inside its own input slice (survivors
  // cannot exceed the slice), then the slices are closed up in order, so the
  // output ordering matches the serial version the moments are accumulated in.
  const int n_chunks = std::clamp(
    max_threads, 1, static_cast<int>(std::max<size_t>(1, n_src / 4096)));
  std::vector<size_t> chunk_count(static_cast<size_t>(n_chunks), 0);

  #pragma omp parallel for schedule(static) num_threads(n_chunks)
  for (int c = 0; c < n_chunks; ++c) {
    const size_t begin = static_cast<size_t>(c) * n_src / n_chunks;
    const size_t end = static_cast<size_t>(c + 1) * n_src / n_chunks;
    size_t n = begin;
    for (size_t k = begin; k < end; ++k) {
      const float x = src.x[k];
      const float y = src.y[k];
      const float t = src.t[k];
      float vx, vy;
      sample_tile_velocity(F, tiles, tiles, img_w, img_h, x, y, vx, vy);
      float wx = x + t * vx;
      float wy = y + t * vy;
      if (A != nullptr) {
        float ax, ay;
        sample_tile_velocity(*A, tiles, tiles, img_w, img_h, x, y, ax, ay);
        wx += 0.5f * t * t * ax;
        wy += 0.5f * t * t * ay;
      }
      if (!(wx >= 0.0f && wy >= 0.0f && wx < img_w && wy < img_h)) {
        continue;
      }
      dst.x[n] = wx;
      dst.y[n] = wy;
      dst.t[n] = t;
      ++n;
    }
    chunk_count[static_cast<size_t>(c)] = n - begin;
  }

  size_t n = chunk_count[0];
  for (int c = 1; c < n_chunks; ++c) {
    const size_t begin = static_cast<size_t>(c) * n_src / n_chunks;
    const size_t count = chunk_count[static_cast<size_t>(c)];
    if (count > 0 && begin > n) {
      std::copy(&dst.x[begin], &dst.x[begin] + count, &dst.x[n]);
      std::copy(&dst.y[begin], &dst.y[begin] + count, &dst.y[n]);
      std::copy(&dst.t[begin], &dst.t[begin] + count, &dst.t[n]);
    }
    n += count;
  }

  dst.x.resize(n);
  dst.y.resize(n);
  dst.t.resize(n);
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
  float splat_offset_iwe_px,
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
    wx = wx * inv_scale + splat_offset_iwe_px;
    wy = wy * inv_scale + splat_offset_iwe_px;

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

/**
 * Full-resolution event-support mask: marks the bilinear footprint of every
 * (optionally warped) event. Direct 8U splat — replaces the float IWE render +
 * threshold pass, which dominated the per-window render budget.
 */
cv::Mat render_support_mask(
  const Events & events,
  const Eigen::VectorXf & F,
  const Eigen::VectorXf * A,
  int tiles,
  int img_w,
  int img_h,
  bool warp,
  int max_threads)
{
  // One full-resolution mask per thread, so the team is capped: past a handful
  // of threads the extra allocations and the final OR cost more than the splat
  // work they remove.
  const int n_threads = std::clamp(max_threads, 1, 8);
  std::vector<cv::Mat> masks(static_cast<size_t>(n_threads));
  for (cv::Mat & m : masks) {
    m = cv::Mat(img_h, img_w, CV_8U, cv::Scalar(0));
  }

  #pragma omp parallel num_threads(n_threads)
  {
    cv::Mat & mask = masks[static_cast<size_t>(omp_get_thread_num())];
    #pragma omp for schedule(static)
    for (int64_t k = 0; k < static_cast<int64_t>(events.size()); ++k) {
      float wx = events.x[k];
      float wy = events.y[k];
      if (warp) {
        const float t = events.t[k];
        float vx, vy;
        sample_tile_velocity(F, tiles, tiles, img_w, img_h, wx, wy, vx, vy);
        float px = events.x[k] + t * vx;
        float py = events.y[k] + t * vy;
        if (A != nullptr) {
          float ax, ay;
          sample_tile_velocity(*A, tiles, tiles, img_w, img_h, wx, wy, ax, ay);
          px += 0.5f * t * t * ax;
          py += 0.5f * t * t * ay;
        }
        wx = px;
        wy = py;
      }
      const int x0 = static_cast<int>(std::floor(wx));
      const int y0 = static_cast<int>(std::floor(wy));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= img_w || y0 + 1 >= img_h) {
        continue;
      }
      uint8_t * r0 = mask.ptr<uint8_t>(y0);
      uint8_t * r1 = mask.ptr<uint8_t>(y0 + 1);
      r0[x0] = 255;
      r0[x0 + 1] = 255;
      r1[x0] = 255;
      r1[x0 + 1] = 255;
    }
  }

  cv::Mat mask = masks[0];
  for (int i = 1; i < n_threads; ++i) {
    cv::bitwise_or(mask, masks[static_cast<size_t>(i)], mask);
  }
  return mask;
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

cv::Mat render_flow_hsv_debug(
  const cv::Mat & flow,
  double vis_speed_cap,
  const cv::Size & output_size = cv::Size())
{
  if (flow.empty() || flow.type() != CV_32FC2) {
    return cv::Mat();
  }

  cv::Mat angle(flow.rows, flow.cols, CV_32F, cv::Scalar(0.0f));
  cv::Mat magnitude(flow.rows, flow.cols, CV_32F, cv::Scalar(0.0f));
  std::vector<float> speeds;
  speeds.reserve(static_cast<size_t>(flow.rows) * flow.cols);
  double max_speed = 0.0;

  for (int y = 0; y < flow.rows; ++y) {
    const cv::Vec2f * vrow = flow.ptr<cv::Vec2f>(y);
    float * arow = angle.ptr<float>(y);
    float * mrow = magnitude.ptr<float>(y);
    for (int x = 0; x < flow.cols; ++x) {
      const float vx = vrow[x][0];
      const float vy = vrow[x][1];
      if (!std::isfinite(vx) || !std::isfinite(vy)) {
        continue;
      }
      float a = std::atan2(vy, vx);
      if (a < 0.0f) {
        a += 2.0f * static_cast<float>(CV_PI);
      }
      const float spd = std::hypot(vx, vy);
      arow[x] = a;
      mrow[x] = spd;
      speeds.push_back(spd);
      max_speed = std::max(max_speed, static_cast<double>(spd));
    }
  }

  double robust_max_speed = max_speed;
  if (!speeds.empty()) {
    const size_t nth = std::min(
      speeds.size() - 1,
      static_cast<size_t>(0.95 * static_cast<double>(speeds.size())));
    std::nth_element(speeds.begin(), speeds.begin() + nth, speeds.end());
    robust_max_speed = std::max<double>(speeds[nth], 0.05 * max_speed);
  }
  const double speed_cap = std::max(1e-6, vis_speed_cap);
  const double display_floor_speed = std::max(25.0, 0.10 * speed_cap);
  const double raw_display_max = (robust_max_speed > 1e-6) ? robust_max_speed : speed_cap;
  const double display_max_speed =
    std::clamp(raw_display_max, display_floor_speed, speed_cap);

  cv::Mat hsv_parts[3];
  cv::Mat hue = angle * (180.0f / (2.0f * static_cast<float>(CV_PI)));
  hue.convertTo(hsv_parts[0], CV_8U);
  hsv_parts[1] = cv::Mat(flow.rows, flow.cols, CV_8U, cv::Scalar(255));
  magnitude *= (255.0f / static_cast<float>(display_max_speed));
  cv::threshold(magnitude, magnitude, 255.0, 255.0, cv::THRESH_TRUNC);
  magnitude.convertTo(hsv_parts[2], CV_8U);

  cv::Mat hsv;
  cv::Mat bgr;
  cv::merge(hsv_parts, 3, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  if (output_size.width > 0 && output_size.height > 0 && bgr.size() != output_size) {
    cv::Mat resized;
    cv::resize(bgr, resized, output_size, 0.0, 0.0, cv::INTER_NEAREST);
    return resized;
  }
  return bgr;
}

}  // namespace

EventDetector::FlowResult EventDetector::solve_flow_moment(
  const EventStore & window,
  std::optional<int64_t> t_ref_override_us)
{
  const auto t_total = ProfileClock::now();
  FlowResult result;
  FlowTiming timing;
  if (window.isEmpty() || res_.width <= 0 || res_.height <= 0) {
    return result;
  }

  const int w = res_.width;
  const int h = res_.height;
  const int n_scales = std::max<int>(1, static_cast<int>(flow_num_scales_));
  const int final_tiles = 1 << (n_scales - 1);
  const float vis_speed_cap = static_cast<float>(flow_max_speed_px_s_);

  const int n_threads = resolve_threads(flow_num_threads_);

  const auto t_select = ProfileClock::now();
  const std::size_t total = static_cast<std::size_t>(window.size());
  // Busier windows are strided down so the per-window moment update stays
  // bounded: every warped re-solve iteration re-ingests the whole selection, so
  // the cap trades moment-statistics noise for a hard latency bound on
  // embedded targets. The mass gates are rescaled by 1/stride below, hence the
  // decimation is unbiased in expectation.
  const std::size_t max_solve_events = static_cast<std::size_t>(
    std::max<int64_t>(0, flow_max_solve_events_));
  const bool capped = max_solve_events > 0 && total > max_solve_events;
  const std::size_t stride = capped
    ? (total + max_solve_events - 1) / max_solve_events
    : 1;
  // Selection is positional (every stride-th event of the store), so the store
  // is split into chunks whose kept counts become write offsets: each kept
  // event lands at the index the serial loop would have given it. The window
  // time bounds come from the same pass, so no separate min/max traversal is
  // needed.
  const int n_chunks = std::clamp(
    n_threads, 1, static_cast<int>(std::max<std::size_t>(1, total / 4096)));
  const auto events_begin = window.begin();
  std::vector<std::size_t> chunk_offset(static_cast<std::size_t>(n_chunks) + 1, 0);
  int64_t t_lo_us = std::numeric_limits<int64_t>::max();
  int64_t t_hi_us = std::numeric_limits<int64_t>::lowest();

  #pragma omp parallel for schedule(static) num_threads(n_chunks) \
  reduction(min : t_lo_us) reduction(max : t_hi_us)
  for (int c = 0; c < n_chunks; ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * total / n_chunks;
    const std::size_t end = static_cast<std::size_t>(c + 1) * total / n_chunks;
    std::size_t kept = 0;
    for (std::size_t k = ((begin + stride - 1) / stride) * stride; k < end; k += stride) {
      const auto & e = events_begin[static_cast<std::ptrdiff_t>(k)];
      if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
        continue;
      }
      t_lo_us = std::min(t_lo_us, e.timestamp());
      t_hi_us = std::max(t_hi_us, e.timestamp());
      kept += 1;
    }
    chunk_offset[static_cast<std::size_t>(c) + 1] = kept;
  }
  for (int c = 0; c < n_chunks; ++c) {
    chunk_offset[static_cast<std::size_t>(c) + 1] +=
      chunk_offset[static_cast<std::size_t>(c)];
  }
  const std::size_t n_selected = chunk_offset[static_cast<std::size_t>(n_chunks)];
  timing.select_events_ms = elapsed_ms(t_select);
  if (n_selected < 2) {
    if (debug_) {
      RCLCPP_INFO(
        get_logger(),
        "Flow profile: skipped moment flow, raw_events=%zu solve_events=%zu "
        "stride=%zu select=%.3f ms",
        total, n_selected, stride, timing.select_events_ms);
    }
    return result;
  }

  const auto t_pack = ProfileClock::now();
  const int64_t t_ref_us = t_ref_override_us.value_or(t_lo_us + (t_hi_us - t_lo_us) / 2);

  Events ev;
  ev.t_ref_us = t_ref_us;
  ev.x.resize(n_selected);
  ev.y.resize(n_selected);
  ev.t.resize(n_selected);
  #pragma omp parallel for schedule(static) num_threads(n_chunks)
  for (int c = 0; c < n_chunks; ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * total / n_chunks;
    const std::size_t end = static_cast<std::size_t>(c + 1) * total / n_chunks;
    std::size_t n = chunk_offset[static_cast<std::size_t>(c)];
    for (std::size_t k = ((begin + stride - 1) / stride) * stride; k < end; k += stride) {
      const auto & e = events_begin[static_cast<std::ptrdiff_t>(k)];
      if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
        continue;
      }
      ev.x[n] = static_cast<float>(e.x());
      ev.y[n] = static_cast<float>(e.y());
      ev.t[n] = static_cast<float>(e.timestamp() - t_ref_us) * 1e-6f;
      ++n;
    }
  }
  timing.pack_events_ms = elapsed_ms(t_pack);

  MomentFlowParams params;
  params.num_scales = n_scales;
  params.cell_size_px = static_cast<int>(flow_cell_size_px_);
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
  params.max_speed_px_s = static_cast<float>(flow_max_speed_px_s_);

  if (!moment_flow_.has_value() || !moment_flow_->compatible(w, h, params)) {
    moment_flow_.emplace(w, h, params);
  }
  moment_flow_->set_mass_scale(1.0f / static_cast<float>(stride));
  moment_flow_->set_max_threads(static_cast<int>(flow_num_threads_));

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
  // Coarse-to-fine residual solve with per-scale event re-warping: the warm
  // start (previous tracked field) seeds the composed field, every scale
  // estimates a residual on events warped by the field composed so far, and
  // fine scales re-warp before solving. This keeps the within-tile dispersion
  // regression in its unbiased small-residual regime — solving raw events
  // would systematically underestimate any displacement that crosses a tile
  // within the window (dense texture -> data term biased toward zero).
  Eigen::VectorXf F(moment_flow_->num_vars());
  moment_flow_->set_prior_scale(0.5f);
  moment_flow_->solve_coarse_to_fine(ev, warm_start, F, kRewarpMinTiles);
  moment_flow_->set_prior_scale(1.0f);
  sanitize_field(F);

  // ---- Stage C: compositional warped re-solve (Gauss-Newton on dispersion) ----
  // The single-pass tile regression truncates large displacements: with long
  // windows an edge crosses several tiles/cells and the linear moment model
  // underestimates speed. Warping events by the current field and re-solving
  // for the residual is the moment-domain analogue of the CMax multi-iteration
  // alignment (Shiba et al.): each pass sharpens the warped cloud, the residual
  // field shrinks, and the composition F <- F + dF converges to the dispersion
  // optimum without the truncation bias.
  int refine_done = 0;
  if (flow_refine_enabled_ && flow_refine_iters_ > 0) {
    const auto t_refine = ProfileClock::now();
    // Residual speeds below this leave sub-pixel displacement over the window:
    // further iterations cannot sharpen the IWE, so stop early.
    const float t_span_s = std::max(
      1e-4f, static_cast<float>(t_hi_us - t_lo_us) * 1e-6f);
    const float res_speed_eps = 0.5f / t_span_s;  // 0.5 px over the window
    Events wev;
    Eigen::VectorXf F_res(moment_flow_->num_vars());
    const Eigen::VectorXf no_warm;
    // Residual passes: a moderate prior (toward zero residual) keeps
    // degenerate tiles stable while letting corrections through; the full
    // prior would damp every step by ~1/(1+prior_lambda).
    moment_flow_->set_prior_scale(0.5f);
    for (int it = 0; it < static_cast<int>(flow_refine_iters_); ++it) {
      warp_events_by_field(
        ev, F, nullptr,
        final_tiles, w, h, n_threads, wev);
      if (wev.size() < 2) {
        break;
      }
      moment_flow_->reset();
      moment_flow_->ingest(wev);
      moment_flow_->solve(no_warm, F_res);
      sanitize_field(F_res);
      float max_res = 0.0f;
      for (int k = 0; k < F_res.size() / 2; ++k) {
        max_res = std::max(max_res, std::hypot(F_res[2 * k], F_res[2 * k + 1]));
      }
      F += F_res;
      refine_done += 1;
      if (max_res < res_speed_eps) {
        break;
      }
    }
    moment_flow_->set_prior_scale(1.0f);
    clamp_field_speed(F, static_cast<float>(flow_max_speed_px_s_));
    timing.refine_ms = elapsed_ms(t_refine);
  }
  {
    // Regularize the composed field: the in-solve regularizer only saw the
    // last residual pass.
    std::vector<float> smooth_conf;
    moment_flow_->final_tile_confidence(smooth_conf);
    smooth_field_confidence(
      F, smooth_conf, final_tiles,
      static_cast<int>(flow_smooth_sweeps_), static_cast<float>(flow_smooth_beta_));
  }
  timing.solve_moments_ms = elapsed_ms(t_moment);
  const auto profile = moment_flow_->profile();

  const bool need_iwe = iwe_enabled_ || debug_;
  const bool need_event_mask = flow_events_enabled_;
  const bool need_render_events = need_iwe || need_event_mask;

  // Without striding, the packed solver input already is every in-frame event
  // of the window at this reference time: the render set is the same array, so
  // it is aliased instead of rebuilt.
  Events render_ev_strided;
  if (need_render_events && stride != 1) {
    const auto t_render_events = ProfileClock::now();
    render_ev_strided.t_ref_us = t_ref_us;
    render_ev_strided.x.reserve(total);
    render_ev_strided.y.reserve(total);
    render_ev_strided.t.reserve(total);
    for (const auto & e : window) {
      if (e.x() < 0 || e.y() < 0 || e.x() >= w || e.y() >= h) {
        continue;
      }
      render_ev_strided.x.push_back(static_cast<float>(e.x()));
      render_ev_strided.y.push_back(static_cast<float>(e.y()));
      render_ev_strided.t.push_back(static_cast<float>(e.timestamp() - t_ref_us) * 1e-6f);
    }
    timing.render_events_ms = elapsed_ms(t_render_events);
  }
  const Events & render_ev = (stride == 1) ? ev : render_ev_strided;

  // ---- Field-level multi-reference focus diagnostic ----
  // Warp to explicit target times in the event window coordinate system,
  // combine gradient-magnitude contrasts: f = (G_lo + 2 G_mid + G_hi)/(4 G_id).
  // f <= 1 means the field did not improve this particular focus metric. Keep
  // the candidate visible anyway; otherwise a strict gate can black out the
  // first frame and keep every later warm start at zero.
  const Eigen::VectorXf * accel_ptr = nullptr;
  const int iwe_scale = std::max<int>(1, static_cast<int>(flow_iwe_scale_));
  const float t_lo_ref_s = static_cast<float>(t_lo_us - t_ref_us) * 1e-6f;
  const float t_hi_ref_s = static_cast<float>(t_hi_us - t_ref_us) * 1e-6f;
  double g_id = 0.0;
  double g_mid = 0.0;
  double g_lo = 0.0;
  double g_hi = 0.0;
  double focus_f = 1.0;
  bool flow_rejected = false;

  // The focus diagnostic uses contrast ratios, which are stable under event
  // subsampling: cap the events fed to the four focus renders so the
  // diagnostic cost stays bounded on busy windows.
  constexpr std::size_t kMaxFocusEvents = 150000;
  cv::Mat focus_id, focus_mid, focus_lo, focus_hi;
  Events focus_ev;
  const Events * focus_src = &render_ev;
  if (need_iwe) {
    const auto t_focus = ProfileClock::now();
    if (render_ev.size() > kMaxFocusEvents) {
      const std::size_t fstride = (render_ev.size() + kMaxFocusEvents - 1) / kMaxFocusEvents;
      focus_ev.t_ref_us = render_ev.t_ref_us;
      focus_ev.x.reserve(render_ev.size() / fstride + 1);
      focus_ev.y.reserve(render_ev.size() / fstride + 1);
      focus_ev.t.reserve(render_ev.size() / fstride + 1);
      for (size_t k = 0; k < render_ev.size(); k += fstride) {
        focus_ev.x.push_back(render_ev.x[k]);
        focus_ev.y.push_back(render_ev.y[k]);
        focus_ev.t.push_back(render_ev.t[k]);
      }
      focus_src = &focus_ev;
    }

    render_iwe_bilinear(
      *focus_src, F, nullptr, final_tiles, w, h, iwe_scale, false, 0.0f,
      kFocusSplatOffsetIwePx, focus_id);
    render_iwe_bilinear(
      *focus_src, F, accel_ptr, final_tiles, w, h, iwe_scale, true, 0.0f,
      kFocusSplatOffsetIwePx, focus_mid);
    render_iwe_bilinear(
      *focus_src, F, accel_ptr, final_tiles, w, h, iwe_scale, true, t_lo_ref_s,
      kFocusSplatOffsetIwePx, focus_lo);
    render_iwe_bilinear(
      *focus_src, F, accel_ptr, final_tiles, w, h, iwe_scale, true, t_hi_ref_s,
      kFocusSplatOffsetIwePx, focus_hi);
    // The common sub-pixel offset gives integer identity splats variance
    // eps*(1-eps) ~= 1/6, the average bilinear variance of fractional warps.
    // Applying it to every focus render keeps zero-motion focus_f exactly neutral.
    auto focus_contrast = [](const cv::Mat & iwe) {
      cv::Mat b;
      cv::GaussianBlur(iwe, b, cv::Size(0, 0), 1.0);
      return iwe_contrast(b);
    };
    g_id  = std::max(focus_contrast(focus_id), 1e-12);
    g_mid = focus_contrast(focus_mid);
    g_lo  = focus_contrast(focus_lo);
    g_hi  = focus_contrast(focus_hi);
    focus_f = (g_lo + 2.0 * g_mid + g_hi) / (4.0 * g_id);
    flow_rejected = !(focus_f > 1.0);
    timing.iwe_focus_ms = elapsed_ms(t_focus);
  }

  prev_flow_field_ = F;
  prev_flow_tiles_ = final_tiles;

  const auto t_tile_flow = ProfileClock::now();
  result.flow_tiles = cv::Mat(final_tiles, final_tiles, CV_32FC2);
  for (int ty = 0; ty < final_tiles; ++ty) {
    cv::Vec2f * row = result.flow_tiles.ptr<cv::Vec2f>(ty);
    for (int tx = 0; tx < final_tiles; ++tx) {
      const int k = ty * final_tiles + tx;
      row[tx] = cv::Vec2f(-F[2 * k], -F[2 * k + 1]);
    }
  }
  timing.tile_flow_ms = elapsed_ms(t_tile_flow);

  const auto t_tile_debug = ProfileClock::now();
  result.flow_tile_debug =
    render_flow_hsv_debug(result.flow_tiles, vis_speed_cap, cv::Size(w, h));
  timing.tile_debug_ms = elapsed_ms(t_tile_debug);

  // Dense flow field: the actual estimator output. Always computed, since
  // it (and flow_events derived from it below) is the useful data this node
  // exists to produce, regardless of `debug_`.
  const auto t_dense_flow = ProfileClock::now();
  result.flow_dense = cv::Mat(h, w, CV_32FC2);
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    cv::Vec2f * vrow = result.flow_dense.ptr<cv::Vec2f>(y);
    for (int x = 0; x < w; ++x) {
      float vx, vy;
      sample_tile_velocity(
        F, final_tiles, final_tiles, w, h,
        static_cast<float>(x), static_cast<float>(y), vx, vy);
      vrow[x] = cv::Vec2f(-vx, -vy);
    }
  }
  timing.dense_flow_ms = elapsed_ms(t_dense_flow);

  // ---- Debug-only: HSV flow visualization + speed stats/logging ----
  if (debug_) {
    const auto t_flow_image = ProfileClock::now();
    cv::Mat angle(h, w, CV_32F);
    cv::Mat magnitude(h, w, CV_32F);
    double vx_sum = 0.0;
    double vy_sum = 0.0;
    double vx2_sum = 0.0;
    double vy2_sum = 0.0;
    #pragma omp parallel for schedule(static) \
    reduction(+ : vx_sum, vy_sum, vx2_sum, vy2_sum)
    for (int y = 0; y < h; ++y) {
      const cv::Vec2f * vrow = result.flow_dense.ptr<cv::Vec2f>(y);
      float * arow = angle.ptr<float>(y);
      float * mrow = magnitude.ptr<float>(y);
      for (int x = 0; x < w; ++x) {
        const float vx = vrow[x][0];
        const float vy = vrow[x][1];
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
    timing.flow_debug_ms = elapsed_ms(t_flow_image);
  }

  // The event-supported flow is a sparse benchmark/diagnostic product. It is
  // intentionally optional because splatting every event is expensive.
  cv::Mat support_mask;
  if (need_event_mask) {
    const auto t_support_mask = ProfileClock::now();
    support_mask = render_support_mask(
      render_ev, F, flow_rejected ? nullptr : accel_ptr,
      final_tiles, w, h, /*warp=*/!flow_rejected, n_threads);
    timing.support_mask_ms = elapsed_ms(t_support_mask);

    const auto t_events_mask = ProfileClock::now();
    result.flow_events = mask_flow_by_support(result.flow_dense, support_mask);
    timing.events_mask_ms = elapsed_ms(t_events_mask);
  }

  // ---- Debug-only: IWE preview image + event-supported debug visualization ----
  if (need_iwe) {
    const auto t_iwe = ProfileClock::now();
    // The published IWE is a visualization: the (subsampled, possibly
    // downscaled) focus render is sufficient, upscaled back to sensor size.
    cv::Mat iwe_vis;
    cv::normalize(
      flow_rejected ? focus_id : focus_mid, iwe_vis, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    if (iwe_vis.size() != cv::Size(w, h)) {
      cv::resize(iwe_vis, result.iwe, cv::Size(w, h), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
      result.iwe = iwe_vis;
    }
    if (debug_ && need_event_mask) {
      result.flow_events_debug = mask_debug_by_support(result.flow_dense_debug, support_mask);
    }
    timing.iwe_debug_ms = elapsed_ms(t_iwe);
  }

  if (debug_) {
    const double reg_modified_fraction =
      (profile.reg_total_tiles > 0)
        ? static_cast<double>(profile.reg_modified_tiles) /
          static_cast<double>(profile.reg_total_tiles)
        : 0.0;

    RCLCPP_DEBUG(
      get_logger(),
      "Flow profile MomentFlow: ingest=%.3f ms stage_a=%.3f ms "
      "stage_b=%.3f ms spatial=%.3f ms solve_total=%.3f ms total=%.3f ms events=%d "
      "active_cells=%d valid_cells=%d reject(residual/speed)=%d/%d "
      "tiles_total(full/aperture/fallback/prior)=%d/%d/%d/%d "
      "tiles_final(full/aperture/fallback)=%d/%d/%d "
      "reg(modified_frac/mean_delta legacy_geom/warped_geom) %.3f/%.3f %d/%d",
      profile.ingest_ms, profile.stage_a_ms,
      profile.stage_b_ms, profile.smooth_ms, profile.total_solve_ms, timing.solve_moments_ms,
      profile.events_ingested, profile.active_cells, profile.valid_cells,
      profile.residual_reject_cells, profile.speed_reject_cells,
      profile.full_rank_tiles, profile.aperture_tiles, profile.fallback_tiles,
      profile.prior_tiles,
      profile.final_full_rank_tiles, profile.final_aperture_tiles, profile.final_fallback_tiles,
      reg_modified_fraction, profile.reg_mean_delta_speed,
      profile.reg_legacy_geometry_tiles, profile.reg_warped_geometry_tiles);

    // FWL standard (single-reference, Stoffregen & Kleeman 2019): ratio between
    // the motion-compensated IWE variance and the identity-IWE variance. It is
    // the metric reported by Shiba et al. (Table II), i.e. the term directly
    // comparable to CMax. No GaussianBlur (unlike focus_f), computed on the
    // raw float IWE before any normalization. Diagnostic only: never affects
    // flow_rejected or any published output.
    auto iwe_variance = [](const cv::Mat & iwe) {
      cv::Scalar mean, stddev;
      cv::meanStdDev(iwe, mean, stddev);
      return stddev[0] * stddev[0];
    };
    const double var_id  = std::max(iwe_variance(focus_id), 1e-12);
    const double var_mid = iwe_variance(focus_mid);
    const double fwl = var_mid / var_id;

    RCLCPP_DEBUG(
      get_logger(),
      "Flow profile IWE: render_events=%zu focus_events=%zu event_pack=%.3f ms "
      "focus=%.3f ms flow_image=%.3f ms "
      "iwe_render_norm=%.3f ms solver=anisotropic focus_f=%.4f "
      "(lo=%.6f mid=%.6f hi=%.6f id=%.6f) "
      "fwl=%.4f (var_warp=%.6f var_id=%.6f) rejected=%d refine_iters=%d refine=%.3f ms",
      render_ev.size(), focus_src->size(), timing.render_events_ms, timing.iwe_focus_ms,
      timing.flow_debug_ms, timing.iwe_debug_ms,
      focus_f, g_lo, g_mid, g_hi, g_id,
      fwl, var_mid, var_id,
      flow_rejected ? 1 : 0, refine_done, timing.refine_ms);

    timing.total_ms = elapsed_ms(t_total);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Flow profile summary: raw_events=%zu solve_events=%zu stride=%zu span=%.3f ms "
      "setup_select=%.3f ms setup_pack=%.3f ms "
      "moment=%.3f ms render=%.3f ms total=%.3f ms achieved_hz=%.3f",
      total, n_selected, stride, static_cast<double>(t_hi_us - t_lo_us) * 1e-3,
      timing.select_events_ms, timing.pack_events_ms, timing.solve_moments_ms,
      timing.render_events_ms + timing.iwe_focus_ms + timing.tile_flow_ms +
      timing.tile_debug_ms + timing.dense_flow_ms + timing.flow_debug_ms +
      timing.support_mask_ms + timing.events_mask_ms + timing.iwe_debug_ms,
      timing.total_ms, 1000.0 / std::max(1e-9, timing.total_ms));
  }

  // End-to-end timing CSV: useful data, always logged regardless of `debug_`.
  timing.total_ms = elapsed_ms(t_total);
  log_timing(t_lo_us, t_hi_us, n_selected, timing);

  return result;
}

}  // namespace moment_flow
