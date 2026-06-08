/**
 * Moment-based incremental optical-flow estimator.
 *
 * Maintains spatio-temporal moments per sensor cell, optionally with causal
 * exponential decay, and solves small closed-form structure-tensor systems on
 * the tile pyramid. The output field uses the same convention as the previous
 * dense-flow publisher:
 * F[2*k], F[2*k+1] are warp parameters in x' = x + t * F.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
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

#pragma once

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace event_detector_cpp::flow
{

/// Events for one window: pixel coords and time relative to t_ref_us [s].
struct Events
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> t;
  int64_t t_ref_us = 0;

  size_t size() const { return x.size(); }
};

struct MomentFlowParams
{
  int num_scales = 1;
  int cell_size_px = 16;
  bool decay_enabled = false;
  int decay_tau_us = 30000;
  bool tau_adaptive = false;
  float cell_min_mass = 3.0f;
  float cell_min_lambda = 1e-3f;
  float cell_max_residual_ratio = 0.6f;
  float tile_min_mass = 10.0f;
  int tile_min_cells = 3;
  float tile_min_lambda = 1e-6f;
  float aperture_ratio = 0.05f;
  float tikhonov_eps = 1e-3f;
  float prior_lambda = 0.05f;
  int smooth_iters = 0;
  float smooth_alpha = 0.0f;
  bool refine_enabled = false;
  int refine_iters = 2;
  float refine_huber_delta = 0.01f;
  float max_speed_px_s = 4000.0f;
};

inline bool operator==(const MomentFlowParams & a, const MomentFlowParams & b)
{
  return a.num_scales == b.num_scales &&
         a.cell_size_px == b.cell_size_px &&
         a.decay_enabled == b.decay_enabled &&
         a.decay_tau_us == b.decay_tau_us &&
         a.tau_adaptive == b.tau_adaptive &&
         a.cell_min_mass == b.cell_min_mass &&
         a.cell_min_lambda == b.cell_min_lambda &&
         a.cell_max_residual_ratio == b.cell_max_residual_ratio &&
         a.tile_min_mass == b.tile_min_mass &&
         a.tile_min_cells == b.tile_min_cells &&
         a.tile_min_lambda == b.tile_min_lambda &&
         a.aperture_ratio == b.aperture_ratio &&
         a.tikhonov_eps == b.tikhonov_eps &&
         a.prior_lambda == b.prior_lambda &&
         a.smooth_iters == b.smooth_iters &&
         a.smooth_alpha == b.smooth_alpha &&
         a.refine_enabled == b.refine_enabled &&
         a.refine_iters == b.refine_iters &&
         a.refine_huber_delta == b.refine_huber_delta &&
         a.max_speed_px_s == b.max_speed_px_s;
}

inline bool operator!=(const MomentFlowParams & a, const MomentFlowParams & b)
{
  return !(a == b);
}

struct MomentFlowProfile
{
  int events_ingested = 0;
  int active_cells = 0;
  int valid_cells = 0;
  int residual_reject_cells = 0;
  int speed_reject_cells = 0;
  int full_rank_tiles = 0;
  int aperture_tiles = 0;
  int fallback_tiles = 0;
  int prior_tiles = 0;
  double ingest_ms = 0.0;
  double decay_ms = 0.0;
  double stage_a_ms = 0.0;
  double stage_b_ms = 0.0;
  double smooth_ms = 0.0;
  double total_solve_ms = 0.0;
};

struct alignas(64) CellMoments
{
  float w = 0.0f;
  float sx = 0.0f;
  float sy = 0.0f;
  float st = 0.0f;
  float sxx = 0.0f;
  float sxy = 0.0f;
  float syy = 0.0f;
  float sxt = 0.0f;
  float syt = 0.0f;
  float stt = 0.0f;
  int64_t t_last_us = 0;
};

static_assert(alignof(CellMoments) == 64, "CellMoments must be cache-line aligned");
static_assert(sizeof(CellMoments) == 64, "CellMoments must occupy one cache line");

class MomentFlow
{
public:
  MomentFlow(int img_w, int img_h, MomentFlowParams params)
  : img_w_(img_w),
    img_h_(img_h),
    params_(sanitize_params(params)),
    cells_x_((img_w_ + params_.cell_size_px - 1) / params_.cell_size_px),
    cells_y_((img_h_ + params_.cell_size_px - 1) / params_.cell_size_px),
    final_tiles_(1 << (params_.num_scales - 1)),
    final_vars_(2 * final_tiles_ * final_tiles_),
    cells_(static_cast<size_t>(cells_x_) * cells_y_),
    fits_(cells_.size()),
    cell_center_x_(cells_.size(), 0.0f),
    cell_center_y_(cells_.size(), 0.0f),
    tile_accum_(static_cast<size_t>(final_tiles_) * final_tiles_),
    smooth_scratch_(final_vars_)
  {
    for (int cy = 0; cy < cells_y_; ++cy) {
      for (int cx = 0; cx < cells_x_; ++cx) {
        const size_t k = cell_index(cx, cy);
        cell_center_x_[k] = std::min(
          (static_cast<float>(cx) + 0.5f) * params_.cell_size_px,
          static_cast<float>(img_w_) - 0.5f);
        cell_center_y_[k] = std::min(
          (static_cast<float>(cy) + 0.5f) * params_.cell_size_px,
          static_cast<float>(img_h_) - 0.5f);
      }
    }

    scale_fields_.reserve(static_cast<size_t>(params_.num_scales));
    scale_fallback_.reserve(static_cast<size_t>(params_.num_scales));
    for (int l = 0; l < params_.num_scales; ++l) {
      const int tiles = 1 << l;
      scale_fields_.push_back(Eigen::VectorXf(2 * tiles * tiles));
      scale_fallback_.push_back(Eigen::VectorXf(2 * tiles * tiles));
    }
  }

  bool compatible(int img_w, int img_h, const MomentFlowParams & params) const
  {
    return img_w_ == img_w && img_h_ == img_h && params_ == sanitize_params(params);
  }

  int final_tiles() const { return final_tiles_; }
  int num_vars() const { return final_vars_; }
  const MomentFlowProfile & profile() const { return profile_; }

  void reset()
  {
    for (CellMoments & c : cells_) {
      clear_cell(c);
    }
    time_origin_us_ = 0;
    last_event_us_ = 0;
    has_time_origin_ = false;
    profile_ = MomentFlowProfile{};
  }

  void ingest(const Events & events)
  {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    profile_ = MomentFlowProfile{};

    if (events.size() == 0 || img_w_ <= 0 || img_h_ <= 0) {
      return;
    }
    if (!has_time_origin_) {
      time_origin_us_ = events.t_ref_us;
      has_time_origin_ = true;
    }
    if (std::llabs(events.t_ref_us - time_origin_us_) > kRebaseThresholdUs) {
      rebase_time_origin(events.t_ref_us);
    }

    const float tau_s = std::max(1, params_.decay_tau_us) * 1e-6f;
    for (size_t k = 0; k < events.size(); ++k) {
      const float x = events.x[k];
      const float y = events.y[k];
      if (!(x >= 0.0f && y >= 0.0f && x < img_w_ && y < img_h_)) {
        continue;
      }
      const int cx = std::clamp(
        static_cast<int>(x) / params_.cell_size_px, 0, cells_x_ - 1);
      const int cy = std::clamp(
        static_cast<int>(y) / params_.cell_size_px, 0, cells_y_ - 1);
      CellMoments & c = cells_[cell_index(cx, cy)];
      const int64_t t_us = events.t_ref_us +
        static_cast<int64_t>(std::llround(static_cast<double>(events.t[k]) * 1e6));
      if (std::llabs(t_us - time_origin_us_) > kRebaseThresholdUs) {
        rebase_time_origin(t_us);
      }
      float a = 1.0f;
      if (params_.decay_enabled) {
        if (c.w > 0.0f && c.t_last_us > 0 && t_us < c.t_last_us) {
          a = std::exp(-static_cast<float>(c.t_last_us - t_us) * 1e-6f / tau_s);
          if (a < 1e-6f) {
            continue;
          }
        } else {
          decay_cell_to(c, t_us, tau_s);
        }
      }

      const float dx = x - cell_center_x_[cell_index(cx, cy)];
      const float dy = y - cell_center_y_[cell_index(cx, cy)];
      const float dt = static_cast<float>(t_us - time_origin_us_) * 1e-6f;

      c.w += a;
      c.sx += a * dx;
      c.sy += a * dy;
      c.st += a * dt;
      c.sxx += a * dx * dx;
      c.sxy += a * dx * dy;
      c.syy += a * dy * dy;
      c.sxt += a * dx * dt;
      c.syt += a * dy * dt;
      c.stt += a * dt * dt;
      c.t_last_us = std::max(c.t_last_us, t_us);
      last_event_us_ = std::max(last_event_us_, t_us);
      profile_.events_ingested += 1;
    }

    profile_.ingest_ms = elapsed_ms(t0, Clock::now());
  }

  void solve(const Eigen::VectorXf & warm_start, Eigen::VectorXf & F_out)
  {
    using Clock = std::chrono::steady_clock;
    const auto t_total = Clock::now();
    profile_.decay_ms = 0.0;
    profile_.stage_a_ms = 0.0;
    profile_.stage_b_ms = 0.0;
    profile_.smooth_ms = 0.0;
    profile_.total_solve_ms = 0.0;
    profile_.active_cells = 0;
    profile_.valid_cells = 0;
    profile_.residual_reject_cells = 0;
    profile_.speed_reject_cells = 0;
    profile_.full_rank_tiles = 0;
    profile_.aperture_tiles = 0;
    profile_.fallback_tiles = 0;
    profile_.prior_tiles = 0;

    if (F_out.size() != final_vars_) {
      return;
    }

    const auto t_decay = Clock::now();
    if (params_.decay_enabled && last_event_us_ > 0) {
      decay_all_to(last_event_us_);
    }
    profile_.decay_ms = elapsed_ms(t_decay, Clock::now());

    const auto t_a = Clock::now();
    compute_cell_fits();
    profile_.stage_a_ms = elapsed_ms(t_a, Clock::now());

    const bool have_warm = warm_start.size() == final_vars_;
    if (profile_.valid_cells == 0) {
      if (have_warm) {
        copy_vector(warm_start, F_out);
      } else {
        set_zero(F_out);
      }
      profile_.fallback_tiles += final_tiles_ * final_tiles_;
      profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
      return;
    }

    for (int l = 0; l < params_.num_scales; ++l) {
      const int tiles = 1 << l;
      Eigen::VectorXf & fallback = scale_fallback_[static_cast<size_t>(l)];
      Eigen::VectorXf & field = scale_fields_[static_cast<size_t>(l)];

      if (l == 0) {
        if (have_warm) {
          resample_field(warm_start, final_tiles_, fallback, tiles);
        } else {
          set_zero(fallback);
        }
      } else {
        resample_field(
          scale_fields_[static_cast<size_t>(l - 1)], 1 << (l - 1), fallback, tiles);
        if (have_warm) {
          average_with_resampled(warm_start, final_tiles_, fallback, tiles);
        }
      }

      const auto t_b = Clock::now();
      solve_scale(tiles, fallback, field);
      profile_.stage_b_ms += elapsed_ms(t_b, Clock::now());

      const auto t_smooth = Clock::now();
      smooth_field(tiles, field);
      profile_.smooth_ms += elapsed_ms(t_smooth, Clock::now());
    }

    copy_vector(scale_fields_.back(), F_out);
    profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
  }

private:
  struct CellFit
  {
    float gx = 0.0f;
    float gy = 0.0f;
    float rho = 0.0f;
    float sc = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
  };

  struct TileAccum
  {
    float mxx = 0.0f;
    float mxy = 0.0f;
    float myy = 0.0f;
    float bx = 0.0f;
    float by = 0.0f;
    float rho = 0.0f;
    int count = 0;
  };

  static constexpr int64_t kRebaseThresholdUs = 250000;
  static constexpr float kPruneMass = 1e-4f;
  static constexpr float kMinTimeVariance = 1e-12f;

  int img_w_;
  int img_h_;
  MomentFlowParams params_;
  int cells_x_;
  int cells_y_;
  int final_tiles_;
  int final_vars_;
  std::vector<CellMoments> cells_;
  std::vector<CellFit> fits_;
  std::vector<float> cell_center_x_;
  std::vector<float> cell_center_y_;
  std::vector<TileAccum> tile_accum_;
  std::vector<Eigen::VectorXf> scale_fields_;
  std::vector<Eigen::VectorXf> scale_fallback_;
  Eigen::VectorXf smooth_scratch_;
  MomentFlowProfile profile_;
  int64_t time_origin_us_ = 0;
  int64_t last_event_us_ = 0;
  bool has_time_origin_ = false;

  static MomentFlowParams sanitize_params(MomentFlowParams p)
  {
    p.num_scales = std::clamp(p.num_scales, 1, 8);
    p.cell_size_px = std::max(1, p.cell_size_px);
    p.decay_tau_us = std::max(1, p.decay_tau_us);
    p.cell_min_mass = std::max(0.0f, p.cell_min_mass);
    p.cell_min_lambda = std::max(0.0f, p.cell_min_lambda);
    p.cell_max_residual_ratio = std::clamp(p.cell_max_residual_ratio, 1e-6f, 1.0f);
    p.tile_min_mass = std::max(0.0f, p.tile_min_mass);
    p.tile_min_cells = std::max(0, p.tile_min_cells);
    p.tile_min_lambda = std::max(0.0f, p.tile_min_lambda);
    p.aperture_ratio = std::clamp(p.aperture_ratio, 0.0f, 1.0f);
    p.tikhonov_eps = std::max(1e-12f, p.tikhonov_eps);
    p.prior_lambda = std::max(0.0f, p.prior_lambda);
    p.smooth_iters = std::clamp(p.smooth_iters, 0, 16);
    p.smooth_alpha = std::clamp(p.smooth_alpha, 0.0f, 1.0f);
    p.refine_iters = std::max(0, p.refine_iters);
    p.refine_huber_delta = std::max(1e-9f, p.refine_huber_delta);
    p.max_speed_px_s = std::max(1.0f, p.max_speed_px_s);
    return p;
  }

  static double elapsed_ms(
    const std::chrono::steady_clock::time_point & start,
    const std::chrono::steady_clock::time_point & end)
  {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  size_t cell_index(int cx, int cy) const
  {
    return static_cast<size_t>(cy) * cells_x_ + cx;
  }

  static void clear_cell(CellMoments & c)
  {
    c = CellMoments{};
  }

  static void scale_cell(CellMoments & c, float s)
  {
    c.w *= s;
    c.sx *= s;
    c.sy *= s;
    c.st *= s;
    c.sxx *= s;
    c.sxy *= s;
    c.syy *= s;
    c.sxt *= s;
    c.syt *= s;
    c.stt *= s;
  }

  void decay_cell_to(CellMoments & c, int64_t t_us, float tau_s) const
  {
    if (c.w <= 0.0f || c.t_last_us == 0) {
      c.t_last_us = t_us;
      return;
    }
    const int64_t dt_us = t_us - c.t_last_us;
    if (dt_us <= 0) {
      c.t_last_us = t_us;
      return;
    }
    const float decay = std::exp(-static_cast<float>(dt_us) * 1e-6f / tau_s);
    if (decay < 1e-6f) {
      clear_cell(c);
    } else {
      scale_cell(c, decay);
      if (c.w < kPruneMass) {
        clear_cell(c);
      }
    }
    c.t_last_us = t_us;
  }

  void decay_all_to(int64_t t_us)
  {
    const float tau_s = std::max(1, params_.decay_tau_us) * 1e-6f;
    for (CellMoments & c : cells_) {
      decay_cell_to(c, t_us, tau_s);
    }
  }

  void rebase_time_origin(int64_t new_origin_us)
  {
    if (!has_time_origin_) {
      time_origin_us_ = new_origin_us;
      has_time_origin_ = true;
      return;
    }
    const float delta = static_cast<float>(new_origin_us - time_origin_us_) * 1e-6f;
    if (delta == 0.0f) {
      return;
    }
    for (CellMoments & c : cells_) {
      if (c.w <= 0.0f) {
        continue;
      }
      const float old_st = c.st;
      c.st -= delta * c.w;
      c.sxt -= delta * c.sx;
      c.syt -= delta * c.sy;
      c.stt = c.stt - 2.0f * delta * old_st + delta * delta * c.w;
    }
    time_origin_us_ = new_origin_us;
  }

  static void set_zero(Eigen::VectorXf & v)
  {
    for (int i = 0; i < v.size(); ++i) {
      v[i] = 0.0f;
    }
  }

  static void copy_vector(const Eigen::VectorXf & src, Eigen::VectorXf & dst)
  {
    for (int i = 0; i < src.size(); ++i) {
      dst[i] = src[i];
    }
  }

  static void eig2(float a, float b, float c, float & lmin, float & lmax)
  {
    const float tr = a + c;
    const float diff = a - c;
    const float disc = std::sqrt(std::max(0.0f, diff * diff + 4.0f * b * b));
    lmax = 0.5f * (tr + disc);
    lmin = 0.5f * (tr - disc);
  }

  static bool solve2(float a, float b, float c, float rx, float ry, float & x, float & y)
  {
    const float det = a * c - b * b;
    if (!(std::abs(det) > 1e-20f) || !std::isfinite(det)) {
      x = 0.0f;
      y = 0.0f;
      return false;
    }
    x = (c * rx - b * ry) / det;
    y = (-b * rx + a * ry) / det;
    return std::isfinite(x) && std::isfinite(y);
  }

  static void dominant_eigenvector(
    float a, float b, float c, float lmax, float & ex, float & ey)
  {
    if (std::abs(b) > 1e-12f) {
      ex = b;
      ey = lmax - a;
    } else if (a >= c) {
      ex = 1.0f;
      ey = 0.0f;
    } else {
      ex = 0.0f;
      ey = 1.0f;
    }
    const float n = std::hypot(ex, ey);
    if (n > 1e-12f) {
      ex /= n;
      ey /= n;
    } else {
      ex = 1.0f;
      ey = 0.0f;
    }
  }

  void compute_cell_fits()
  {
    for (CellFit & f : fits_) {
      f = CellFit{};
    }

    for (size_t k = 0; k < cells_.size(); ++k) {
      const CellMoments & c = cells_[k];
      if (c.w > 0.0f) {
        profile_.active_cells += 1;
      }
      if (c.w < params_.cell_min_mass) {
        continue;
      }

      const float inv_w = 1.0f / c.w;
      const float mx = c.sx * inv_w;
      const float my = c.sy * inv_w;
      const float mt = c.st * inv_w;
      const float cxx = c.sxx * inv_w - mx * mx;
      const float cxy = c.sxy * inv_w - mx * my;
      const float cyy = c.syy * inv_w - my * my;
      const float dx = c.sxt * inv_w - mx * mt;
      const float dy = c.syt * inv_w - my * mt;
      const float sc = std::max(0.0f, c.stt * inv_w - mt * mt);
      if (!(sc > kMinTimeVariance)) {
        continue;
      }

      float spatial_lmin, spatial_lmax;
      eig2(cxx, cxy, cyy, spatial_lmin, spatial_lmax);
      if (!(spatial_lmax >= params_.cell_min_lambda)) {
        continue;
      }

      float gx = 0.0f;
      float gy = 0.0f;
      if (!solve2(cxx, cxy, cyy, dx, dy, gx, gy)) {
        continue;
      }

      const float residual = std::max(
        0.0f,
        sc - 2.0f * (gx * dx + gy * dy) +
        gx * (cxx * gx + cxy * gy) +
        gy * (cxy * gx + cyy * gy));
      const float residual_ratio = residual / std::max(sc, kMinTimeVariance);
      if (!(residual_ratio <= params_.cell_max_residual_ratio)) {
        profile_.residual_reject_cells += 1;
        continue;
      }
      const float g2 = gx * gx + gy * gy;
      const float min_g2 = 1.0f / (params_.max_speed_px_s * params_.max_speed_px_s);
      if (!(g2 >= min_g2)) {
        profile_.speed_reject_cells += 1;
        continue;
      }
      const float residual_conf = std::max(0.0f, 1.0f - residual_ratio);
      const float spatial_conf = spatial_lmax / (spatial_lmax + params_.tikhonov_eps);
      const float rho = c.w * spatial_conf * residual_conf * residual_conf;
      if (!(rho > 0.0f) || !std::isfinite(rho)) {
        continue;
      }

      fits_[k].gx = gx;
      fits_[k].gy = gy;
      fits_[k].rho = rho;
      fits_[k].sc = sc;
      fits_[k].dx = dx;
      fits_[k].dy = dy;
      profile_.valid_cells += 1;
    }
  }

  void solve_scale(
    int tiles,
    const Eigen::VectorXf & fallback_F,
    Eigen::VectorXf & out_F)
  {
    const int n_tiles = tiles * tiles;
    for (int i = 0; i < n_tiles; ++i) {
      tile_accum_[static_cast<size_t>(i)] = TileAccum{};
    }

    for (size_t k = 0; k < fits_.size(); ++k) {
      const CellFit & f = fits_[k];
      if (!(f.rho > 0.0f)) {
        continue;
      }
      const int tx = std::clamp(
        static_cast<int>(cell_center_x_[k] * tiles / std::max(1, img_w_)), 0, tiles - 1);
      const int ty = std::clamp(
        static_cast<int>(cell_center_y_[k] * tiles / std::max(1, img_h_)), 0, tiles - 1);
      TileAccum & a = tile_accum_[static_cast<size_t>(ty * tiles + tx)];
      a.mxx += f.rho * f.gx * f.gx;
      a.mxy += f.rho * f.gx * f.gy;
      a.myy += f.rho * f.gy * f.gy;
      a.bx += f.rho * f.gx;
      a.by += f.rho * f.gy;
      a.rho += f.rho;
      a.count += 1;
    }

    for (int ty = 0; ty < tiles; ++ty) {
      for (int tx = 0; tx < tiles; ++tx) {
        const int k = ty * tiles + tx;
        const TileAccum & a = tile_accum_[static_cast<size_t>(k)];
        const float fb_Fx = fallback_F[2 * k];
        const float fb_Fy = fallback_F[2 * k + 1];
        const float fb_px = -fb_Fx;
        const float fb_py = -fb_Fy;

        float lmin, lmax;
        eig2(a.mxx, a.mxy, a.myy, lmin, lmax);
        if (a.count < params_.tile_min_cells ||
            a.rho < params_.tile_min_mass ||
            !(lmax >= params_.tile_min_lambda))
        {
          out_F[2 * k] = fb_Fx;
          out_F[2 * k + 1] = fb_Fy;
          profile_.fallback_tiles += 1;
          continue;
        }

        float vx_phys = 0.0f;
        float vy_phys = 0.0f;
        const float tile_eps = params_.tikhonov_eps * std::max(lmax, 1e-12f);
        const float prior = params_.prior_lambda * std::max(lmax, 1e-12f);
        const bool solved = solve2(
          a.mxx + tile_eps + prior, a.mxy,
          a.myy + tile_eps + prior,
          a.bx + prior * fb_px, a.by + prior * fb_py,
          vx_phys, vy_phys);
        if (!solved) {
          out_F[2 * k] = fb_Fx;
          out_F[2 * k + 1] = fb_Fy;
          profile_.fallback_tiles += 1;
          continue;
        }

        if (lmin / std::max(lmax, 1e-12f) < params_.aperture_ratio) {
          float ex, ey;
          dominant_eigenvector(a.mxx, a.mxy, a.myy, lmax, ex, ey);
          const float tx_tan = -ey;
          const float ty_tan = ex;
          const float normal = vx_phys * ex + vy_phys * ey;
          const float tangent = fb_px * tx_tan + fb_py * ty_tan;
          vx_phys = normal * ex + tangent * tx_tan;
          vy_phys = normal * ey + tangent * ty_tan;
          profile_.aperture_tiles += 1;
        } else {
          profile_.full_rank_tiles += 1;
        }
        if (prior > 0.0f) {
          profile_.prior_tiles += 1;
        }

        clamp_speed(vx_phys, vy_phys);
        out_F[2 * k] = -vx_phys;
        out_F[2 * k + 1] = -vy_phys;
      }
    }
  }

  void clamp_speed(float & vx, float & vy) const
  {
    const float speed = std::hypot(vx, vy);
    if (speed > params_.max_speed_px_s) {
      const float s = params_.max_speed_px_s / speed;
      vx *= s;
      vy *= s;
    }
  }

  void smooth_field(int tiles, Eigen::VectorXf & field)
  {
    if (params_.smooth_iters <= 0 || params_.smooth_alpha <= 0.0f || tiles <= 1) {
      return;
    }

    const float alpha = params_.smooth_alpha;
    const float keep = 1.0f - alpha;
    for (int iter = 0; iter < params_.smooth_iters; ++iter) {
      for (int ty = 0; ty < tiles; ++ty) {
        for (int tx = 0; tx < tiles; ++tx) {
          const int k = ty * tiles + tx;
          float sum_x = field[2 * k];
          float sum_y = field[2 * k + 1];
          float weight = 1.0f;

          auto add_neighbor = [&](int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= tiles || ny >= tiles) {
              return;
            }
            const int nk = ny * tiles + nx;
            sum_x += field[2 * nk];
            sum_y += field[2 * nk + 1];
            weight += 1.0f;
          };
          add_neighbor(tx - 1, ty);
          add_neighbor(tx + 1, ty);
          add_neighbor(tx, ty - 1);
          add_neighbor(tx, ty + 1);

          smooth_scratch_[2 * k] = keep * field[2 * k] + alpha * (sum_x / weight);
          smooth_scratch_[2 * k + 1] =
            keep * field[2 * k + 1] + alpha * (sum_y / weight);
        }
      }

      const int n = 2 * tiles * tiles;
      for (int i = 0; i < n; ++i) {
        field[i] = smooth_scratch_[i];
      }
    }
  }

  void sample_field(
    const Eigen::VectorXf & F, int tiles,
    float px, float py, float & vx, float & vy) const
  {
    const float sx = static_cast<float>(img_w_) / tiles;
    const float sy = static_cast<float>(img_h_) / tiles;
    const float gx = px / sx - 0.5f;
    const float gy = py / sy - 0.5f;
    const int i0 = static_cast<int>(std::floor(gx));
    const int j0 = static_cast<int>(std::floor(gy));
    const float fx = gx - i0;
    const float fy = gy - j0;
    const int i0c = std::clamp(i0, 0, tiles - 1);
    const int i1c = std::clamp(i0 + 1, 0, tiles - 1);
    const int j0c = std::clamp(j0, 0, tiles - 1);
    const int j1c = std::clamp(j0 + 1, 0, tiles - 1);
    const int k00 = j0c * tiles + i0c;
    const int k10 = j0c * tiles + i1c;
    const int k01 = j1c * tiles + i0c;
    const int k11 = j1c * tiles + i1c;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    vx = w00 * F[2 * k00] + w10 * F[2 * k10] +
         w01 * F[2 * k01] + w11 * F[2 * k11];
    vy = w00 * F[2 * k00 + 1] + w10 * F[2 * k10 + 1] +
         w01 * F[2 * k01 + 1] + w11 * F[2 * k11 + 1];
  }

  void resample_field(
    const Eigen::VectorXf & src,
    int src_tiles,
    Eigen::VectorXf & dst,
    int dst_tiles) const
  {
    for (int j = 0; j < dst_tiles; ++j) {
      for (int i = 0; i < dst_tiles; ++i) {
        const float px = (static_cast<float>(i) + 0.5f) / dst_tiles * img_w_;
        const float py = (static_cast<float>(j) + 0.5f) / dst_tiles * img_h_;
        float vx, vy;
        sample_field(src, src_tiles, px, py, vx, vy);
        const int k = j * dst_tiles + i;
        dst[2 * k] = vx;
        dst[2 * k + 1] = vy;
      }
    }
  }

  void average_with_resampled(
    const Eigen::VectorXf & src,
    int src_tiles,
    Eigen::VectorXf & dst,
    int dst_tiles) const
  {
    for (int j = 0; j < dst_tiles; ++j) {
      for (int i = 0; i < dst_tiles; ++i) {
        const float px = (static_cast<float>(i) + 0.5f) / dst_tiles * img_w_;
        const float py = (static_cast<float>(j) + 0.5f) / dst_tiles * img_h_;
        float vx, vy;
        sample_field(src, src_tiles, px, py, vx, vy);
        const int k = j * dst_tiles + i;
        dst[2 * k] = 0.5f * (dst[2 * k] + vx);
        dst[2 * k + 1] = 0.5f * (dst[2 * k + 1] + vy);
      }
    }
  }
};

}  // namespace event_detector_cpp::flow
