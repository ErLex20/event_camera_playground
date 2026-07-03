/**
 * Moment-based incremental optical-flow estimator.
 *
 * Maintains spatio-temporal moments per sensor cell, optionally with causal
 * exponential decay, and solves either the legacy structure-tensor system or a
 * closed-form moment-domain surrogate of contrast maximization based on
 * minimizing warped event-cloud dispersion. The output field uses the same
 * convention as the previous dense-flow publisher:
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
  float flow_reg_lambda = 0.0f;
  int flow_reg_sweeps = 0;
  float flow_reg_sigma = 1e9f;
  int smooth_iters = 0;
  float smooth_alpha = 0.0f;
  bool refine_enabled = false;
  int refine_iters = 2;
  float refine_huber_delta = 0.01f;
  float max_speed_px_s = 4000.0f;
  int time_aware_order = 0;  // 0 legacy tensor, 1 const-vel dispersion, 2 local accel model
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
         a.flow_reg_lambda == b.flow_reg_lambda &&
         a.flow_reg_sweeps == b.flow_reg_sweeps &&
         a.flow_reg_sigma == b.flow_reg_sigma &&
         a.smooth_iters == b.smooth_iters &&
         a.smooth_alpha == b.smooth_alpha &&
         a.refine_enabled == b.refine_enabled &&
         a.refine_iters == b.refine_iters &&
         a.refine_huber_delta == b.refine_huber_delta &&
         a.max_speed_px_s == b.max_speed_px_s &&
         a.time_aware_order == b.time_aware_order;
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
  int final_full_rank_tiles = 0;
  int final_aperture_tiles = 0;
  int final_fallback_tiles = 0;
  int timeaware_support_fallback_tiles = 0;
  int timeaware_reject_fallback_tiles = 0;
  int final_timeaware_support_fallback_tiles = 0;
  int final_timeaware_reject_fallback_tiles = 0;
  int timeaware_low_cells_fallback_tiles = 0;
  int timeaware_low_mass_fallback_tiles = 0;
  int timeaware_no_time_fallback_tiles = 0;
  int timeaware_low_lambda_fallback_tiles = 0;
  int timeaware_solve_fail_fallback_tiles = 0;
  int timeaware_no_improve_fallback_tiles = 0;
  int timeaware_no_focus_fallback_tiles = 0;
  int final_timeaware_low_cells_fallback_tiles = 0;
  int final_timeaware_low_mass_fallback_tiles = 0;
  int final_timeaware_no_time_fallback_tiles = 0;
  int final_timeaware_low_lambda_fallback_tiles = 0;
  int final_timeaware_solve_fail_fallback_tiles = 0;
  int final_timeaware_no_improve_fallback_tiles = 0;
  int final_timeaware_no_focus_fallback_tiles = 0;
  int reg_total_tiles = 0;
  int reg_modified_tiles = 0;
  int reg_legacy_geometry_tiles = 0;
  int reg_warped_geometry_tiles = 0;
  double reg_mean_delta_speed = 0.0;
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

/// Extra per-cell temporal moments for the local acceleration estimator
/// (time_aware_order == 2): sums of a*tau^3, a*tau^4, a*dx*tau^2, a*dy*tau^2
/// (dx,dy cell-centre-relative, tau == dt). This is a local Taylor model, not
/// the Shiba/Gallego time-aware PDE formulation. Kept out of CellMoments so the
/// constant-velocity fast path stays one cache line.
struct CellMomentsT
{
  float stau3 = 0.0f;
  float stau4 = 0.0f;
  float sxtau2 = 0.0f;
  float sytau2 = 0.0f;
};

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
    tile_warp_geom_(params_.time_aware_order > 0
      ? static_cast<size_t>(final_tiles_) * final_tiles_
      : 0U),
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

    accel_final_ = Eigen::VectorXf::Zero(final_vars_);
    if (params_.time_aware_order > 0) {
      tile_accum_t_.assign(
        static_cast<size_t>(final_tiles_) * final_tiles_, TileMomentsT{});
    }
    if (params_.time_aware_order >= 2) {
      cells_t_.assign(cells_.size(), CellMomentsT{});
    }
  }

  bool compatible(int img_w, int img_h, const MomentFlowParams & params) const
  {
    return img_w_ == img_w && img_h_ == img_h && params_ == sanitize_params(params);
  }

  int final_tiles() const { return final_tiles_; }
  int num_vars() const { return final_vars_; }
  const MomentFlowProfile & profile() const { return profile_; }
  const Eigen::VectorXf & acceleration() const { return accel_final_; }

  /// Runtime multiplier on prior_lambda for the next solve(s). Residual passes
  /// of the compositional refinement use a weaker prior so the warm-start pull
  /// does not damp the correction steps (which would leave a systematic
  /// magnitude deficit after few iterations).
  void set_prior_scale(float s) { prior_scale_ = std::max(0.0f, s); }
  float prior_scale() const { return prior_scale_; }

  /// Runtime multiplier on the cell/tile mass gates. The gates are calibrated
  /// in raw event counts; when a busy window is strided down, scale them by
  /// the kept fraction so acceptance does not depend on the event rate.
  void set_mass_scale(float s) { mass_scale_ = std::clamp(s, 1e-3f, 1.0f); }
  float mass_scale() const { return mass_scale_; }

  void reset()
  {
    for (CellMoments & c : cells_) {
      clear_cell(c);
    }
    for (CellMomentsT & ct : cells_t_) {
      ct = CellMomentsT{};
    }
    for (TileWarpGeometry & g : tile_warp_geom_) {
      g = TileWarpGeometry{};
    }
    accel_final_.setZero();
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
      if (!cells_t_.empty()) {
        CellMomentsT & ct = cells_t_[cell_index(cx, cy)];
        const float dt2 = dt * dt;
        ct.stau3 += a * dt2 * dt;
        ct.stau4 += a * dt2 * dt2;
        ct.sxtau2 += a * dx * dt2;
        ct.sytau2 += a * dy * dt2;
      }
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
    profile_.final_full_rank_tiles = 0;
    profile_.final_aperture_tiles = 0;
    profile_.final_fallback_tiles = 0;
    profile_.timeaware_support_fallback_tiles = 0;
    profile_.timeaware_reject_fallback_tiles = 0;
    profile_.final_timeaware_support_fallback_tiles = 0;
    profile_.final_timeaware_reject_fallback_tiles = 0;
    profile_.timeaware_low_cells_fallback_tiles = 0;
    profile_.timeaware_low_mass_fallback_tiles = 0;
    profile_.timeaware_no_time_fallback_tiles = 0;
    profile_.timeaware_low_lambda_fallback_tiles = 0;
    profile_.timeaware_solve_fail_fallback_tiles = 0;
    profile_.timeaware_no_improve_fallback_tiles = 0;
    profile_.timeaware_no_focus_fallback_tiles = 0;
    profile_.final_timeaware_low_cells_fallback_tiles = 0;
    profile_.final_timeaware_low_mass_fallback_tiles = 0;
    profile_.final_timeaware_no_time_fallback_tiles = 0;
    profile_.final_timeaware_low_lambda_fallback_tiles = 0;
    profile_.final_timeaware_solve_fail_fallback_tiles = 0;
    profile_.final_timeaware_no_improve_fallback_tiles = 0;
    profile_.final_timeaware_no_focus_fallback_tiles = 0;
    profile_.reg_total_tiles = 0;
    profile_.reg_modified_tiles = 0;
    profile_.reg_legacy_geometry_tiles = 0;
    profile_.reg_warped_geometry_tiles = 0;
    profile_.reg_mean_delta_speed = 0.0;

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
    const bool has_solver_support = profile_.valid_cells > 0;
    if (!has_solver_support) {
      if (have_warm) {
        copy_vector(warm_start, F_out);
      } else {
        set_zero(F_out);
      }
      profile_.fallback_tiles += final_tiles_ * final_tiles_;
      profile_.final_fallback_tiles += final_tiles_ * final_tiles_;
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
      if (params_.time_aware_order == 0) {
        solve_scale(tiles, fallback, field);
      } else {
        Eigen::VectorXf stable_fallback(field.size());
        const int saved_full_rank = profile_.full_rank_tiles;
        const int saved_aperture = profile_.aperture_tiles;
        const int saved_fallback = profile_.fallback_tiles;
        const int saved_prior = profile_.prior_tiles;
        const int saved_final_full_rank = profile_.final_full_rank_tiles;
        const int saved_final_aperture = profile_.final_aperture_tiles;
        const int saved_final_fallback = profile_.final_fallback_tiles;
        solve_scale(tiles, fallback, stable_fallback);
        profile_.full_rank_tiles = saved_full_rank;
        profile_.aperture_tiles = saved_aperture;
        profile_.fallback_tiles = saved_fallback;
        profile_.prior_tiles = saved_prior;
        profile_.final_full_rank_tiles = saved_final_full_rank;
        profile_.final_aperture_tiles = saved_final_aperture;
        profile_.final_fallback_tiles = saved_final_fallback;
        Eigen::VectorXf * accel_out =
          (l == params_.num_scales - 1 && params_.time_aware_order >= 2)
            ? &accel_final_ : nullptr;
        solve_scale_timeaware(tiles, stable_fallback, field, accel_out);
      }
      profile_.stage_b_ms += elapsed_ms(t_b, Clock::now());

      const auto t_spatial = Clock::now();
      regularize_field_coupled(tiles, field);
      profile_.smooth_ms += elapsed_ms(t_spatial, Clock::now());
    }

    copy_vector(scale_fields_.back(), F_out);
    profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
  }

  void final_tile_confidence(std::vector<float> & conf) const
  {
    const int n = final_tiles_ * final_tiles_;
    conf.resize(static_cast<size_t>(n));
    if (params_.time_aware_order >= 1 && tile_warp_geom_.size() >= static_cast<size_t>(n)) {
      for (int k = 0; k < n; ++k) {
        const TileWarpGeometry & g = tile_warp_geom_[static_cast<size_t>(k)];
        const bool usable =
          g.valid && std::isfinite(g.confidence) && g.confidence > 0.0f;
        conf[k] = usable ? g.confidence : 0.0f;
      }
      return;
    }

    for (int k = 0; k < n; ++k) {
      const TileAccum & a = tile_accum_[static_cast<size_t>(k)];
      float lmin = 0.0f, lmax = 0.0f;
      eig2(a.mxx, a.mxy, a.myy, lmin, lmax);
      conf[k] = (std::isfinite(lmin) && lmin > 0.0f) ? lmin : 0.0f;
    }
  }

private:
#ifdef EVENT_DETECTOR_CPP_MOMENT_FLOW_TEST_ACCESS
  friend struct MomentFlowTestAccess;
#endif

  enum class TimeAwareFallbackReason
  {
    LowCells,
    LowMass,
    NoTimeVariance,
    LowLambda,
    SolveFail,
    NoImprove,
    NoFocus
  };

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

  struct TileMomentsT
  {
    double P0 = 0.0, P1 = 0.0, P2 = 0.0, P3 = 0.0, P4 = 0.0;
    double Qx0 = 0.0, Qx1 = 0.0, Qx2 = 0.0;
    double Qy0 = 0.0, Qy1 = 0.0, Qy2 = 0.0;
    double Rxx = 0.0, Rxy = 0.0, Ryy = 0.0;
    int count = 0;
  };

  struct TileWarpGeometry
  {
    bool valid = false;
    float cov_xx = 0.0f;
    float cov_xy = 0.0f;
    float cov_yy = 0.0f;
    float lmin = 0.0f;
    float lmax = 0.0f;
    float tangent_x = 1.0f;
    float tangent_y = 0.0f;
    float normal_x = 0.0f;
    float normal_y = 1.0f;
    float mass = 0.0f;
    float confidence = 0.0f;
    float focus_phi = 0.0f;
    float data_mxx = 0.0f;
    float data_mxy = 0.0f;
    float data_myy = 0.0f;
    float data_bx = 0.0f;
    float data_by = 0.0f;
  };

  static constexpr int64_t kRebaseThresholdUs = 250000;
  static constexpr float kPruneMass = 1e-4f;
  static constexpr float kMinTimeVariance = 1e-12f;
  // Significance multiplier for the NoImprove gate: the candidate's scatter
  // reduction over the fallback warp must exceed this many times the expected
  // spurious reduction from fitting `dof` motion parameters to noise.
  static constexpr double kNoImproveSignificance = 5.0;

  int img_w_;
  int img_h_;
  MomentFlowParams params_;
  int cells_x_;
  int cells_y_;
  int final_tiles_;
  int final_vars_;
  std::vector<CellMoments> cells_;
  std::vector<CellMomentsT> cells_t_;
  std::vector<CellFit> fits_;
  std::vector<float> cell_center_x_;
  std::vector<float> cell_center_y_;
  std::vector<TileAccum> tile_accum_;
  std::vector<TileMomentsT> tile_accum_t_;
  std::vector<TileWarpGeometry> tile_warp_geom_;
  std::vector<Eigen::VectorXf> scale_fields_;
  std::vector<Eigen::VectorXf> scale_fallback_;
  Eigen::VectorXf smooth_scratch_;
  Eigen::VectorXf accel_final_;
  MomentFlowProfile profile_;
  float prior_scale_ = 1.0f;
  float mass_scale_ = 1.0f;
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
    p.flow_reg_lambda = std::max(0.0f, p.flow_reg_lambda);
    p.flow_reg_sweeps = std::clamp(p.flow_reg_sweeps, 0, 16);
    p.flow_reg_sigma = std::max(1e-6f, p.flow_reg_sigma);
    p.smooth_iters = std::clamp(p.smooth_iters, 0, 16);
    p.smooth_alpha = std::clamp(p.smooth_alpha, 0.0f, 1.0f);
    p.refine_iters = std::max(0, p.refine_iters);
    p.refine_huber_delta = std::max(1e-9f, p.refine_huber_delta);
    p.max_speed_px_s = std::max(1.0f, p.max_speed_px_s);
    p.time_aware_order = std::clamp(p.time_aware_order, 0, 2);
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
    for (size_t k = 0; k < cells_.size(); ++k) {
      CellMoments & c = cells_[k];
      if (c.w <= 0.0f) {
        continue;
      }
      const float old_st = c.st;
      const float old_stt = c.stt;
      if (!cells_t_.empty()) {
        CellMomentsT & ct = cells_t_[k];
        const float d = delta;
        const float old_stau3 = ct.stau3;
        ct.stau4 = ct.stau4 - 4.0f * d * old_stau3 + 6.0f * d * d * old_stt
                   - 4.0f * d * d * d * old_st + d * d * d * d * c.w;
        ct.stau3 = old_stau3 - 3.0f * d * old_stt + 3.0f * d * d * old_st
                   - d * d * d * c.w;
        ct.sxtau2 = ct.sxtau2 - 2.0f * d * c.sxt + d * d * c.sx;
        ct.sytau2 = ct.sytau2 - 2.0f * d * c.syt + d * d * c.sy;
      }
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

  struct Cholesky3
  {
    double l00 = 0.0, l10 = 0.0, l20 = 0.0;
    double l11 = 0.0, l21 = 0.0, l22 = 0.0;
  };

  static bool factor3_spd(
    double a00, double a01, double a02,
    double a11, double a12, double a22,
    Cholesky3 & f)
  {
    constexpr double eps = 1e-24;
    if (!(a00 > eps) || !std::isfinite(a00)) {
      return false;
    }
    f.l00 = std::sqrt(a00);
    f.l10 = a01 / f.l00;
    f.l20 = a02 / f.l00;

    const double d11 = a11 - f.l10 * f.l10;
    if (!(d11 > eps) || !std::isfinite(d11)) {
      return false;
    }
    f.l11 = std::sqrt(d11);
    f.l21 = (a12 - f.l20 * f.l10) / f.l11;

    const double d22 = a22 - f.l20 * f.l20 - f.l21 * f.l21;
    if (!(d22 > eps) || !std::isfinite(d22)) {
      return false;
    }
    f.l22 = std::sqrt(d22);
    return std::isfinite(f.l00) && std::isfinite(f.l11) && std::isfinite(f.l22);
  }

  static bool solve3_cholesky(
    const Cholesky3 & f, double r0, double r1, double r2,
    double & x0, double & x1, double & x2)
  {
    const double y0 = r0 / f.l00;
    const double y1 = (r1 - f.l10 * y0) / f.l11;
    const double y2 = (r2 - f.l20 * y0 - f.l21 * y1) / f.l22;

    x2 = y2 / f.l22;
    x1 = (y1 - f.l21 * x2) / f.l11;
    x0 = (y0 - f.l10 * x1 - f.l20 * x2) / f.l00;
    return std::isfinite(x0) && std::isfinite(x1) && std::isfinite(x2);
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
      if (c.w < mass_scale_ * params_.cell_min_mass) {
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
            a.rho < mass_scale_ * params_.tile_min_mass ||
            !(lmax >= params_.tile_min_lambda))
        {
          out_F[2 * k] = fb_Fx;
          out_F[2 * k + 1] = fb_Fy;
          profile_.fallback_tiles += 1;
          if (tiles == final_tiles_) {
            profile_.final_fallback_tiles += 1;
          }
          continue;
        }

        float vx_phys = 0.0f;
        float vy_phys = 0.0f;
        const float tile_eps = params_.tikhonov_eps * std::max(lmax, 1e-12f);
        const float prior = prior_scale_ * params_.prior_lambda * std::max(lmax, 1e-12f);
        const bool solved = solve2(
          a.mxx + tile_eps + prior, a.mxy,
          a.myy + tile_eps + prior,
          a.bx + prior * fb_px, a.by + prior * fb_py,
          vx_phys, vy_phys);
        if (!solved) {
          out_F[2 * k] = fb_Fx;
          out_F[2 * k + 1] = fb_Fy;
          profile_.fallback_tiles += 1;
          if (tiles == final_tiles_) {
            profile_.final_fallback_tiles += 1;
          }
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
          if (tiles == final_tiles_) {
            profile_.final_aperture_tiles += 1;
          }
        } else {
          profile_.full_rank_tiles += 1;
          if (tiles == final_tiles_) {
            profile_.final_full_rank_tiles += 1;
          }
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

  // Closed-form moment-domain surrogate of contrast maximization based on
  // minimizing warped event-cloud dispersion. Under locally rigid translation
  // and stationary edge sampling the dispersion optimum coincides with the true
  // velocity / CMax optimum. order=1 is reference-time invariant; order=2 is a
  // local acceleration model solved as a joint quadratic OLS, not a time-aware
  // PDE formulation. No iteration, re-warp, or autodiff.
  void solve_scale_timeaware(
    int tiles,
    const Eigen::VectorXf & fallback_F,
    Eigen::VectorXf & out_F,
    Eigen::VectorXf * out_A)
  {
    const int order = params_.time_aware_order;
    const bool have_t = (order >= 2) && !cells_t_.empty();
    const int n_tiles = tiles * tiles;
    for (int i = 0; i < n_tiles; ++i) {
      tile_accum_t_[static_cast<size_t>(i)] = TileMomentsT{};
      if (static_cast<size_t>(i) < tile_warp_geom_.size()) {
        tile_warp_geom_[static_cast<size_t>(i)] = TileWarpGeometry{};
      }
    }

    // Scatter per-cell moments into their tile, rebuilding absolute-frame
    // monomials sum a*x*tau^j from the cell-centre-relative base moments.
    for (size_t k = 0; k < cells_.size(); ++k) {
      const CellMoments & c = cells_[k];
      if (!(c.w >= mass_scale_ * params_.cell_min_mass)) {
        continue;
      }
      const double q = 1.0;  // event-uniform: dispersion-min wants every massive
                             // cell, not only the clean single-orientation fits
      const int tx = std::clamp(
        static_cast<int>(cell_center_x_[k] * tiles / std::max(1, img_w_)), 0, tiles - 1);
      const int ty = std::clamp(
        static_cast<int>(cell_center_y_[k] * tiles / std::max(1, img_h_)), 0, tiles - 1);
      TileMomentsT & a = tile_accum_t_[static_cast<size_t>(ty * tiles + tx)];
      const double cx = cell_center_x_[k];
      const double cy = cell_center_y_[k];
      a.P0 += q * c.w;
      a.P1 += q * c.st;
      a.P2 += q * c.stt;
      a.Qx0 += q * (c.sx + cx * c.w);
      a.Qy0 += q * (c.sy + cy * c.w);
      a.Qx1 += q * (c.sxt + cx * c.st);
      a.Qy1 += q * (c.syt + cy * c.st);
      a.Rxx += q * (c.sxx + 2.0 * cx * c.sx + cx * cx * c.w);
      a.Rxy += q * (c.sxy + cx * c.sy + cy * c.sx + cx * cy * c.w);
      a.Ryy += q * (c.syy + 2.0 * cy * c.sy + cy * cy * c.w);
      a.count += 1;
      if (have_t) {
        const CellMomentsT & ct = cells_t_[k];
        a.P3 += q * ct.stau3;
        a.P4 += q * ct.stau4;
        a.Qx2 += q * (ct.sxtau2 + cx * c.stt);
        a.Qy2 += q * (ct.sytau2 + cy * c.stt);
      }
    }

    const double ridge = std::max(1e-12f, params_.tikhonov_eps);
    const double prior = std::max(0.0f, prior_scale_ * params_.prior_lambda);

    for (int ty = 0; ty < tiles; ++ty) {
      for (int tx = 0; tx < tiles; ++tx) {
        const int k = ty * tiles + tx;
        const TileMomentsT & a = tile_accum_t_[static_cast<size_t>(k)];
        const float fb_Fx = fallback_F[2 * k];
        const float fb_Fy = fallback_F[2 * k + 1];
        const double fb_px = -fb_Fx;
        const double fb_py = -fb_Fy;

        auto fall_back = [&](TimeAwareFallbackReason reason) {
          const bool candidate_rejected =
            reason == TimeAwareFallbackReason::NoImprove ||
            reason == TimeAwareFallbackReason::NoFocus;
          out_F[2 * k] = fb_Fx;
          out_F[2 * k + 1] = fb_Fy;
          if (static_cast<size_t>(k) < tile_warp_geom_.size()) {
            tile_warp_geom_[static_cast<size_t>(k)] = TileWarpGeometry{};
          }
          if (out_A) { (*out_A)[2 * k] = 0.0f; (*out_A)[2 * k + 1] = 0.0f; }
          profile_.fallback_tiles += 1;
          const bool final_scale = tiles == final_tiles_;
          if (candidate_rejected) {
            profile_.timeaware_reject_fallback_tiles += 1;
            if (final_scale) {
              profile_.final_timeaware_reject_fallback_tiles += 1;
            }
          } else {
            profile_.timeaware_support_fallback_tiles += 1;
            if (final_scale) {
              profile_.final_timeaware_support_fallback_tiles += 1;
            }
          }
          if (final_scale) {
            profile_.final_fallback_tiles += 1;
          }

          switch (reason) {
            case TimeAwareFallbackReason::LowCells:
              profile_.timeaware_low_cells_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_low_cells_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::LowMass:
              profile_.timeaware_low_mass_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_low_mass_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::NoTimeVariance:
              profile_.timeaware_no_time_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_no_time_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::LowLambda:
              profile_.timeaware_low_lambda_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_low_lambda_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::SolveFail:
              profile_.timeaware_solve_fail_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_solve_fail_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::NoImprove:
              profile_.timeaware_no_improve_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_no_improve_fallback_tiles += 1; }
              break;
            case TimeAwareFallbackReason::NoFocus:
              profile_.timeaware_no_focus_fallback_tiles += 1;
              if (final_scale) { profile_.final_timeaware_no_focus_fallback_tiles += 1; }
              break;
          }
        };

        if (a.count < std::max(1, params_.tile_min_cells)) {
          fall_back(TimeAwareFallbackReason::LowCells);
          continue;
        }
        if (a.P0 < mass_scale_ * params_.tile_min_mass) {
          fall_back(TimeAwareFallbackReason::LowMass);
          continue;
        }
        if (!(a.P2 > 0.0)) {
          fall_back(TimeAwareFallbackReason::NoTimeVariance);
          continue;
        }

        const double inv_p0 = 1.0 / a.P0;
        const double cxx = std::max(0.0, a.Rxx * inv_p0 - a.Qx0 * a.Qx0 * inv_p0 * inv_p0);
        const double cxy = a.Rxy * inv_p0 - a.Qx0 * a.Qy0 * inv_p0 * inv_p0;
        const double cyy = std::max(0.0, a.Ryy * inv_p0 - a.Qy0 * a.Qy0 * inv_p0 * inv_p0);
        float spatial_lmin = 0.0f;
        float spatial_lmax = 0.0f;
        eig2(
          static_cast<float>(cxx), static_cast<float>(cxy), static_cast<float>(cyy),
          spatial_lmin, spatial_lmax);
        if (!(spatial_lmax >= params_.tile_min_lambda)) {
          fall_back(TimeAwareFallbackReason::LowLambda);
          continue;
        }

        const double denom_v = a.P2 - a.P1 * a.P1 / a.P0;
        if (!(std::abs(denom_v) > 1e-20) || !std::isfinite(denom_v)) {
          fall_back(TimeAwareFallbackReason::NoTimeVariance);
          continue;
        }

        double vx_phys = 0.0, vy_phys = 0.0, ax_phys = 0.0, ay_phys = 0.0;
        double s_tau = std::sqrt(std::max(0.0, denom_v / a.P0));
        bool ok = false;

        if (order >= 2 && have_t) {
          const double tau_mean = a.P1 / a.P0;
          const double tm2 = tau_mean * tau_mean;
          const double tm3 = tm2 * tau_mean;
          const double tm4 = tm2 * tm2;

          const double S0 = a.P0;
          const double S1 = a.P1 - tau_mean * a.P0;
          const double S2 = a.P2 - 2.0 * tau_mean * a.P1 + tm2 * a.P0;
          const double S3 = a.P3 - 3.0 * tau_mean * a.P2 + 3.0 * tm2 * a.P1 -
                            tm3 * a.P0;
          const double S4 = a.P4 - 4.0 * tau_mean * a.P3 + 6.0 * tm2 * a.P2 -
                            4.0 * tm3 * a.P1 + tm4 * a.P0;

          if (!(S2 > 1e-20) || !std::isfinite(S2) || !std::isfinite(S4)) {
            fall_back(TimeAwareFallbackReason::NoTimeVariance);
            continue;
          }
          s_tau = std::sqrt(std::max(0.0, S2 / a.P0));

          const double Cx0 = a.Qx0;
          const double Cy0 = a.Qy0;
          const double Cx1 = a.Qx1 - tau_mean * a.Qx0;
          const double Cy1 = a.Qy1 - tau_mean * a.Qy0;
          const double Cx2 = a.Qx2 - 2.0 * tau_mean * a.Qx1 + tm2 * a.Qx0;
          const double Cy2 = a.Qy2 - 2.0 * tau_mean * a.Qy1 + tm2 * a.Qy0;

          const double vel_scale = std::max(S2, 1e-24);
          const double acc_scale = std::max(0.25 * S4, 1e-24);
          const double vel_prior_mass = prior * vel_scale;
          const double vel_ridge_mass = ridge * vel_scale;
          const double accel_ridge_mass = std::max(ridge, 8.0 * prior) * acc_scale;
          const double vel_prior_tau = vel_prior_mass * tau_mean;

          // The warm-start prior targets physical v(0)=vx_c-a*tau_mean, so add
          // vel_prior_mass * [0,1,-tau_mean]^T[0,1,-tau_mean] to the centered OLS.

          Cholesky3 fact;
          const bool factored = factor3_spd(
            S0, S1, 0.5 * S2,
            S2 + vel_ridge_mass + vel_prior_mass, 0.5 * S3 - vel_prior_tau,
            0.25 * S4 + accel_ridge_mass + vel_prior_tau * tau_mean,
            fact);
          if (factored) {
            double x0_c = 0.0, vx_c = 0.0, ax_c = 0.0;
            double y0_c = 0.0, vy_c = 0.0, ay_c = 0.0;
            const bool sx_ok = solve3_cholesky(
              fact,
              Cx0,
              Cx1 + vel_prior_mass * fb_px,
              0.5 * Cx2 - vel_prior_tau * fb_px,
              x0_c, vx_c, ax_c);
            const bool sy_ok = solve3_cholesky(
              fact,
              Cy0,
              Cy1 + vel_prior_mass * fb_py,
              0.5 * Cy2 - vel_prior_tau * fb_py,
              y0_c, vy_c, ay_c);
            (void)x0_c;
            (void)y0_c;
            if (sx_ok && sy_ok) {
              vx_phys = vx_c - ax_c * tau_mean;
              vy_phys = vy_c - ay_c * tau_mean;
              ax_phys = ax_c;
              ay_phys = ay_c;
              ok = std::isfinite(vx_phys) && std::isfinite(vy_phys) &&
                   std::isfinite(ax_phys) && std::isfinite(ay_phys);
            }
          }
        } else {
          // order=1 is reference-time invariant: a centred covariance ratio
          // estimates the constant-velocity dispersion optimum.
          const double prior_mass = prior * denom_v;
          vx_phys = (a.Qx1 - a.Qx0 * a.P1 / a.P0 + prior_mass * fb_px) /
                    (denom_v + prior_mass);
          vy_phys = (a.Qy1 - a.Qy0 * a.P1 / a.P0 + prior_mass * fb_py) /
                    (denom_v + prior_mass);
          ok = std::isfinite(vx_phys) && std::isfinite(vy_phys);
        }
        if (s_tau > 0.0) {
          const double da = std::hypot(ax_phys, ay_phys) * s_tau;
          const double max_delta_v = 0.25 * static_cast<double>(params_.max_speed_px_s);
          if (da > max_delta_v && da > 0.0) {
            const double sc = max_delta_v / da;
            ax_phys *= sc;
            ay_phys *= sc;
          }
        }
        if (!ok) {
          fall_back(TimeAwareFallbackReason::SolveFail);
          continue;
        }

        // Moment-domain multi-reference focus phi = Var_id / Var_warp. phi <= 1
        // means the warp does not sharpen -> reject (analogue of the paper's f<=1).
        const double var_id =
          (a.Rxx - a.Qx0 * a.Qx0 / a.P0) + (a.Ryy - a.Qy0 * a.Qy0 / a.P0);
        auto warped_stats = [&](double vx, double vy, double ax, double ay,
            double & var_w, TileWarpGeometry & geom) {
            const double sxp = a.Qx0 - vx * a.P1 - 0.5 * ax * a.P2;
            const double syp = a.Qy0 - vy * a.P1 - 0.5 * ay * a.P2;
            const double sxxp = a.Rxx - 2.0 * vx * a.Qx1 - ax * a.Qx2
              + vx * vx * a.P2 + vx * ax * a.P3 + 0.25 * ax * ax * a.P4;
            const double sxyp = a.Rxy - vx * a.Qy1 - vy * a.Qx1
              - 0.5 * ax * a.Qy2 - 0.5 * ay * a.Qx2 + vx * vy * a.P2
              + 0.5 * (vx * ay + vy * ax) * a.P3 + 0.25 * ax * ay * a.P4;
            const double syyp = a.Ryy - 2.0 * vy * a.Qy1 - ay * a.Qy2
              + vy * vy * a.P2 + vy * ay * a.P3 + 0.25 * ay * ay * a.P4;
            const double inv_p0 = 1.0 / a.P0;
            const double cov_xx = std::max(0.0, sxxp * inv_p0 - sxp * sxp * inv_p0 * inv_p0);
            const double cov_xy = sxyp * inv_p0 - sxp * syp * inv_p0 * inv_p0;
            const double cov_yy = std::max(0.0, syyp * inv_p0 - syp * syp * inv_p0 * inv_p0);
            geom = TileWarpGeometry{};
            geom.valid = true;
            geom.cov_xx = static_cast<float>(cov_xx);
            geom.cov_xy = static_cast<float>(cov_xy);
            geom.cov_yy = static_cast<float>(cov_yy);
            eig2(
              static_cast<float>(cov_xx), static_cast<float>(cov_xy),
              static_cast<float>(cov_yy), geom.lmin, geom.lmax);
            dominant_eigenvector(
              static_cast<float>(cov_xx), static_cast<float>(cov_xy),
              static_cast<float>(cov_yy), geom.lmax, geom.tangent_x, geom.tangent_y);
            geom.normal_x = -geom.tangent_y;
            geom.normal_y = geom.tangent_x;
            geom.mass = static_cast<float>(a.P0);
            var_w = (sxxp - sxp * sxp * inv_p0) + (syyp - syp * syp * inv_p0);
          };

        double var_w = 0.0;
        TileWarpGeometry warp_geom;
        warped_stats(vx_phys, vy_phys, ax_phys, ay_phys, var_w, warp_geom);

        const bool aperture_limited =
          warp_geom.lmin / std::max(warp_geom.lmax, 1e-12f) < params_.aperture_ratio;
        if (aperture_limited) {
          const double nx_norm = warp_geom.normal_x;
          const double ny_norm = warp_geom.normal_y;
          const double normal_v = vx_phys * nx_norm + vy_phys * ny_norm;
          const double tangent_v =
            fb_px * warp_geom.tangent_x + fb_py * warp_geom.tangent_y;
          vx_phys = normal_v * nx_norm + tangent_v * warp_geom.tangent_x;
          vy_phys = normal_v * ny_norm + tangent_v * warp_geom.tangent_y;

          const double normal_a = ax_phys * nx_norm + ay_phys * ny_norm;
          ax_phys = normal_a * nx_norm;
          ay_phys = normal_a * ny_norm;
          warped_stats(vx_phys, vy_phys, ax_phys, ay_phys, var_w, warp_geom);
        }

        double fallback_var_w = 0.0;
        TileWarpGeometry fallback_geom;
        warped_stats(
          fb_px, fb_py, 0.0, 0.0,
          fallback_var_w, fallback_geom);
        // NoImprove: significance test on the scatter reduction. The previous
        // criterion (var_w > 0.98 * fallback_var_w) required a 2% reduction of
        // the TOTAL scatter, which is dominated by the static geometric extent
        // of the tile, while the motion-explainable part is only
        // O(Var(tau) * |dv|^2) per unit mass; with short windows this rejected
        // essentially every candidate. Instead, accept only if the observed
        // reduction exceeds the expected spurious reduction from fitting `dof`
        // motion parameters to noise: kNoImproveSignificance * dof * sigma^2,
        // with sigma^2 = per-event residual scatter of the candidate warp.
        // n_eff = P0 is a conservative effective sample size (weights <= 1).
        const double n_eff = std::max(1.0, a.P0);
        const double dof = (order >= 2 && have_t) ? 4.0 : 2.0;
        const double noise_floor =
          kNoImproveSignificance * dof * (std::max(0.0, var_w) / n_eff);
        // if (fallback_var_w > 1e-12 &&
        //   (fallback_var_w - var_w) < noise_floor)
        // {
        //   fall_back(TimeAwareFallbackReason::NoImprove);
        //   continue;
        // }

        const double phi = (var_w > 1e-12) ? var_id / var_w : 0.0;
        // if (!(phi > 1.0) || !std::isfinite(phi)) {
        //   fall_back(TimeAwareFallbackReason::NoFocus);
        //   continue;
        // }

        if (aperture_limited) {
          profile_.aperture_tiles += 1;
          if (tiles == final_tiles_) {
            profile_.final_aperture_tiles += 1;
          }
        } else {
          profile_.full_rank_tiles += 1;
          if (tiles == final_tiles_) {
            profile_.final_full_rank_tiles += 1;
          }
        }
        if (prior > 0.0) {
          profile_.prior_tiles += 1;
        }

        float vxf = static_cast<float>(vx_phys);
        float vyf = static_cast<float>(vy_phys);
        clamp_speed(vxf, vyf);
        vx_phys = vxf;
        vy_phys = vyf;
        out_F[2 * k] = -vxf;
        out_F[2 * k + 1] = -vyf;

        if (out_A) {
          float axf = static_cast<float>(ax_phys);
          float ayf = static_cast<float>(ay_phys);
          const float da = std::hypot(axf, ayf) * static_cast<float>(s_tau);
          const float max_delta_v = 0.25f * params_.max_speed_px_s;
          if (da > max_delta_v && da > 0.0f) {
            const float sc = max_delta_v / da;
            axf *= sc; ayf *= sc;
          }
          ax_phys = axf;
          ay_phys = ayf;
          (*out_A)[2 * k] = -axf;
          (*out_A)[2 * k + 1] = -ayf;
        }

        if (static_cast<size_t>(k) < tile_warp_geom_.size()) {
          double reg_var_w = 0.0;
          TileWarpGeometry reg_geom;
          warped_stats(vx_phys, vy_phys, ax_phys, ay_phys, reg_var_w, reg_geom);
          const float ratio =
            reg_geom.lmin / std::max(reg_geom.lmax, 1e-12f);
          const float data_conf =
            static_cast<float>(std::max(denom_v, 1e-12) *
            std::clamp((phi - 1.0) / std::max(phi, 1e-12), 0.05, 1.0));
          const bool reg_aperture =
            ratio < params_.aperture_ratio;
          const float normal_w = data_conf;
          const float tangent_w = reg_aperture
            ? data_conf * std::max(0.0f, ratio)
            : data_conf;
          const float nx = reg_geom.normal_x;
          const float ny = reg_geom.normal_y;
          const float txg = reg_geom.tangent_x;
          const float tyg = reg_geom.tangent_y;
          reg_geom.confidence = data_conf;
          reg_geom.focus_phi = static_cast<float>(phi);
          reg_geom.data_mxx = normal_w * nx * nx + tangent_w * txg * txg;
          reg_geom.data_mxy = normal_w * nx * ny + tangent_w * txg * tyg;
          reg_geom.data_myy = normal_w * ny * ny + tangent_w * tyg * tyg;
          reg_geom.data_bx = reg_geom.data_mxx * vxf + reg_geom.data_mxy * vyf;
          reg_geom.data_by = reg_geom.data_mxy * vxf + reg_geom.data_myy * vyf;
          tile_warp_geom_[static_cast<size_t>(k)] = reg_geom;
          (void)reg_var_w;
        }
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

  const Eigen::VectorXf * fallback_for_tiles(int tiles) const
  {
    const int n = 2 * tiles * tiles;
    for (const Eigen::VectorXf & fallback : scale_fallback_) {
      if (fallback.size() == n) {
        return &fallback;
      }
    }
    return nullptr;
  }

  float tile_data_confidence(int k, bool use_warped_geometry) const
  {
    if (use_warped_geometry) {
      if (static_cast<size_t>(k) >= tile_warp_geom_.size()) {
        return 0.0f;
      }
      const TileWarpGeometry & g = tile_warp_geom_[static_cast<size_t>(k)];
      if (g.valid && g.confidence > 0.0f && std::isfinite(g.confidence)) {
        return g.confidence;
      }
      return 0.0f;
    }

    const TileAccum & a = tile_accum_[static_cast<size_t>(k)];
    const float trace = a.mxx + a.myy;
    if (a.rho > 0.0f && trace > 0.0f && std::isfinite(trace)) {
      return trace;
    }
    return 0.0f;
  }

  float robust_coupling_psi(const Eigen::VectorXf & field, int k, int nk) const
  {
    const float sigma = params_.flow_reg_sigma;
    if (!(sigma < 1e8f)) {
      return 1.0f;
    }
    const float dx = field[2 * k] - field[2 * nk];
    const float dy = field[2 * k + 1] - field[2 * nk + 1];
    const float z = (dx * dx + dy * dy) / (sigma * sigma);
    return 1.0f / std::sqrt(1.0f + z);
  }

  void regularize_field_coupled(int tiles, Eigen::VectorXf & field)
  {
    if (params_.flow_reg_lambda <= 0.0f || params_.flow_reg_sweeps <= 0 || tiles <= 1) {
      return;
    }

    const int n_tiles = tiles * tiles;
    const int n_vars = 2 * n_tiles;
    if (field.size() < n_vars || smooth_scratch_.size() < n_vars) {
      return;
    }

    for (int i = 0; i < n_vars; ++i) {
      smooth_scratch_[i] = field[i];
    }

    const Eigen::VectorXf * fallback = fallback_for_tiles(tiles);
    const float lambda_s = params_.flow_reg_lambda;
    const bool use_warped_geometry =
      params_.time_aware_order >= 1 && tile_warp_geom_.size() >= static_cast<size_t>(n_tiles);
    if (use_warped_geometry) {
      profile_.reg_warped_geometry_tiles += n_tiles;
    } else {
      profile_.reg_legacy_geometry_tiles += n_tiles;
    }

    for (int sweep = 0; sweep < params_.flow_reg_sweeps; ++sweep) {
      for (int ty = 0; ty < tiles; ++ty) {
        for (int tx = 0; tx < tiles; ++tx) {
          const int k = ty * tiles + tx;

          float data_mxx = 0.0f;
          float data_mxy = 0.0f;
          float data_myy = 0.0f;
          float data_bx = 0.0f;
          float data_by = 0.0f;
          float ex = 1.0f;
          float ey = 0.0f;
          float tgx = 0.0f;
          float tgy = 1.0f;

          if (use_warped_geometry) {
            const TileWarpGeometry & g = tile_warp_geom_[static_cast<size_t>(k)];
            if (!g.valid || !(g.confidence > 0.0f)) {
              continue;
            }
            const float data_scale = std::max(g.confidence, 1e-12f);
            const float tile_eps = params_.tikhonov_eps * data_scale;
            const float anchor_vx = -smooth_scratch_[2 * k];
            const float anchor_vy = -smooth_scratch_[2 * k + 1];
            data_mxx = g.data_mxx + tile_eps;
            data_mxy = g.data_mxy;
            data_myy = g.data_myy + tile_eps;
            data_bx = g.data_bx + tile_eps * anchor_vx;
            data_by = g.data_by + tile_eps * anchor_vy;
            tgx = g.tangent_x;
            tgy = g.tangent_y;
          } else {
            const TileAccum & a = tile_accum_[static_cast<size_t>(k)];
            float lmin = 0.0f;
            float lmax = 0.0f;
            eig2(a.mxx, a.mxy, a.myy, lmin, lmax);
            const float data_scale = std::max(lmax, 1e-12f);
            const float tile_eps = params_.tikhonov_eps * data_scale;
            const float prior = prior_scale_ * params_.prior_lambda * data_scale;
            const float fb_px =
              fallback ? -(*fallback)[2 * k] : -smooth_scratch_[2 * k];
            const float fb_py =
              fallback ? -(*fallback)[2 * k + 1] : -smooth_scratch_[2 * k + 1];

            data_mxx = a.mxx + tile_eps + prior;
            data_mxy = a.mxy;
            data_myy = a.myy + tile_eps + prior;
            data_bx = a.bx + prior * fb_px;
            data_by = a.by + prior * fb_py;

            // Legacy structure tensor: dominant eigenvector is the edge normal,
            // so the weak/tangential direction is its perpendicular.
            dominant_eigenvector(a.mxx, a.mxy, a.myy, lmax, ex, ey);
            tgx = -ey;
            tgy = ex;
          }

          // Accumulo rank-1: penalita' sum_n w_n * (t . (v - v_n))^2.
          //   LHS += (sum_n w_n) * t t^T      (Cxx,Cxy,Cyy)
          //   RHS += (sum_n w_n * t.v_n) * t  (rt)
          float Cxx = 0.0f;
          float Cxy = 0.0f;
          float Cyy = 0.0f;
          float rt = 0.0f;
          float wt_sum = 0.0f;

          auto add_neighbor = [&](int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= tiles || ny >= tiles) {
              return;
            }
            const int nk = ny * tiles + nx;
            const float conf = tile_data_confidence(nk, use_warped_geometry);
            if (!(conf > 0.0f)) {
              return;
            }
            const float psi = robust_coupling_psi(field, k, nk);
            const float w = lambda_s * conf * psi;
            if (!(w > 0.0f) || !std::isfinite(w)) {
              return;
            }
            // velocita' fisica del vicino (field = -v_phys) e sua proiezione tangenziale
            const float vnx = -field[2 * nk];
            const float vny = -field[2 * nk + 1];
            const float proj = vnx * tgx + vny * tgy;
            wt_sum += w;
            Cxx += w * tgx * tgx;
            Cxy += w * tgx * tgy;
            Cyy += w * tgy * tgy;
            rt += w * proj;
          };

          add_neighbor(tx - 1, ty);
          add_neighbor(tx + 1, ty);
          add_neighbor(tx, ty - 1);
          add_neighbor(tx, ty + 1);

          if (!(wt_sum > 0.0f)) {
            continue;
          }

          // Normale: M_dato + tikhonov + prior (intatta). Tangenziale: + coupling.
          float vx_phys = -field[2 * k];
          float vy_phys = -field[2 * k + 1];
          const bool solved = solve2(
            data_mxx + Cxx, data_mxy + Cxy,
            data_myy + Cyy,
            data_bx + rt * tgx,
            data_by + rt * tgy,
            vx_phys, vy_phys);
          if (!solved) {
            continue;
          }

          clamp_speed(vx_phys, vy_phys);
          field[2 * k] = -vx_phys;
          field[2 * k + 1] = -vy_phys;
        }
      }
    }

    int modified = 0;
    double delta_sum = 0.0;
    for (int k = 0; k < n_tiles; ++k) {
      const double dx = static_cast<double>(field[2 * k] - smooth_scratch_[2 * k]);
      const double dy = static_cast<double>(field[2 * k + 1] - smooth_scratch_[2 * k + 1]);
      const double delta = std::hypot(dx, dy);
      delta_sum += delta;
      if (delta > 1e-4) {
        modified += 1;
      }
    }

    const double prev_sum = profile_.reg_mean_delta_speed * profile_.reg_total_tiles;
    profile_.reg_total_tiles += n_tiles;
    profile_.reg_modified_tiles += modified;
    profile_.reg_mean_delta_speed =
      (profile_.reg_total_tiles > 0)
        ? (prev_sum + delta_sum) / static_cast<double>(profile_.reg_total_tiles)
        : 0.0;
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
          int neighbors[9];
          int n = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int nx = tx + dx;
              const int ny = ty + dy;
              if (nx < 0 || ny < 0 || nx >= tiles || ny >= tiles) {
                continue;
              }
              neighbors[n++] = ny * tiles + nx;
            }
          }

          int best = k;
          float best_score = std::numeric_limits<float>::max();
          for (int ci = 0; ci < n; ++ci) {
            const int ck = neighbors[ci];
            const float cx = field[2 * ck];
            const float cy = field[2 * ck + 1];
            float score = 0.0f;
            for (int ni = 0; ni < n; ++ni) {
              const int nk = neighbors[ni];
              score += std::hypot(cx - field[2 * nk], cy - field[2 * nk + 1]);
            }
            if (score < best_score) {
              best_score = score;
              best = ck;
            }
          }

          smooth_scratch_[2 * k] = keep * field[2 * k] + alpha * field[2 * best];
          smooth_scratch_[2 * k + 1] =
            keep * field[2 * k + 1] + alpha * field[2 * best + 1];
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
