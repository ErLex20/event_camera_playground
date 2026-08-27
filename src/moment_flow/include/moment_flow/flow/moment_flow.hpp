/**
 * Moment-based incremental optical-flow estimator.
 *
 * Maintains spatio-temporal moments per sensor cell and solves a
 * closed-form moment-domain surrogate of contrast maximization based on
 * anisotropic normal-projected dispersion. A legacy structure-tensor solve is
 * still used internally as a stable fallback. The output field uses the same
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>

namespace moment_flow::flow
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
  float max_speed_px_s = 4000.0f;
};

inline bool operator==(const MomentFlowParams & a, const MomentFlowParams & b)
{
  return a.num_scales == b.num_scales &&
         a.cell_size_px == b.cell_size_px &&
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
  int final_full_rank_tiles = 0;
  int final_aperture_tiles = 0;
  int final_fallback_tiles = 0;
  int reg_total_tiles = 0;
  int reg_modified_tiles = 0;
  int reg_legacy_geometry_tiles = 0;
  int reg_warped_geometry_tiles = 0;
  double reg_mean_delta_speed = 0.0;
  double ingest_ms = 0.0;
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
    cell_normals_(cells_.size()),
    cell_center_x_(cells_.size(), 0.0f),
    cell_center_y_(cells_.size(), 0.0f),
    tile_accum_(static_cast<size_t>(final_tiles_) * final_tiles_),
    tile_aniso_(static_cast<size_t>(final_tiles_) * final_tiles_),
    tile_warp_geom_(static_cast<size_t>(final_tiles_) * final_tiles_),
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

  /// Threads for the per-event loops (accumulation, warping). 0 (the default)
  /// leaves the choice to the OpenMP runtime. Results do not depend on it: the
  /// work is partitioned so that every cell keeps the serial accumulation
  /// order, so any value yields bit-identical moments.
  void set_max_threads(int t) { max_threads_ = std::max(0, t); }
  int max_threads() const { return max_threads_; }

  void reset()
  {
    for (CellMoments & c : cells_) {
      clear_cell(c);
    }
    for (TileWarpGeometry & g : tile_warp_geom_) {
      g = TileWarpGeometry{};
    }
    time_origin_us_ = 0;
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

    // A mid-loop time rebase rewrites every cell, so it cannot happen while
    // threads accumulate concurrently. The window's largest |t| bounds the
    // reachable distance from the origin: when no event can trigger a rebase
    // the banded path is taken, otherwise the serial loop keeps the exact
    // original semantics (a rebase mid-window changes all later dt values).
    const int n_bands =
      (events.size() >= kMinParallelEvents) ? accumulation_bands() : 1;
    if (n_bands > 1 && !rebase_reachable(events, n_bands)) {
      ingest_banded(events, n_bands);
    } else {
      ingest_serial(events);
    }

    profile_.ingest_ms = elapsed_ms(t0, Clock::now());
  }

  void solve(const Eigen::VectorXf & warm_start, Eigen::VectorXf & F_out)
  {
    using Clock = std::chrono::steady_clock;
    const auto t_total = Clock::now();
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
    profile_.reg_total_tiles = 0;
    profile_.reg_modified_tiles = 0;
    profile_.reg_legacy_geometry_tiles = 0;
    profile_.reg_warped_geometry_tiles = 0;
    profile_.reg_mean_delta_speed = 0.0;

    if (F_out.size() != final_vars_) {
      return;
    }

    const auto t_a = Clock::now();
    compute_cell_fits();
    compute_cell_normals();
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

      solve_one_scale(l, fallback, field);
    }

    copy_vector(scale_fields_.back(), F_out);
    profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
  }

  /**
   * Coarse-to-fine solve (residual formulation). The composed field G starts
   * from the warm start; every scale estimates a residual with respect to the
   * events warped by the current G, and each level takes its fallback from the
   * level above. F_out follows the same warp convention as solve().
   */
  void solve_coarse_to_fine(
    const Events & events,
    const Eigen::VectorXf & warm_start,
    Eigen::VectorXf & F_out)
  {
    using Clock = std::chrono::steady_clock;
    const auto t_total = Clock::now();
    if (F_out.size() != final_vars_) {
      return;
    }

    const bool have_warm = warm_start.size() == final_vars_;
    Eigen::VectorXf G = Eigen::VectorXf::Zero(final_vars_);
    if (have_warm) {
      copy_vector(warm_start, G);
    }

    double ingest_ms_acc = 0.0;
    double stage_a_ms_acc = 0.0;
    auto ingest_current = [&](bool warp) {
        const auto t_i = Clock::now();
        reset();
        if (warp) {
          warp_events(events, G, warp_scratch_);
          ingest(warp_scratch_);
        } else {
          ingest(events);
        }
        ingest_ms_acc += elapsed_ms(t_i, Clock::now());
        const auto t_a = Clock::now();
        compute_cell_fits();
        compute_cell_normals();
        stage_a_ms_acc += elapsed_ms(t_a, Clock::now());
      };

    // Largest |t| relative to the reference time: converts residual tile
    // speeds into worst-case displacements for the adaptive re-warp trigger.
    float t_absmax = 0.0f;
    for (size_t k = 0; k < events.size(); ++k) {
      t_absmax = std::max(t_absmax, std::abs(events.t[k]));
    }

    ingest_current(have_warm);
    if (profile_.valid_cells == 0 && profile_.active_cells == 0) {
      copy_vector(G, F_out);
      profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
      return;
    }

    for (int l = 0; l < params_.num_scales; ++l) {
      const int tiles = 1 << l;
      Eigen::VectorXf & fallback = scale_fallback_[static_cast<size_t>(l)];
      Eigen::VectorXf & field = scale_fields_[static_cast<size_t>(l)];

      // Each level takes its fallback from the level above. Re-warping the
      // events between levels was measured to be dormant in slow scenes and
      // unpredictable in fast ones, so the pyramid propagates the field only.
      if (l == 0) {
        set_zero(fallback);
      } else {
        resample_field(
          scale_fields_[static_cast<size_t>(l - 1)], 1 << (l - 1), fallback, tiles);
      }

      solve_one_scale(l, fallback, field);
    }

    // Fold the final scale's residual and clamp the composed speed.
    for (int k = 0; k < final_tiles_ * final_tiles_; ++k) {
      float vx = G[2 * k] + scale_fields_.back()[2 * k];
      float vy = G[2 * k + 1] + scale_fields_.back()[2 * k + 1];
      if (!std::isfinite(vx) || !std::isfinite(vy)) {
        vx = have_warm ? warm_start[2 * k] : 0.0f;
        vy = have_warm ? warm_start[2 * k + 1] : 0.0f;
      }
      clamp_speed(vx, vy);
      F_out[2 * k] = vx;
      F_out[2 * k + 1] = vy;
    }
    profile_.ingest_ms = ingest_ms_acc;
    profile_.stage_a_ms = stage_a_ms_acc;
    profile_.total_solve_ms = elapsed_ms(t_total, Clock::now());
  }

  void final_tile_confidence(std::vector<float> & conf) const
  {
    const int n = final_tiles_ * final_tiles_;
    conf.resize(static_cast<size_t>(n));
    if (tile_warp_geom_.size() >= static_cast<size_t>(n)) {
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
#ifdef MOMENT_FLOW_CPP_MOMENT_FLOW_TEST_ACCESS
  friend struct MomentFlowTestAccess;
#endif

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

  /// Fused per-cell normal-flow information for the anisotropic solver:
  /// M = sum w n n^T (2x2 information matrix), b = sum w v_n n.
  struct TileAniso
  {
    double mxx = 0.0, mxy = 0.0, myy = 0.0;
    double bx = 0.0, by = 0.0;
    double mass = 0.0;
    int count = 0;
  };

  /// Tile-grid-independent part of a cell's contribution to the anisotropic
  /// GLS: computed once per ingest, then binned by every pyramid scale.
  struct CellNormal
  {
    double mxx = 0.0, mxy = 0.0, myy = 0.0;
    double bx = 0.0, by = 0.0;
    double mass = 0.0;
    bool valid = false;
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

  // Below this many events a window is not worth a thread team: spawning one
  // costs more than the work it removes from the critical path.
  static constexpr size_t kMinParallelEvents = 20000;
  static constexpr int64_t kRebaseThresholdUs = 250000;
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
  std::vector<CellNormal> cell_normals_;
  std::vector<float> cell_center_x_;
  std::vector<float> cell_center_y_;
  std::vector<TileAccum> tile_accum_;
  std::vector<TileAniso> tile_aniso_;
  std::vector<TileWarpGeometry> tile_warp_geom_;
  std::vector<Eigen::VectorXf> scale_fields_;
  std::vector<Eigen::VectorXf> scale_fallback_;
  Eigen::VectorXf smooth_scratch_;
  Events warp_scratch_;
  MomentFlowProfile profile_;
  float prior_scale_ = 1.0f;
  float mass_scale_ = 1.0f;
  int max_threads_ = 0;
  int64_t time_origin_us_ = 0;
  bool has_time_origin_ = false;

  static MomentFlowParams sanitize_params(MomentFlowParams p)
  {
    p.num_scales = std::clamp(p.num_scales, 1, 8);
    p.cell_size_px = std::max(1, p.cell_size_px);
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
      c.st -= delta * c.w;
      c.sxt -= delta * c.sx;
      c.syt -= delta * c.sy;
      c.stt = c.stt - 2.0f * delta * old_st + delta * delta * c.w;
    }
    time_origin_us_ = new_origin_us;
  }

  /// Threads used by the per-event and per-cell loops. Work is split into
  /// stripes of cell rows: every cell is touched by exactly one thread, in
  /// event/cell index order, so results do not depend on the thread count.
  int accumulation_bands() const
  {
#ifdef _OPENMP
    const int budget = (max_threads_ > 0) ? max_threads_ : omp_get_max_threads();
    return std::clamp(budget, 1, std::max(1, cells_y_));
#else
    return 1;
#endif
  }

  /// Whether any event in the window could move the time origin while
  /// accumulating: |t_ref - origin| + max|t| bounds the reachable distance.
  bool rebase_reachable(const Events & events, int n_bands) const
  {
    float t_absmax = 0.0f;
    const int64_t n = static_cast<int64_t>(events.size());
    #pragma omp parallel for schedule(static) num_threads(n_bands) \
    reduction(max : t_absmax)
    for (int64_t k = 0; k < n; ++k) {
      t_absmax = std::max(t_absmax, std::abs(events.t[static_cast<size_t>(k)]));
    }
    const int64_t reach =
      static_cast<int64_t>(std::llround(static_cast<double>(t_absmax) * 1e6));
    return std::llabs(events.t_ref_us - time_origin_us_) + reach > kRebaseThresholdUs;
  }

  /// Accumulate one in-frame event into its cell.
  void accumulate_event(float x, float y, int64_t t_us, size_t cell_k)
  {
    const float a = 1.0f;
    const float dx = x - cell_center_x_[cell_k];
    const float dy = y - cell_center_y_[cell_k];
    const float dt = static_cast<float>(t_us - time_origin_us_) * 1e-6f;

    CellMoments & c = cells_[cell_k];
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
  }

  void ingest_serial(const Events & events)
  {
    int ingested = 0;
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
      const int64_t t_us = events.t_ref_us +
        static_cast<int64_t>(std::llround(static_cast<double>(events.t[k]) * 1e6));
      if (std::llabs(t_us - time_origin_us_) > kRebaseThresholdUs) {
        rebase_time_origin(t_us);
      }
      accumulate_event(x, y, t_us, cell_index(cx, cy));
      ingested += 1;
    }
    profile_.events_ingested = ingested;
  }

  /// Every band streams the whole event array and keeps only the events whose
  /// cell row it owns: no atomics, no per-thread cell copies (each band's cell
  /// stripe stays cache-resident), and the moments are bit-identical to the
  /// serial accumulation because a cell's events are still applied in order.
  void ingest_banded(const Events & events, int n_bands)
  {
    const int64_t n = static_cast<int64_t>(events.size());
    int ingested = 0;
    #pragma omp parallel for schedule(static) num_threads(n_bands) \
    reduction(+ : ingested)
    for (int band = 0; band < n_bands; ++band) {
      const int cy_begin =
        static_cast<int>(static_cast<int64_t>(band) * cells_y_ / n_bands);
      const int cy_end =
        static_cast<int>(static_cast<int64_t>(band + 1) * cells_y_ / n_bands);
      if (cy_begin >= cy_end) {
        continue;
      }
      // Row ownership is tested in pixels, not cell indices: for an in-frame y
      // the cell row is trunc(y / cell_size) with no clamping, so the two are
      // equivalent and the rejected events cost two compares instead of an
      // integer division each.
      const float y_lo = static_cast<float>(cy_begin * params_.cell_size_px);
      const float y_hi = static_cast<float>(cy_end * params_.cell_size_px);
      for (int64_t i = 0; i < n; ++i) {
        const size_t k = static_cast<size_t>(i);
        const float y = events.y[k];
        if (!(y >= y_lo && y < y_hi)) {
          continue;
        }
        if (!(y < img_h_)) {
          continue;
        }
        const int cy = std::clamp(
          static_cast<int>(y) / params_.cell_size_px, 0, cells_y_ - 1);
        const float x = events.x[k];
        if (!(x >= 0.0f && x < img_w_)) {
          continue;
        }
        const int cx = std::clamp(
          static_cast<int>(x) / params_.cell_size_px, 0, cells_x_ - 1);
        const int64_t t_us = events.t_ref_us +
          static_cast<int64_t>(std::llround(static_cast<double>(events.t[k]) * 1e6));
        accumulate_event(x, y, t_us, cell_index(cx, cy));
        ingested += 1;
      }
    }
    profile_.events_ingested = ingested;
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

  /// Per-cell plane fits. Kept single-threaded on purpose: the loop is ~150 us
  /// for a full grid, short enough that spawning a thread team costs more than
  /// the work it saves.
  void compute_cell_fits()
  {
    const int64_t n_cells = static_cast<int64_t>(cells_.size());
    int active_cells = 0;
    int valid_cells = 0;
    int residual_reject_cells = 0;
    int speed_reject_cells = 0;

    for (int64_t kk = 0; kk < n_cells; ++kk) {
      const size_t k = static_cast<size_t>(kk);
      fits_[k] = CellFit{};
      const CellMoments & c = cells_[k];
      if (c.w > 0.0f) {
        active_cells += 1;
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
        residual_reject_cells += 1;
        continue;
      }
      const float g2 = gx * gx + gy * gy;
      const float min_g2 = 1.0f / (params_.max_speed_px_s * params_.max_speed_px_s);
      if (!(g2 >= min_g2)) {
        speed_reject_cells += 1;
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
      valid_cells += 1;
    }

    profile_.active_cells += active_cells;
    profile_.valid_cells += valid_cells;
    profile_.residual_reject_cells += residual_reject_cells;
    profile_.speed_reject_cells += speed_reject_cells;
  }

  /// Per-cell normal-flow constraints for the anisotropic tile GLS. These
  /// depend only on the cell moments, not on the tile grid, so they are
  /// computed once per ingest instead of once per pyramid scale (which is where
  /// the saving comes from). Single-threaded for the same reason as
  /// compute_cell_fits.
  void compute_cell_normals()
  {
    // Residual-variance floor of the per-cell normal-position regression
    // [px^2]: bounds the GLS weight of a perfectly explained cell (sub-pixel
    // splat jitter makes anything sharper unphysical).
    constexpr double kSigmaFloor2 = 0.25;

    const float cell_gate = mass_scale_ * params_.cell_min_mass;
    const int64_t n_cells = static_cast<int64_t>(cells_.size());

    for (int64_t kk = 0; kk < n_cells; ++kk) {
      const size_t k = static_cast<size_t>(kk);
      cell_normals_[k] = CellNormal{};
      const CellMoments & c = cells_[k];
      if (!(c.w >= cell_gate)) {
        continue;
      }
      const double inv_w = 1.0 / c.w;
      const double mx = c.sx * inv_w;
      const double my = c.sy * inv_w;
      const double mt = c.st * inv_w;
      const double cxx = std::max(0.0, c.sxx * inv_w - mx * mx);
      const double cxy = c.sxy * inv_w - mx * my;
      const double cyy = std::max(0.0, c.syy * inv_w - my * my);
      const double dx = c.sxt * inv_w - mx * mt;
      const double dy = c.syt * inv_w - my * mt;
      const double sc = c.stt * inv_w - mt * mt;
      if (!(sc > kMinTimeVariance)) {
        continue;
      }

      float spatial_lmin = 0.0f;
      float spatial_lmax = 0.0f;
      eig2(
        static_cast<float>(cxx), static_cast<float>(cxy), static_cast<float>(cyy),
        spatial_lmin, spatial_lmax);
      if (!(spatial_lmax >= params_.cell_min_lambda)) {
        continue;
      }

      const double dn2 = dx * dx + dy * dy;
      if (!(dn2 > 0.0) || !std::isfinite(dn2)) {
        continue;
      }
      // Normal-direction spatial variance n^T C n.
      const double nCn =
        (dx * (cxx * dx + cxy * dy) + dy * (cxy * dx + cyy * dy)) / dn2;
      if (!(nCn > 1e-12)) {
        continue;
      }
      // R^2 of the normal-position-vs-time regression: squared correlation
      // between n^T x and tau. Clean moving edge -> 1, noise -> 0.
      const double r2 = std::clamp(dn2 / (sc * nCn), 0.0, 1.0);
      if (!(1.0 - r2 <= params_.cell_max_residual_ratio)) {
        continue;
      }
      // Speed sanity: |v_n| beyond the clamp is noise, not motion.
      const double vn = std::sqrt(dn2) / sc;
      if (!(vn <= params_.max_speed_px_s) || !std::isfinite(vn)) {
        continue;
      }

      // GLS precision of v_n: (mass * Var(tau)) / residual variance.
      const double sigma2 = std::max(kSigmaFloor2, nCn * (1.0 - r2));
      const double w = c.w * sc / sigma2;
      if (!(w > 0.0) || !std::isfinite(w)) {
        continue;
      }

      const double inv_dn2 = 1.0 / dn2;
      CellNormal & cn = cell_normals_[k];
      cn.mxx = w * dx * dx * inv_dn2;
      cn.mxy = w * dx * dy * inv_dn2;
      cn.myy = w * dy * dy * inv_dn2;
      cn.bx = (w / sc) * dx;
      cn.by = (w / sc) * dy;
      cn.mass = c.w;
      cn.valid = true;
    }
  }

  /// One pyramid scale: legacy stable fallback, anisotropic solve, and the
  /// coupled spatial regularizer. Shared by solve() and solve_coarse_to_fine().
  void solve_one_scale(int l, const Eigen::VectorXf & fallback, Eigen::VectorXf & field)
  {
    using Clock = std::chrono::steady_clock;
    const int tiles = 1 << l;
    const auto t_b = Clock::now();
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
    solve_scale_anisotropic(tiles, stable_fallback, field);
    profile_.stage_b_ms += elapsed_ms(t_b, Clock::now());

    const auto t_spatial = Clock::now();
    regularize_field_coupled(tiles, field);
    profile_.smooth_ms += elapsed_ms(t_spatial, Clock::now());
  }

  /// Warp events to the reference time with a final-grid field (F warp
  /// convention: x' = x + t * F(x)). Out-of-bounds warps are dropped.
  /// Events are warped in independent chunks, each writing into its own slice
  /// of the destination (survivors can never exceed the chunk's input count);
  /// the slices are then closed up in order, so the surviving events keep the
  /// serial ordering the moment accumulation depends on.
  void warp_events(const Events & src, const Eigen::VectorXf & F, Events & dst) const
  {
    const size_t n_src = src.size();
    dst.t_ref_us = src.t_ref_us;
    dst.x.resize(n_src);
    dst.y.resize(n_src);
    dst.t.resize(n_src);

    const int n_chunks = std::clamp(
      accumulation_bands(), 1, static_cast<int>(std::max<size_t>(1, n_src / 4096)));
    std::vector<size_t> chunk_count(static_cast<size_t>(n_chunks), 0);

    #pragma omp parallel for schedule(static) num_threads(n_chunks)
    for (int c = 0; c < n_chunks; ++c) {
      const size_t begin = static_cast<size_t>(c) * n_src / n_chunks;
      const size_t end = static_cast<size_t>(c + 1) * n_src / n_chunks;
      size_t n = begin;
      for (size_t k = begin; k < end; ++k) {
        const float t = src.t[k];
        float vx, vy;
        sample_field(F, final_tiles_, src.x[k], src.y[k], vx, vy);
        const float wx = src.x[k] + t * vx;
        const float wy = src.y[k] + t * vy;
        if (!(wx >= 0.0f && wy >= 0.0f && wx < img_w_ && wy < img_h_)) {
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

  // Anisotropic normal-projected dispersion.
  //
  // The isotropic tile dispersion penalizes the warped scatter along the edge
  // TANGENT as well, a component that motion cannot reduce: for dense texture
  // crossing cells/tiles it acts as a zero-velocity attractor. Here every
  // cell contributes only the component it actually observes. The cell drift
  // d = Cov(x, tau) points along the local edge normal (tangential motion
  // produces no space-time correlation), so with
  //   n   = d / |d|            (edge normal)
  //   v_n = |d| / Var(tau)     (normal speed)
  //   R^2 = |d|^2 / (Var(tau) * n^T C n)   (regression quality, in [0,1])
  // the tile fuses the per-cell normal-flow constraints n^T v = v_n by GLS:
  //   M v = b,  M = sum w n n^T,  b = sum w v_n n = sum (w / Var(tau)) d,
  // with w = the OLS precision of v_n (residual variance floored to avoid
  // single-cell blowups). M's eigenstructure IS the aperture geometry of the
  // tile: full-rank tiles solve 2D flow, degenerate ones keep the tangential
  // component of the fallback.
  void solve_scale_anisotropic(
    int tiles,
    const Eigen::VectorXf & fallback_F,
    Eigen::VectorXf & out_F)
  {
    const int n_tiles = tiles * tiles;
    for (int i = 0; i < n_tiles; ++i) {
      tile_aniso_[static_cast<size_t>(i)] = TileAniso{};
      if (static_cast<size_t>(i) < tile_warp_geom_.size()) {
        tile_warp_geom_[static_cast<size_t>(i)] = TileWarpGeometry{};
      }
    }

    // Per-cell normal-flow constraints do not depend on the tile grid, so they
    // are computed once per ingest (compute_cell_normals) and here only binned
    // and accumulated, in cell-index order as before.
    for (size_t k = 0; k < cell_normals_.size(); ++k) {
      const CellNormal & cn = cell_normals_[k];
      if (!cn.valid) {
        continue;
      }
      const int tx = std::clamp(
        static_cast<int>(cell_center_x_[k] * tiles / std::max(1, img_w_)), 0, tiles - 1);
      const int ty = std::clamp(
        static_cast<int>(cell_center_y_[k] * tiles / std::max(1, img_h_)), 0, tiles - 1);
      TileAniso & a = tile_aniso_[static_cast<size_t>(ty * tiles + tx)];
      a.mxx += cn.mxx;
      a.mxy += cn.mxy;
      a.myy += cn.myy;
      a.bx += cn.bx;
      a.by += cn.by;
      a.mass += cn.mass;
      a.count += 1;
    }

    const double ridge = std::max(1e-12f, params_.tikhonov_eps);
    const double prior = std::max(0.0f, prior_scale_ * params_.prior_lambda);

    for (int ty = 0; ty < tiles; ++ty) {
      for (int tx = 0; tx < tiles; ++tx) {
        const int k = ty * tiles + tx;
        const TileAniso & a = tile_aniso_[static_cast<size_t>(k)];
        const float fb_Fx = fallback_F[2 * k];
        const float fb_Fy = fallback_F[2 * k + 1];
        const float fb_px = -fb_Fx;
        const float fb_py = -fb_Fy;

        auto fall_back = [&]() {
            out_F[2 * k] = fb_Fx;
            out_F[2 * k + 1] = fb_Fy;
            if (static_cast<size_t>(k) < tile_warp_geom_.size()) {
              tile_warp_geom_[static_cast<size_t>(k)] = TileWarpGeometry{};
            }
            profile_.fallback_tiles += 1;
            if (tiles == final_tiles_) {
              profile_.final_fallback_tiles += 1;
            }
          };

        float lminM = 0.0f;
        float lmaxM = 0.0f;
        eig2(
          static_cast<float>(a.mxx), static_cast<float>(a.mxy),
          static_cast<float>(a.myy), lminM, lmaxM);
        if (a.count < std::max(1, params_.tile_min_cells) ||
          a.mass < mass_scale_ * params_.tile_min_mass ||
          !(lmaxM > 0.0f) || !std::isfinite(lmaxM))
        {
          fall_back();
          continue;
        }

        const double tile_eps = ridge * lmaxM;
        const double prior_mass = prior * lmaxM;
        float vx_phys = 0.0f;
        float vy_phys = 0.0f;
        const bool solved = solve2(
          static_cast<float>(a.mxx + tile_eps + prior_mass),
          static_cast<float>(a.mxy),
          static_cast<float>(a.myy + tile_eps + prior_mass),
          static_cast<float>(a.bx + prior_mass * fb_px),
          static_cast<float>(a.by + prior_mass * fb_py),
          vx_phys, vy_phys);
        if (!solved) {
          fall_back();
          continue;
        }

        // M's eigenstructure is the aperture geometry: dominant eigenvector
        // = best-constrained direction (edge normal of the dominant
        // structure), weak direction takes the fallback's component.
        const bool aperture_limited =
          lminM / std::max(lmaxM, 1e-12f) < params_.aperture_ratio;
        float nx = 1.0f;
        float ny = 0.0f;
        dominant_eigenvector(
          static_cast<float>(a.mxx), static_cast<float>(a.mxy),
          static_cast<float>(a.myy), lmaxM, nx, ny);
        const float tgx = -ny;
        const float tgy = nx;
        if (aperture_limited) {
          const float normal_v = vx_phys * nx + vy_phys * ny;
          const float tangent_v = fb_px * tgx + fb_py * tgy;
          vx_phys = normal_v * nx + tangent_v * tgx;
          vy_phys = normal_v * ny + tangent_v * tgy;
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
        if (prior_mass > 0.0) {
          profile_.prior_tiles += 1;
        }

        clamp_speed(vx_phys, vy_phys);
        out_F[2 * k] = -vx_phys;
        out_F[2 * k + 1] = -vy_phys;

        if (static_cast<size_t>(k) < tile_warp_geom_.size()) {
          TileWarpGeometry geom;
          geom.valid = true;
          geom.normal_x = nx;
          geom.normal_y = ny;
          geom.tangent_x = tgx;
          geom.tangent_y = tgy;
          geom.lmin = lminM;
          geom.lmax = lmaxM;
          geom.mass = static_cast<float>(a.mass);
          const float ratio = lminM / std::max(lmaxM, 1e-12f);
          const float data_conf = 0.5f * (lminM + lmaxM);
          const float normal_w = data_conf;
          const float tangent_w = aperture_limited
            ? data_conf * std::max(0.0f, ratio)
            : data_conf;
          geom.confidence = data_conf;
          geom.focus_phi = 1.0f + ratio;
          geom.data_mxx = normal_w * nx * nx + tangent_w * tgx * tgx;
          geom.data_mxy = normal_w * nx * ny + tangent_w * tgx * tgy;
          geom.data_myy = normal_w * ny * ny + tangent_w * tgy * tgy;
          geom.data_bx = geom.data_mxx * vx_phys + geom.data_mxy * vy_phys;
          geom.data_by = geom.data_mxy * vx_phys + geom.data_myy * vy_phys;
          tile_warp_geom_[static_cast<size_t>(k)] = geom;
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
      tile_warp_geom_.size() >= static_cast<size_t>(n_tiles);
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

}  // namespace moment_flow::flow
