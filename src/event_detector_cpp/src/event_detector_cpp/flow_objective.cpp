/**
 * Contrast-maximization flow objective implementation.
 *
 * All accumulation is in double precision so the finite-difference gradient
 * checks are numerically clean; results are returned as float.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#include "event_detector_cpp/flow/objective.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <utility>

#include <omp.h>

namespace event_detector_cpp::flow
{

namespace
{

using ProfileClock = std::chrono::steady_clock;

double elapsed_ms(
  const ProfileClock::time_point & start,
  const ProfileClock::time_point & end = ProfileClock::now())
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void add_profile(ObjectiveProfile * profile, double ObjectiveProfile::* field, double value)
{
  if (profile != nullptr) {
    (profile->*field) += value;
  }
}

void inc_profile(ObjectiveProfile * profile, int ObjectiveProfile::* field)
{
  if (profile != nullptr) {
    (profile->*field) += 1;
  }
}

// Mean IWE gradient-magnitude focus G (Eq. 6) and, optionally, its adjoint
// dG/dI scaled by `seed` (= dE/dG) accumulated into `adj`.
double contrast(
  const std::vector<double> & I, int w, int h, ContrastNorm norm,
  const double * seed, std::vector<double> * adj)
{
  const double n = static_cast<double>((w - 1) * (h - 1));

  if (adj == nullptr) {
    // Forward focus: parallel sum over rows.
    double g = 0.0;
    #pragma omp parallel for reduction(+ : g) schedule(static)
    for (int y = 0; y < h - 1; ++y) {
      const double * row = &I[static_cast<size_t>(y) * w];
      const double * rowd = &I[static_cast<size_t>(y + 1) * w];
      double acc = 0.0;
      for (int x = 0; x < w - 1; ++x) {
        const double gx = row[x + 1] - row[x];
        const double gy = rowd[x] - row[x];
        acc += (norm == ContrastNorm::L1)
          ? (std::abs(gx) + std::abs(gy))
          : (gx * gx + gy * gy);
      }
      g += acc;
    }
    return g / n;
  }

  // Adjoint dG/dI scaled by *seed, written as a gather so each output pixel is
  // computed independently (race-free, parallel). `adj` is zero on entry. This
  // is the exact transpose of the forward scatter: pixel q receives a term as
  // the central pixel p=q and as the right/bottom neighbour of p=q-1 / p=q-w.
  const double s = *seed / n;
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t q = static_cast<size_t>(y) * w + x;
      double a = 0.0;
      if (x < w - 1 && y < h - 1) {           // q acts as the central pixel p
        const double gx = I[q + 1] - I[q];
        const double gy = I[q + w] - I[q];
        if (norm == ContrastNorm::L1) {
          a -= s * ((gx > 0) - (gx < 0));
          a -= s * ((gy > 0) - (gy < 0));
        } else {
          a -= s * 2.0 * gx;
          a -= s * 2.0 * gy;
        }
      }
      if (x >= 1 && y < h - 1) {              // q acts as p+1 for p=q-1
        const double gx = I[q] - I[q - 1];
        a += (norm == ContrastNorm::L1) ? s * ((gx > 0) - (gx < 0)) : s * 2.0 * gx;
      }
      if (y >= 1 && x < w - 1) {              // q acts as p+w for p=q-w
        const double gy = I[q] - I[q - w];
        a += (norm == ContrastNorm::L1) ? s * ((gy > 0) - (gy < 0)) : s * 2.0 * gy;
      }
      (*adj)[q] += a;
    }
  }
  return 0.0;  // forward value unused when computing the adjoint
}

// Total-variation (isotropic, Charbonnier) of the tile field and, optionally,
// its gradient accumulated into `grad` scaled by `lambda`.
double tv(
  const Eigen::VectorXf & F, int tx, int ty, double eps,
  double lambda, Eigen::VectorXf * grad)
{
  const double eps2 = eps * eps;
  const double norm = static_cast<double>(std::max(1, tx * ty));
  double R = 0.0;
  auto vx = [&](int i, int j) { return static_cast<double>(F[2 * (j * tx + i)]); };
  auto vy = [&](int i, int j) { return static_cast<double>(F[2 * (j * tx + i) + 1]); };
  for (int j = 0; j < ty; ++j) {
    for (int i = 0; i < tx; ++i) {
      double dxx = 0, dxy = 0, dyx = 0, dyy = 0;
      const bool hx = (i + 1 < tx);
      const bool hy = (j + 1 < ty);
      if (hx) { dxx = vx(i + 1, j) - vx(i, j); dxy = vy(i + 1, j) - vy(i, j); }
      if (hy) { dyx = vx(i, j + 1) - vx(i, j); dyy = vy(i, j + 1) - vy(i, j); }
      const double m = std::sqrt(dxx * dxx + dxy * dxy + dyx * dyx + dyy * dyy + eps2);
      R += m;
      if (grad) {
        const double inv = lambda / (norm * m);
        const int k = j * tx + i;
        if (hx) {
          const int kr = j * tx + (i + 1);
          (*grad)[2 * kr]     += static_cast<float>(inv * dxx);
          (*grad)[2 * kr + 1] += static_cast<float>(inv * dxy);
          (*grad)[2 * k]      -= static_cast<float>(inv * dxx);
          (*grad)[2 * k + 1]  -= static_cast<float>(inv * dxy);
        }
        if (hy) {
          const int kd = (j + 1) * tx + i;
          (*grad)[2 * kd]     += static_cast<float>(inv * dyx);
          (*grad)[2 * kd + 1] += static_cast<float>(inv * dyy);
          (*grad)[2 * k]      -= static_cast<float>(inv * dyx);
          (*grad)[2 * k + 1]  -= static_cast<float>(inv * dyy);
        }
      }
    }
  }
  return R / norm;
}

// Bilinear stencil for a query pixel over a regular grid of nx*ny nodes whose
// centers sit at ((i+0.5)*sx, (j+0.5)*sy); edges replicate (Neumann). Fills the
// 4 node indices and weights.
inline void bilinear(
  double px, double py, int nx, int ny, double sx, double sy,
  int * idx, float * wt)
{
  const double gx = px / sx - 0.5;
  const double gy = py / sy - 0.5;
  const int i0 = static_cast<int>(std::floor(gx));
  const int j0 = static_cast<int>(std::floor(gy));
  const double fx = gx - i0;
  const double fy = gy - j0;
  auto cl = [](int v, int n) { return std::clamp(v, 0, n - 1); };
  const int i0c = cl(i0, nx), i1c = cl(i0 + 1, nx);
  const int j0c = cl(j0, ny), j1c = cl(j0 + 1, ny);
  idx[0] = j0c * nx + i0c; wt[0] = static_cast<float>((1 - fx) * (1 - fy));
  idx[1] = j0c * nx + i1c; wt[1] = static_cast<float>(fx * (1 - fy));
  idx[2] = j1c * nx + i0c; wt[2] = static_cast<float>((1 - fx) * fy);
  idx[3] = j1c * nx + i1c; wt[3] = static_cast<float>(fx * fy);
}

// IWE accumulation kernel (Eq. 2): each warped event is splatted as a Gaussian
// of sigma = epsilon = 1 px (the paper's Dirac-delta approximation), truncated
// at radius kIweR. The smooth kernel (vs. a 4-pixel bilinear vote) is essential
// to the contrast objective's landscape and convergence.
constexpr int kIweR = 2;            // truncation radius [px]
constexpr double kIweInv2Sig2 = 0.5;  // 1 / (2 sigma^2), sigma = 1 px

inline void iwe_splat(
  double * I, int w, int h, double xp, double yp, double amp = 1.0)
{
  const int ixlo = static_cast<int>(std::ceil(xp - kIweR));
  const int ixhi = static_cast<int>(std::floor(xp + kIweR));
  const int iylo = static_cast<int>(std::ceil(yp - kIweR));
  const int iyhi = static_cast<int>(std::floor(yp + kIweR));
  for (int iy = iylo; iy <= iyhi; ++iy) {
    if (iy < 0 || iy >= h) continue;
    const double dy = iy - yp;
    for (int ix = ixlo; ix <= ixhi; ++ix) {
      if (ix < 0 || ix >= w) continue;
      const double dx = ix - xp;
      I[static_cast<size_t>(iy) * w + ix] +=
        amp * std::exp(-(dx * dx + dy * dy) * kIweInv2Sig2);
    }
  }
}

// Adjoint of iwe_splat: given dL/dI in `adjI`, return dL/dxp, dL/dyp.
inline void iwe_splat_adj(
  const std::vector<double> & adjI, int w, int h, double xp, double yp,
  double amp, double & dxp, double & dyp)
{
  dxp = 0.0;
  dyp = 0.0;
  const int ixlo = static_cast<int>(std::ceil(xp - kIweR));
  const int ixhi = static_cast<int>(std::floor(xp + kIweR));
  const int iylo = static_cast<int>(std::ceil(yp - kIweR));
  const int iyhi = static_cast<int>(std::floor(yp + kIweR));
  for (int iy = iylo; iy <= iyhi; ++iy) {
    if (iy < 0 || iy >= h) continue;
    const double dy = iy - yp;
    for (int ix = ixlo; ix <= ixhi; ++ix) {
      if (ix < 0 || ix >= w) continue;
      const double dx = ix - xp;
      const double wgt = std::exp(-(dx * dx + dy * dy) * kIweInv2Sig2);
      // d(wgt)/d(xp) = wgt * (ix - xp) / sigma^2 = wgt * dx  (sigma = 1).
      const double common = amp * wgt * adjI[static_cast<size_t>(iy) * w + ix];
      dxp += common * dx;
      dyp += common * dy;
    }
  }
}

// Race-free IWE renderer. Events are parallelized, but each worker splats into
// a private image; private images are then reduced into the output image.
template <class WarpFn>
std::vector<double> render_iwe(
  size_t n_events, int w, int h, const WarpFn & warp)
{
  const size_t n_pix = static_cast<size_t>(w) * h;
  std::vector<double> I(n_pix, 0.0);
  if (n_events == 0 || n_pix == 0) {
    return I;
  }

  #pragma omp parallel
  {
    std::vector<double> local(n_pix, 0.0);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t kk = 0; kk < static_cast<std::ptrdiff_t>(n_events); ++kk) {
      double xp = 0.0;
      double yp = 0.0;
      warp(static_cast<size_t>(kk), xp, yp);
      iwe_splat(local.data(), w, h, xp, yp, 1.0);
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < n_pix; ++i) {
        I[i] += local[i];
      }
    }
  }

  return I;
}

}  // namespace

Objective::Objective(Events events, ObjectiveParams params)
: events_(std::move(events)), params_(std::move(params))
{
  build_stencils();
  if (params_.time_aware) {
    build_time_aware();
  }
}

void Objective::build_stencils()
{
  const auto t_stencils = ProfileClock::now();
  const int tx = params_.tiles_x;
  const int ty = params_.tiles_y;
  const double sx = static_cast<double>(params_.img_w) / tx;  // tile spacing [px]
  const double sy = static_cast<double>(params_.img_h) / ty;

  stencils_.resize(events_.size());
  for (size_t k = 0; k < events_.size(); ++k) {
    // Pixel -> tile-center grid (centers at (i+0.5)*spacing, replicate edges).
    bilinear(events_.x[k], events_.y[k], tx, ty, sx, sy,
             stencils_[k].idx, stencils_[k].w);
  }
  add_profile(
    params_.profile, &ObjectiveProfile::build_stencils_ms, elapsed_ms(t_stencils));

  // Zero-flow normalization G0: events splat unwarped, identical for all refs.
  // G0 is tile-independent, so a caller solving multiple scales can compute it
  // once and inject it via params_.g0_override to avoid recomputation.
  if (params_.g0_override > 0.0f) {
    g0_ = params_.g0_override;
    return;
  }
  const int w = params_.img_w, h = params_.img_h;
  const auto t_g0_render = ProfileClock::now();
  std::vector<double> I = render_iwe(
    events_.size(), w, h,
    [this](size_t k, double & xp, double & yp) {
      xp = events_.x[k];
      yp = events_.y[k];
    });
  add_profile(
    params_.profile, &ObjectiveProfile::g0_render_ms, elapsed_ms(t_g0_render));
  const auto t_g0_contrast = ProfileClock::now();
  g0_ = static_cast<float>(contrast(I, w, h, params_.norm, nullptr, nullptr));
  add_profile(
    params_.profile, &ObjectiveProfile::g0_contrast_ms, elapsed_ms(t_g0_contrast));
  if (!(g0_ > 0.0f)) g0_ = 1.0f;  // guard empty/degenerate windows
}

namespace
{
// Per-event interpolated velocity from the tile field.
inline void sample_v(
  const Eigen::VectorXf & F, const int * idx, const float * w,
  double & vx, double & vy)
{
  vx = vy = 0.0;
  for (int c = 0; c < 4; ++c) {
    vx += static_cast<double>(w[c]) * F[2 * idx[c]];
    vy += static_cast<double>(w[c]) * F[2 * idx[c] + 1];
  }
}
}  // namespace

float Objective::focus(const Eigen::VectorXf & F) const
{
  if (params_.time_aware) {
    return focus_time_aware(F, nullptr);
  }
  inc_profile(params_.profile, &ObjectiveProfile::focus_calls);
  const auto t_focus = ProfileClock::now();
  const int w = params_.img_w, h = params_.img_h;
  const std::array<float, 3> refs = {params_.t_lo, 0.0f, params_.t_hi};
  std::array<double, 3> G{};
  for (int r = 0; r < 3; ++r) {
    const auto t_render = ProfileClock::now();
    std::vector<double> I = render_iwe(
      events_.size(), w, h,
      [this, &F, &refs, r](size_t k, double & xp, double & yp) {
        double vx, vy;
        sample_v(F, stencils_[k].idx, stencils_[k].w, vx, vy);
        const double f = events_.t[k] - refs[r];
        xp = events_.x[k] + f * vx;
        yp = events_.y[k] + f * vy;
      });
    add_profile(params_.profile, &ObjectiveProfile::render_iwe_ms, elapsed_ms(t_render));
    const auto t_contrast = ProfileClock::now();
    G[r] = contrast(I, w, h, params_.norm, nullptr, nullptr);
    add_profile(params_.profile, &ObjectiveProfile::contrast_ms, elapsed_ms(t_contrast));
  }
  const float f = static_cast<float>((G[0] + 2.0 * G[1] + G[2]) / (4.0 * g0_));
  add_profile(params_.profile, &ObjectiveProfile::focus_ms, elapsed_ms(t_focus));
  return f;
}

void Objective::event_flow(
  const Eigen::VectorXf & F,
  std::vector<float> & vx,
  std::vector<float> & vy) const
{
  inc_profile(params_.profile, &ObjectiveProfile::event_flow_calls);
  const auto t_event_flow = ProfileClock::now();
  const size_t N = events_.size();
  vx.assign(N, 0.0f);
  vy.assign(N, 0.0f);

  if (!params_.time_aware) {
    const auto t_sample = ProfileClock::now();
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t kk = 0; kk < static_cast<std::ptrdiff_t>(N); ++kk) {
      double sx = 0.0;
      double sy = 0.0;
      const size_t k = static_cast<size_t>(kk);
      sample_v(F, stencils_[k].idx, stencils_[k].w, sx, sy);
      vx[k] = static_cast<float>(sx);
      vy[k] = static_cast<float>(sy);
    }
    add_profile(params_.profile, &ObjectiveProfile::event_sample_ms, elapsed_ms(t_sample));
    add_profile(params_.profile, &ObjectiveProfile::event_flow_ms, elapsed_ms(t_event_flow));
    return;
  }

  const auto t_boundary = ProfileClock::now();
  const Grid v0 = boundary_field(F);
  add_profile(params_.profile, &ObjectiveProfile::boundary_ms, elapsed_ms(t_boundary));
  const auto t_propagate = ProfileClock::now();
  const std::vector<Grid> V =
    propagate(v0, bin_times_, params_.scheme, ds_, params_.cfl);
  add_profile(params_.profile, &ObjectiveProfile::propagate_ms, elapsed_ms(t_propagate));

  const auto t_sample = ProfileClock::now();
  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t kk = 0; kk < static_cast<std::ptrdiff_t>(N); ++kk) {
    const size_t k = static_cast<size_t>(kk);
    const Stencil & st = ev_prop_stencils_[k];
    const Grid & g = V[static_cast<size_t>(ev_bin_[k])];
    double sx = 0.0;
    double sy = 0.0;
    for (int c = 0; c < 4; ++c) {
      sx += static_cast<double>(st.w[c]) * g.vx[st.idx[c]];
      sy += static_cast<double>(st.w[c]) * g.vy[st.idx[c]];
    }
    vx[k] = static_cast<float>(sx);
    vy[k] = static_cast<float>(sy);
  }
  add_profile(params_.profile, &ObjectiveProfile::event_sample_ms, elapsed_ms(t_sample));
  add_profile(params_.profile, &ObjectiveProfile::event_flow_ms, elapsed_ms(t_event_flow));
}

float Objective::value(const Eigen::VectorXf & F) const
{
  inc_profile(params_.profile, &ObjectiveProfile::value_calls);
  const auto t_value = ProfileClock::now();
  const double f = focus(F);
  double E = 1.0 / static_cast<double>(f);
  if (params_.tv_weight > 0.0f) {
    const auto t_tv = ProfileClock::now();
    E += tv(F, params_.tiles_x, params_.tiles_y, params_.tv_eps,
            params_.tv_weight, nullptr) * params_.tv_weight;
    add_profile(params_.profile, &ObjectiveProfile::tv_ms, elapsed_ms(t_tv));
  }
  add_profile(params_.profile, &ObjectiveProfile::value_ms, elapsed_ms(t_value));
  return static_cast<float>(E);
}

float Objective::value_and_grad(const Eigen::VectorXf & F, Eigen::VectorXf & grad) const
{
  inc_profile(params_.profile, &ObjectiveProfile::value_and_grad_calls);
  const auto t_value_and_grad = ProfileClock::now();
  if (params_.time_aware) {
    grad = Eigen::VectorXf::Zero(F.size());
    const double f = focus_time_aware(F, &grad);
    double E = 1.0 / f;
    if (params_.tv_weight > 0.0f) {
      const auto t_tv = ProfileClock::now();
      E += tv(F, params_.tiles_x, params_.tiles_y, params_.tv_eps,
              params_.tv_weight, &grad) * params_.tv_weight;
      add_profile(params_.profile, &ObjectiveProfile::tv_ms, elapsed_ms(t_tv));
    }
    add_profile(
      params_.profile, &ObjectiveProfile::value_and_grad_ms, elapsed_ms(t_value_and_grad));
    return static_cast<float>(E);
  }

  const int w = params_.img_w, h = params_.img_h;
  const std::array<float, 3> refs = {params_.t_lo, 0.0f, params_.t_hi};
  const std::array<double, 3> coef = {1.0, 2.0, 1.0};
  grad = Eigen::VectorXf::Zero(F.size());

  // Forward: keep each reference IWE so we can backprop.
  std::array<std::vector<double>, 3> Is;
  std::array<double, 3> G{};
  for (int r = 0; r < 3; ++r) {
    const auto t_render = ProfileClock::now();
    Is[r] = render_iwe(
      events_.size(), w, h,
      [this, &F, &refs, r](size_t k, double & xp, double & yp) {
        double vx, vy;
        sample_v(F, stencils_[k].idx, stencils_[k].w, vx, vy);
        const double fct = events_.t[k] - refs[r];
        xp = events_.x[k] + fct * vx;
        yp = events_.y[k] + fct * vy;
      });
    add_profile(params_.profile, &ObjectiveProfile::render_iwe_ms, elapsed_ms(t_render));
    const auto t_contrast = ProfileClock::now();
    G[r] = contrast(Is[r], w, h, params_.norm, nullptr, nullptr);
    add_profile(params_.profile, &ObjectiveProfile::contrast_ms, elapsed_ms(t_contrast));
  }

  const double f = (G[0] + 2.0 * G[1] + G[2]) / (4.0 * g0_);
  const double dE_df = -1.0 / (f * f);

  // Backprop each reference.
  for (int r = 0; r < 3; ++r) {
    // dE/dG_r = dE/df * df/dG_r = dE_df * coef_r / (4 G0).
    double seed = dE_df * coef[r] / (4.0 * static_cast<double>(g0_));
    std::vector<double> adjI(static_cast<size_t>(w) * h, 0.0);
    const auto t_contrast = ProfileClock::now();
    contrast(Is[r], w, h, params_.norm, &seed, &adjI);
    add_profile(params_.profile, &ObjectiveProfile::contrast_ms, elapsed_ms(t_contrast));

    // Per-event velocity gradient (independent -> parallel); the cheap scatter
    // into the shared tile gradient is then done serially to avoid races.
    const auto t_backprop = ProfileClock::now();
    const size_t N = events_.size();
    std::vector<double> dvx(N), dvy(N);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < N; ++k) {
      double vx, vy;
      sample_v(F, stencils_[k].idx, stencils_[k].w, vx, vy);
      const double fct = events_.t[k] - refs[r];
      const double xp = events_.x[k] + fct * vx;
      const double yp = events_.y[k] + fct * vy;
      double dxp, dyp;
      iwe_splat_adj(adjI, w, h, xp, yp, 1.0, dxp, dyp);
      dvx[k] = dxp * fct;
      dvy[k] = dyp * fct;
    }
    add_profile(params_.profile, &ObjectiveProfile::backprop_ms, elapsed_ms(t_backprop));
    const auto t_scatter = ProfileClock::now();
    for (size_t k = 0; k < N; ++k) {
      const int * idx = stencils_[k].idx;
      const float * wv = stencils_[k].w;
      for (int c = 0; c < 4; ++c) {
        grad[2 * idx[c]]     += static_cast<float>(wv[c] * dvx[k]);
        grad[2 * idx[c] + 1] += static_cast<float>(wv[c] * dvy[k]);
      }
    }
    add_profile(params_.profile, &ObjectiveProfile::scatter_ms, elapsed_ms(t_scatter));
  }

  double E = 1.0 / f;
  if (params_.tv_weight > 0.0f) {
    const auto t_tv = ProfileClock::now();
    E += tv(F, params_.tiles_x, params_.tiles_y, params_.tv_eps,
            params_.tv_weight, &grad) * params_.tv_weight;
    add_profile(params_.profile, &ObjectiveProfile::tv_ms, elapsed_ms(t_tv));
  }
  add_profile(
    params_.profile, &ObjectiveProfile::value_and_grad_ms, elapsed_ms(t_value_and_grad));
  return static_cast<float>(E);
}

// ─────────────────────────── Time-aware flow (Sec. III-C) ───────────────────

void Objective::build_time_aware()
{
  const auto t_build = ProfileClock::now();
  const int w = params_.img_w, h = params_.img_h;
  const int tx = params_.tiles_x, ty = params_.tiles_y;
  pw_ = std::max(2, params_.prop_w);
  ph_ = std::max(2, params_.prop_h);
  ds_ = static_cast<float>(w) / pw_;   // square-cell spacing [px]

  // Boundary stencils: each propagation node samples the tile field F at its
  // pixel-space center.
  const double sx_tile = static_cast<double>(w) / tx;
  const double sy_tile = static_cast<double>(h) / ty;
  bnd_stencils_.resize(static_cast<size_t>(pw_) * ph_);
  for (int r = 0; r < ph_; ++r) {
    for (int c = 0; c < pw_; ++c) {
      const double px = (c + 0.5) * ds_;
      const double py = (r + 0.5) * ds_;
      const int n = r * pw_ + c;
      bilinear(px, py, tx, ty, sx_tile, sy_tile,
               bnd_stencils_[n].idx, bnd_stencils_[n].w);
    }
  }

  // Per-event: bilinear stencil over the propagation grid and the time bin.
  const int nb = std::max(1, params_.time_bins);
  const double t_lo = params_.t_lo, t_hi = params_.t_hi;
  const double span = t_hi - t_lo;
  bin_times_.assign(nb, 0.0f);
  for (int b = 0; b < nb; ++b) {
    // Match the paper/reference implementation's "flow at t_mid" convention:
    // bin nb/2 is exactly the t=0 boundary field, with neighbours propagated by
    // one temporal-bin step. For odd bin counts this is equivalent to bin centers.
    bin_times_[b] = (span > 0.0)
      ? static_cast<float>((b - nb / 2) * span / nb)
      : 0.0f;
  }
  ev_prop_stencils_.resize(events_.size());
  ev_bin_.assign(events_.size(), 0);
  for (size_t k = 0; k < events_.size(); ++k) {
    bilinear(events_.x[k], events_.y[k], pw_, ph_, ds_, ds_,
             ev_prop_stencils_[k].idx, ev_prop_stencils_[k].w);
    if (span > 0.0) {
      int b = static_cast<int>(std::floor((events_.t[k] - t_lo) / span * nb));
      ev_bin_[k] = std::clamp(b, 0, nb - 1);
    }
  }
  add_profile(
    params_.profile, &ObjectiveProfile::build_time_aware_ms, elapsed_ms(t_build));
}

Grid Objective::boundary_field(const Eigen::VectorXf & F) const
{
  Grid v0(pw_, ph_);
  for (size_t n = 0; n < bnd_stencils_.size(); ++n) {
    const Stencil & st = bnd_stencils_[n];
    double vx = 0.0, vy = 0.0;
    for (int c = 0; c < 4; ++c) {
      vx += static_cast<double>(st.w[c]) * F[2 * st.idx[c]];
      vy += static_cast<double>(st.w[c]) * F[2 * st.idx[c] + 1];
    }
    v0.vx[n] = static_cast<float>(vx);
    v0.vy[n] = static_cast<float>(vy);
  }
  return v0;
}

float Objective::focus_time_aware(const Eigen::VectorXf & F, Eigen::VectorXf * grad) const
{
  inc_profile(params_.profile, &ObjectiveProfile::focus_calls);
  const auto t_focus = ProfileClock::now();
  const int w = params_.img_w, h = params_.img_h;
  const std::array<float, 3> refs = {params_.t_lo, 0.0f, params_.t_hi};
  const std::array<double, 3> coef = {1.0, 2.0, 1.0};
  const size_t N = events_.size();

  // Transport the boundary field at t_mid to each time bin (Eq. 7).
  const auto t_boundary = ProfileClock::now();
  const Grid v0 = boundary_field(F);
  add_profile(params_.profile, &ObjectiveProfile::boundary_ms, elapsed_ms(t_boundary));
  const auto t_propagate = ProfileClock::now();
  const std::vector<Grid> V =
    propagate(v0, bin_times_, params_.scheme, ds_, params_.cfl);
  add_profile(params_.profile, &ObjectiveProfile::propagate_ms, elapsed_ms(t_propagate));

  // Flow value v_hat sampled at each event's own (x, t) bin (Eq. 8).
  const auto t_sample = ProfileClock::now();
  std::vector<double> vhx(N), vhy(N);
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < N; ++k) {
    const Stencil & st = ev_prop_stencils_[k];
    const Grid & g = V[static_cast<size_t>(ev_bin_[k])];
    double ax = 0.0, ay = 0.0;
    for (int c = 0; c < 4; ++c) {
      ax += static_cast<double>(st.w[c]) * g.vx[st.idx[c]];
      ay += static_cast<double>(st.w[c]) * g.vy[st.idx[c]];
    }
    vhx[k] = ax;
    vhy[k] = ay;
  }
  add_profile(params_.profile, &ObjectiveProfile::event_sample_ms, elapsed_ms(t_sample));

  // Forward: build the three reference IWEs and their focus values.
  std::array<std::vector<double>, 3> Is;
  std::array<double, 3> G{};
  for (int r = 0; r < 3; ++r) {
    const auto t_render = ProfileClock::now();
    Is[r] = render_iwe(
      N, w, h,
      [this, &refs, &vhx, &vhy, r](size_t k, double & xp, double & yp) {
        const double fct = events_.t[k] - refs[r];
        xp = events_.x[k] + fct * vhx[k];
        yp = events_.y[k] + fct * vhy[k];
      });
    add_profile(params_.profile, &ObjectiveProfile::render_iwe_ms, elapsed_ms(t_render));
    const auto t_contrast = ProfileClock::now();
    G[r] = contrast(Is[r], w, h, params_.norm, nullptr, nullptr);
    add_profile(params_.profile, &ObjectiveProfile::contrast_ms, elapsed_ms(t_contrast));
  }

  const double f = (G[0] + 2.0 * G[1] + G[2]) / (4.0 * static_cast<double>(g0_));
  if (grad == nullptr) {
    add_profile(params_.profile, &ObjectiveProfile::focus_ms, elapsed_ms(t_focus));
    return static_cast<float>(f);
  }

  // Backprop: dE/dv_hat per event (summed over the three references).
  const double dE_df = -1.0 / (f * f);
  std::vector<double> dvhx(N, 0.0), dvhy(N, 0.0);
  for (int r = 0; r < 3; ++r) {
    double seed = dE_df * coef[r] / (4.0 * static_cast<double>(g0_));
    std::vector<double> adjI(static_cast<size_t>(w) * h, 0.0);
    const auto t_contrast = ProfileClock::now();
    contrast(Is[r], w, h, params_.norm, &seed, &adjI);
    add_profile(params_.profile, &ObjectiveProfile::contrast_ms, elapsed_ms(t_contrast));
    const auto t_backprop = ProfileClock::now();
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < N; ++k) {
      const double fct = events_.t[k] - refs[r];
      const double xp = events_.x[k] + fct * vhx[k];
      const double yp = events_.y[k] + fct * vhy[k];
      double dxp, dyp;
      iwe_splat_adj(adjI, w, h, xp, yp, 1.0, dxp, dyp);
      dvhx[k] += dxp * fct;
      dvhy[k] += dyp * fct;
    }
    add_profile(params_.profile, &ObjectiveProfile::backprop_ms, elapsed_ms(t_backprop));
  }

  // Scatter dE/dv_hat onto the per-bin propagated grids.
  const auto t_scatter_events = ProfileClock::now();
  std::vector<Grid> gV(bin_times_.size(), Grid(pw_, ph_));
  for (size_t k = 0; k < N; ++k) {
    Grid & g = gV[static_cast<size_t>(ev_bin_[k])];
    const Stencil & st = ev_prop_stencils_[k];
    for (int c = 0; c < 4; ++c) {
      g.vx[st.idx[c]] += static_cast<float>(st.w[c] * dvhx[k]);
      g.vy[st.idx[c]] += static_cast<float>(st.w[c] * dvhy[k]);
    }
  }
  add_profile(params_.profile, &ObjectiveProfile::scatter_ms, elapsed_ms(t_scatter_events));

  // Adjoint of the transport, then of the boundary sampling, into dE/dF.
  const auto t_propagate_vjp = ProfileClock::now();
  const Grid g_v0 =
    propagate_vjp(v0, bin_times_, params_.scheme, ds_, params_.cfl, gV);
  add_profile(
    params_.profile, &ObjectiveProfile::propagate_vjp_ms, elapsed_ms(t_propagate_vjp));
  const auto t_scatter_boundary = ProfileClock::now();
  for (size_t n = 0; n < bnd_stencils_.size(); ++n) {
    const Stencil & st = bnd_stencils_[n];
    for (int c = 0; c < 4; ++c) {
      (*grad)[2 * st.idx[c]]     += static_cast<float>(st.w[c] * g_v0.vx[n]);
      (*grad)[2 * st.idx[c] + 1] += static_cast<float>(st.w[c] * g_v0.vy[n]);
    }
  }
  add_profile(params_.profile, &ObjectiveProfile::scatter_ms, elapsed_ms(t_scatter_boundary));
  add_profile(params_.profile, &ObjectiveProfile::focus_ms, elapsed_ms(t_focus));
  return static_cast<float>(f);
}

}  // namespace event_detector_cpp::flow
