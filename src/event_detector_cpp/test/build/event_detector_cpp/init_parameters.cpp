#include <event_detector_cpp/event_detector_cpp.hpp>

namespace event_detector_cpp
{

/**
 * @brief Routine to initialize node parameters.
 */
void EventDetector::init_parameters()
{
  // autostart
  pmanager_->declare_bool_parameter(
    "autostart",
    false,
    "Start detection right after node initialization.",
    "Cannot be changed.",
    true,
    &autostart_,
    nullptr);

  // ba_filter_dt_ms
  pmanager_->declare_double_parameter(
    "ba_filter_dt_ms",
    1.0, 0.01, 100.0, 0.0,
    "BA filter correlation time window [ms]. Events with no neighbour within this window are rejected.",
    "Cannot be changed at runtime.",
    true,
    &ba_filter_dt_ms_,
    nullptr);

  // ba_filter_enabled
  pmanager_->declare_bool_parameter(
    "ba_filter_enabled",
    true,
    "Enable Background Activity filter to suppress hot-pixel and shot noise.",
    "Cannot be changed at runtime.",
    true,
    &ba_filter_enabled_,
    nullptr);

  // flow_cg_max_iter
  pmanager_->declare_integer_parameter(
    "flow_cg_max_iter",
    20, 1, 500, 1,
    "Maximum inner conjugate-gradient iterations per Newton step.",
    "Must be positive.",
    false,
    &flow_cg_max_iter_,
    nullptr);

  // flow_cg_tol
  pmanager_->declare_double_parameter(
    "flow_cg_tol",
    0.1, 1e-06, 1.0, 0.0,
    "Forcing-tolerance factor for the inner CG solve (truncated Newton).",
    "Must be positive.",
    false,
    &flow_cg_tol_,
    nullptr);

  // flow_contrast_l2
  pmanager_->declare_bool_parameter(
    "flow_contrast_l2",
    false,
    "Use squared-L2 IWE gradient focus instead of the L1 default (Eq. 6).",
    "Cannot be changed at runtime.",
    true,
    &flow_contrast_l2_,
    nullptr);

  // flow_enabled
  pmanager_->declare_bool_parameter(
    "flow_enabled",
    true,
    "Enable dense optical-flow estimation via contrast maximization.",
    "Cannot be changed at runtime.",
    true,
    &flow_enabled_,
    nullptr);

  // flow_iwe_scale
  pmanager_->declare_integer_parameter(
    "flow_iwe_scale",
    1, 1, 8, 1,
    "Downscale factor for the rendered IWE image (1 = full resolution).",
    "Must be positive.",
    false,
    &flow_iwe_scale_,
    nullptr);

  // flow_max_speed_px_s
  pmanager_->declare_double_parameter(
    "flow_max_speed_px_s",
    4000.0, 1.0, 100000.0, 0.0,
    "Speed used to normalize flow colour and to clamp the estimated velocity [px/s].",
    "Must be positive.",
    false,
    &flow_max_speed_px_s_,
    nullptr);

  // flow_newton_max_iter
  pmanager_->declare_integer_parameter(
    "flow_newton_max_iter",
    30, 1, 500, 1,
    "Maximum outer Newton iterations per scale for the truncated-Newton solver.",
    "Must be positive.",
    false,
    &flow_newton_max_iter_,
    nullptr);

  // flow_num_events
  pmanager_->declare_integer_parameter(
    "flow_num_events",
    30000, 100, 5000000, 1,
    "Events accumulated per flow estimate (the paper's set E_i of N events).",
    "Must be positive.",
    false,
    &flow_num_events_,
    nullptr);

  // flow_num_scales
  pmanager_->declare_integer_parameter(
    "flow_num_scales",
    5, 1, 8, 1,
    "Tile-pyramid scales; scale l uses 2^(l-1) tiles per side (Sec. III-D).",
    "Cannot be changed at runtime.",
    true,
    &flow_num_scales_,
    nullptr);

  // flow_pde_burgers
  pmanager_->declare_bool_parameter(
    "flow_pde_burgers",
    false,
    "Use the Burgers flux scheme for time-aware propagation instead of upwind.",
    "Cannot be changed at runtime.",
    true,
    &flow_pde_burgers_,
    nullptr);

  // flow_prop_grid
  pmanager_->declare_integer_parameter(
    "flow_prop_grid",
    32, 4, 256, 1,
    "Side length of the square propagation grid for time-aware flow.",
    "Must be positive.",
    false,
    &flow_prop_grid_,
    nullptr);

  // flow_time_aware
  pmanager_->declare_bool_parameter(
    "flow_time_aware",
    false,
    "Enable time-aware flow propagation (Sec. III-C) when supported by the objective.",
    "Cannot be changed at runtime.",
    true,
    &flow_time_aware_,
    nullptr);

  // flow_time_bins
  pmanager_->declare_integer_parameter(
    "flow_time_bins",
    5, 1, 200, 1,
    "Number of temporal bins for time-aware propagation (5 MVSEC / 40 DSEC).",
    "Must be positive.",
    false,
    &flow_time_bins_,
    nullptr);

  // flow_tv_charbonnier_eps
  pmanager_->declare_double_parameter(
    "flow_tv_charbonnier_eps",
    0.001, 1e-06, 1.0, 0.0,
    "Charbonnier epsilon smoothing the isotropic TV norm (Sec. III-E).",
    "Must be positive.",
    false,
    &flow_tv_charbonnier_eps_,
    nullptr);

  // flow_tv_weight
  pmanager_->declare_double_parameter(
    "flow_tv_weight",
    0.0025, 0.0, 10.0, 0.0,
    "Total-variation regularizer weight lambda (Eq. 9).",
    "Must be non-negative.",
    false,
    &flow_tv_weight_,
    nullptr);

  // iwe_enabled
  pmanager_->declare_bool_parameter(
    "iwe_enabled",
    true,
    "Enable global Image of Warped Events rendering and publishing.",
    "Cannot be changed at runtime.",
    true,
    &iwe_enabled_,
    nullptr);

  // sae_enabled
  pmanager_->declare_bool_parameter(
    "sae_enabled",
    true,
    "Enable Surface of Active Events rendering and publishing.",
    "Cannot be changed at runtime.",
    true,
    &sae_enabled_,
    nullptr);

  // time_window_ms
  pmanager_->declare_double_parameter(
    "time_window_ms",
    33.0, 1.0, 1000.0, 0.0,
    "SAE normalization time window [ms]. Events older than this are rendered as black.",
    "Must be positive.",
    false,
    &time_window_ms_,
    nullptr);
}

} // namespace event_detector_cpp
