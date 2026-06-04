/**
 * Time-aware flow propagation (Sec. III-C of Shiba-Gallego "Secrets").
 *
 * Solves the self-advection PDE  v_t + (v.grad) v = 0  with the flow field at
 * t_mid as boundary condition, transporting it outward in time so each event
 * can be warped by the flow value at its own (x, t) (Eq. 8).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#pragma once

#include <vector>

#include "event_detector_cpp/flow/grid.hpp"

namespace event_detector_cpp::flow
{

/**
 * @brief PDE discretization scheme.
 *
 * Upwind: non-conservative upwind differences.
 * Burgers: conservative scheme for the inviscid-Burgers flux.
 */
enum class Scheme
{
  Upwind,
  Burgers
};

/**
 * @brief Propagate a boundary flow field to a set of output times.
 *
 * @param v0 Boundary flow field at t_mid (t = 0).
 * @param t_out_s Output times relative to t_mid [s]; may be negative (before
 *   t_mid) or positive (after). The boundary t = 0 yields v0 unchanged.
 * @param scheme PDE discretization scheme.
 * @param ds Grid spacing [px] between adjacent nodes.
 * @param cfl CFL number bounding the explicit sub-step size (<= 1 for stability).
 * @return One grid per requested output time, in the same order as t_out_s.
 */
std::vector<Grid> propagate(
  const Grid & v0,
  const std::vector<float> & t_out_s,
  Scheme scheme,
  float ds,
  float cfl = 0.5f);

/**
 * @brief Reverse-mode (vector-Jacobian product) of @ref propagate.
 *
 * Given the gradient of a downstream scalar loss w.r.t. each output grid,
 * returns the gradient w.r.t. the boundary field @p v0. Upwind/flux directions
 * are frozen at the forward states (standard frozen-coefficient adjoint).
 *
 * @param v0 Boundary field used in the forward pass.
 * @param t_out_s Same output times as the forward pass.
 * @param scheme Same scheme as the forward pass.
 * @param ds Same grid spacing.
 * @param cfl Same CFL number (must match so sub-step counts agree).
 * @param grad_out Cotangent per output grid (dL/d(output[n])); size must match.
 * @return dL/d(v0).
 */
Grid propagate_vjp(
  const Grid & v0,
  const std::vector<float> & t_out_s,
  Scheme scheme,
  float ds,
  float cfl,
  const std::vector<Grid> & grad_out);

}  // namespace event_detector_cpp::flow
