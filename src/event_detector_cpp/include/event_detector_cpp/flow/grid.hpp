/**
 * Regular 2D vector-field grid used by the contrast-maximization flow core.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

#pragma once

#include <vector>

namespace event_detector_cpp::flow
{

/**
 * @brief A row-major 2D grid of 2D vectors (a flow field sampled on a regular
 * lattice). vx[r*w + c] / vy[r*w + c] hold the components at column c, row r.
 *
 * Kept deliberately free of OpenCV/ROS types so the flow math can be unit
 * tested in isolation.
 */
struct Grid
{
  int w = 0;
  int h = 0;
  std::vector<float> vx;
  std::vector<float> vy;

  Grid() = default;
  Grid(int width, int height)
  : w(width), h(height), vx(static_cast<size_t>(width) * height, 0.0f),
    vy(static_cast<size_t>(width) * height, 0.0f) {}
  Grid(int width, int height, std::vector<float> vx_in, std::vector<float> vy_in)
  : w(width), h(height), vx(std::move(vx_in)), vy(std::move(vy_in)) {}

  size_t size() const { return vx.size(); }
};

}  // namespace event_detector_cpp::flow
