#pragma once

#ifndef EVO_UTILS_CAMERA_HPP
#define EVO_UTILS_CAMERA_HPP

#include <glog/logging.h>

#include <image_geometry/pinhole_camera_model.hpp>

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include <Eigen/Core>

namespace evo_utils::camera {

/**
 * Precompute the rectification table for the given camera model.
 *
 * @param rectified_points [output] rectified pixel coordinates, row-major
 * @param cam pinhole camera model
 */
void precomputeRectificationTable(
    std::vector<cv::Point2f>& rectified_points,
    const image_geometry::PinholeCameraModel& cam);

/**
 * Load a pinhole camera model from a ROS calibration YAML file.
 *
 * Replaces the original ROS1 camera_info_manager based loader. The calibration
 * file is parsed with camera_calibration_parsers into a CameraInfo message,
 * which is then loaded into the pinhole camera model.
 *
 * @param camera_name expected camera name (for logging / validation)
 * @param calib_file path to the calibration YAML file
 *
 * @return the retrieved pinhole camera model
 */
image_geometry::PinholeCameraModel loadPinholeCamera(
    const std::string& camera_name, const std::string& calib_file);

typedef Eigen::Vector3d BearingVector;
typedef Eigen::Vector2d Keypoint;  // subpixel pixel coordinate
typedef Eigen::Vector3d Point;

/**
 * Pinhole camera model
 *
 * Provides basic functionality to project 3d point to pixel coordinates and
 * backwards
 */
class PinholeCamera {
   public:
    PinholeCamera() {}
    /**
     * Construct a PinholeCamera object
     *
     * @param width Width of the image
     * @param height Height of the image
     * @param fx Camera's horizontal focal length
     * @param fy Camera's vertical focal length
     * @param cx Camera's horizontal principal point
     * @param cy Camera's vertical principal point
     */
    PinholeCamera(int width, int height, float fx, float fy, float cx, float cy)
        : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
        CHECK_GT(width_, 0);
        CHECK_GT(height_, 0);
        CHECK_GT(fx_, 0.0);
        CHECK_GT(fy_, 0.0);
        CHECK_GT(cx_, 0.0);
        CHECK_GT(cy_, 0.0);

        K_ << (float)fx_, 0.f, (float)cx_, 0.f, (float)fy_, (float)cy_, 0.f,
            0.f, 1.f;

        Kinv_ = K_.inverse();
    }

    /**
     * Projects a 3d point (in camera frame) to the pixel coordinates
     *
     * @param P 3d point in camera frame
     *
     * @return Pixel coordinates of the projected point
     */
    inline Keypoint project3dToPixel(const Point& P) {
        CHECK_GE(std::fabs(P[2]), 1e-6);
        return Keypoint(fx_ * P[0] / P[2] + cx_, fy_ * P[1] / P[2] + cy_);
    }

    /**
     * Backprojects the pixel coordinates to 3d ray
     *
     * @param u Keypoint representing the pixel coordinates
     *
     * @return Bearing vector being the 3d ray, backprojection of P
     */
    inline BearingVector projectPixelTo3dRay(const Keypoint& u) {
        return BearingVector((u[0] - cx_) / fx_, (u[1] - cy_) / fy_, 1.0);
    }

    Eigen::Matrix3f K() const { return K_; }

    int width_;   ///< image size: width
    int height_;  ///< image size: height
    float fx_;    ///< camera's horizontal focal length
    float fy_;    ///< camera's vertical focal lenght
    float cx_;    ///< camera's principal point (x)
    float cy_;    ///< camera's principal point (y)

    Eigen::Matrix3f K_;     ///< camera's intrinsic matrix
    Eigen::Matrix3f Kinv_;  ///< camera's intrinsic matrix inverse
};

}  // namespace evo_utils::camera

#endif  // EVO_UTILS_CAMERA_HPP
