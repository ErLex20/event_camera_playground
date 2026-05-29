#include "evo_utils/camera.hpp"

#include <camera_calibration_parsers/parse.hpp>

#include <sensor_msgs/msg/camera_info.hpp>

namespace evo_utils::camera {

void precomputeRectificationTable(
    std::vector<cv::Point2f>& rectified_points,
    const image_geometry::PinholeCameraModel& cam) {
    static std::vector<cv::Point2f> points;
    points.clear();
    rectified_points.clear();

    cv::Size fr = cam.fullResolution();

    for (int y = 0; y != fr.height; ++y) {
        for (int x = 0; x != fr.width; ++x) {
            points.push_back(cv::Point2f(x, y));
        }
    }

    const std::string distortion_type = cam.cameraInfo().distortion_model;
    LOG(INFO) << "Distortion type: " << distortion_type;

    if (distortion_type == "plumb_bob") {
        auto K = cv::getOptimalNewCameraMatrix(cam.fullIntrinsicMatrix(),
                                               cam.distortionCoeffs(), fr, 0.);
        cv::undistortPoints(points, rectified_points, cam.fullIntrinsicMatrix(),
                            cam.distortionCoeffs(), cv::Matx33f::eye(), K);
    } else {
        CHECK(false) << "Distortion model: " << distortion_type
                     << " is not supported.";
    }
}

image_geometry::PinholeCameraModel loadPinholeCamera(
    const std::string& camera_name, const std::string& calib_file) {
    sensor_msgs::msg::CameraInfo cam_info;
    std::string parsed_name;

    const bool ok = camera_calibration_parsers::readCalibration(
        calib_file, parsed_name, cam_info);
    CHECK(ok) << "Could not read calibration file: " << calib_file;

    if (!camera_name.empty() && parsed_name != camera_name) {
        LOG(WARNING) << "Calibration camera name '" << parsed_name
                     << "' differs from requested '" << camera_name << "'";
    }

    image_geometry::PinholeCameraModel cam;
    cam.fromCameraInfo(cam_info);

    return cam;
}

}  // namespace evo_utils::camera
