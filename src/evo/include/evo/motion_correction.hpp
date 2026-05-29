#pragma once

#ifndef MOTION_CORRECTION_HPP
#define MOTION_CORRECTION_HPP

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>

#include "evo/event_types.hpp"

namespace motion_correction {

using EventArray = std::vector<evo::Event>;

/**
 * Custom floor function, reduces overhead avoiding overflows check (not
 * required in our use cases)
 */
static inline int int_floor(float x) {
    int i = (int)x;     /* truncate */
    return i - (i > x); /* convert trunc to floor */
}

/**
 * Resets matrix if already initialized, otherwise instantiates a zero matrix
 */
static inline void resetMat(cv::Mat& arr, const cv::Size& size,
                            int type = CV_32F) {
    if (arr.cols == 0 || arr.rows == 0) {
        arr = cv::Mat::zeros(size, type);
    } else {
        arr.setTo(0);
    }
}

/**
 * Embeds all the parameters required for the estimation of the warp
 */
class WarpUpdateParams {
   public:
    int warp_mode;              ///< currently supported: cv::MOTION_HOMOGRAPHY
    int num_pyramid_levels;     ///< # pyramid levels used to estimate the warp
    cv::Size sensor_size;       ///< image size
    cv::TermCriteria criteria;  ///< termination criteria for the optimization
    WarpUpdateParams() {}

    WarpUpdateParams(int nIt, double eps, int mode, int lvls,
                     cv::Size sensor_size)
        : warp_mode(mode),
          num_pyramid_levels(lvls),
          criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, nIt, eps),
          sensor_size(sensor_size) {}
};

void initWarp(cv::Mat& warp, const WarpUpdateParams& params);

void updateWarp(cv::Mat& warp, const cv::Mat& img0, const cv::Mat& img1,
                const WarpUpdateParams& params);

cv::Mat computeFlowFromWarp(const cv::Mat& warp, double dt,
                            cv::Size sensor_size,
                            std::vector<cv::Point2f> rectified_points);

void drawEventsUndistorted(EventArray::iterator ev_first,
                           EventArray::iterator ev_last, cv::Mat& out,
                           cv::Size sensor_size,
                           const std::vector<cv::Point2f>& rectified_points,
                           const bool use_polarity);

void drawEventsMotionCorrectedOpticalFlow(
    EventArray::iterator ev_first, EventArray::iterator ev_last,
    const cv::Mat& flow_field, cv::Mat& out, cv::Size sensor_size,
    const std::vector<cv::Point2f>& rectified_points, const bool use_polarity);

}  // namespace motion_correction

#endif
