#pragma once

#include <kindr/minimal/quat-transformation.h>

#include <opencv2/core/core.hpp>

namespace evo_utils::geometry {

using Transformation = kindr::minimal::QuatTransformation;
using Quaternion = kindr::minimal::RotationQuaternion;

// Aliases for clarity
typedef float Depth;
typedef float InverseDepth;

/**
 * Naive 2D median filter. Complexity: O(p^2NM) where p is the patch size, and
 * N, M the image size
 *
 * @see Remark: use the huangMedianFilter, this function is here because it is
 * used as a reference in the unit tests
 *
 * @param img Input image
 * @param out_img Filtered image
 * @param mask Which pixels have to be considered
 * @param patch_size Filter kernel size
 */
void naiveMedianFilter(const cv::Mat& img, cv::Mat& out_img,
                       const cv::Mat& mask, int patch_size);

/**
 * 2D median filter. Complexity: O(pNM) where p is the patch size, and N, M the
 * image size
 *
 * "T. Huang, G. Yang, and G. Tang, "A Fast Two-Dimensional Median Filtering
 * Algorithm"
 *
 * Code adapted from
 * http://www.sergejusz.com/engineering_tips/median_filter.htm
 *
 * @param img Input image
 * @param out_img Filtered image
 * @param mask Which pixels have to be considered
 * @param patch_size Filter kernel size
 */
void huangMedianFilter(const cv::Mat& img, cv::Mat& out_img,
                       const cv::Mat& mask, int patch_size);

}  // namespace evo_utils::geometry
