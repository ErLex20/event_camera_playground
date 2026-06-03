/**
 * Mosaic and Reconstruction implementation (minimal ROS2 port).
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * May 29, 2026
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

#include "evo/mosaic.hpp"

#include <glog/logging.h>

#include <chrono>
#include <cmath>

#include <cv_bridge/cv_bridge.hpp>

#include "evo/poisson_solver/laplace.h"
#include "evo_utils/interpolation.hpp"

namespace evo
{

// ── Mosaic ──────────────────────────────────────────────────────────────

void Mosaic::setup(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
                   float sigma_m, float init_cov, int window_size, int map_blur) {
    node_ = node;
    nh_ = {node, ""};
    nhp_ = {node, "reco."};
    tf_buffer_ = tf_buffer;
    R_(0, 0) = sigma_m;
    init_cov_ = init_cov;
    window_size_ = static_cast<size_t>(window_size);
    map_blur_ = static_cast<size_t>(map_blur);

    frame_id_ = rpg_common_ros::param<std::string>(nh_, "dvs_frame_id", std::string("dvs"));
    world_frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "world_frame_id", std::string("world"));

    pub_img_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "~/reconstruction/image", rclcpp::QoS(1));
    pub_depth_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "~/reconstruction/depthmap", rclcpp::QoS(1));
}

void Mosaic::setCamModel(image_geometry::PinholeCameraModel &c) {
    static const double s =
        rpg_common_ros::param<double>(nhp_, "super_resolution_factor", 1.);
    c_ = c;
    K_ = tf2::Matrix3x3(c_.fx(), 0., c_.cx(), 0., c_.fy(), c_.cy(), 0., 0., 1.);
    KInv_ = K_.inverse();
    w_ = static_cast<size_t>(c_.fullResolution().width);
    h_ = static_cast<size_t>(c_.fullResolution().height);

    K_virt_ = tf2::Matrix3x3(s * c_.fx(), 0., s * c_.cx(), 0., s * c_.fy(),
                             s * c_.cy(), 0., 0., 1.);
    KInv_virt_ = K_virt_.inverse();
    w_virt_ = static_cast<size_t>(s * c_.fullResolution().width);
    h_virt_ = static_cast<size_t>(s * c_.fullResolution().height);

    precomputeRectificationMap();
}

namespace {

// Convert a geometry_msgs transform to a tf2::Transform.
tf2::Transform toTf2(const geometry_msgs::msg::Transform &m) {
    tf2::Transform t;
    t.setOrigin(tf2::Vector3(m.translation.x, m.translation.y, m.translation.z));
    t.setRotation(tf2::Quaternion(m.rotation.x, m.rotation.y, m.rotation.z,
                                  m.rotation.w));
    return t;
}

}  // namespace

void Mosaic::compute(const std::vector<Event> &evs,
                     const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &map) {
    const int pose_update_step = 500, map_update_step = 10000;

    if (evs.empty()) return;

    EventTime t1 = ts_ref_ = evs.back().ts;

    cv::Mat img(static_cast<int>(h_virt_), static_cast<int>(w_virt_), CV_32F,
                cv::Scalar(0.));
    cv::Mat depth(static_cast<int>(h_), static_cast<int>(w_), CV_32F,
                  cv::Scalar(0.));

    // Convert map to internal representation.
    map_.clear();
    for (const auto &p : map->points) {
        map_.push_back(tf2::Vector3(p.x, p.y, p.z));
    }

    // Reproject map into the virtual camera at the reference time to obtain the
    // super-resolved depthmap used by the EKF.
    try {
        auto ref_t = tf2::TimePoint(std::chrono::nanoseconds(ts_ref_.toNSec()));
        tf2::Transform T_map =
            toTf2(tf_buffer_
                      ->lookupTransform(world_frame_id_, frame_id_, ref_t)
                      .transform);
        reprojectDepthmap(T_map, depthmap, true);
    } catch (const tf2::TransformException &e) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                             "Mosaic TF lookup failed: %s", e.what());
        return;
    }

    grad = cv::Mat_<cv::Vec2f>(static_cast<int>(h_virt_),
                               static_cast<int>(w_virt_), cv::Vec2f(0.f, 0.f));
    cov = cv::Mat_<Covariance>(static_cast<int>(h_virt_),
                               static_cast<int>(w_virt_));
    for (size_t y = 0; y < h_virt_; ++y)
        for (size_t x = 0; x < w_virt_; ++x)
            cov(static_cast<int>(y), static_cast<int>(x)) =
                Covariance(init_cov_, 0.f, 0.f, init_cov_);

    tf2::Transform T;       // pose at current event, relative to ts_ref_
    bool have_pose = false;

    for (size_t i = 0; i < evs.size(); ++i) {
        int idx = static_cast<int>(evs.size()) - 1 - static_cast<int>(i);
        const auto &e = evs[static_cast<size_t>(idx)];

        // Periodically refresh the dense depthmap at the current event time.
        if (i % static_cast<size_t>(map_update_step) == 0) {
            try {
                auto t_ev =
                    tf2::TimePoint(std::chrono::nanoseconds(e.ts.toNSec()));
                tf2::Transform T_md = toTf2(
                    tf_buffer_
                        ->lookupTransform(world_frame_id_, frame_id_, t_ev)
                        .transform);
                reprojectDepthmap(T_md, depth);
            } catch (const tf2::TransformException &) {
                // skip
            }
        }

        // Periodically refresh the relative pose (event frame -> ref frame).
        if (i % static_cast<size_t>(pose_update_step) == 0) {
            try {
                T = toTf2(
                    tf_buffer_
                        ->lookupTransform(
                            frame_id_,
                            tf2::TimePoint(std::chrono::nanoseconds(
                                ts_ref_.toNSec())),
                            frame_id_,
                            tf2::TimePoint(std::chrono::nanoseconds(
                                e.ts.toNSec())),
                            world_frame_id_)
                        .transform);
                have_pose = true;
            } catch (const tf2::TransformException &) {
                // keep previous pose
            }
        }

        // EKF update once a window of events has been integrated.
        if (i != 0 && ((i + 1) % window_size_ == 0)) {
            update(img, e.ts, t1);
            img = cv::Scalar(0.);
            t1 = e.ts;
        }

        if (!have_pose) continue;

        size_t idx_rect = static_cast<size_t>(e.x) + w_ * e.y;
        if (idx_rect >= rectification_map_.size()) continue;
        cv::Point2d px = rectification_map_[idx_rect];
        if (px.x < 0) continue;

        float z = depth.at<float>(static_cast<int>(px.y),
                                  static_cast<int>(px.x));

        // Project the event onto the map and into the virtual camera.
        tf2::Vector3 p(px.x, px.y, 1.);
        p = T * (z * (KInv_ * p));

        if (p.z() < 0.01f) continue;

        p = K_virt_ * (p / p.z());

        float pol = e.polarity ? +1.f : -1.f;
        evo_utils::interpolate::draw(img, p.x(), p.y(), pol);
    }

    publishImageReconstruction();
    publishDepthmap(depthmap);
}

cv::Mat Mosaic::getAbsoluteGradient() {
    cv::Mat img(grad.size(), CV_32F);

    for (size_t y = 0; y < h_virt_; ++y)
        for (size_t x = 0; x < w_virt_; ++x)
            img.at<float>(static_cast<int>(y), static_cast<int>(x)) =
                cv::norm(grad(y, x));

    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    return img;
}

cv::Mat Mosaic::getReconstructedImage() {
    int w_ = grad.cols, h_ = grad.rows;
    boost::multi_array<double, 2> F(boost::extents[h_][w_]);
    boost::multi_array<double, 2> U;

    // Calculate Second Derivative
    for (int y = 0; y < h_ - 1; ++y)
        for (int x = 0; x < w_ - 1; ++x) {
            float uxx = grad(y, x + 1)[0] - grad(y, x)[0],
                  vxx = grad(y + 1, x)[1] - grad(y, x)[1];
            F[y][x] = uxx + vxx;
        }
    F[h_ - 1][w_ - 1] = 0.;

    // Solve for M_xx + M_yy = F using Poisson solver.
    double boundary = 0.;
    pde::poisolve(U, F, 1., 1., 1., 1., boundary,
                  pde::types::boundary::Dirichlet, false);

    // Copy Multiarray to cv::Mat
    cv::Mat img(h_, w_, CV_32F);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x) img.at<float>(y, x) = U[y][x];

    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    return img;
}

void Mosaic::reprojectDepthmap(const tf2::Transform &T, cv::Mat &_depthmap,
                               bool virtual_cam) {
    const tf2::Matrix3x3 &K = virtual_cam ? K_virt_ : K_;
    const size_t h_orig = virtual_cam ? h_virt_ : h_,
                 w_orig = virtual_cam ? w_virt_ : w_,
                 h = h_orig + map_blur_ - 1,
                 w = w_orig + map_blur_ - 1;

    cv::Mat weights(h, w, CV_32F, cv::Scalar(0.));
    cv::Mat depth(h, w, CV_32F, cv::Scalar(0.));

    cv::Mat kern = cv::getGaussianKernel(static_cast<int>(map_blur_), 0.,
                                         CV_32F);
    kern = kern * kern.t();
    tf2::Transform TInv = T.inverse();

    static std::vector<float> z_values;
    z_values.clear();

    for (size_t i = 0; i < map_.size(); ++i) {
        tf2::Vector3 p = TInv * map_[i];
        float z = p.z();
        z_values.push_back(z);

        if (z < 0.01f) continue;

        p = K * (p / z);
        if (p.x() < map_blur_ / 2 || p.y() < map_blur_ / 2) continue;

        size_t x = static_cast<size_t>(p.x() + .5f);
        size_t y = static_cast<size_t>(p.y() + .5f);

        if (x >= w_orig || y >= h_orig) continue;

        cv::Mat patch = z * kern;

        cv::Rect dst(static_cast<int>(x), static_cast<int>(y),
                     static_cast<int>(map_blur_),
                     static_cast<int>(map_blur_));
        depth(dst) += patch;
        weights(dst) += kern;
    }

    depth /= weights;

    // Fill with Median Depth
    std::nth_element(z_values.begin(),
                     z_values.begin() + z_values.size() / 2,
                     z_values.end());
    depth_median_ = z_values[z_values.size() / 2];

    // Pixels with no contribution have weights==0, so `depth /= weights`
    // leaves them NaN (0/0). A plain `< .1f` test does NOT catch NaN/Inf
    // (comparisons with NaN are false), which would propagate non-finite
    // event coordinates into draw() and segfault. Treat non-finite as invalid.
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            const float d = depth.at<float>(y, x);
            if (!std::isfinite(d) || d < .1f)
                depth.at<float>(y, x) = depth_median_;
        }

    cv::medianBlur(depth, depth, 3);

    int b2 = static_cast<int>(map_blur_) / 2;
    _depthmap =
        depth(cv::Rect(b2, b2, static_cast<int>(w_orig),
                       static_cast<int>(h_orig)))
            .clone();
}

void Mosaic::precomputeRectificationMap() {
    for (size_t y = 0; y < h_; ++y) {
        for (size_t x = 0; x < w_; ++x) {
            cv::Point2d p(c_.rectifyPoint(cv::Point2d(x, y)));
            cv::Point2i p_i = p;

            if (p_i.x < 0 || p_i.x >= static_cast<int>(w_) ||
                p_i.y < 0 || p_i.y >= static_cast<int>(h_))
                rectification_map_.push_back(cv::Point2d(-1, -1));
            else
                rectification_map_.push_back(p);
        }
    }
}

void Mosaic::update(const cv::Mat &new_grad, const EventTime &t0,
                    const EventTime &t1) {
    float min_grad = .0001f;

    tf2::Transform T0, T1;
    try {
        geometry_msgs::msg::TransformStamped m0 = tf_buffer_->lookupTransform(
            frame_id_, tf2::TimePoint(std::chrono::nanoseconds(t0.toNSec())),
            frame_id_,
            tf2::TimePoint(std::chrono::nanoseconds(ts_ref_.toNSec())),
            world_frame_id_);
        T0.setOrigin(tf2::Vector3(m0.transform.translation.x,
                                  m0.transform.translation.y,
                                  m0.transform.translation.z));
        T0.setRotation(tf2::Quaternion(m0.transform.rotation.x,
                                       m0.transform.rotation.y,
                                       m0.transform.rotation.z,
                                       m0.transform.rotation.w));

        geometry_msgs::msg::TransformStamped m1 = tf_buffer_->lookupTransform(
            frame_id_, tf2::TimePoint(std::chrono::nanoseconds(t1.toNSec())),
            frame_id_,
            tf2::TimePoint(std::chrono::nanoseconds(ts_ref_.toNSec())),
            world_frame_id_);
        T1.setOrigin(tf2::Vector3(m1.transform.translation.x,
                                  m1.transform.translation.y,
                                  m1.transform.translation.z));
        T1.setRotation(tf2::Quaternion(m1.transform.rotation.x,
                                       m1.transform.rotation.y,
                                       m1.transform.rotation.z,
                                       m1.transform.rotation.w));
    } catch (const tf2::TransformException &) {
        return;
    }

    int h = static_cast<int>(h_virt_), w = static_cast<int>(w_virt_);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float z = new_grad.at<float>(y, x);

            if (std::abs(z) < min_grad) continue;

            const float depth =
                depthmap.at<float>(static_cast<int>(y), static_cast<int>(x));

            Gradient &g = grad(y, x);
            Covariance &P = cov(y, x);

            // Velocity
            tf2::Vector3 p_ref =
                depth * (KInv_virt_ * tf2::Vector3(x, y, 1.));
            tf2::Vector3 p0 = T0 * p_ref, p1 = T1 * p_ref;

            if (p0.z() < 0.01f || p1.z() < 0.01f) continue;

            p0 = K_virt_ * (p0 / p0.z());
            p1 = K_virt_ * (p1 / p1.z());

            const Velocity v(p0.x() - p1.x(), p0.y() - p1.y());

            // EKF Update Equations (h = g . v)
            const cv::Matx12f dhdg(v[0], v[1]);
            const float h_pred = g[0] * v[0] + g[1] * v[1];
            const cv::Matx<float, 1, 1> nu(z - h_pred);
            const cv::Matx<float, 1, 1> S = dhdg * P * dhdg.t() + R_;
            const cv::Matx21f W = P * dhdg.t() * S.inv();

            // Apply Update: g += W * nu;  P -= W * S * W^T
            const cv::Matx21f Wnu = W * nu;
            g[0] += Wnu(0, 0);
            g[1] += Wnu(1, 0);
            P -= W * S * W.t();
        }
}

void Mosaic::publishImageReconstruction() {
    if (!pub_img_ || pub_img_->get_subscription_count() == 0) return;

    cv::Mat img = getReconstructedImage();
    cv::convertScaleAbs(img, img, 255);

    std_msgs::msg::Header header;
    header.stamp = rclcpp::Time(ts_ref_.toNSec());

    sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(header, "mono8", img).toImageMsg();
    pub_img_->publish(*msg);
}

void Mosaic::publishDepthmap(const cv::Mat &depth) {
    if (!pub_depth_ || pub_depth_->get_subscription_count() == 0) return;

    cv::Mat img;
    cv::normalize(depth, img, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(img, img);
    cv::applyColorMap(img, img, cv::COLORMAP_JET);
    sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
            .toImageMsg();
    pub_depth_->publish(*msg);
}

// ── Reconstruction ──────────────────────────────────────────────────────

void Reconstruction::setup(rclcpp::Node *node,
                           std::shared_ptr<tf2::BufferCore> tf_buffer,
                           const image_geometry::PinholeCameraModel &cam) {
    node_ = node;
    nh_ = {node, ""};
    nhp_ = {node, "reco."};
    tf_buffer_ = tf_buffer;

    frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "dvs_frame_id",
                                           std::string("dvs"));
    world_frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "world_frame_id",
                                           std::string("world"));

    c_ = cam;
    cur_ev_ = 0;

    // Initialize the Mosaic (EKF/Poisson reconstructor). Parameters mirror the
    // original dvs_reconstruction node (sigma_m, init_cov, window_size,
    // map_blur), read from the reco. namespace.
    const float sigma_m = rpg_common_ros::param<float>(nhp_, "sigma_m", 10.f);
    const float init_cov = rpg_common_ros::param<float>(nhp_, "init_cov", 10.f);
    const int window_size =
        rpg_common_ros::param<int>(nhp_, "window_size", 5000);
    const int map_blur = rpg_common_ros::param<int>(nhp_, "map_blur", 15);
    mosaic_.setup(node_, tf_buffer_, sigma_m, init_cov, window_size, map_blur);
    mosaic_.setCamModel(c_);
}

void Reconstruction::addEvents(const std::vector<Event> &events) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    events_.insert(events_.end(), events.begin(), events.end());
}

void Reconstruction::setMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &map,
    const EventTime &stamp) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    map_ = map;
    map_stamp_ = stamp;
    doReconstruction();
}

void Reconstruction::doReconstruction() {
    static const size_t events_for_reconstruction =
        rpg_common_ros::param<int>(nhp_, "events_for_reconstruction", 100000);

    if (!map_ || events_.empty()) return;

    // Advance the event cursor up to the map timestamp.
    while (cur_ev_ < events_.size() && events_[cur_ev_].ts < map_stamp_)
        ++cur_ev_;

    // Drop processed history beyond the reconstruction window.
    if (cur_ev_ > events_for_reconstruction) {
        size_t remove_events = cur_ev_ - events_for_reconstruction;
        events_.erase(events_.begin(), events_.begin() + remove_events);
        cur_ev_ -= remove_events;
    }

    if (cur_ev_ == 0) return;

    // Require camera poses at both ends of the window (the original aborted the
    // reconstruction otherwise).
    const size_t last = std::min(cur_ev_, events_.size() - 1);
    const auto t0 =
        tf2::TimePoint(std::chrono::nanoseconds(events_[0].ts.toNSec()));
    const auto t1 =
        tf2::TimePoint(std::chrono::nanoseconds(events_[last].ts.toNSec()));
    if (!tf_buffer_->canTransform(world_frame_id_, frame_id_, t0) ||
        !tf_buffer_->canTransform(world_frame_id_, frame_id_, t1))
        return;

    std::vector<Event> evs(events_.begin(), events_.begin() + cur_ev_);
    mosaic_.compute(evs, map_);
}

}  // namespace evo
