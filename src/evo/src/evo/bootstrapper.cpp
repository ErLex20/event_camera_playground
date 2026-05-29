#include "evo/bootstrapper.hpp"

#include <glog/logging.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cv_bridge/cv_bridge.hpp>

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include "evo_utils/camera.hpp"
#include "evo_utils/geometry.hpp"

namespace dvs_bootstrapping {

// ---------------------------------------------------------------------------
// Bootstrapper (base)
// ---------------------------------------------------------------------------

void Bootstrapper::setupBase(rclcpp::Node *node,
                             std::shared_ptr<tf2::BufferCore> tf_buffer) {
    node_ = node;
    nh_ = {node, ""};
    nhp_ = {node, "boot."};
    tf_buffer_ = tf_buffer;

    idle_ = !rpg_common_ros::param<bool>(nhp_, "auto_trigger", false);

    LOG(INFO) << "Bootstrapper initially idle: " << idle_;

    const std::string camera_name =
        rpg_common_ros::param<std::string>(nh_, "camera_name", "");
    const std::string calib_file =
        "/home/neo/workspace/src/evo/config/evo_calibration.yaml";
    cam_ = evo_utils::camera::loadPinholeCamera(camera_name, calib_file);

    world_frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "world_frame_id", "world");
    bootstrap_frame_id_ = rpg_common_ros::param<std::string>(
        nh_, "dvs_bootstrap_frame_id", "dvs_evo");
}

void Bootstrapper::onRemoteKey(const std::string &cmd) {
    if (cmd == "bootstrap") {
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            eventQueue_.clear();
        }

        LOG(INFO) << "Bootstrapping not idle anymore.";
        idle_ = false;

        postBootstrapCalled();
    }
}

void Bootstrapper::pauseBootstrapper() {
    LOG(INFO) << "Bootstrapper is now idle.";
    idle_ = true;
}

void Bootstrapper::addEvents(const std::vector<evo::Event> &events) {
    if (idle_) return;

    std::lock_guard<std::mutex> lock(data_mutex_);
    eventQueue_.insert(eventQueue_.end(), events.begin(), events.end());
}

// ---------------------------------------------------------------------------
// EventsFramesBootstrapper
// ---------------------------------------------------------------------------

void EventsFramesBootstrapper::setupEventsFrames() {
    postCameraLoaded();

    rate_hz_ = rpg_common_ros::param<int>(nhp_, "rate_hz", 25);
    frame_size_ = rpg_common_ros::param<int>(nhp_, "frame_size", 50000);
    local_frame_size_ =
        rpg_common_ros::param<int>(nhp_, "local_frame_size", 10000);
    CHECK_GE(frame_size_, local_frame_size_);
    enable_visuals_ =
        rpg_common_ros::param<bool>(nhp_, "enable_visualizations", false);
    if (enable_visuals_) {
        pub_event_img_ = node_->create_publisher<sensor_msgs::msg::Image>(
            "~/bootstrap/event_frame", rclcpp::QoS(1));
        pub_optical_flow_ = node_->create_publisher<sensor_msgs::msg::Image>(
            "~/bootstrap/optical_flow", rclcpp::QoS(1));
    }
}

void EventsFramesBootstrapper::postCameraLoaded() {
    static int nIt =
        rpg_common_ros::param<int>(nhp_, "unwarp_estimate_n_it", 50);
    static double eps =
        rpg_common_ros::param<double>(nhp_, "unwarp_estimate_eps", 1e-2);
    static int lvls =
        rpg_common_ros::param<int>(nhp_, "unwarp_estimate_pyramid_lvls", 2);

    std::lock_guard<std::mutex> lock(data_mutex_);
    sensor_size_ = cam_.fullResolution();
    height_ = sensor_size_.height;
    width_ = sensor_size_.width;

    // pre compute rectification table and weights
    cv::initUndistortRectifyMap(cam_.intrinsicMatrix(), cam_.distortionCoeffs(),
                                cv::noArray(), cv::noArray(), sensor_size_,
                                CV_32FC1, undistort_mapx_, undistort_mapy_);

    unwarp_params_ = motion_correction::WarpUpdateParams(
        nIt, eps, cv::MOTION_HOMOGRAPHY, lvls, sensor_size_);
    eventQueue_.clear();

    // Precompute rectification table
    evo_utils::camera::precomputeRectificationTable(rectified_points_, cam_);
}

void EventsFramesBootstrapper::integratingThread(std::atomic<bool> &running) {
    const auto period = std::chrono::milliseconds(
        rate_hz_ > 0 ? static_cast<int>(1000 / rate_hz_) : 40);

    while (running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(period);

        if (!idle_) {
            if (integrateEvents()) {
                clearEventQueue();
            }
        }
    }
}

void EventsFramesBootstrapper::clearEventQueue() {
    if (newest_processed_event_ <= frame_size_) return;

    std::lock_guard<std::mutex> lock(data_mutex_);
    eventQueue_.erase(
        eventQueue_.begin(),
        eventQueue_.begin() + (newest_processed_event_ - frame_size_));
}

bool EventsFramesBootstrapper::integrateEvents() {
    static cv::Mat img0, img1, warp, event_img;
    static cv::Mat flow_field;
    static size_t min_step_size =
        rpg_common_ros::param<int>(nhp_, "min_step_size", 5000);
    static float events_scale_factor =
        rpg_common_ros::param<float>(nhp_, "events_scale_factor", 7);
    static size_t th_min =
        rpg_common_ros::param<float>(nhp_, "activation_threshold_min", 200);
    static bool median_filtering =
        rpg_common_ros::param<bool>(nhp_, "median_filtering", true);
    static bool adaptive_thresholding =
        rpg_common_ros::param<bool>(nhp_, "adaptive_thresholding", true);
    static size_t median_filter_size =
        rpg_common_ros::param<int>(nhp_, "median_filter_size", 3);

    // perform this step in a thread safe manner
    std::lock_guard<std::mutex> lock_events(data_mutex_);
    if (eventQueue_.size() <= static_cast<size_t>(frame_size_)) return false;

    size_t last_event = eventQueue_.size() - 1;
    if (last_event - newest_processed_event_ < min_step_size) return false;

    auto frame0_begin = eventQueue_.end() - frame_size_;
    auto frame0_end = frame0_begin + local_frame_size_;
    auto frame1_begin = eventQueue_.end() - local_frame_size_;
    auto frame1_end = eventQueue_.end() - 1;
    const double dt = (frame1_end->ts - frame0_begin->ts).toSec();

    motion_correction::resetMat(img0, sensor_size_, CV_32F);
    motion_correction::resetMat(img1, sensor_size_, CV_32F);
    motion_correction::resetMat(event_img, sensor_size_, CV_32F);

    motion_correction::drawEventsUndistorted(
        frame0_begin, frame0_end, img0, sensor_size_, rectified_points_, false);

    motion_correction::drawEventsUndistorted(
        frame1_begin, frame1_end, img1, sensor_size_, rectified_points_, false);

    static cv::Mat I33 = cv::Mat::eye(3, 3, CV_32F);
    warp = I33;

    motion_correction::updateWarp(warp, img0, img1, unwarp_params_);

    flow_field = motion_correction::computeFlowFromWarp(warp, dt, sensor_size_,
                                                        rectified_points_);

    motion_correction::drawEventsMotionCorrectedOpticalFlow(
        frame0_begin, frame1_end, flow_field, event_img, sensor_size_,
        rectified_points_, false);

    cv::Mat frame = (255.0 / events_scale_factor) * (event_img);
    frame.convertTo(frame, CV_8U);

    if (median_filtering)
        evo_utils::geometry::huangMedianFilter(
            frame, frame, cv::Mat(height_, width_, CV_8U, 1),
            median_filter_size);
    if (adaptive_thresholding)
        cv::threshold(frame, frame, th_min, 255,
                      cv::ThresholdTypes::THRESH_TOZERO);

    ts_ = frame1_end->ts;
    if (enable_visuals_) {
        publishOpticalFlowVectors(flow_field);
        publishEventImage(frame, ts_);
    }

    newest_processed_event_ = last_event;

    // save img for later use
    std::lock_guard<std::mutex> lock_frames(img_mutex_);
    if (keep_frames_) {
        frames_.push_back(frame);
    }

    return true;
}

void EventsFramesBootstrapper::publishEventImage(const cv::Mat &img,
                                                 const evo::EventTime &ts) {
    if (!pub_event_img_ || pub_event_img_->get_subscription_count() == 0)
        return;

    cv_bridge::CvImage cv_event_image;
    cv_event_image.encoding = "mono8";
    img.convertTo(cv_event_image.image, CV_8U);

    auto aux = cv_event_image.toImageMsg();
    aux->header.stamp = rclcpp::Time(ts.toNSec());
    pub_event_img_->publish(*aux);
}

void EventsFramesBootstrapper::publishOpticalFlowVectors(
    const cv::Mat &flow_field) {
    if (!pub_optical_flow_ || pub_optical_flow_->get_subscription_count() == 0)
        return;

    static cv::Mat disp;
    motion_correction::resetMat(disp, sensor_size_, CV_8UC3);

    static const int step = 18;
    static const float scale = 0.04;

    cv::Point2f origin;
    cv::Point2f end;
    for (int y = 0; y < flow_field.rows; y += step) {
        for (int x = 0; x < flow_field.cols; x += step) {
            const cv::Vec2f &flow_vec = flow_field.at<cv::Vec2f>(y, x);

            origin.x = x;
            origin.y = y;
            end.x = origin.x + scale * flow_vec[0];
            end.y = origin.y + scale * flow_vec[1];

            cv::arrowedLine(disp, origin, end, cv::Scalar(0, 0, 255), 1, 8, 0,
                            0.15);
        }
    }

    cv_bridge::CvImage cv_event_image;
    cv_event_image.encoding = "bgr8";
    cv_event_image.image = disp;
    auto aux = cv_event_image.toImageMsg();
    pub_optical_flow_->publish(*aux);
}

// ---------------------------------------------------------------------------
// FrontoPlanarBootstrapper
// ---------------------------------------------------------------------------

void FrontoPlanarBootstrapper::setup(
    rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
    std::function<void(const PointCloud::Ptr &, const rclcpp::Time &)>
        map_callback) {
    setupBase(node, tf_buffer);
    setupEventsFrames();

    pcl_ = PointCloud::Ptr(new PointCloud);
    map_callback_ = map_callback;
    pub_tf_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    start_ = rpg_common_ros::param<bool>(nhp_, "auto_trigger", false);

    pub_pc_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/bootstrap/pointcloud", rclcpp::QoS(1));
    pcl_->header.frame_id = world_frame_id_;
}

void FrontoPlanarBootstrapper::bootstrappingThread(std::atomic<bool> &running) {
    static const bool one_shot =
        rpg_common_ros::param<bool>(nhp_, "one_shot", false);
    static const int n_subscribers_to_wait =
        rpg_common_ros::param<int>(nhp_, "n_subscribers_to_wait", 0);

    idle_ = false;  // this way events frames are published before actually
                    // bootstrapping

    const auto period = std::chrono::milliseconds(
        rate_hz_ > 0 ? static_cast<int>(1000 / rate_hz_) : 40);

    while (running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(period);
        std::lock_guard<std::mutex> lock_frames(img_mutex_);
        if (start_) {
            keep_frames_ = true;
            if (!frames_.empty()) {
                if (static_cast<int>(pub_pc_->get_subscription_count()) >=
                        n_subscribers_to_wait ||
                    map_callback_) {
                    if (bootstrap()) {
                        if (one_shot) {
                            start_ = false;
                        }
                    }
                }
            }
        } else if (frames_.size() > 1) {
            frames_.erase(frames_.begin(), frames_.end() - 2);
        }
    }
}

void FrontoPlanarBootstrapper::postBootstrapCalled() { start_ = true; }

bool FrontoPlanarBootstrapper::bootstrap() {
    static float plane_distance =
        rpg_common_ros::param<float>(nhp_, "plane_distance", 1.);

    static cv::Mat activation_mask;
    activation_mask = frames_.back();
    frames_.clear();
    pcl_->clear();

    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            if (activation_mask.at<u_char>(y, x) > 0) {
                cv::Point3d f_ref = cam_.projectPixelTo3dRay(cv::Point2d(x, y));
                cv::Point3d p = f_ref * plane_distance / f_ref.z;

                PointType p_world;
                p_world.x = p.x;
                p_world.y = p.y;
                p_world.z = p.z;
                p_world.intensity = 1.0 / p_world.z;
                pcl_->push_back(p_world);
            }
        }
    }

    return publishPcl();
}

bool FrontoPlanarBootstrapper::publishPcl() {
    static const float radius_search =
        rpg_common_ros::param<float>(nhp_, "radius_search", 0.5);
    static const float min_num_neighbors =
        rpg_common_ros::param<int>(nhp_, "min_num_neighbors", 2);

    if (pcl_->empty()) {
        LOG(WARNING) << "Point cloud empty!";
        return false;
    }

    // filter point cloud to remove outliers
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::RadiusOutlierRemoval<PointType> outrem;
    outrem.setInputCloud(pcl_);
    outrem.setRadiusSearch(radius_search);
    outrem.setMinNeighborsInRadius(min_num_neighbors);
    outrem.filter(*cloud_filtered);
    pcl_->swap(*cloud_filtered);

    if (pcl_->empty()) {
        LOG(WARNING) << "Point cloud empty after filtering!";
        return false;
    }

    const rclcpp::Time stamp(ts_.toNSec());

    // Broadcast and insert the initial (identity) bootstrap pose so that the
    // tracker can look it up to initialize (replaces tf::TransformBroadcaster).
    geometry_msgs::msg::TransformStamped first_pose;
    first_pose.header.stamp = stamp;
    first_pose.header.frame_id = world_frame_id_;
    first_pose.child_frame_id = bootstrap_frame_id_;
    first_pose.transform.rotation.w = 1.0;
    pub_tf_->sendTransform(first_pose);
    tf_buffer_->setTransform(first_pose, "evo_bootstrap");

    pcl_->header.frame_id = world_frame_id_;

    LOG(INFO) << "Fronto planar map published";
    sensor_msgs::msg::PointCloud2 pc_to_publish;
    pcl::toROSMsg(*pcl_, pc_to_publish);
    pc_to_publish.header.frame_id = world_frame_id_;
    pc_to_publish.header.stamp = stamp;
    pub_pc_->publish(pc_to_publish);

    // Feed the bootstrap map to the tracker/mapper (replaces the inter-node
    // pointcloud topic).
    if (map_callback_) {
        map_callback_(pcl_, stamp);
    }

    return true;
}

}  // namespace dvs_bootstrapping
