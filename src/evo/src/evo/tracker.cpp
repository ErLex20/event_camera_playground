#include "evo/tracker.hpp"

#include <glog/logging.h>

#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.hpp>
#include <thread>

#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/io.h>

#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>

#include <kindr/minimal/quat-transformation.h>

#include "evo_utils/camera.hpp"

using Transformation = kindr::minimal::QuatTransformation;
using Quaternion = kindr::minimal::RotationQuaternion;

namespace evo {

namespace {

tf2::Transform affineToTf2(const Eigen::Affine3d & T) {
    const Eigen::Quaterniond q(T.linear());
    const Eigen::Vector3d t = T.translation();
    tf2::Transform out;
    out.setOrigin(tf2::Vector3(t.x(), t.y(), t.z()));
    out.setRotation(tf2::Quaternion(q.x(), q.y(), q.z(), q.w()));
    return out;
}

Eigen::Affine3d tf2ToAffine(const tf2::Transform & tf) {
    const tf2::Vector3 & t = tf.getOrigin();
    const tf2::Quaternion & q = tf.getRotation();
    Eigen::Affine3d out = Eigen::Affine3d::Identity();
    out.translation() = Eigen::Vector3d(t.x(), t.y(), t.z());
    out.linear() =
        Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()).toRotationMatrix();
    return out;
}

tf2::Transform msgToTf2(const geometry_msgs::msg::Transform & m) {
    tf2::Transform out;
    out.setOrigin(
        tf2::Vector3(m.translation.x, m.translation.y, m.translation.z));
    out.setRotation(
        tf2::Quaternion(m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w));
    return out;
}

geometry_msgs::msg::TransformStamped toTransformStamped(const StampedPose & p) {
    geometry_msgs::msg::TransformStamped g;
    g.header.stamp = p.stamp_;
    g.header.frame_id = p.frame_id_;
    g.child_frame_id = p.child_frame_id_;
    const tf2::Vector3 & t = p.getOrigin();
    const tf2::Quaternion & q = p.getRotation();
    g.transform.translation.x = t.x();
    g.transform.translation.y = t.y();
    g.transform.translation.z = t.z();
    g.transform.rotation.x = q.x();
    g.transform.rotation.y = q.y();
    g.transform.rotation.z = q.z();
    g.transform.rotation.w = q.w();
    return g;
}

}  // namespace

void Tracker::postCameraLoaded() {
    width_ = c_.fullResolution().width;
    height_ = c_.fullResolution().height;
    fx_ = c_.fx();
    fy_ = c_.fy();
    cx_ = c_.cx();
    cy_ = c_.cy();
    rect_ = cv::Rect(0, 0, width_, height_);

    float fov = 2. * std::atan(c_.fullResolution().width / 2. / c_.fx());
    LOG(INFO) << "Field of view: " << fov / M_PI * 180.;

    new_img_ = cv::Mat(c_.fullResolution(), CV_32F, cv::Scalar(0));

    sensor_msgs::msg::CameraInfo cam_ref = c_.cameraInfo();
    cam_ref.width = rpg_common_ros::param<int>(nh_, "virtual_width",
                                               c_.fullResolution().width);
    cam_ref.height = rpg_common_ros::param<int>(nh_, "virtual_height",
                                                c_.fullResolution().height);
    cam_ref.p[0 * 4 + 2] = cam_ref.k[0 * 3 + 2] = 0.5 * (float)cam_ref.width;
    cam_ref.p[1 * 4 + 2] = cam_ref.k[1 * 3 + 2] = 0.5 * (float)cam_ref.height;

    float f_ref = rpg_common_ros::param<float>(nh_, "fov_virtual_camera_deg", 0.);
    if (f_ref == 0.)
        f_ref = c_.fx();
    else {
        const float f_ref_rad = f_ref * CV_PI / 180.0;
        f_ref = 0.5 * (float)cam_ref.width / std::tan(0.5 * f_ref_rad);
    }
    cam_ref.p[0 * 4 + 0] = cam_ref.k[0 * 3 + 0] = f_ref;
    cam_ref.p[1 * 4 + 1] = cam_ref.k[1 * 3 + 1] = f_ref;
    c_ref_.fromCameraInfo(cam_ref);

    reset();
}

void Tracker::setup(rclcpp::Node * node,
                    std::shared_ptr<tf2::BufferCore> tf_buffer) {
    node_ = node;
    nh_ = {node, ""};
    nhp_ = {node, "track."};
    tf_buffer_ = tf_buffer;
    tf_pub_ = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    cur_ev_ = 0;
    kf_ev_ = 0;
    noise_rate_ = rpg_common_ros::param<int>(nhp_, "noise_rate", 10000);
    frame_size_ = rpg_common_ros::param<int>(nhp_, "frame_size", 2500);
    step_size_ = rpg_common_ros::param<int>(nhp_, "step_size", 2500);
    idle_ = true;

    batch_size_ = rpg_common_ros::param<int>(nhp_, "batch_size", 500);
    max_iterations_ = rpg_common_ros::param<int>(nhp_, "max_iterations", 100);
    map_blur_ = rpg_common_ros::param<int>(nhp_, "map_blur", 5);

    pyramid_levels_ = rpg_common_ros::param<int>(nhp_, "pyramid_levels", 1);

    weight_scale_trans_ =
        rpg_common_ros::param<float>(nhp_, "weight_scale_translation", 0.);
    weight_scale_rot_ =
        rpg_common_ros::param<float>(nhp_, "weight_scale_rotation", 0.);

    T_world_kf_ = T_kf_ref_ = T_ref_cam_ = T_cur_ref_ =
        Eigen::Affine3f::Identity();

    map_ = PointCloud::Ptr(new PointCloud);
    map_local_ = PointCloud::Ptr(new PointCloud);

    // Load camera calibration
    const std::string camera_name =
        rpg_common_ros::param<std::string>(nh_, "camera_name", "");
    const std::string calib_file =
        rpg_common_ros::param<std::string>(nh_, "calib_file", "");
    c_ = evo_utils::camera::loadPinholeCamera(camera_name, calib_file);
    postCameraLoaded();

    // Setup Publishers
    poses_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "~/pose", rclcpp::QoS(10));
    overlap_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "~/event_map_overlap", rclcpp::QoS(1));

    frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "dvs_frame_id", "dvs_evo");
    world_frame_id_ =
        rpg_common_ros::param<std::string>(nh_, "world_frame_id", "world");
    auto_trigger_ = rpg_common_ros::param<bool>(nhp_, "auto_trigger", false);
}

void Tracker::trackingThread(std::atomic<bool> & running) {
    LOG(INFO) << "Spawned tracking thread.";

    while (running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 100 Hz

        if (!idle_ && keypoints_.size() > 0) {
            estimateTrajectory();
        }
    }
}

void Tracker::onRemoteKey(const std::string & cmd) {
    if (cmd == "switch")
        initialize(EventTime(0));
    else if (cmd == "reset")
        reset();
    else if (cmd == "bootstrap")
        auto_trigger_ = true;
}

void Tracker::initialize(const EventTime & /*ts*/) {
    std::string bootstrap_frame_id = rpg_common_ros::param<std::string>(
        nh_, "dvs_bootstrap_frame_id", std::string("camera_0"));

    geometry_msgs::msg::TransformStamped g;
    try {
        // ros::Time(0) -> latest available transform
        g = tf_buffer_->lookupTransform(bootstrap_frame_id, world_frame_id_,
                                        tf2::TimePointZero);
    } catch (const tf2::TransformException & e) {
        LOG(WARNING) << "Could not initialize tracker: " << e.what();
        return;
    }

    Eigen::Affine3d T_kf_world = tf2ToAffine(msgToTf2(g.transform));

    T_world_kf_ = T_kf_world.cast<float>().inverse();
    T_kf_ref_ = Eigen::Affine3f::Identity();
    T_ref_cam_ = Eigen::Affine3f::Identity();

    const int64_t stamp_ns = rclcpp::Time(g.header.stamp).nanoseconds();
    while (cur_ev_ + 1 < events_.size() &&
           events_[cur_ev_].ts.toNSec() < stamp_ns)
        ++cur_ev_;

    updateMap();

    idle_ = false;
}

void Tracker::reset() {
    idle_ = true;

    events_.clear();
    poses_.clear();
    poses_filtered_.clear();
    cur_ev_ = kf_ev_ = 0;
}

void Tracker::addEvents(const std::vector<Event> & events) {
    static const bool discard_events_when_idle =
        rpg_common_ros::param<bool>(nhp_, "discard_events_when_idle", false);

    std::lock_guard<std::mutex> lock(data_mutex_);
    if (discard_events_when_idle && idle_) return;

    clearEventQueue();
    for (const auto & e : events) events_.push_back(e);
}

void Tracker::setMap(const PointCloud::Ptr & map) {
    static size_t min_map_size =
        rpg_common_ros::param<int>(nhp_, "min_map_size", 0);

    std::lock_guard<std::mutex> lock(data_mutex_);

    *map_ = *map;

    LOG(INFO) << "Received new map: " << map_->size() << " points";

    if (map_->size() > min_map_size && auto_trigger_) {
        LOG(INFO) << "Auto-triggering tracking";

        initialize(EventTime(0));
        auto_trigger_ = false;
    }
}

void Tracker::updateMap() {
    static size_t min_map_size =
        rpg_common_ros::param<int>(nhp_, "min_map_size", 0);
    static size_t min_n_keypoints =
        rpg_common_ros::param<int>(nhp_, "min_n_keypoints", 0);

    if (map_->size() <= min_map_size) {
        LOG(WARNING) << "Unreliable map! Can not update map.";
        return;
    }

    T_kf_ref_ = T_kf_ref_ * T_ref_cam_;
    T_ref_cam_ = Eigen::Affine3f::Identity();
    kf_ev_ = cur_ev_;

    projectMap();

    if (keypoints_.size() < min_n_keypoints) {
        LOG(WARNING) << "Losing track!";
        // TODO: do something about it
    }
}

void Tracker::clearEventQueue() {
    static size_t event_history_size_ = TRACKER_EVENT_HISTORY_SIZE;

    if (idle_) {
        if (events_.size() > event_history_size_) {
            events_.erase(events_.begin(), events_.begin() + events_.size() -
                                               event_history_size_);

            cur_ev_ = kf_ev_ = 0;
        }
    } else {
        events_.erase(events_.begin(), events_.begin() + kf_ev_);

        cur_ev_ -= kf_ev_;
        kf_ev_ = 0;
    }
}

void Tracker::publishMapOverlapThread(std::atomic<bool> & running) {
    const int rate_hz =
        rpg_common_ros::param<int>(nhp_, "event_map_overlap_rate", 25);
    const float z0 = 1. / rpg_common_ros::param<float>(nh_, "max_depth", 10.),
                z1 = 1. / rpg_common_ros::param<float>(nh_, "min_depth", .1),
                z_range = z1 - z0;

    cv::Mat cmap;
    {
        cv::Mat gray(256, 1, CV_8U);
        for (int i = 0; i != gray.rows; ++i) gray.at<uchar>(i) = i;
        cv::applyColorMap(gray, cmap, cv::COLORMAP_RAINBOW);
    }

    cv::Mat ev_img, img;

    const auto period = std::chrono::milliseconds(
        rate_hz > 0 ? static_cast<int>(1000 / rate_hz) : 40);

    while (running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(period);

        if (idle_ || overlap_pub_->get_subscription_count() == 0 ||
            event_rate_ < noise_rate_)
            continue;

        cv::convertScaleAbs(1. - .25 * new_img_, ev_img, 255);
        cv::cvtColor(ev_img, img, cv::COLOR_GRAY2RGB);

        Eigen::Affine3f T_cam_w =
            (T_world_kf_ * T_kf_ref_ * T_ref_cam_).inverse();

        const int s = 2;

        for (const auto & P : map_local_->points) {
            Eigen::Vector3f p = T_cam_w * Eigen::Vector3f(P.x, P.y, P.z);
            p[0] = p[0] / p[2] * fx_ + cx_;
            p[1] = p[1] / p[2] * fy_ + cy_;

            int x = std::round(s * p[0]), y = std::round(s * p[1]);
            float z = p[2];

            if (x < 0 || x >= s * (int)width_ || y < 0 || y >= s * (int)height_)
                continue;

            cv::Vec3b c = cmap.at<cv::Vec3b>(
                std::min(255., std::max(255. * (1. / z - z0) / z_range, 0.)));
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(c[0], c[1], c[2]),
                       -1, cv::LINE_AA, 1);
        }

        std_msgs::msg::Header header;
        header.stamp = rclcpp::Time(events_[cur_ev_].ts.toNSec());

        sensor_msgs::msg::Image::SharedPtr msg =
            cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
        overlap_pub_->publish(*msg);
    }
}

void Tracker::publishTF() {
    Eigen::Affine3f T_world_cam = T_world_kf_ * T_kf_ref_ * T_ref_cam_;
    tf2::Transform pose_tf = affineToTf2(T_world_cam.cast<double>());
    rclcpp::Time stamp(events_[cur_ev_ + frame_size_].ts.toNSec());
    StampedPose new_pose(pose_tf, stamp, world_frame_id_, "dvs_evo_raw");
    poses_.push_back(new_pose);
    tf_pub_->sendTransform(toTransformStamped(new_pose));

    StampedPose filtered_pose;
    if (getFilteredPose(filtered_pose)) {
        filtered_pose.frame_id_ = world_frame_id_;
        filtered_pose.child_frame_id_ = frame_id_;
        tf_pub_->sendTransform(toTransformStamped(filtered_pose));
        // Insert into the shared buffer so the mapper can look up the camera
        // pose by timestamp (replaces the cross-node TF system).
        tf_buffer_->setTransform(toTransformStamped(filtered_pose),
                                 "evo_tracker");
        poses_filtered_.push_back(filtered_pose);

        publishPose();
    }
}

void Tracker::publishPose() {
    const StampedPose & T_world_cam = poses_.back();

    const tf2::Vector3 & p = T_world_cam.getOrigin();
    const tf2::Quaternion & q = T_world_cam.getRotation();
    geometry_msgs::msg::PoseStamped msg_pose;
    msg_pose.header.stamp = T_world_cam.stamp_;
    msg_pose.header.frame_id = frame_id_;
    msg_pose.pose.position.x = p.x();
    msg_pose.pose.position.y = p.y();
    msg_pose.pose.position.z = p.z();
    msg_pose.pose.orientation.x = q.x();
    msg_pose.pose.orientation.y = q.y();
    msg_pose.pose.orientation.z = q.z();
    msg_pose.pose.orientation.w = q.w();
    poses_pub_->publish(msg_pose);
}

bool Tracker::getFilteredPose(StampedPose & pose) {
    static const size_t mean_filter_size =
        rpg_common_ros::param<int>(nhp_, "pose_mean_filter_size", 10);

    if (mean_filter_size < 2) {
        pose = poses_.back();
        return true;
    }

    if (poses_.size() < mean_filter_size) {
        return false;
    }

    static Eigen::VectorXd P(7);
    P.setZero();

    // Take the first rotation q0 as the reference
    // Then, for the remainders rotations qi, instead of
    // averaging directly the qi's, average the incremental rotations
    // q0^-1 * q_i (in the Lie algebra), and then get the original mean
    // rotation by multiplying the mean incremental rotation on the left
    // by q0.

    tf2::Quaternion tf_q0 =
        poses_[poses_.size() - mean_filter_size].getRotation();
    const Quaternion q0(tf_q0.w(), tf_q0.x(), tf_q0.y(), tf_q0.z());
    const Quaternion q0_inv = q0.inverse();

    for (size_t i = poses_.size() - mean_filter_size; i != poses_.size(); ++i) {
        const tf2::Quaternion & tf_q = poses_[i].getRotation();
        const Quaternion q(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());
        const Quaternion q_inc = q0_inv * q;

        const tf2::Vector3 & t = poses_[i].getOrigin();

        Transformation T(q_inc, Eigen::Vector3d(t.x(), t.y(), t.z()));

        P.head<6>() += T.log();
        P[6] += poses_[i].stamp_.seconds();
    }

    P /= mean_filter_size;
    Transformation T(Transformation::Vector6(P.head<6>()));

    const Eigen::Vector3d & t_mean = T.getPosition();
    const Quaternion q_mean = q0 * T.getRotation();

    StampedPose filtered_pose;
    filtered_pose.setOrigin(tf2::Vector3(t_mean[0], t_mean[1], t_mean[2]));
    filtered_pose.setRotation(
        tf2::Quaternion(q_mean.x(), q_mean.y(), q_mean.z(), q_mean.w()));
    filtered_pose.stamp_ = rclcpp::Time(static_cast<int64_t>(P[6] * 1e9));

    pose = filtered_pose;
    return true;
}

void Tracker::estimateTrajectory() {
    static const size_t max_event_rate =
                            rpg_common_ros::param<int>(nhp_, "max_event_rate",
                                                       8000000),
                        events_per_kf =
                            rpg_common_ros::param<int>(nhp_, "events_per_kf",
                                                       100000);

    std::lock_guard<std::mutex> lock(data_mutex_);

    while (true) {
        if (cur_ev_ + std::max(step_size_, frame_size_) > events_.size()) break;

        if (cur_ev_ - kf_ev_ >= events_per_kf) updateMap();

        if (idle_) break;

        size_t frame_end = cur_ev_ + frame_size_;

        // Skip frame if event rate below noise rate
        double frameduration =
            (events_[frame_end].ts - events_[cur_ev_].ts).toSec();
        event_rate_ =
            std::round(static_cast<double>(frame_size_) / frameduration);
        if (event_rate_ < noise_rate_) {
            LOG(WARNING) << "Event rate below NOISE RATE. Skipping frame.";
            cur_ev_ += step_size_;
            continue;
        }
        if (event_rate_ > max_event_rate) {
            LOG(WARNING) << "Event rate above MAX EVENT RATE. Skipping frame.";
            cur_ev_ += step_size_;
            continue;
        }

        drawEvents(events_.begin() + cur_ev_, events_.begin() + frame_end,
                   new_img_);
        cv::buildPyramid(new_img_, pyr_new_, pyramid_levels_);
        trackFrame();

        T_ref_cam_ *= SE3::exp(-x_).matrix();

        publishTF();
        cur_ev_ += step_size_;
    }
}

}  // namespace evo
