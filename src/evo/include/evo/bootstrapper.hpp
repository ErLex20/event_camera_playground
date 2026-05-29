#pragma once

#ifndef EVO_BOOTSTRAPPER_HPP
#define EVO_BOOTSTRAPPER_HPP

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/buffer_core.h>
#include <tf2_ros/transform_broadcaster.h>

#include <image_geometry/pinhole_camera_model.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/core.hpp>

#include <rpg_common_ros/params_helper.hpp>

#include "evo/event_types.hpp"
#include "evo/motion_correction.hpp"

namespace dvs_bootstrapping {

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

/**
 * Base bootstrapper: collects events and reacts to the bootstrap command.
 * (Internalized port of dvs_bootstrapping::Bootstrapper.)
 */
class Bootstrapper {
   public:
    virtual ~Bootstrapper() {}

    /// Push a freshly decoded batch of events (replaces eventCallback).
    void addEvents(const std::vector<evo::Event> &events);

    /// Handle a remote-key command (replaces startCommandCallback).
    void onRemoteKey(const std::string &cmd);

    void pauseBootstrapper();

   protected:
    virtual void postCameraLoaded() {}
    virtual void postBootstrapCalled() {}

    /// Common setup: store node/buffer, load camera and shared params.
    void setupBase(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer);

    rclcpp::Node *node_ = nullptr;
    rpg_common_ros::ParamProvider nh_, nhp_;
    std::shared_ptr<tf2::BufferCore> tf_buffer_;
    image_geometry::PinholeCameraModel cam_;

    bool idle_ = true;

    std::vector<evo::Event> eventQueue_;
    std::mutex data_mutex_;

    std::string world_frame_id_;
    std::string bootstrap_frame_id_;
};

/**
 * Collects events and produces motion-undistorted events frames.
 * (Internalized port of dvs_bootstrapping::EventsFramesBootstrapper.)
 */
class EventsFramesBootstrapper : public Bootstrapper {
   public:
    virtual ~EventsFramesBootstrapper() {}

    /// Bootstrapping integrating thread @ rate_hz, gated by running.
    void integratingThread(std::atomic<bool> &running);

   protected:
    void postCameraLoaded() override;

    size_t width_;
    size_t height_;
    cv::Size sensor_size_;
    std::vector<cv::Point2f> rectified_points_;
    bool keep_frames_ = false;
    std::mutex img_mutex_;
    std::deque<cv::Mat> frames_;
    int rate_hz_;
    evo::EventTime ts_;  ///< last timestamped image (last received event)

    void setupEventsFrames();

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_event_img_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_optical_flow_;

    int frame_size_;
    int local_frame_size_;
    int newest_processed_event_ = 0;
    bool enable_visuals_;
    cv::Mat undistort_mapx_;
    cv::Mat undistort_mapy_;

    motion_correction::WarpUpdateParams unwarp_params_;

    bool integrateEvents();
    void clearEventQueue();
    void publishEventImage(const cv::Mat &img, const evo::EventTime &ts);
    void publishOpticalFlowVectors(const cv::Mat &flow_field);
};

/**
 * Fronto-planar bootstrapper (SVO-free): assumes a fronto-planar scene and
 * integrates a fixed number of events to bootstrap the first map and pose.
 * (Internalized port of dvs_bootstrapping::FrontoPlanarBootstrapper.)
 */
class FrontoPlanarBootstrapper : public EventsFramesBootstrapper {
   public:
    /**
     * Initialize the fronto-planar bootstrapper.
     * @param map_callback invoked with the bootstrap map and its timestamp when
     *        a fronto-planar map is produced (used to feed tracker/mapper).
     */
    void setup(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
               std::function<void(const PointCloud::Ptr &, const rclcpp::Time &)>
                   map_callback = nullptr);

    /// Bootstrapping thread @ rate_hz, gated by running.
    void bootstrappingThread(std::atomic<bool> &running);

   protected:
    void postBootstrapCalled() override;

   private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc_;
    PointCloud::Ptr pcl_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> pub_tf_;
    std::function<void(const PointCloud::Ptr &, const rclcpp::Time &)>
        map_callback_;

    bool start_ = false;

    bool bootstrap();
    bool publishPcl();
};

}  // namespace dvs_bootstrapping

#endif  // EVO_BOOTSTRAPPER_HPP
