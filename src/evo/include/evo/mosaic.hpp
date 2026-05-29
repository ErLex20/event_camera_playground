#ifndef GRADIENT_ESTIMATION_H
#define GRADIENT_ESTIMATION_H

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <tf2/buffer_core.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <image_geometry/pinhole_camera_model.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include <rpg_common_ros/params_helper.hpp>

#include "evo/event_types.hpp"

namespace evo {

/**
 * Reconstructs an image from events using a Poisson equation solver
 * (internalized port of dvs_reconstruction::Mosaic). EKF gradient estimation +
 * Poisson integration. tf is replaced by the shared tf2::BufferCore.
 */
class Mosaic {
   public:
    typedef cv::Vec2f Velocity;
    typedef cv::Vec2f Gradient;
    typedef cv::Matx<float, 2, 2> Covariance;
    typedef cv::Matx<float, 1, 1> Scalar;
    typedef double Time;

    Mosaic() : R_(0.f) {}

    /**
     * Initialize the reconstructor (replaces the original constructor).
     */
    void setup(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
               float sigma_m, float init_cov, int window_size, int map_blur);

    void setCamModel(image_geometry::PinholeCameraModel &c);

    void compute(const std::vector<evo::Event> &evs,
                 const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &map);

    cv::Mat getAbsoluteGradient();
    cv::Mat getReconstructedImage();

    void reprojectDepthmap(const tf2::Transform &T, cv::Mat &_depthmap,
                           bool virtual_cam = false);

    cv::Mat_<Gradient> grad;
    cv::Mat_<Covariance> cov;
    cv::Mat depthmap;
    cv::Mat image;

   protected:
    rclcpp::Node *node_ = nullptr;
    rpg_common_ros::ParamProvider nh_, nhp_;
    std::shared_ptr<tf2::BufferCore> tf_buffer_;

    Scalar R_;
    float init_cov_;
    size_t window_size_;
    size_t map_blur_;
    float depth_median_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_;

    evo::EventTime ts_ref_;

    std::string frame_id_;
    std::string world_frame_id_;

    std::vector<cv::Point2d> rectification_map_;

    image_geometry::PinholeCameraModel c_;
    size_t w_;
    size_t h_;
    tf2::Matrix3x3 K_;
    tf2::Matrix3x3 KInv_;

    size_t w_virt_;
    size_t h_virt_;
    tf2::Matrix3x3 K_virt_;
    tf2::Matrix3x3 KInv_virt_;

    std::vector<tf2::Vector3> map_;

    void update(const cv::Mat &new_grad, const evo::EventTime &t0,
                const evo::EventTime &t1);
    void precomputeRectificationMap();
    void publishImageReconstruction();
    void publishDepthmap(const cv::Mat &depth);
};

/**
 * Thin driver around Mosaic (internalized port of the dvs_reconstruction node):
 * buffers events and triggers a reconstruction on each new map.
 */
class Reconstruction {
   public:
    void setup(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
               const image_geometry::PinholeCameraModel &cam);

    /// Push events (replaces eventCallback).
    void addEvents(const std::vector<evo::Event> &events);

    /// Supply a new map and run reconstruction (replaces mapCallback).
    void setMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &map,
                const evo::EventTime &stamp);

   private:
    void doReconstruction();

    rclcpp::Node *node_ = nullptr;
    rpg_common_ros::ParamProvider nh_, nhp_;
    std::shared_ptr<tf2::BufferCore> tf_buffer_;

    std::string frame_id_;
    std::string world_frame_id_;

    std::deque<evo::Event> events_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_;
    evo::EventTime map_stamp_;

    Mosaic mosaic_;
    image_geometry::PinholeCameraModel c_;
    size_t cur_ev_ = 0;

    /// Serializes addEvents() (event thread) and setMap() (mapper/bootstrap
    /// threads); the original reconstruction node ran single-threaded.
    std::mutex data_mutex_;
};

}  // namespace evo

#endif  // GRADIENT_ESTIMATION_H
