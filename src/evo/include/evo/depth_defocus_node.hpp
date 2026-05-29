#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/buffer_core.h>

#include <image_geometry/pinhole_camera_model.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <dvs_slam_msgs/msg/voxel_grid.hpp>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <rpg_common_ros/params_helper.hpp>

#include "evo/depth_vector.hpp"
#include "evo/event_types.hpp"
#include "evo_utils/camera.hpp"
#include "evo_utils/geometry.hpp"

namespace depth_from_defocus {

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef float Vote;

/**
 * State of the mapping algorithm (see remoteKeyCallback / onRemoteKey).
 */
enum MapperState { IDLE, MAPPING };

/**
 * Mapping stage (internalized). Reproduces the original dvs_mapping
 * DepthFromDefocusNode: event-based multi-view stereo voting into a DSI voxel
 * grid and depth extraction via a focus measure. ROS plumbing is replaced by
 * node injection, the shared tf2::BufferCore for time-indexed pose lookup, and
 * direct event delivery.
 */
class DepthFromDefocusNode {
   public:
    DepthFromDefocusNode() = default;

    /**
     * Initializes the mapper: stores the node and shared TF buffer, loads
     * parameters and camera calibration, sets up the voxel grid and publishers.
     * Mirrors the original constructor.
     */
    void setup(rclcpp::Node *node, std::shared_ptr<tf2::BufferCore> tf_buffer,
               const image_geometry::PinholeCameraModel &cam);

    /// Push a freshly decoded batch of events (replaces processEventArray).
    void addEvents(const std::vector<evo::Event> &events);

    /**
     * Notifies the mapper that a pose is now known up to time `stamp`
     * (replaces the bookkeeping side of the original tfCallback). Advances
     * newest_tracked_event_ and possibly auto-triggers the first map update.
     */
    void onTrackedPose(const rclcpp::Time &stamp);

    /// Handle a remote-key command (replaces remoteKeyCallback).
    void onRemoteKey(const std::string &cmd);

    /// Switch the queried frame between bootstrap and regular (copilotCallback).
    void setCopilot(bool regular);

    /**
     * Register a callback invoked whenever a new local map is produced
     * (replaces the original dvs_mapping/pointcloud topic that fed the tracker
     * and the reconstruction node). Called with the freshly accumulated cloud
     * and its timestamp.
     */
    void setMapCallback(
        std::function<void(const PointCloud::Ptr &, const evo::EventTime &)> cb) {
        map_callback_ = std::move(cb);
    }

    /**
     * Performs a map update (the heavy DSI computation). Called by the
     * map-expansion monitor thread / on "update" command.
     */
    void update();

    /// Local map produced by the last update (world frame, PointXYZI).
    PointCloud::Ptr localMap() const { return pc_; }

    MapperState state() const { return state_; }

   private:
    void resetMapper();
    void projectEventsToVoxelGrid(const std::vector<Eigen::Vector4f> &events,
                                  const std::vector<Eigen::Vector3f> &centers);
    void resetVoxelGrid();
    void processEventQueue(bool reset = false);
    bool getPoseAt(const evo::EventTime &t,
                   evo_utils::geometry::Transformation &T);
    void setupVoxelGrid();

    void voteForCell(const float x_f, const float y_f, Vote *grid) {
        const int x = (int)(x_f + 0.5), y = (int)(y_f + 0.5);

        if (x >= 0 && x < virtual_width_ && y >= 0 && y < virtual_height_) {
            grid[x + y * virtual_width_] += 1.f;
        }
    }

    void voteForCellBilinear(const float x_f, const float y_f, Vote *grid) {
        if (x_f >= 0.f && y_f >= 0.f) {
            const int x = x_f, y = y_f;
            if (x + 1 < virtual_width_ && y + 1 < virtual_height_) {
                Vote *g = grid + x + y * virtual_width_;
                const float fx = x_f - x, fy = y_f - y, fx1 = 1.f - fx,
                            fy1 = 1.f - fy;

                g[0] += fx1 * fy1;
                g[1] += fx * fy1;
                g[virtual_width_] += fx1 * fy;
                g[virtual_width_ + 1] += fx * fy;
            }
        }
    }

    void precomputeRectifiedPoints();
    void synthesizeAndPublishDepth();
    void synthesizePointCloudFromVoxelGrid(cv::Mat &depth, cv::Mat &confidence);
    void synthesizePointCloudFromVoxelGridLinf(cv::Mat &depth_cell_indices,
                                               cv::Mat &confidence);
    void synthesizePointCloudFromVoxelGridContrast(cv::Mat &depth_cell_indices,
                                                   cv::Mat &confidence);
    void synthesizePointCloudFromVoxelGridGradMag(cv::Mat &depth_cell_indices,
                                                  cv::Mat &confidence);
    void accumulatePointcloud(const cv::Mat &depth, const cv::Mat &mask,
                              const cv::Mat &confidence,
                              const Eigen::Matrix3d R_world_ref,
                              const Eigen::Vector3d t_world_ref,
                              const evo::EventTime &timestamp);
    void publishVoxelGrid(
        const evo_utils::geometry::Transformation &T_w_ref) const;
    void publishDepthmap(const cv::Mat &depth, const cv::Mat &mask);
    void publishGlobalMap();
    void clearEventQueue();
    void postCameraLoaded();

    inline const Vote &voxelGridAt(const std::vector<Vote> &grid, int x, int y,
                                   size_t z) const {
        return grid[x + virtual_width_ * (y + z * virtual_height_)];
    }

    inline Vote &voxelGridAt(std::vector<Vote> &grid, int x, int y, size_t z) {
        return grid[x + virtual_width_ * (y + z * virtual_height_)];
    }

    /** ROS */
    rclcpp::Node *node_ = nullptr;
    rpg_common_ros::ParamProvider nh_, pnh_;
    std::shared_ptr<tf2::BufferCore> tf_buffer_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc_global_;
    rclcpp::Publisher<dvs_slam_msgs::msg::VoxelGrid>::SharedPtr pub_voxel_grid_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_map_;

    MapperState state_ = IDLE;

    std::string world_frame_id_;
    std::string frame_id_;
    std::string regular_frame_id_;
    std::string bootstrap_frame_id_;

    evo_utils::geometry::Transformation T_ref_w_;  ///< camera pose
    image_geometry::PinholeCameraModel dvs_cam_;
    int width_;
    int height_;

    evo_utils::camera::PinholeCamera virtual_cam_;
    int virtual_width_;
    int virtual_height_;

    Eigen::Matrix3f K_;

    std::vector<Vote> ref_voxel_grid_;
    cv::Mat confidence_mask_;
    float voxel_filter_leaf_size_;
    size_t events_to_recreate_kf_;
    int type_focus_measure_;

    Eigen::Matrix2Xf precomputed_rectified_points_;

    size_t current_event_;
    size_t newest_tracked_event_;
    size_t last_kf_update_event_;
    std::deque<evo::Event> event_queue_;

    PointCloud::Ptr pc_;
    PointCloud::Ptr pc_global_;

    evo_utils::geometry::Depth min_depth_;
    evo_utils::geometry::Depth max_depth_;
    size_t num_depth_cells_;
    InverseDepthVector depths_vec_;
    std::vector<float> raw_depths_vec_;

    float radius_search_;
    int min_num_neighbors_;
    int median_filter_size_;

    int adaptive_threshold_kernel_size_;
    int adaptive_threshold_c_;

    bool auto_trigger_;

    /// Serializes the mapper's public entry points. The original mapper ran
    /// single-threaded under ros::spin; in the merged node addEvents(),
    /// onTrackedPose() and onRemoteKey() are invoked from different threads.
    std::mutex data_mutex_;

    /// Invoked with the local map whenever one is produced (see setMapCallback).
    std::function<void(const PointCloud::Ptr &, const evo::EventTime &)>
        map_callback_;
};

}  // namespace depth_from_defocus
