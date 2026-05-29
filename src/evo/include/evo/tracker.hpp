#ifndef EVO_TRACKER_H
#define EVO_TRACKER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <tf2/buffer_core.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>

#include <rpg_common_ros/params_helper.hpp>

#include "evo/event_types.hpp"
#include "evo/lk_se3.hpp"

#define TRACKER_EVENT_HISTORY_SIZE \
    500000  // size of the history of events (already processed) to keep

#define TRACKER_DEBUG_REFERENCE_IMAGE  // define this to start overlap thread

namespace evo {

/**
 * Stamped transform mirroring the subset of the ROS1 tf::StampedTransform API
 * used by the tracker (origin/rotation accessors plus a stamp and frame ids).
 */
struct StampedPose : public tf2::Transform {
  rclcpp::Time stamp_;
  std::string frame_id_;
  std::string child_frame_id_;

  StampedPose() = default;
  StampedPose(const tf2::Transform & t, const rclcpp::Time & stamp,
              const std::string & frame_id, const std::string & child_frame_id)
  : tf2::Transform(t), stamp_(stamp), frame_id_(frame_id), child_frame_id_(child_frame_id) {}
};

/**
 * Tracking stage (internalized). Reproduces the original dvs_tracking node's
 * Tracker : LKSE3 logic, with ROS plumbing replaced by direct in-memory data
 * flow: events are pushed in via addEvents(), the map is supplied via setMap(),
 * cross-node TF is replaced by the shared tf2::BufferCore, and poses are still
 * broadcast/published externally for visualization.
 */
class Tracker : public LKSE3 {
   public:
    Tracker() = default;

    /**
     * Initializes the tracker: loads parameters and camera calibration, sets up
     * publishers and the TF broadcaster. Mirrors the original constructor minus
     * the thread spawning (threads are started by the owning node in activate).
     */
    void setup(rclcpp::Node * node, std::shared_ptr<tf2::BufferCore> tf_buffer);

    /// Push a freshly decoded batch of events (replaces eventCallback).
    void addEvents(const std::vector<Event> & events);

    /// Supply a new map point cloud (replaces mapCallback).
    void setMap(const PointCloud::Ptr & map);

    /// Handle a remote-key command (replaces remoteCallback).
    void onRemoteKey(const std::string & cmd);

    /// 100 Hz tracking loop, gated by the owning node's running flag.
    void trackingThread(std::atomic<bool> & running);

    /// Debug thread overlaying the local map onto the event image.
    void publishMapOverlapThread(std::atomic<bool> & running);

   private:
    rclcpp::Node * node_ = nullptr;
    rpg_common_ros::ParamProvider nh_, nhp_;
    std::shared_ptr<tf2::BufferCore> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_pub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr poses_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlap_pub_;

    std::string frame_id_;
    std::string world_frame_id_;

    bool idle_ = true;  ///< whether the tracking is idle or active

    EventQueue events_;  ///< already processed and to process events

    size_t cur_ev_ = 0, kf_ev_ = 0, noise_rate_ = 0, frame_size_ = 0,
           step_size_ = 0, event_rate_ = 0;

    std::vector<StampedPose> poses_;           // estimated poses
    std::vector<StampedPose> poses_filtered_;  // median filtered poses

    std::mutex data_mutex_;  ///< mutex to own when accessing data

    bool auto_trigger_ = false;

    void initialize(const EventTime & ts);
    void reset();
    void estimateTrajectory();
    void updateMap();
    bool getFilteredPose(StampedPose & pose);
    void publishTF();
    void publishPose();
    void clearEventQueue();
    void postCameraLoaded();
};

}  // namespace evo

#endif  // EVO_TRACKER_H
