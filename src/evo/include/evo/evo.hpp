/**
 * EVO node definition.
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

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <dua_common_interfaces/msg/command_result_stamped.hpp>

#include <std_msgs/msg/string.hpp>

#include <event_camera_msgs/msg/event_packet.hpp>
#include <event_camera_codecs/decoder.h>
#include <event_camera_codecs/decoder_factory.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/buffer_core.h>

#include <rpg_common_ros/params_helper.hpp>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "evo/bootstrapper.hpp"
#include "evo/depth_defocus_node.hpp"
#include "evo/event_decoder.hpp"
#include "evo/event_types.hpp"
#include "evo/mosaic.hpp"
#include "evo/tracker.hpp"

namespace evo
{

using EventPacket = event_camera_msgs::msg::EventPacket;

class EVO : public dua_node::NodeBase
{
public:
  /**
   * @brief Constructor.
   */
  EVO(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());

  /**
   * @brief Destructor.
   */
  ~EVO();

private:
  /* Init functions. */
  void init_parameters() override;
  void init_cgroups() override;
  void init_publishers() override;
  void init_subscribers() override;

  /**
   * @brief Activates the node (spawns worker threads).
   */
  void activate();

  /**
   * @brief Deactivates the node (joins worker threads).
   */
  void deactivate();

  /* Parameter helpers. */
  /// Shared (global) parameter namespace, mirroring the original ROS1 nh_.
  rpg_common_ros::ParamProvider shared_params() { return {this, ""}; }
  /// Stage-private parameter namespace, mirroring the original ROS1 nhp_.
  rpg_common_ros::ParamProvider stage_params(const std::string & stage)
  {
    return {this, stage + "."};
  }

  /* Event input. */
  /**
   * @brief Decodes an incoming event packet and fans the events out to the
   *        per-stage buffers.
   */
  void event_packet_callback(const EventPacket::ConstSharedPtr msg);

  /**
   * @brief Dispatches a freshly decoded batch of events to every active stage.
   */
  void dispatch_events(const std::vector<Event> & events);

  /**
   * @brief Handles remote-key commands for all stages.
   */
  void on_remote_key(const std_msgs::msg::String::ConstSharedPtr msg);

  /**
   * @brief Relays tracked poses to the mapper so it can advance its event
   *        cursor and auto-trigger map updates.
   */
  void on_tracked_pose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg);

  /**
   * @brief Converts a mapper/bootstrap map (PointXYZI) to PointXYZ and feeds it
   *        to the tracker and the reconstruction stage. This replaces the
   *        original dvs_mapping/pointcloud topic that the tracker and the
   *        reconstruction node subscribed to.
   */
  void feed_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr & map,
                const rclcpp::Time & stamp, bool from_mapper = false);

  /* Map expansion (internal port of the original trigger_map_expansion.py).
   * As the camera moves, the single keyframe map loses coverage/visibility; this
   * stage monitors the current map against the latest tracked pose and asks the
   * mapper to build a fresh keyframe map (onRemoteKey("update")) when it degrades.
   * Without it the camera eventually leaves the mapped region and tracking
   * diverges. */
  void map_expansion_thread(std::atomic<bool> & running);
  void check_map_expansion();
  void note_expansion_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr & map_world);

  enum class MapExpansionState { DISABLED, WAIT_FOR_MAP, CHECKING };

  /* Callback Groups. */
  rclcpp::CallbackGroup::SharedPtr cgroup_event_packet_;

  /* Publishers. */

  /* Subscribers. */
  rclcpp::Subscription<EventPacket>::SharedPtr sub_event_packet_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_remote_key_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_relay_;

  /* Event decoding. */
  event_camera_codecs::DecoderFactory<EventPacket, EventCollector> decoder_factory_;
  std::vector<Event> decoded_events_;  ///< scratch buffer reused per packet

  /* Internal pose buffer (replaces cross-node TF). The tracking stage inserts
   * stamped transforms here; the mapper queries it by timestamp. */
  std::shared_ptr<tf2::BufferCore> tf_buffer_;

  /* Pipeline stages. */
  std::unique_ptr<dvs_bootstrapping::FrontoPlanarBootstrapper> bootstrapper_;
  std::unique_ptr<evo::Tracker> tracker_;
  std::unique_ptr<depth_from_defocus::DepthFromDefocusNode> mapper_;
  std::unique_ptr<evo::Reconstruction> reconstruction_;

  /* Stage thread handles. */
  std::thread thread_integrate_;   ///< bootstrapper integrating thread
  std::thread thread_bootstrap_;   ///< fronto-planar bootstrapping thread
  std::thread thread_tracking_;    ///< 100 Hz tracking thread
  std::thread thread_overlap_;     ///< debug map-overlay thread
  std::thread thread_map_expansion_;  ///< map-expansion monitor thread

  /* Map-expansion state (guarded by me_mutex_). */
  std::mutex me_mutex_;
  MapExpansionState me_state_{MapExpansionState::WAIT_FOR_MAP};
  pcl::PointCloud<pcl::PointXYZ>::Ptr me_map_;  ///< last map fed to the tracker (world frame)
  Eigen::Vector3d me_t_map_{0.0, 0.0, 0.0};     ///< world-origin-in-camera translation when me_map_ arrived
  bool me_have_t_map_{false};
  int me_maps_seen_{0};

  /* Map-expansion configuration. */
  bool me_enabled_{true};
  int me_rate_hz_{3};
  double me_visibility_th_{0.9};
  double me_coverage_th_{0.4};
  double me_baseline_th_{0.1};
  int me_skip_first_{0};

  /* Real-camera intrinsics used to project the map for the heuristics. */
  cv::Matx33d me_K_;
  int me_img_w_{0};
  int me_img_h_{0};

  /* Callbacks and synchronization. */
  std::mutex stage_mutex_;         ///< protects pose_msg_ relay to mapper
  std::mutex bootstrap_mutex_;     ///< protects bootstrap callback setup
  std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> pose_pub_;
  std::atomic<bool> pose_available_{false};

  /* Callbacks for inter-stage communication. */
  /// Bootstrap map → tracker + mapper
  std::function<void(const pcl::PointCloud<pcl::PointXYZI>::Ptr &, const rclcpp::Time &)>
      bootstrap_map_callback_;
  /// Tracked pose → mapper (relay)
  std::mutex pose_mutex_;
  std::shared_ptr<geometry_msgs::msg::PoseStamped> pose_msg_;

  /* Shared parameters (read by several stages). */
  std::string world_frame_id_;
  std::string dvs_frame_id_;
  std::string bootstrap_frame_id_;
  std::string camera_name_;
  std::string calib_file_;
  double min_depth_{0.0};
  double max_depth_{0.0};
  int num_depth_cells_{0};
  double fov_virtual_camera_deg_{0.0};
  int virtual_width_{0};
  int virtual_height_{0};

  /* Node parameters (DUA codegen). */
  bool autostart_;

  /* Threads. */
  std::thread thread_worker_;

  /* Synchronization primitives. */
  std::mutex data_mutex_;
  std::atomic<bool> running_{false};
};

} // namespace evo
