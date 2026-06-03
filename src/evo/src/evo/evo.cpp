/**
 * EVO node implementation.
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

#include "evo/evo.hpp"

#include <pcl/common/io.h>

namespace evo
{

void EVO::feed_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr & map,
                   const rclcpp::Time & stamp)
{
  // The tracker and the reconstruction operate on PointXYZ; the mapper and the
  // bootstrapper produce PointXYZI. copyPointCloud transfers the common x/y/z
  // fields (PointXYZI and PointXYZ have different memory layouts, so a raw cast
  // is invalid).
  auto map_xyz = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::copyPointCloud(*map, *map_xyz);
  map_xyz->header = map->header;

  tracker_->setMap(map_xyz);
  reconstruction_->setMap(map_xyz, evo::EventTime(stamp.nanoseconds()));
}

EVO::EVO(const rclcpp::NodeOptions & node_options)
: NodeBase("evo", node_options, true)
{
  dua_init_node();

  // Internal pose buffer that replaces the cross-node TF system of the
  // original four-node pipeline.
  tf_buffer_ = std::make_shared<tf2::BufferCore>();

  // Load parameters shared by several stages (mirror of the original ROS1
  // global node handle parameters).
  auto sp = shared_params();
  world_frame_id_ = rpg_common_ros::param<std::string>(sp, "world_frame_id", "world");
  dvs_frame_id_ = rpg_common_ros::param<std::string>(sp, "dvs_frame_id", "dvs_evo");
  bootstrap_frame_id_ =
    rpg_common_ros::param<std::string>(sp, "dvs_bootstrap_frame_id", "camera_0");
  camera_name_ = rpg_common_ros::param<std::string>(sp, "camera_name", "DAVIS-ijrr");
  calib_file_ = "/home/neo/workspace/src/evo/config/evo_calibration.yaml";
  min_depth_ = rpg_common_ros::param<double>(sp, "min_depth", 0.4);
  max_depth_ = rpg_common_ros::param<double>(sp, "max_depth", 5.0);
  num_depth_cells_ = rpg_common_ros::param<int>(sp, "num_depth_cells", 100);
  fov_virtual_camera_deg_ =
    rpg_common_ros::param<double>(sp, "fov_virtual_camera_deg", 80.0);
  virtual_width_ = rpg_common_ros::param<int>(sp, "virtual_width", 240);
  virtual_height_ = rpg_common_ros::param<int>(sp, "virtual_height", 180);

  RCLCPP_INFO(this->get_logger(), "Node initialized");

  // ── Pipeline stage creation ──────────────────────────────────────────────
  bootstrapper_ =
      std::make_unique<dvs_bootstrapping::FrontoPlanarBootstrapper>();
  tracker_ = std::make_unique<evo::Tracker>();
  mapper_ = std::make_unique<depth_from_defocus::DepthFromDefocusNode>();
  reconstruction_ = std::make_unique<evo::Reconstruction>();

  // Bootstrap map callback: feed the produced map into tracker, reconstruction
  // and switch the mapper into MAPPING mode.
  bootstrap_map_callback_ = [this](
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &map,
      const rclcpp::Time &stamp) {
    RCLCPP_INFO(this->get_logger(), "Bootstrapper produced a map (%zu pts)",
                map->size());

    // Switch the mapper into MAPPING mode (replaces the "bootstrap" remote key
    // the mapper received on the original /evo/remote_key topic).
    mapper_->onRemoteKey("bootstrap");

    // Feed the bootstrap map to the tracker and the reconstruction stage.
    feed_map(map, stamp);
  };

  // Mapper map callback: every time the mapper produces a new local map, feed
  // it to the tracker and the reconstruction (replaces the original
  // dvs_mapping/pointcloud topic that both subscribed to).
  mapper_->setMapCallback(
      [this](const pcl::PointCloud<pcl::PointXYZI>::Ptr &map,
             const evo::EventTime &stamp) {
        feed_map(map, rclcpp::Time(stamp.toNSec()));
      });

  // Wire the callback into the bootstrapper so it can invoke it.
  {
    std::lock_guard<std::mutex> lock(bootstrap_mutex_);
    bootstrapper_->setup(this, tf_buffer_, bootstrap_map_callback_);
  }

  // Tracker: receives events from dispatch_events(), map from bootstrapper.
  tracker_->setup(this, tf_buffer_);

  // Mapper: receives events and TF poses (via tf_buffer_).
  mapper_->setup(this, tf_buffer_, bootstrapper_->getCamModel());

  // Reconstruction: optional Poisson mosaic.
  reconstruction_->setup(this, tf_buffer_, bootstrapper_->getCamModel());

  // ── Remote key subscription (replaces the original ROS1 /evo/remote_key) ─
  rclcpp::SubscriptionOptions remote_opts;
  remote_opts.callback_group = cgroup_event_packet_;
  sub_remote_key_ = this->create_subscription<std_msgs::msg::String>(
      "~/remote_key", 10,
      std::bind(&EVO::on_remote_key, this, std::placeholders::_1),
      remote_opts);

  // ── Pose relay subscription (mapper needs tracked pose timestamps) ──────
  sub_pose_relay_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/pose", 10,
      std::bind(&EVO::on_tracked_pose, this, std::placeholders::_1));

  if (autostart_) {
    activate();
  }
}

EVO::~EVO()
{
  deactivate();
}

void EVO::init_cgroups()
{
  // Event-packet decoding mutates shared state (the reused decoded_events_
  // buffer and the stateful event_camera_codecs decoder), so the callback must
  // NOT run concurrently with itself. A mutually-exclusive group serializes
  // event decoding, mirroring the original pipeline where each node consumed
  // the event stream from its own single-threaded callback queue. The heavy
  // per-stage computation runs on dedicated worker threads, not here.
  cgroup_event_packet_ = dua_create_exclusive_cgroup();
}

void EVO::init_publishers()
{}

void EVO::init_subscribers()
{
  rclcpp::SubscriptionOptions opts;
  opts.callback_group = cgroup_event_packet_;

  // Event-stream QoS. For a live event camera BestEffort is correct (drop
  // stale data rather than build latency). For OFFLINE replay (e.g. the
  // dsec_publisher) the producer outruns EVO's compute, and BestEffort would
  // silently drop most packets -> the tracker never gets a usable stream and
  // no pose is produced. Setting `event_sub_reliable: true` (with a deep
  // queue) makes delivery lossless: EVO buffers and processes every event at
  // its own pace (slower than wall-clock, but complete).
  const bool reliable =
    this->has_parameter("event_sub_reliable")
      ? this->get_parameter("event_sub_reliable").as_bool()
      : this->declare_parameter<bool>("event_sub_reliable", false);
  const int depth =
    this->has_parameter("event_sub_depth")
      ? static_cast<int>(this->get_parameter("event_sub_depth").as_int())
      : this->declare_parameter<int>("event_sub_depth", 5000);

  const rclcpp::QoS qos = reliable
    ? dua_qos::Reliable::get_datum_qos(static_cast<unsigned int>(depth))
    : dua_qos::BestEffort::get_datum_qos();

  RCLCPP_INFO(
    this->get_logger(), "Event subscription QoS: %s (depth %d)",
    reliable ? "RELIABLE" : "BEST_EFFORT", depth);

  sub_event_packet_ = this->create_subscription<EventPacket>(
    "~/events",
    qos,
    std::bind(&EVO::event_packet_callback, this, std::placeholders::_1),
    opts);
}

} // namespace evo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(evo::EVO)
