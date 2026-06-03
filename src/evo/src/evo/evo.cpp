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

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <glog/logging.h>

#include <chrono>
#include <thread>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <pcl/common/io.h>

namespace evo
{

void EVO::feed_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr & map,
                   const rclcpp::Time & stamp, bool from_mapper)
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

  // Arm the map-expansion monitor with the latest map the tracker is using.
  // Only the mapper's keyframe maps drive expansion (the original
  // trigger_map_expansion.py subscribed to dvs_mapping/pointcloud, not the
  // bootstrap map).
  if (from_mapper && me_enabled_) {
    note_expansion_map(map_xyz);
  }
}

void EVO::note_expansion_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr & map_world)
{
  // Record the camera pose (world-origin-in-camera translation) at the instant
  // this map was produced, so baseline-over-depth can be measured against it.
  Eigen::Vector3d t_map(0.0, 0.0, 0.0);
  bool have = false;
  try {
    const auto tf =
      tf_buffer_->lookupTransform(dvs_frame_id_, world_frame_id_, tf2::TimePointZero);
    t_map = {tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z};
    have = true;
  } catch (const tf2::TransformException &) {
  }

  std::lock_guard<std::mutex> lk(me_mutex_);
  me_map_ = map_world;  // shared ownership; the tracker keeps its own copy
  me_t_map_ = t_map;
  me_have_t_map_ = have;
  ++me_maps_seen_;
  me_state_ = MapExpansionState::CHECKING;
}

void EVO::check_map_expansion()
{
  // Snapshot the shared state, then release the lock before any heavy work or
  // any call back into the mapper (which takes its own lock) to avoid an
  // AB-BA deadlock with note_expansion_map().
  pcl::PointCloud<pcl::PointXYZ>::Ptr map;
  Eigen::Vector3d t_map;
  bool have_t_map;
  int seen;
  MapExpansionState st;
  {
    std::lock_guard<std::mutex> lk(me_mutex_);
    st = me_state_;
    map = me_map_;
    t_map = me_t_map_;
    have_t_map = me_have_t_map_;
    seen = me_maps_seen_;
  }

  if (st != MapExpansionState::CHECKING) return;
  if (!map || map->empty()) return;
  if (seen <= me_skip_first_) return;

  // Latest tracked pose: this transform maps world points into the camera frame.
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(dvs_frame_id_, world_frame_id_, tf2::TimePointZero);
  } catch (const tf2::TransformException &) {
    return;
  }

  const Eigen::Quaterniond q(tf.transform.rotation.w, tf.transform.rotation.x,
                             tf.transform.rotation.y, tf.transform.rotation.z);
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const Eigen::Vector3d t(tf.transform.translation.x, tf.transform.translation.y,
                          tf.transform.translation.z);

  const double fx = me_K_(0, 0), fy = me_K_(1, 1), cx = me_K_(0, 2),
               cy = me_K_(1, 2);

  cv::Mat mask = cv::Mat::zeros(me_img_h_, me_img_w_, CV_8U);
  size_t N = 0, in_bounds = 0;
  double depth_sum = 0.0;

  for (const auto & P : map->points) {
    const Eigen::Vector3d pc = R * Eigen::Vector3d(P.x, P.y, P.z) + t;
    if (pc.z() <= 0.0) continue;
    ++N;
    depth_sum += pc.norm();
    const double u = fx * pc.x() / pc.z() + cx;
    const double v = fy * pc.y() / pc.z() + cy;
    cv::circle(mask, cv::Point(cvRound(u), cvRound(v)), 7, 255, -1);
    if (u >= 0.0 && v >= 0.0 && u < me_img_w_ && v < me_img_h_) ++in_bounds;
  }

  if (N == 0) return;

  const double coverage = static_cast<double>(cv::countNonZero(mask)) /
                          static_cast<double>(me_img_w_ * me_img_h_);
  const double visibility =
    static_cast<double>(in_bounds) / static_cast<double>(N);
  const double avg_depth = depth_sum / static_cast<double>(N);
  const double bod = have_t_map ? (t - t_map).norm() / avg_depth : 0.0;

  if (coverage < me_coverage_th_ || visibility < me_visibility_th_ ||
      bod > me_baseline_th_) {
    // Set WAIT_FOR_MAP before triggering: the update is synchronous and will
    // produce a fresh map -> note_expansion_map() flips us back to CHECKING.
    {
      std::lock_guard<std::mutex> lk(me_mutex_);
      if (me_state_ == MapExpansionState::CHECKING)
        me_state_ = MapExpansionState::WAIT_FOR_MAP;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Map expansion: coverage=%.1f%% visibility=%.1f%% baseline/depth=%.3f -> update",
      coverage * 100.0, visibility * 100.0, bod);
    mapper_->onRemoteKey("update");
  }
}

void EVO::map_expansion_thread(std::atomic<bool> & running)
{
  const int rate = me_rate_hz_ > 0 ? me_rate_hz_ : 3;
  const auto period = std::chrono::milliseconds(1000 / rate);
  LOG(INFO) << "Spawned map-expansion thread.";
  while (running.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(period);
    check_map_expansion();
  }
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
  // Camera calibration file, settable from config/evo.yaml via the `calib_file`
  // parameter. An empty value (the default) resolves to the calibration
  // installed in the package share directory (config/evo_calibration.yaml).
  // Declared here so the tracker/bootstrapper stages just read the shared value;
  // we write the resolved path back so those stages see the absolute path.
  calib_file_ = rpg_common_ros::param<std::string>(sp, "calib_file", "");
  if (calib_file_.empty()) {
    calib_file_ = ament_index_cpp::get_package_share_directory("evo") +
                  "/config/evo_calibration.yaml";
    this->set_parameter(rclcpp::Parameter("calib_file", calib_file_));
  }
  RCLCPP_INFO(this->get_logger(), "Using camera calibration: %s",
              calib_file_.c_str());
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
        feed_map(map, rclcpp::Time(stamp.toNSec()), /*from_mapper=*/true);
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

  // ── Map-expansion monitor config + real-camera intrinsics ───────────────
  {
    auto ep = stage_params("expand");
    me_enabled_ = rpg_common_ros::param<bool>(ep, "enabled", true);
    me_rate_hz_ = rpg_common_ros::param<int>(ep, "rate", 3);
    me_visibility_th_ =
      rpg_common_ros::param<double>(ep, "visibility_threshold", 0.9);
    me_coverage_th_ =
      rpg_common_ros::param<double>(ep, "coverage_threshold", 0.4);
    me_baseline_th_ =
      rpg_common_ros::param<double>(ep, "baseline_threshold", 0.1);
    me_skip_first_ =
      rpg_common_ros::param<int>(ep, "number_of_initial_maps_to_skip", 0);

    const auto & cam = bootstrapper_->getCamModel();
    me_K_ = cam.fullIntrinsicMatrix();
    const cv::Size res = cam.fullResolution();
    me_img_w_ = res.width;
    me_img_h_ = res.height;
  }

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
