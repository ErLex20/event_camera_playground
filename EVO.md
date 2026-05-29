# EVO (ROS2 port) — current state of the codebase

This document describes the **as-built** state of `src/evo`: a single ROS2 (ament/colcon)
DUA node that ports RPG's **EVO** (*Event-based 6-DOF Parallel Tracking and Mapping*,
originally a 4-node ROS1 system in `tools/rpg_dvs_evo_open/`). The algorithm is meant to be
numerically unchanged; only the ROS plumbing (build, message types, node lifecycle,
topics/TF → internal data flow, parameter loading) was rewritten.

> History/plan context and non-obvious decisions also live in the auto-memory
> `evo-port-decisions.md`. This file is the codebase reference; keep it in sync when the
> architecture changes.

## Scope decisions (confirmed with user)

- **Single merged node.** The four original ROS1 nodes (bootstrapping, mapping, tracking,
  reconstruction) are one DUA node; stages communicate by direct calls / shared in-memory
  structures + internal threads, not topics+TF.
- **SVO is dropped** ("ignore SVO, I only want EVO"). The bootstrapper is therefore the
  **fronto-planar** one (`FrontoPlanarBootstrapper : EventsFramesBootstrapper`), the
  SVO-free path. All `svo*`, `svo_vikit/*`, `svo_msgs`, OpenGV, fast, GTSAM, CUDA are skipped.
- **Event input** is the workspace-native `event_camera_msgs/msg/EventPacket`, decoded once
  via `event_camera_codecs`.

## Build

```bash
bash /tmp/evo_build.sh --packages-select evo      # clean-env helper (see note)
# or, with a clean environment already sourced:
colcon build --symlink-install --packages-select evo
source install/setup.zsh
```

Note: the login shell has stale `workspace/install/*` entries on its prefix paths;
`/tmp/evo_build.sh` unsets `AMENT_PREFIX_PATH`/`CMAKE_PREFIX_PATH`/`COLCON_PREFIX_PATH` and
re-sources `/opt/ros/jazzy`, zenoh, and dua-utils before building. Produces
`libevo.so` (composable component `evo::EVO`) and `evo_app` (standalone). Build type
`RelWithDebInfo`, C++17.

## Support packages (under `src/`)

| Package | Role |
|---|---|
| `evo_utils` | `evo_utils::camera::PinholeCamera`, `evo_utils::geometry` (kindr `Transformation`, median filters), `rpg_common_ros::ParamProvider` parameter shim, interpolation helpers. |
| `dvs_slam_msgs` | `VoxelGrid.msg` (ROS2 rosidl). |
| `sophus_vendor` | header-only Sophus (INTERFACE). EVO uses only `T.log()` and `SE3f::exp(v)`. |
| `minkindr` | header-only kindr/minimal (INTERFACE), needs Eigen + glog. |
| `prophesee_evk4_driver`, `v4l2_camera_driver`, `event_camera_*` | event source + codecs/tools. |

## The `evo` node

Mirrors `event_detector_cpp`: `class evo::EVO : public dua_node::NodeBase`, composable
component + `evo_app.cpp` standalone (`ROS2AppManager` + `SignalHandler` +
`MultiThreadedExecutor`). `dua_init_node()` runs `init_parameters()` (DUA codegen from
`src/evo/src/evo/params.yaml`) → `init_cgroups()` → `init_publishers()` → `init_subscribers()`.

### File map

| Path | Contents |
|---|---|
| `include/evo/evo.hpp`, `src/evo/evo.cpp` | `EVO` node: stage ownership, constructor wiring, `feed_map`, `init_*`. |
| `src/evo/subscriptions.cpp` | `event_packet_callback` (decode), `dispatch_events`, `on_remote_key`, `on_tracked_pose`. |
| `src/evo/utils.cpp` | `activate()` / `deactivate()` (spawn/join the 4 worker threads). |
| `include/evo/event_types.hpp` | `Event{x,y,ts,polarity}` + `EventTime` (signed-ns `ros::Time` stand-in: `toSec/toNSec`, +/-, ordering). |
| `include/evo/event_decoder.hpp` | `EventCollector : event_camera_codecs::EventProcessor` → appends decoded CD events. |
| `include/evo/bootstrapper.hpp`, `src/evo/bootstrapper.cpp` | `Bootstrapper` / `EventsFramesBootstrapper` / `FrontoPlanarBootstrapper`. |
| `include/evo/motion_correction.hpp`, `src/evo/motion_correction.cpp` | events-frame warping / optical flow (used by the bootstrapper). |
| `include/evo/tracker.hpp`, `src/evo/tracker.cpp` | `Tracker : LKSE3` — pose tracking, `StampedPose`, TF/pose publishing. |
| `include/evo/lk_se3.hpp`, `src/evo/lk_se3.cpp` | `LKSE3` base: `projectMap`, `precomputeReferenceFrame`, `trackFrame`, `updateTransformation`, Jacobians, pyramid. |
| `include/evo/weight_functions.hpp` | robust weight functions for the LK optimizer. |
| `include/evo/depth_defocus_node.hpp`, `src/evo/depth_defocus_node.cpp` | `DepthFromDefocusNode` — DSI mapping (EMVS). |
| `include/evo/depth_vector.hpp` | `InverseDepthVector` (depth-cell ↔ depth mapping). |
| `include/evo/mosaic.hpp`, `src/evo/mosaic.cpp` | `Mosaic` (EKF gradient + Poisson) + `Reconstruction` driver. |
| `src/evo/laplace.cpp` (`poisson_solver/laplace.h`) | Poisson/SOR solver used by `Mosaic::getReconstructedImage`. |
| `config/evo.yaml` | runtime params (autostart, frame ids, depth range). |
| `config/evo_calibration.yaml` | camera calibration (loaded by every stage via an **absolute path** hardcoded in the stage `setup()`s — see Known issues). |
| `launch/evo.launch.py` | brings up driver + renderer + republisher + `evo::EVO` in `dua_component_container_mt`, remaps `~/events`→`event_camera/events`. |

### Stages (members of `EVO`)

```
std::unique_ptr<dvs_bootstrapping::FrontoPlanarBootstrapper> bootstrapper_;
std::unique_ptr<evo::Tracker>                                tracker_;
std::unique_ptr<depth_from_defocus::DepthFromDefocusNode>    mapper_;
std::unique_ptr<evo::Reconstruction>                         reconstruction_;
```

Each stage is constructed default and initialized with a `setup(node, tf_buffer, …)` call
(no ROS1 node-handle constructor). Their compute logic is lifted from the originals; only
ROS touchpoints changed (`ros::Time`→`EventTime`/`rclcpp::Time`, `tf::`→`tf2::`,
`dvs_msgs::Event`→`evo::Event`).

## Data flow (replaces the original topics + cross-node TF)

```
~/events (EventPacket)
   └─ event_packet_callback: DecoderFactory.getInstance(msg).decode(msg, EventCollector)
        └─ dispatch_events(events):  bootstrapper_->addEvents
                                     mapper_->addEvents
                                     tracker_->addEvents
                                     reconstruction_->addEvents

FrontoPlanarBootstrapper.bootstrap()  (bootstrappingThread)
   ├─ tf_buffer_->setTransform(world → camera_0, identity)   ← initial pose
   └─ bootstrap_map_callback_(pcl, stamp):
        ├─ mapper_->onRemoteKey("bootstrap")   → mapper state IDLE→MAPPING, auto_trigger
        └─ EVO::feed_map(pcl, stamp)           → tracker_->setMap + reconstruction_->setMap

Tracker.publishTF()  (trackingThread, 100 Hz)
   ├─ tf_pub_->sendTransform(world → dvs_evo)        (external viz)
   ├─ tf_buffer_->setTransform(world → dvs_evo)      ← internal, for mapper/mosaic lookup
   └─ poses_pub_->publish(~/pose)
        └─ EVO::on_tracked_pose → mapper_->onTrackedPose(stamp)
             advances newest_tracked_event_; auto-fires the first update()

DepthFromDefocusNode.update()  (driven by onTrackedPose / "update" remote key)
   └─ accumulatePointcloud(): publishes ~/pointcloud AND
        map_callback_(pc_, ts) → EVO::feed_map → tracker_->setMap + reconstruction_->setMap
```

- **`EVO::feed_map`** converts the mapper/bootstrap `PointXYZI` cloud to `PointXYZ`
  (`pcl::copyPointCloud`) and hands it to the tracker (`setMap`) and reconstruction
  (`setMap`). This is the single internal replacement for the original
  `dvs_mapping/pointcloud` topic that both the tracker and reconstruction subscribed to.
- **Internal `tf2::BufferCore tf_buffer_`** (shared `shared_ptr`) replaces cross-node TF:
  bootstrapper inserts `world→camera_0`, tracker inserts the filtered `world→dvs_evo`, and
  the mapper/mosaic query it by event-batch timestamp. TF is still broadcast externally for
  rviz via per-stage `tf2_ros::TransformBroadcaster`s.
- **Remote-key state machine**: `~/remote_key` (`std_msgs/String`) →
  `on_remote_key` → fans `bootstrap`/`switch`/`reset`/`update`/`enable_map_expansion` to the
  stages (each interprets the subset it knows, as in the originals).

## Threading & lifecycle

`activate()` (called on autostart) spawns, gated by `std::atomic<bool> running_`:

| Thread | Body | Rate |
|---|---|---|
| `thread_integrate_` | `bootstrapper_->integratingThread` (motion-corrected events frames) | `boot.rate_hz` |
| `thread_bootstrap_` | `bootstrapper_->bootstrappingThread` (produce fronto-planar map) | `boot.rate_hz` |
| `thread_tracking_` | `tracker_->trackingThread` → `estimateTrajectory` | 100 Hz |
| `thread_overlap_` | `tracker_->publishMapOverlapThread` (debug overlay) | `track.event_map_overlap_rate` |

`deactivate()` clears `running_` and joins all four. The mapper's heavy `update()` runs
inline on the pose-relay / remote-key callback thread (not a dedicated thread).
`init_cgroups()` puts the event + remote-key subscriptions in a **mutually-exclusive**
group (`dua_create_exclusive_cgroup`); `~/pose` uses the default group. Executor is
`MultiThreadedExecutor`.

> **Important — do not make the event group reentrant.** `event_packet_callback` mutates
> shared state (the reused `decoded_events_` scratch buffer and the stateful
> `event_camera_codecs` decoder). A reentrant group lets the executor dispatch the callback
> concurrently with itself, which corrupts that state and aborts with
> `double free or corruption`. The exclusive group serializes decoding, matching the
> original pipeline where each node consumed events from its own single-threaded queue.

### Concurrency guards (the originals were single-threaded under `ros::spin`)

- `Tracker::data_mutex_` — events_/poses_/map_ (kept from the original).
- `DepthFromDefocusNode::data_mutex_` — guards `addEvents` / `onTrackedPose` / `onRemoteKey`
  (added during the port; mapper is now hit by 3 threads).
- `Reconstruction::data_mutex_` — guards `addEvents` / `setMap`.
- Lock order is **mapper → tracker → reconstruction** (feed_map is only called downward), so
  there is no deadlock.

## Parameters

`rpg_common_ros::param<T>(provider, name, default)` (in `evo_utils/.../params_helper.hpp`)
replaces `ros::NodeHandle::param`. A `ParamProvider{rclcpp::Node*, prefix}` carries a
namespace prefix and **declares each param lazily on first read** (so YAML/launch overrides
apply, and any param absent from yaml falls back to its in-code default). Prefixes:

- `nh_  = {node, ""}`   → shared params: `world_frame_id`, `dvs_frame_id`,
  `dvs_bootstrap_frame_id`, `camera_name`, `min_depth`, `max_depth`, `num_depth_cells`,
  `fov_virtual_camera_deg`, `virtual_width`, `virtual_height`.
- stage-private prefixes avoid collisions (both mapping and tracking define e.g.
  `frame_size`): bootstrapper `boot.`, mapper `map.`, tracker `track.`, reconstruction `reco.`.

Frame ids (note: **no** leading `/`, unlike the ROS1 launch): `world`, `dvs_evo`, `camera_0`.

`src/evo/src/evo/params.yaml` now declares the **full tunable schema** (shared + every
stage prefix) via the DUA `params_manager` codegen. Stage params are declared **declare-only**
(no `var_name` → bound to `nullptr`): the params_manager declares them so they appear in the
node schema / `ros2 param`, and the per-stage shim then reads the already-declared value
(its `has_parameter` check prevents a double declaration). Types must match the shim read
(`<int>`→`integer`, `<float>`/`<double>`→`double`, `<bool>`→`bool`); ranges are generous so
config overrides never fail validation. Only `autostart` binds to a node member.

`config/evo.yaml` carries the recommended values (mirroring `dvs_tracking/launch/live.launch`):
`autostart: true`, `boot.auto_trigger`/`track.auto_trigger: true` (so the pipeline starts on
first data), the depth range, and the bootstrapping/mapping/tracking/reconstruction knobs
(`boot.frame_size 10000`, `boot.events_scale_factor 4.0`, `boot.activation_threshold_min 10`,
`map.frame_size 4096`, `map.type_focus_measure 0`, `track.frame_size 5000`,
`track.step_size 15000`, `reco.window_size 5000`, …). Edit `config/evo.yaml` to tune;
edit `params.yaml` to change a default/description/range/constraint.

## Stage notes

- **Bootstrapper** (`bootstrapper.cpp`): integrates events into a motion-corrected frame
  (`integrateEvents` + `motion_correction`), then `FrontoPlanarBootstrapper::bootstrap()`
  back-projects the activation mask onto a plane at `boot.plane_distance` to make the first
  map, inserts the identity `world→camera_0` pose, and fires `bootstrap_map_callback_`.
- **Tracker** (`tracker.cpp` + `lk_se3.cpp`): faithful LK-SE3. Pose chain
  `T_world_kf_ · T_kf_ref_ · T_ref_cam_`; `estimateTrajectory` draws an event frame, builds a
  pyramid, runs `trackFrame`, applies `SE3::exp(-x_)`, publishes filtered pose. `initialize`
  looks up `camera_0→world` from `tf_buffer_` and advances `cur_ev_` to the bootstrap stamp.
- **Mapper** (`depth_defocus_node.cpp`): EMVS DSI. `processEventQueue` warps events into the
  DSI via `getPoseAt(t)` (queries `world→frame_id_` from `tf_buffer_`), `projectEventsToVoxelGrid`
  votes (`voteForCellBilinear`), `synthesizePointCloudFromVoxelGrid{Linf,Contrast,GradMag}`
  extracts depth via the focus measure, adaptive-threshold + Huang median filter, radius
  outlier removal, then publishes/feeds the cloud. `frame_id_` starts at `camera_0` and
  switches to `dvs_evo` after the first auto-triggered `update()`.
- **Reconstruction** (`mosaic.cpp`, optional): `Reconstruction::setMap` triggers
  `Mosaic::compute`, which reprojects the map to a depthmap and runs a per-pixel EKF on the
  image gradient every `reco.window_size` events (`update`, using the tf2 6-arg time-travel
  lookup), then integrates with the Poisson solver (`getReconstructedImage`). Publishes
  `~/reconstruction/image` and `~/reconstruction/depthmap`. Does **not** feed back into
  tracking/mapping (viz only; `required=false` in the original).

## Behavioral-equivalence audit (2026-05-29)

Every numeric/algorithmic file was line-diffed against the ROS1 original and is **verbatim**
except for mechanical changes that do not affect results: type substitutions
(`dvs_msgs::Event`→`evo::Event`, `ros::Time`→`EventTime`, `tf`→`tf2`), `namespace`/include/
`inline`/`#pragma once`, signed-vs-unsigned loop counters, removed comments, and ROS I/O
plumbing (calibration loaded via `camera_calibration_parsers` instead of `camera_info_manager`;
cross-node TF replaced by the shared `tf2::BufferCore`; topics→callbacks). Verified files:
`lk_se3.cpp/.hpp`, `tracker.cpp`, `depth_defocus_node.cpp`, `bootstrapper.cpp`,
`motion_correction.cpp/.hpp`, `weight_functions.hpp`, `depth_vector.hpp`, `laplace.cpp`,
`evo_utils/{camera,utils_geometry}.cpp`. The only code that had to be re-ported was
`Mosaic::compute`/`update` (an earlier edit had simplified it); it is now line-equivalent to
the original EKF + windowing.

## Known issues / open risks (validate with a dataset)

- **Runs without crashing** on `logs/20260528_1_evk4`. That bag is short hand-motion test data
  (not a proper EVO sequence), so the fronto-planar bootstrap produces an empty cloud
  (`Point cloud empty!`) — expected for this input, not a bug.
- **Bootstrap→mapper frame sequencing**: the mapper queries `world→camera_0` until its first
  auto-triggered `update()` switches `frame_id_` to `dvs_evo`. Only one `camera_0` TF exists
  (the bootstrap identity), so `getPoseAt` can fail for that first DSI. In the original SVO
  path `camera_0` had continuous poses; fronto-planar mode does not.
- **Calibration path is hardcoded absolute** (`/home/neo/workspace/src/evo/config/evo_calibration.yaml`)
  in `Bootstrapper::setupBase`, `Tracker::setup`, and `EVO`'s ctor — not portable; should be
  derived from `camera_name`/share dir.
- Reconstruction runs synchronously under the mapper lock (perf, not correctness).

## Running it

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd &          # rmw_zenoh needs a router for discovery
ros2 launch evo evo.launch.py                # driver fails w/o an EVK4 camera (harmless)
ros2 bag play logs/<sequence>                # feed /event_camera/events
```
The launch also starts the metavision driver, which logs `driver initialization failed!` with
no camera attached — harmless; the EVO node still loads and consumes the bag's events.

## Verification (end-to-end)

1. Build per above; `source install/setup.zsh`.
2. Feed an `event_camera_msgs/EventPacket` stream matching `config/evo_calibration.yaml`
   (driver, or republish a bag via `event_camera_tools`).
3. `ros2 launch evo evo.launch.py` (or run `evo_app`). Confirm: bootstrap fires, `~/pose`
   advances, TF `world→dvs_evo` broadcasts, `~/pointcloud` populates, depthmap +
   reconstructed image publish; visualize in rviz2.
4. **Numeric equivalence**: feed an identical event window through one stage (e.g. mapper
   `update()`) and compare the DSI / output cloud against the ROS1 build within float
   tolerance — the core proof the algorithm is unchanged.
