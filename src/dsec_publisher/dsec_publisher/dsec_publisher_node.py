"""
DSEC event replay node.

Reads a DSEC ``events.h5`` recording, slices it into fixed time windows
(``dsec_preprocess.py`` style, see :mod:`dsec_publisher.event_slicer`) and
publishes each window as an ``event_camera_msgs/msg/EventPacket`` using the
``mono`` codec so that the EVO pipeline (``src/evo``) can consume it.

The ``mono`` byte layout (8 bytes/event, little-endian uint64) is::

    (polarity << 63) | (y << 48) | (x << 32) | dt_ns

where ``dt_ns`` is the event time relative to ``msg.time_base``. The matching
decoder reconstructs ``sensor_time = time_base + dt_ns``. This is produced here
directly with NumPy bit-packing (no C++ encoder needed).
"""

import os
import re
from array import array
from pathlib import Path

import h5py
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from event_camera_msgs.msg import EventPacket
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster

from dsec_publisher.event_slicer import EventSlicer
from dsec_publisher.rosbag1_imu_reader import Rosbag1Error, Rosbag1ImuReader


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class DsecPublisher(Node):
    def __init__(self):
        super().__init__('dsec_publisher')

        # --- Parameters ---
        self.events_h5 = self.declare_parameter(
            'events_h5',
            'logs/dsec/zurich_city_10_a/events_left/events.h5').value
        self.topic = self.declare_parameter('topic', 'event_camera/events').value
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.window_ms = float(self.declare_parameter('window_ms', 10.0).value)
        self.realtime_factor = float(self.declare_parameter('realtime_factor', 1.0).value)
        self.loop = _as_bool(self.declare_parameter('loop', False).value)
        self.start_offset_s = float(self.declare_parameter('start_offset_s', 0.0).value)
        self.frame_id = self.declare_parameter('frame_id', 'camera_0').value
        self.publish_imu = _as_bool(self.declare_parameter('publish_imu', True).value)
        self.imu_topic = self.declare_parameter('imu_topic', 'imu/data').value
        self.imu_bag_topic = self.declare_parameter('imu_bag_topic', '/imu/data').value
        self.imu_bag = self.declare_parameter('imu_bag', '').value
        self.lidar_imu_root = self.declare_parameter('lidar_imu_root', '').value
        self.imu_frame_id = self.declare_parameter('imu_frame_id', 'imu0').value
        self.publish_tf = _as_bool(self.declare_parameter('publish_tf', True).value)
        self.lidar_frame_id = self.declare_parameter('lidar_frame_id', 'lidar').value
        self.cam_to_imu_yaml = self.declare_parameter('cam_to_imu_yaml', '').value
        self.cam_to_lidar_yaml = self.declare_parameter('cam_to_lidar_yaml', '').value
        # Reliable + deep queue = lossless offline replay. EVO is compute-bound
        # and far slower than DSEC's realtime event rate; with BestEffort it
        # would drop ~98% of packets and never track. Reliable lets EVO buffer
        # and process every event at its own pace. Set reliable:=false only to
        # emulate a live (lossy) camera.
        self.reliable = _as_bool(self.declare_parameter('reliable', True).value)
        self.qos_depth = int(self.declare_parameter('qos_depth', 5000).value)

        self.events_h5 = os.path.abspath(os.path.expanduser(self.events_h5))
        if not os.path.isfile(self.events_h5):
            raise FileNotFoundError(f'events.h5 not found: {self.events_h5}')
        
        rect_path = os.path.join(os.path.dirname(self.events_h5), 'rectify_map.h5')
        with h5py.File(rect_path, 'r') as rf:
            self.rectify_map = np.asarray(rf['rectify_map'], dtype=np.float32)

        # --- Data source ---
        self._h5f = h5py.File(self.events_h5, 'r')
        self.slicer = EventSlicer(self._h5f)
        self.t_start_us = self.slicer.get_start_time_us() + int(self.start_offset_s * 1e6)
        self.t_final_us = self.slicer.get_final_time_us()
        self.window_us = int(self.window_ms * 1000)
        self.cursor_us = self.t_start_us
        self.short_sequence = self._short_sequence_name()
        self.full_sequence = self._full_sequence_name(self.short_sequence)

        # --- Publisher ---
        qos = QoSProfile(
            reliability=(QoSReliabilityPolicy.RELIABLE if self.reliable
                         else QoSReliabilityPolicy.BEST_EFFORT),
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.qos_depth)
        self.pub = self.create_publisher(EventPacket, self.topic, qos)
        self.imu_pub = None
        self.imu_reader = None
        self.imu_iter = None
        self.next_imu = None
        self.total_imu = 0
        if self.publish_imu:
            self._init_imu_publisher(qos)
        if self.publish_tf:
            self._publish_static_tfs()

        self.seq = 0
        self.total_events = 0

        period = (self.window_ms / 1000.0) / max(self.realtime_factor, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"DSEC replay: '{self.events_h5}' -> topic '{self.topic}' "
            f"({self.width}x{self.height}, window={self.window_ms} ms, "
            f"rt_factor={self.realtime_factor}, loop={self.loop}). "
            f"Sequence '{self.short_sequence}' uses IMU base '{self.full_sequence}'. "
            f"Duration ~{(self.t_final_us - self.t_start_us) / 1e6:.2f} s.")

    def _short_sequence_name(self) -> str:
        return Path(self.events_h5).parent.parent.name

    @staticmethod
    def _full_sequence_name(short_sequence: str) -> str:
        match = re.match(r'(.+)_[a-z]$', short_sequence)
        return match.group(1) if match else short_sequence

    def _lidar_imu_root(self) -> str:
        if self.lidar_imu_root:
            return os.path.abspath(os.path.expanduser(self.lidar_imu_root))
        dsec_root = Path(self.events_h5).parent.parent.parent
        return str(dsec_root / 'lidar_imu')

    def _resolve_path(self, configured_path: str, default_path: str) -> str:
        path = configured_path if configured_path else default_path
        return os.path.abspath(os.path.expanduser(path))

    def _default_imu_bag(self) -> str:
        return str(Path(self._lidar_imu_root()) / 'data' / self.full_sequence / 'lidar_imu.bag')

    def _default_cam_to_imu_yaml(self) -> str:
        return str(Path(self._lidar_imu_root()) / 'imu_calibration' / 'cam0_to_imu0.yaml')

    def _default_cam_to_lidar_yaml(self) -> str:
        return str(Path(self._lidar_imu_root()) / 'data' / self.full_sequence / 'cam_to_lidar.yaml')

    def _init_imu_publisher(self, qos: QoSProfile) -> None:
        self.imu_bag = self._resolve_path(self.imu_bag, self._default_imu_bag())
        if not os.path.isfile(self.imu_bag):
            raise FileNotFoundError(f'IMU bag not found: {self.imu_bag}')

        try:
            self.imu_reader = Rosbag1ImuReader(self.imu_bag, self.imu_bag_topic)
        except Rosbag1Error as exc:
            raise RuntimeError(f'Failed to open IMU bag {self.imu_bag}: {exc}') from exc

        self.imu_pub = self.create_publisher(Imu, self.imu_topic, qos)
        self._reset_imu_stream()
        self.get_logger().info(
            f"IMU replay: '{self.imu_bag}' topic '{self.imu_bag_topic}' "
            f"-> '{self.imu_topic}' frame '{self.imu_frame_id}' "
            f"within [{self.t_start_us}, {self.t_final_us}] us.")

    def _reset_imu_stream(self) -> None:
        if self.imu_reader is None:
            return
        self.imu_iter = self.imu_reader.iter_range(self.t_start_us, self.t_final_us + 1)
        self._advance_imu()

    def _advance_imu(self) -> None:
        if self.imu_iter is None:
            self.next_imu = None
            return
        try:
            self.next_imu = next(self.imu_iter)
        except StopIteration:
            self.next_imu = None

    def _publish_imu_until(self, t_end_us: int) -> None:
        if self.imu_pub is None:
            return
        while self.next_imu is not None and self.next_imu.stamp_us < t_end_us:
            self._publish_imu(self.next_imu)
            self._advance_imu()

    def _publish_imu(self, sample) -> None:
        msg = Imu()
        msg.header.stamp.sec = int(sample.stamp_sec)
        msg.header.stamp.nanosec = int(sample.stamp_nanosec)
        msg.header.frame_id = self.imu_frame_id or sample.frame_id

        msg.orientation.x = float(sample.orientation[0])
        msg.orientation.y = float(sample.orientation[1])
        msg.orientation.z = float(sample.orientation[2])
        msg.orientation.w = float(sample.orientation[3])
        msg.orientation_covariance = list(sample.orientation_covariance)

        msg.angular_velocity.x = float(sample.angular_velocity[0])
        msg.angular_velocity.y = float(sample.angular_velocity[1])
        msg.angular_velocity.z = float(sample.angular_velocity[2])
        msg.angular_velocity_covariance = list(sample.angular_velocity_covariance)

        msg.linear_acceleration.x = float(sample.linear_acceleration[0])
        msg.linear_acceleration.y = float(sample.linear_acceleration[1])
        msg.linear_acceleration.z = float(sample.linear_acceleration[2])
        msg.linear_acceleration_covariance = list(sample.linear_acceleration_covariance)

        self.imu_pub.publish(msg)
        self.total_imu += 1

    def _publish_static_tfs(self) -> None:
        transforms = []

        cam_to_imu_yaml = self._resolve_path(
            self.cam_to_imu_yaml, self._default_cam_to_imu_yaml())
        if os.path.isfile(cam_to_imu_yaml):
            matrix = self._load_matrix(cam_to_imu_yaml, 'T_cam0_imu0')
            transforms.append(self._transform_from_matrix(
                self.frame_id, self.imu_frame_id, matrix))
        else:
            self.get_logger().warn(f'Camera-to-IMU calibration not found: {cam_to_imu_yaml}')

        cam_to_lidar_yaml = self._resolve_path(
            self.cam_to_lidar_yaml, self._default_cam_to_lidar_yaml())
        if os.path.isfile(cam_to_lidar_yaml):
            matrix = self._load_matrix(cam_to_lidar_yaml, 'T_lidar_camRect1')
            transforms.append(self._transform_from_matrix(
                self.lidar_frame_id, self.frame_id, matrix))
        else:
            self.get_logger().warn(f'Camera-to-lidar calibration not found: {cam_to_lidar_yaml}')

        if transforms:
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            self.static_tf_broadcaster.sendTransform(transforms)
            names = ', '.join(
                f'{tf.header.frame_id}->{tf.child_frame_id}' for tf in transforms)
            self.get_logger().info(f'Published static TFs: {names}')

    def _load_matrix(self, path: str, key: str) -> np.ndarray:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if key not in data:
            raise KeyError(f'{key} not found in {path}')
        matrix = np.asarray(data[key], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f'{key} in {path} must be a 4x4 matrix')
        return matrix

    def _transform_from_matrix(self, parent: str, child: str, matrix: np.ndarray):
        quat = self._quaternion_from_matrix(matrix[:3, :3])

        msg = TransformStamped()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(matrix[0, 3])
        msg.transform.translation.y = float(matrix[1, 3])
        msg.transform.translation.z = float(matrix[2, 3])
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])
        return msg

    @staticmethod
    def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quat = np.array([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ])
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quat = np.array([
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ])
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quat = np.array([
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ])
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quat = np.array([
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ])
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            raise ValueError('Cannot convert zero-norm rotation matrix to quaternion')
        return quat / norm

    def _rectify(self, ev):
        x = ev['x'].astype(np.int64)
        y = ev['y'].astype(np.int64)
        rc = self.rectify_map[y, x]
        xr = np.rint(rc[:, 0]).astype(np.int64)
        yr = np.rint(rc[:, 1]).astype(np.int64)
        m = (xr >= 0) & (xr < self.width) & (yr >= 0) & (yr < self.height)
        if not np.any(m):
            return None
        return {'t': ev['t'][m], 'p': ev['p'][m],
                'x': xr[m].astype(ev['x'].dtype), 'y': yr[m].astype(ev['y'].dtype)}

    def _on_timer(self):
        if self.cursor_us >= self.t_final_us:
            if self.loop:
                self.get_logger().info('Reached end of sequence, looping.')
                self.cursor_us = self.t_start_us
                self.seq = 0
                self._reset_imu_stream()
            else:
                self.get_logger().info(
                    f'Reached end of sequence ({self.total_events} events, '
                    f'{self.total_imu} IMU messages published). '
                    f'Shutting down.')
                self.timer.cancel()
                rclpy.shutdown()
            return

        t0 = self.cursor_us
        t1 = min(self.cursor_us + self.window_us, self.t_final_us + 1)
        self.cursor_us += self.window_us
        self._publish_imu_until(t1)

        ev = self.slicer.get_events(t0, t1)
        if ev is None or ev['t'].size == 0:
            return
        ev = self._rectify(ev)
        if ev is None or ev['t'].size == 0:
            return

        self._publish(ev)

    def _publish(self, ev):
        t_abs_us = ev['t'].astype(np.int64)          # absolute microseconds
        time_base_ns = int(t_abs_us[0]) * 1000
        dt_ns = ((t_abs_us - t_abs_us[0]) * 1000).astype(np.uint64)

        x = ev['x'].astype(np.uint64)
        y = ev['y'].astype(np.uint64)
        p = ev['p'].astype(np.uint64)

        packed = (p << 63) | (y << 48) | (x << 32) | dt_ns
        # Force little-endian byte order regardless of host.
        buf = packed.astype('<u8').tobytes()

        msg = EventPacket()
        msg.header.stamp.sec = int(time_base_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(time_base_ns % 1_000_000_000)
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.seq = self.seq
        msg.time_base = time_base_ns
        msg.encoding = 'mono'
        msg.is_bigendian = False
        msg.events = array('B', buf)

        self.pub.publish(msg)
        self.seq += 1
        self.total_events += int(t_abs_us.size)

    def destroy_node(self):
        try:
            self._h5f.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DsecPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
