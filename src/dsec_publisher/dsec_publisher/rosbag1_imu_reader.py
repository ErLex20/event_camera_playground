"""Minimal ROS 1 bag reader for DSEC IMU messages.

The DSEC lidar/IMU download ships ROS 1 ``.bag`` files. This reader supports
the uncompressed/bz2 ROS bag v2.0 layout used by those files and deserializes
only ``sensor_msgs/Imu`` records, keeping the ROS 2 replay node free of a
ROS 1 runtime dependency. See :mod:`dsec_publisher.rosbag1_reader_base` for
the shared connection/chunk-index parsing.
"""

import struct
from dataclasses import dataclass
from typing import Iterator, Tuple

from dsec_publisher.rosbag1_reader_base import Rosbag1Error, Rosbag1IndexReader

__all__ = ['Rosbag1Error', 'ImuSample', 'Rosbag1ImuReader']


@dataclass(frozen=True)
class ImuSample:
    stamp_us: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    orientation: Tuple[float, float, float, float]
    orientation_covariance: Tuple[float, ...]
    angular_velocity: Tuple[float, float, float]
    angular_velocity_covariance: Tuple[float, ...]
    linear_acceleration: Tuple[float, float, float]
    linear_acceleration_covariance: Tuple[float, ...]


class Rosbag1ImuReader(Rosbag1IndexReader):
    """Lazily reads ``sensor_msgs/Imu`` samples from a ROS 1 bag."""

    def __init__(self, bag_path: str, topic: str = '/imu/data'):
        super().__init__(bag_path, topic, 'sensor_msgs/Imu')

    def iter_range(self, start_us: int, end_us: int) -> Iterator[ImuSample]:
        """Yield IMU samples with ``start_us <= stamp < end_us``."""
        for record_stamp_us, payload in self._iter_messages_in_range(start_us, end_us):
            sample = self._deserialize_imu(payload, record_stamp_us)
            if sample.stamp_us < start_us:
                continue
            if sample.stamp_us >= end_us:
                return
            yield sample

    def _deserialize_imu(self, data: bytes, record_stamp_us: int) -> ImuSample:
        pos = 0
        seq, sec, nanosec = struct.unpack_from('<III', data, pos)
        del seq
        pos += 12

        frame_len = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        frame_id = data[pos:pos + frame_len].decode('utf-8', errors='replace')
        pos += frame_len

        orientation = struct.unpack_from('<4d', data, pos)
        pos += 4 * 8
        orientation_covariance = struct.unpack_from('<9d', data, pos)
        pos += 9 * 8
        angular_velocity = struct.unpack_from('<3d', data, pos)
        pos += 3 * 8
        angular_velocity_covariance = struct.unpack_from('<9d', data, pos)
        pos += 9 * 8
        linear_acceleration = struct.unpack_from('<3d', data, pos)
        pos += 3 * 8
        linear_acceleration_covariance = struct.unpack_from('<9d', data, pos)

        stamp_us = sec * 1_000_000 + nanosec // 1000
        if stamp_us == 0:
            stamp_us = record_stamp_us
            sec = stamp_us // 1_000_000
            nanosec = (stamp_us % 1_000_000) * 1000

        return ImuSample(
            stamp_us=stamp_us,
            stamp_sec=sec,
            stamp_nanosec=nanosec,
            frame_id=frame_id,
            orientation=orientation,
            orientation_covariance=orientation_covariance,
            angular_velocity=angular_velocity,
            angular_velocity_covariance=angular_velocity_covariance,
            linear_acceleration=linear_acceleration,
            linear_acceleration_covariance=linear_acceleration_covariance)
