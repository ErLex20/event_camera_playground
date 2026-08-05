"""Minimal ROS 1 bag reader for DSEC LiDAR (``sensor_msgs/PointCloud2``) scans.

Mirrors :mod:`dsec_publisher.rosbag1_imu_reader` but decodes point-cloud
messages (e.g. the ``/velodyne_points`` topic present in the DSEC
``lidar_imu.bag`` files alongside the IMU topic). ``PointField`` layouts are
parsed generically from the message itself rather than assumed, so any
field ordering/padding the recording driver used is handled correctly.
"""

import struct
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from dsec_publisher.rosbag1_reader_base import Rosbag1IndexReader

__all__ = ['PointCloudScan', 'Rosbag1PointCloudReader']


# sensor_msgs/PointField datatype constants -> (numpy dtype, size in bytes)
_FIELD_DTYPES = {
    1: ('i1', 1),  # INT8
    2: ('u1', 1),  # UINT8
    3: ('i2', 2),  # INT16
    4: ('u2', 2),  # UINT16
    5: ('i4', 4),  # INT32
    6: ('u4', 4),  # UINT32
    7: ('f4', 4),  # FLOAT32
    8: ('f8', 8),  # FLOAT64
}


@dataclass(frozen=True)
class PointCloudScan:
    stamp_us: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    points_xyz: np.ndarray  # (N, 3) float32, NaNs/infs from the raw scan removed
    intensity: Optional[np.ndarray]  # (N,) float32 or None if the field is absent


def _read_string(data: bytes, pos: int) -> Tuple[str, int]:
    length = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    value = data[pos:pos + length].decode('utf-8', errors='replace')
    pos += length
    return value, pos


class Rosbag1PointCloudReader(Rosbag1IndexReader):
    """Lazily reads ``sensor_msgs/PointCloud2`` scans from a ROS 1 bag."""

    def __init__(self, bag_path: str, topic: str = '/velodyne_points'):
        super().__init__(bag_path, topic, 'sensor_msgs/PointCloud2')

    def iter_range(self, start_us: int, end_us: int) -> Iterator[PointCloudScan]:
        """Yield point-cloud scans with ``start_us <= stamp < end_us``."""
        for record_stamp_us, payload in self._iter_messages_in_range(start_us, end_us):
            scan = self._deserialize_pointcloud2(payload, record_stamp_us)
            if scan.stamp_us < start_us:
                continue
            if scan.stamp_us >= end_us:
                return
            yield scan

    def _deserialize_pointcloud2(self, data: bytes, record_stamp_us: int) -> PointCloudScan:
        pos = 0
        seq, sec, nanosec = struct.unpack_from('<III', data, pos)
        del seq
        pos += 12

        frame_id, pos = _read_string(data, pos)

        height, width = struct.unpack_from('<II', data, pos)
        pos += 8

        num_fields = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        fields = []
        for _ in range(num_fields):
            name, pos = _read_string(data, pos)
            offset, datatype, count = struct.unpack_from('<IBI', data, pos)
            pos += 9
            fields.append((name, offset, datatype, count))

        is_bigendian, = struct.unpack_from('<B', data, pos)
        pos += 1
        point_step, row_step = struct.unpack_from('<II', data, pos)
        pos += 8

        data_len = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        cloud_bytes = data[pos:pos + data_len]
        pos += data_len

        num_points = height * width
        byte_order = '>' if is_bigendian else '<'

        dtype_fields = {}
        for name, offset, datatype, count in fields:
            if count != 1 or datatype not in _FIELD_DTYPES:
                continue
            np_code, _ = _FIELD_DTYPES[datatype]
            dtype_fields[name] = (f'{byte_order}{np_code}', offset)
        struct_dtype = np.dtype({
            'names': list(dtype_fields.keys()),
            'formats': [spec[0] for spec in dtype_fields.values()],
            'offsets': [spec[1] for spec in dtype_fields.values()],
            'itemsize': point_step,
        })

        if num_points == 0 or not {'x', 'y', 'z'}.issubset(dtype_fields):
            points_xyz = np.zeros((0, 3), dtype=np.float32)
            intensity = None
        else:
            raw = np.frombuffer(cloud_bytes, dtype=struct_dtype, count=num_points)
            points_xyz = np.stack(
                [raw['x'], raw['y'], raw['z']], axis=-1).astype(np.float32)
            finite = np.isfinite(points_xyz).all(axis=1)
            points_xyz = points_xyz[finite]
            intensity = None
            if 'intensity' in dtype_fields:
                intensity = raw['intensity'][finite].astype(np.float32)

        stamp_us = sec * 1_000_000 + nanosec // 1000
        if stamp_us == 0:
            stamp_us = record_stamp_us
            sec = stamp_us // 1_000_000
            nanosec = (stamp_us % 1_000_000) * 1000

        return PointCloudScan(
            stamp_us=stamp_us,
            stamp_sec=sec,
            stamp_nanosec=nanosec,
            frame_id=frame_id,
            points_xyz=points_xyz,
            intensity=intensity)
