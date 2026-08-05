"""Shared indexing/decoding primitives for the minimal ROS 1 bag readers.

Factored out of the original IMU-only reader so a second reader (LiDAR
PointCloud2) can reuse the same connection/chunk-index parsing without
duplicating it. Only the uncompressed/bz2 ROS bag v2.0 chunked layout is
supported, matching the DSEC lidar/IMU download.
"""

import bz2
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple


OP_MSG_DATA = 0x02
OP_FILE_HEADER = 0x03
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07


class Rosbag1Error(RuntimeError):
    """Raised when a ROS 1 bag cannot be read by this focused reader."""


@dataclass(frozen=True)
class _ChunkInfo:
    position: int
    start_us: int
    end_us: int
    connection_counts: Dict[int, int]


def _read_exact(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise Rosbag1Error('Unexpected end of bag file')
    return data


def _parse_fields(data: bytes) -> Dict[str, bytes]:
    fields: Dict[str, bytes] = {}
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise Rosbag1Error('Malformed ROS bag header field')
        field_len = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        raw = data[pos:pos + field_len]
        pos += field_len
        if b'=' not in raw:
            raise Rosbag1Error('Malformed ROS bag header field')
        key, value = raw.split(b'=', 1)
        fields[key.decode('ascii')] = value
    return fields


def _read_record(f: BinaryIO, read_data: bool = True):
    header_len_bytes = f.read(4)
    if not header_len_bytes:
        return None
    if len(header_len_bytes) != 4:
        raise Rosbag1Error('Truncated ROS bag record header')

    header_len = struct.unpack('<I', header_len_bytes)[0]
    fields = _parse_fields(_read_exact(f, header_len))
    data_len = struct.unpack('<I', _read_exact(f, 4))[0]
    if read_data:
        data = _read_exact(f, data_len)
    else:
        f.seek(data_len, os.SEEK_CUR)
        data = b''
    return fields, data


def _iter_records(data: bytes):
    pos = 0
    size = len(data)
    while pos < size:
        if pos + 4 > size:
            raise Rosbag1Error('Truncated ROS bag chunk record')
        header_len = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        fields = _parse_fields(data[pos:pos + header_len])
        pos += header_len
        if pos + 4 > size:
            raise Rosbag1Error('Truncated ROS bag chunk data length')
        data_len = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        payload = data[pos:pos + data_len]
        pos += data_len
        yield fields, payload


def _uint32(value: bytes) -> int:
    return struct.unpack('<I', value)[0]


def _uint64(value: bytes) -> int:
    return struct.unpack('<Q', value)[0]


def _time_us(value: bytes) -> int:
    sec, nanosec = struct.unpack('<II', value)
    return sec * 1_000_000 + nanosec // 1000


def _string(value: bytes) -> str:
    return value.decode('utf-8', errors='replace')


class Rosbag1IndexReader:
    """Indexes connections/chunks of a ROS 1 bag; subclasses decode one topic."""

    def __init__(self, bag_path: str, topic: str, msgtype: str):
        self.bag_path = os.path.abspath(os.path.expanduser(bag_path))
        self.topic = topic
        self.msgtype = msgtype
        self._connections: Dict[int, Dict[str, str]] = {}
        self._chunks: List[_ChunkInfo] = []
        self._conn: Optional[int] = None

        self._read_index()
        self._select_connection()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _read_index(self) -> None:
        with open(self.bag_path, 'rb') as f:
            magic = f.readline()
            if magic != b'#ROSBAG V2.0\n':
                raise Rosbag1Error(
                    f'Unsupported ROS bag magic in {self.bag_path!r}: {magic!r}')

            record = _read_record(f)
            if record is None:
                raise Rosbag1Error('Missing ROS bag file header')
            fields, _ = record
            if fields.get('op', b'') != bytes([OP_FILE_HEADER]):
                raise Rosbag1Error('Missing ROS bag file header record')
            if 'index_pos' not in fields:
                raise Rosbag1Error('ROS bag is not indexed')

            f.seek(_uint64(fields['index_pos']))
            while True:
                record = _read_record(f)
                if record is None:
                    break
                fields, data = record
                op = fields.get('op', b'\x00')[0]
                if op == OP_CONNECTION:
                    conn = _uint32(fields['conn'])
                    data_fields = _parse_fields(data)
                    self._connections[conn] = {
                        'topic': _string(data_fields.get('topic', fields.get('topic', b''))),
                        'type': _string(data_fields.get('type', b'')),
                    }
                elif op == OP_CHUNK_INFO:
                    self._chunks.append(self._parse_chunk_info(fields, data))

            self._chunks.sort(key=lambda chunk: chunk.position)

    def _select_connection(self) -> None:
        for conn, info in self._connections.items():
            if info.get('topic') == self.topic and info.get('type') == self.msgtype:
                self._conn = conn
                return

        topics = ', '.join(
            f"{info.get('topic', '?')} ({info.get('type', '?')})"
            for info in self._connections.values())
        raise Rosbag1Error(
            f'Could not find {self.msgtype} topic {self.topic!r} in '
            f'{self.bag_path!r}. Available topics: {topics}')

    def _parse_chunk_info(self, fields: Dict[str, bytes], data: bytes) -> _ChunkInfo:
        counts: Dict[int, int] = {}
        for pos in range(0, len(data), 8):
            conn, count = struct.unpack_from('<II', data, pos)
            counts[conn] = count

        return _ChunkInfo(
            position=_uint64(fields['chunk_pos']),
            start_us=_time_us(fields['start_time']),
            end_us=_time_us(fields['end_time']),
            connection_counts=counts)

    def _iter_chunk_messages(self, chunk_position: int) -> Iterator[Tuple[int, bytes]]:
        """Yield (record_stamp_us, payload) for this reader's connection in one chunk."""
        with open(self.bag_path, 'rb') as f:
            f.seek(chunk_position)
            record = _read_record(f)

        if record is None:
            raise Rosbag1Error(f'Missing chunk at offset {chunk_position}')
        fields, data = record
        if fields.get('op', b'\x00')[0] != OP_CHUNK:
            raise Rosbag1Error(f'Expected chunk at offset {chunk_position}')

        compression = _string(fields.get('compression', b'none'))
        if compression == 'none':
            chunk_data = data
        elif compression == 'bz2':
            chunk_data = bz2.decompress(data)
        else:
            raise Rosbag1Error(
                f'Unsupported ROS bag compression {compression!r} in {self.bag_path!r}')

        for rec_fields, payload in _iter_records(chunk_data):
            if rec_fields.get('op', b'\x00')[0] != OP_MSG_DATA:
                continue
            if _uint32(rec_fields['conn']) != self._conn:
                continue
            yield _time_us(rec_fields['time']), payload

    def _iter_messages_in_range(self, start_us: int, end_us: int) -> Iterator[Tuple[int, bytes]]:
        """Yield (stamp_us, payload) across all overlapping chunks, message-time sorted.

        Chunks are indexed in file-position order, which is NOT guaranteed to
        be monotonic in time for every connection in this bag format (a chunk
        near the start of the file can have a far-later ``end_time`` than
        chunks that follow it, if a low-rate topic's messages get buffered
        into a sparse chunk out of step with a high-rate topic's chunks). So
        this scans every chunk unconditionally rather than breaking early on
        the first ``chunk.start_us >= end_us`` -- that early-exit is only
        valid under a monotonicity assumption this bag violates for at least
        one connection.
        """
        if self._conn is None:
            return
        messages = []
        for chunk in self._chunks:
            if chunk.end_us <= start_us or chunk.start_us >= end_us:
                continue
            if chunk.connection_counts.get(self._conn, 0) == 0:
                continue
            messages.extend(self._iter_chunk_messages(chunk.position))
        messages.sort(key=lambda item: item[0])
        yield from messages
