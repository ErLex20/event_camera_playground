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
from array import array

import h5py
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from event_camera_msgs.msg import EventPacket

from dsec_publisher.event_slicer import EventSlicer


class DsecPublisher(Node):
    def __init__(self):
        super().__init__('dsec_publisher')

        # --- Parameters ---
        self.events_h5 = self.declare_parameter(
            'events_h5',
            'logs/dsec/thun_00__events_left/events.h5').value
        self.topic = self.declare_parameter('topic', 'event_camera/events').value
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.window_ms = float(self.declare_parameter('window_ms', 10.0).value)
        self.realtime_factor = float(self.declare_parameter('realtime_factor', 1.0).value)
        self.loop = bool(self.declare_parameter('loop', False).value)
        self.start_offset_s = float(self.declare_parameter('start_offset_s', 0.0).value)
        self.frame_id = self.declare_parameter('frame_id', 'camera_0').value
        # Reliable + deep queue = lossless offline replay. EVO is compute-bound
        # and far slower than DSEC's realtime event rate; with BestEffort it
        # would drop ~98% of packets and never track. Reliable lets EVO buffer
        # and process every event at its own pace. Set reliable:=false only to
        # emulate a live (lossy) camera.
        self.reliable = bool(self.declare_parameter('reliable', True).value)
        self.qos_depth = int(self.declare_parameter('qos_depth', 5000).value)

        self.events_h5 = os.path.abspath(os.path.expanduser(self.events_h5))
        if not os.path.isfile(self.events_h5):
            raise FileNotFoundError(f'events.h5 not found: {self.events_h5}')

        # --- Data source ---
        self._h5f = h5py.File(self.events_h5, 'r')
        self.slicer = EventSlicer(self._h5f)
        self.t_start_us = self.slicer.get_start_time_us() + int(self.start_offset_s * 1e6)
        self.t_final_us = self.slicer.get_final_time_us()
        self.window_us = int(self.window_ms * 1000)
        self.cursor_us = self.t_start_us

        # --- Publisher ---
        qos = QoSProfile(
            reliability=(QoSReliabilityPolicy.RELIABLE if self.reliable
                         else QoSReliabilityPolicy.BEST_EFFORT),
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.qos_depth)
        self.pub = self.create_publisher(EventPacket, self.topic, qos)

        self.seq = 0
        self.total_events = 0

        period = (self.window_ms / 1000.0) / max(self.realtime_factor, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"DSEC replay: '{self.events_h5}' -> topic '{self.topic}' "
            f"({self.width}x{self.height}, window={self.window_ms} ms, "
            f"rt_factor={self.realtime_factor}, loop={self.loop}). "
            f"Duration ~{(self.t_final_us - self.t_start_us) / 1e6:.2f} s.")

    def _on_timer(self):
        if self.cursor_us >= self.t_final_us:
            if self.loop:
                self.get_logger().info('Reached end of sequence, looping.')
                self.cursor_us = self.t_start_us
                self.seq = 0
            else:
                self.get_logger().info(
                    f'Reached end of sequence ({self.total_events} events published). '
                    f'Shutting down.')
                self.timer.cancel()
                rclpy.shutdown()
            return

        t0 = self.cursor_us
        t1 = min(self.cursor_us + self.window_us, self.t_final_us + 1)
        self.cursor_us += self.window_us

        ev = self.slicer.get_events(t0, t1)
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
