"""
Event Detector node implementation.

dotX Automation s.r.l. <info@dotxautomation.com>

May 6, 2026
"""

# Copyright 2025 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Thread
from event_detector.event_detector_utils import AtomicBool

from dua_node_py.dua_node import NodeBase
import dua_qos_py.dua_qos_besteffort as dua_qos_besteffort

from event_camera_msgs.msg import EventPacket
from event_camera_py import Decoder


class EventDetectorNode(NodeBase):
    """
    Event Detector node implementation.
    """

    def __init__(self, node_name: str, verbose: bool = False) -> None:
        """
        Constructor.

        :param node_name: Name of the node.
        :param verbose: Verbosity flag.
        """
        super().__init__(node_name, verbose)

        self.init_atomics()
        self.init_detector()

        if self._autostart:
            self.activate()

        self.get_logger().info('Node initialized')

    def __del__(self) -> None:
        """
        Destructor.
        """
        self.cleanup()

    def cleanup(self) -> None:
        """
        Cleanup.
        """
        self.deactivate()

    def init_atomics(self) -> None:
        """
        Init atomics.
        """
        self._running = AtomicBool(initial=False)

    def init_detector(self) -> None:
        """
        Init the event detector.
        """
        self._decoder = Decoder()

    def init_subscribers(self) -> None:
        """
        Init subscribers.
        """
        self._event_packet_subscriber = self.dua_create_subscription(
            EventPacket,
            '/event_packet',
            self.event_packet_callback,
            dua_qos_besteffort.get_datum_qos())

    def activate(self) -> None:
        """
        Function to activate the Event Detector node.
        """
        self._running.store(True)
        self._worker = Thread(target=self.worker_thread_routine)
        self._worker.start()

        self.get_logger().warn('Event Detector ACTIVATED')

    def deactivate(self) -> None:
        """
        Function to deactivate the Event Detector node.
        """
        self._running.store(False)
        self._worker.join()

        self.get_logger().warn('Event Detector DEACTIVATED')

    def event_packet_callback(self, msg: EventPacket) -> None:
        """
        Callback for the EventPacket subscriber.

        :param msg: The received EventPacket message.
        """
        self.get_logger().info(f'Received EventPacket message with timestamp: {msg.header.stamp.sec}')
        self._decoder.decode(msg)
        events = self._decoder.get_cd_events()
        self.get_logger().info(f'Decoded {len(events)} events')

    def worker_thread_routine(self) -> None:
        """
        Worker thread routine.
        """
        self.get_logger().info('Worker thread started')

        while self._running.load():
            pass
