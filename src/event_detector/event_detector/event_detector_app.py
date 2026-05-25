#!/opt/dua-venv/bin/python
"""
Event Detector app implementation.

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

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from event_detector.event_detector_node import EventDetectorNode

def main():
    # Initialize ROS 2 context and node
    rclpy.init(args=sys.argv)
    event_detector_node = EventDetectorNode(node_name='event_detector', verbose=True)

    executor = MultiThreadedExecutor()
    executor.add_node(event_detector_node)

    # Run the node
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        event_detector_node.cleanup()
        event_detector_node.destroy_node()

if __name__ == '__main__':
    main()
