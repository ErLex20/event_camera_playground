"""
Event Detector launch file.

May 25, 2025
"""

# Copyright 2024 dotX Automation s.r.l.
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

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    # Build config file path
    config = os.path.join(
        get_package_share_directory('event_detector_cpp'),
        'config',
        'event_detector_cpp.yaml'
    )

    # Declare launch arguments
    ns = LaunchConfiguration('namespace')
    ns_launch_arg = DeclareLaunchArgument(
        'namespace',
        default_value='')
    ld.add_action(ns_launch_arg)

    # Create Event Detector node
    event_detector = Node(
        namespace=ns,
        package='event_detector_cpp',
        executable='event_detector_cpp_app',
        name='event_detector_cpp',
        emulate_tty=True,
        shell=False,
        output='both',
        # prefix='gdbserver localhost:8081',
        parameters=[config],
        remappings=[
            ('/event_packet',                      '/event_camera/events'),
        ]
    )
    ld.add_action(event_detector)

    return ld
