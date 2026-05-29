"""
EVO launch file.

May 29, 2025
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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    ld = LaunchDescription()

    # Build config file path
    config = os.path.join(
        get_package_share_directory('evo'),
        'config',
        'evo.yaml'
    )
    trigger_config = os.path.join(
        get_package_share_directory("prophesee_evk4_driver"),
        "config",
        "trigger_pins.yaml")
    bias_config = os.path.join(
        get_package_share_directory("prophesee_evk4_driver"),
        "config",
        "imx636_CD_standard.bias")

    # Declare launch arguments
    ns = LaunchConfiguration('namespace')
    ns_launch_arg = DeclareLaunchArgument(
        'namespace',
        default_value='')
    ld.add_action(ns_launch_arg)

    container = ComposableNodeContainer(
        name="metavision_driver_container",
        namespace=ns,
        package="dua_app_management",
        executable="dua_component_container_mt",
        emulate_tty=True,
        output='both',
        log_cmd=True,
        # prefix=['gdbserver localhost:3000'],
        # arguments=['--ros-args', '--log-level', 'warn'],
        composable_node_descriptions=[
            # Event Camera Driver
            ComposableNode(
                package="metavision_driver",
                plugin="metavision_driver::DriverROS2",
                namespace=ns,
                name='event_camera_driver',
                parameters=[
                    trigger_config,
                    {
                        "use_multithreading": True,
                        "bias_file": bias_config,
                        "camerainfo_url": "",
                        "frame_id": "",
                        "event_message_time_threshold": 1.0e-3,
                    },
                ],
                remappings=[
                    ("~/events", 'event_camera/events'),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # Event Camera Renderer
            ComposableNode(
                package='event_camera_renderer',
                plugin='event_camera_renderer::Renderer',
                namespace=ns,
                name='event_camera_renderer',
                parameters=[{'fps': 20.0}],
                remappings=[
                    ('~/events', 'event_camera/events')
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # Event Camera Republisher
            ComposableNode(
                package='event_camera_tools',
                plugin='event_camera_tools::RepublishComposable',
                namespace=ns,
                name='event_camera_republisher',
                parameters=[{'output_message_type': 'event_packet'}],
                remappings=[
                    ('~/input_events', 'event_camera' + '/events'),
                    ('~/output_events', 'event_camera' + '/republished_events'),
                    ('~/output_triggers', 'event_camera' + '/republished_triggers'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # EVO
            ComposableNode(
                package='evo',
                plugin='evo::EVO',
                namespace=ns,
                name='evo',
                parameters=[config],
                remappings=[
                    ('/event_packet', '/event_camera/events')
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
    )
    ld.add_action(container)

    return ld
