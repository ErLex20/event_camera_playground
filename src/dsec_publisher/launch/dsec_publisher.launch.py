"""Standalone DSEC replay publisher launch file."""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    default_h5 = '/home/neo/workspace/logs/dsec/thun_00_a_events_left/events.h5'
    config = os.path.join(
        get_package_share_directory('dsec_publisher'), 'config', 'dsec_publisher.yaml')

    args = {
        'namespace': '',
        'events_h5': default_h5,
        'topic': 'event_camera/events',
        'window_ms': '10.0',
        'realtime_factor': '1.0',
        'loop': 'false',
        'start_offset_s': '0.0',
    }
    for name, default in args.items():
        ld.add_action(DeclareLaunchArgument(name, default_value=default))

    ld.add_action(Node(
        package='dsec_publisher',
        executable='dsec_publisher',
        name='dsec_publisher',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        emulate_tty=True,
        parameters=[
            config,
            {
                'events_h5': LaunchConfiguration('events_h5'),
                'topic': LaunchConfiguration('topic'),
                'window_ms': LaunchConfiguration('window_ms'),
                'realtime_factor': LaunchConfiguration('realtime_factor'),
                'loop': LaunchConfiguration('loop'),
                'start_offset_s': LaunchConfiguration('start_offset_s'),
            },
        ],
    ))

    return ld
