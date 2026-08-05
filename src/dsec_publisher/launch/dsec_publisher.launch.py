"""Standalone DSEC replay publisher launch file."""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ld = LaunchDescription()

    default_dsec_root = '/home/neo/workspace/logs/dsec'
    config = os.path.join(
        get_package_share_directory('dsec_publisher'), 'config', 'dsec_publisher.yaml')
    dsec_root = LaunchConfiguration('dsec_root')
    sequence = LaunchConfiguration('sequence')

    ld.add_action(DeclareLaunchArgument('namespace', default_value=''))
    ld.add_action(DeclareLaunchArgument('dsec_root', default_value=default_dsec_root))
    ld.add_action(DeclareLaunchArgument('sequence', default_value='thun_00_a'))
    ld.add_action(DeclareLaunchArgument(
        'events_h5',
        default_value=PathJoinSubstitution([dsec_root, sequence, 'events_left', 'events.h5'])))
    ld.add_action(DeclareLaunchArgument(
        'lidar_imu_root',
        default_value=PathJoinSubstitution([dsec_root, 'lidar_imu'])))
    ld.add_action(DeclareLaunchArgument('topic', default_value='event_camera/events'))
    ld.add_action(DeclareLaunchArgument('imu_topic', default_value='imu/data'))
    ld.add_action(DeclareLaunchArgument('imu_bag_topic', default_value='/imu/data'))
    ld.add_action(DeclareLaunchArgument('imu_bag', default_value=''))
    ld.add_action(DeclareLaunchArgument('publish_imu', default_value='true'))
    ld.add_action(DeclareLaunchArgument('publish_tf', default_value='true'))
    ld.add_action(DeclareLaunchArgument('imu_frame_id', default_value='imu0'))
    ld.add_action(DeclareLaunchArgument('lidar_frame_id', default_value='lidar'))
    ld.add_action(DeclareLaunchArgument('cam_to_imu_yaml', default_value=''))
    ld.add_action(DeclareLaunchArgument('cam_to_lidar_yaml', default_value=''))
    ld.add_action(DeclareLaunchArgument('window_ms', default_value='1.0'))
    ld.add_action(DeclareLaunchArgument('realtime_factor', default_value='1.0'))
    ld.add_action(DeclareLaunchArgument('loop', default_value='false'))
    ld.add_action(DeclareLaunchArgument('start_offset_s', default_value='0.0'))

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
                'lidar_imu_root': LaunchConfiguration('lidar_imu_root'),
                'topic': LaunchConfiguration('topic'),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'imu_bag_topic': LaunchConfiguration('imu_bag_topic'),
                'imu_bag': LaunchConfiguration('imu_bag'),
                'publish_imu': ParameterValue(
                    LaunchConfiguration('publish_imu'), value_type=bool),
                'publish_tf': ParameterValue(
                    LaunchConfiguration('publish_tf'), value_type=bool),
                'imu_frame_id': LaunchConfiguration('imu_frame_id'),
                'lidar_frame_id': LaunchConfiguration('lidar_frame_id'),
                'cam_to_imu_yaml': LaunchConfiguration('cam_to_imu_yaml'),
                'cam_to_lidar_yaml': LaunchConfiguration('cam_to_lidar_yaml'),
                'window_ms': ParameterValue(
                    LaunchConfiguration('window_ms'), value_type=float),
                'realtime_factor': ParameterValue(
                    LaunchConfiguration('realtime_factor'), value_type=float),
                'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool),
                'start_offset_s': ParameterValue(
                    LaunchConfiguration('start_offset_s'), value_type=float),
            },
        ],
    ))

    return ld
