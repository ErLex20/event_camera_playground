"""
End-to-end DSEC -> EVO launch file.

Starts the EVO pipeline (and the event renderer for a live visual) WITHOUT the
metavision camera driver, and feeds it with the DSEC replay publisher. One
command to test EVO offline on a DSEC recording.
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    ld = LaunchDescription()

    evo_config = os.path.join(
        get_package_share_directory('evo'), 'config', 'evo.yaml')
    dsec_config = os.path.join(
        get_package_share_directory('dsec_publisher'), 'config', 'dsec_publisher.yaml')
    default_h5 = '/home/neo/workspace/logs/dsec/thun_00_a_events_left/events.h5'

    events_topic = 'event_camera/events'

    args = {
        'namespace': '',
        'events_h5': default_h5,
        'window_ms': '10.0',
        'realtime_factor': '1.0',
        'loop': 'false',
        'start_offset_s': '0.0',
        'enable_renderer': 'true',
    }
    for name, default in args.items():
        ld.add_action(DeclareLaunchArgument(name, default_value=default))

    ns = LaunchConfiguration('namespace')

    container = ComposableNodeContainer(
        name='evo_container',
        namespace=ns,
        package='dua_app_management',
        executable='dua_component_container_mt',
        emulate_tty=True,
        output='both',
        composable_node_descriptions=[
            # Event Camera Renderer (live visual of the replayed events)
            ComposableNode(
                package='event_camera_renderer',
                plugin='event_camera_renderer::Renderer',
                namespace=ns,
                name='event_camera_renderer',
                parameters=[{'fps': 20.0}],
                remappings=[('~/events', events_topic)],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # EVO
            ComposableNode(
                package='evo',
                plugin='evo::EVO',
                namespace=ns,
                name='evo',
                parameters=[evo_config,
                            # Lossless offline ingestion (see evo.cpp::init_subscribers):
                            # EVO is slower than DSEC's realtime rate, so feed it
                            # reliably and let it process every event at its own pace.
                            {'event_sub_reliable': True, 'event_sub_depth': 5000}],
                remappings=[('~/events', events_topic)],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
    )
    ld.add_action(container)

    # DSEC replay publisher (regular node, normal DDS transport).
    ld.add_action(Node(
        package='dsec_publisher',
        executable='dsec_publisher',
        name='dsec_publisher',
        namespace=ns,
        output='screen',
        emulate_tty=True,
        parameters=[
            dsec_config,
            {
                'events_h5': LaunchConfiguration('events_h5'),
                'topic': events_topic,
                'window_ms': LaunchConfiguration('window_ms'),
                'realtime_factor': LaunchConfiguration('realtime_factor'),
                'loop': LaunchConfiguration('loop'),
                'start_offset_s': LaunchConfiguration('start_offset_s'),
            },
        ],
    ))

    return ld
