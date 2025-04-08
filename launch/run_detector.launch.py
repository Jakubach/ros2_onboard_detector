import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os 

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('onboard_detector'),
        'cfg',
        'detector_param.yaml')
        
    return LaunchDescription([


        DeclareLaunchArgument('detector_param_file', default_value='$(find onboard_detector)/cfg/detector_param.yaml', description='Config file path'),
        Node(
            package='onboard_detector',
            executable='detector_node',
            name='dynamic_detector',
            output='screen',
            parameters=[config]
        ),

    ])
