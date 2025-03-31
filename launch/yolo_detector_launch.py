import os 
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    model = LaunchConfiguration('model')
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            "model",
            default_value="general",
            description="model used in the detections"
        ),

        launch_ros.actions.Node(
            package='yolo_face_detection',
            executable='t265_node',
            name='t265_node',
            output='screen'
        ),

        launch_ros.actions.Node(
            package='yolo_face_detection',
            executable='yolo_detector.py',  
            output='screen',
            arguments=[os.path.join(os.getenv('ROS_WS', '/home/rcasal/ros2_ws'), 'install/yolo_face_detection/lib/yolo_face_detection/yolo_detector.py')]
        ),
    ])