cmake_minimum_required(VERSION 3.8)
project(yolo_face_detection)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# === Dependencies ===
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(realsense2 REQUIRED)
find_package(ament_index_cpp REQUIRED)  # ✅ Added for share path resolution

# === Include paths ===
include_directories(
  include
  /opt/onnxruntime/include        # ONNX Runtime headers
)

# === Link directories ===
link_directories(
  /opt/onnxruntime/lib            # ONNX Runtime libs
)

# === C++ Executable: t265_node ===
add_executable(t265_node src/t265_node.cpp)
ament_target_dependencies(t265_node
  rclcpp
  nav_msgs
  sensor_msgs
  tf2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  cv_bridge
  OpenCV
  realsense2
)

target_link_libraries(t265_node
  ${OpenCV_LIBS}
  ${cv_bridge_LIBRARIES}
  ${realsense2_LIBRARY}
)




# === Install C++ executables ===
install(TARGETS
  t265_node
  DESTINATION lib/${PROJECT_NAME}
)

# === Install Python package ===
ament_python_install_package(${PROJECT_NAME})

# === Install Python executables ===
install(PROGRAMS
  scripts/yolo_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# === Install launch files ===
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# === Install YOLO model files (.pt and .onnx) ===
install(FILES
  yolov9m.onnx
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
