# YOLOv9 + Intel RealSense T265 ROS 2 Package

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)
![YOLOv9](https://img.shields.io/badge/YOLO-v9-red.svg)
![RealSense](https://img.shields.io/badge/Camera-T265-green.svg)

---

## 📁 Overview

This **ROS 2 package** combines real-time **visual odometry and fisheye image capture** from an Intel **RealSense T265 camera** with **YOLOv9 ONNX detection** to run inference on fisheye images.

### Nodes:
- **`t265_node` (C++)**
  - Publishes T265 fisheye images, IMU, and odometry.
- **`yolo_detector.py` (Python)**
  - Runs YOLOv9 object detection (optimized with threading).

### Features:
- ✅ Real-time fisheye image acquisition (left + right)
- ✅ Real-time object detection using YOLOv9 (ONNX)
- ✅ Fully configurable model selection
- ✅ Threaded inference for performance
- ✅ CUDA acceleration via ONNX Runtime

---

## 🚀 Installation & Setup

### 1. Prerequisites
- ROS 2 Humble
- `librealsense2`
- `onnxruntime` GPU runtime installed to `/opt/onnxruntime`
- `ultralytics` Python package (`pip install ultralytics`)

### 2. Clone and Build
```bash
cd ~/ros2_ws/src
# Clone this repo
# git clone <REPO_URL>
cd ~/ros2_ws
colcon build --packages-select yolo_face_detection
source install/setup.bash
```

---

## 🏃️ Run the Nodes

### Launch both camera and detector:
```bash
ros2 launch yolo_face_detection yolo_detector_launch.py model:=person
```
- `model` options: `general`, `person`

### Alternatively, run nodes separately:
```bash
ros2 run yolo_face_detection t265_node
ros2 run yolo_face_detection yolo_detector.py --ros-args -p model:=person
```

---

## 📰 Published Topics

| Topic                              | Type                          | Description                           |
|-----------------------------------|-------------------------------|---------------------------------------|
| `/rs_t265/fisheye_left`           | `sensor_msgs/msg/Image`      | Left camera grayscale image           |
| `/rs_t265/fisheye_right`          | `sensor_msgs/msg/Image`      | Right camera grayscale image          |
| `/rs_t265/imu`                    | `sensor_msgs/msg/Imu`        | Accelerometer + gyroscope             |
| `/rs_t265/odom`                   | `nav_msgs/msg/Odometry`      | Odometry from pose tracking           |
| `/rs_t265/yolo_detector_node`     | `sensor_msgs/msg/Image`      | Annotated image with YOLO detections  |

---

## 🎥 Visualize in RViz2
```bash
rviz2
```
Add the following topics:
- `/rs_t265/fisheye_left` → Image
- `/rs_t265/yolo_detector_node` → Image
- `/rs_t265/odom` → Odometry
- `/tf` → TF

---

## ⚖️ Model Evaluation

You can evaluate your model using:
```python
from ultralytics import YOLO
model = YOLO("yolov9m_person_detection.pt")
metrics = model.val(data="./validation_person_dataset/data.yaml", imgsz=640, conf=0.25, iou=0.5)
print(metrics)
```

---

## 📅 Model Export (PT ➔ ONNX)
```python
from ultralytics import YOLO
model = YOLO("yolov9m_person_detection.pt")
model.export(format="onnx")
```
This is required to run inference inside the ROS node with ONNX Runtime.

---

## 🌐 Project Structure
```
yolo_face_detection/
├── launch/
│   └── yolo_detector_launch.py
├── scripts/
│   ├── yolo_detector.py
│   └── yolo_detector_optimized.py
├── src/
│   └── t265_node.cpp
├── yolov9m.onnx
├── yolov9m_person_detection.onnx
├── CMakeLists.txt
└── package.xml
```

---

## 💡 Future Improvements
- CUDA stream optimization
- TensorRT integration
- Multi-class YOLO head support
- Support for right fisheye camera

---

## 📢 License
This project is licensed under **MIT**. Attribution for pre-trained models belongs to **Ultralytics (YOLOv9)**.

---

## 🔍 References
- [YOLOv9 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [ONNX Runtime](https://onnxruntime.ai/)

---

> Developed by **Ruben Casal** ✨

