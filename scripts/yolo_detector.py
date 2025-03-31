#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data
from threading import Thread
from queue import Queue

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")
        self.bridge = CvBridge()

     
        model_path = os.path.join(os.path.dirname(__file__), "yolov9m.onnx")
        self.image_size = (640, 640)

        # Load trained YOLOv9 model
        self.model = YOLO(model_path, task='detect')
        # Subscribe to fisheye camera topic
        self.subscription = self.create_subscription(
            Image,
            "/rs_t265/fisheye_left",
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publisher for annotated images
        self.publisher = self.create_publisher(Image, "/rs_t265/yolo_detector_node", 1)

        self.get_logger().info(f"✅ YOLO detector node initialized with model: {model_path}")

    def yolo_detect(self, input_image):
        """Preprocess image and run YOLOv9 inference"""
        # Convert grayscale to RGB if needed
        if len(input_image.shape) == 2 or input_image.shape[2] == 1:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        

        # Run inference
        results = self.model.predict(
        source=input_image,
        conf=0.5,
        iou=0.4,
        imgsz=self.image_size,  # force internal shape
        verbose=False
)
        # Overlay predictions on original input
        annotated = results[0].plot()
        return annotated


    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            # Run YOLO and get annotated frame
            detected_img = self.yolo_detect(cv_image)

            # Convert back to ROS Image and publish (rgb8 format)
            ros_msg = self.bridge.cv2_to_imgmsg(detected_img, encoding="rgb8")
            ros_msg.header = msg.header
            self.publisher.publish(ros_msg)

            self.get_logger().info("📦 Published detection result")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
