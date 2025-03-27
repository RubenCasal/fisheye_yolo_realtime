from ultralytics import YOLO

model = YOLO("yolov9m_person_detection.pt")
model.export(format="onnx")
