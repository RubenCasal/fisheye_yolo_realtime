from ultralytics import YOLO

model = YOLO("yolov9m_person_detection.pt")  # or .onnx or last.pt

metrics = model.val(
    data="./validation_person_dataset/data.yaml",  # YAML with val images + labels
    imgsz=640,                 # image size
    conf=0.25,                 # confidence threshold
    iou=0.5,                   # IoU threshold
    split="val",              # or 'test'
    verbose=True
)

print(metrics)  # dict with mAP, precision, recall, etc.
