from ultralytics import YOLO
import os

# Load YOLOv12-nano (assuming weights are available locally or from ultralytics hub)
model = YOLO("yolo12n.pt")

model.export(**{"format": "onnx", "imgsz": 128, "half":True})
