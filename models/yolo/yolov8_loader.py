# models/yolo/yolov8_loader.py

from ultralytics import YOLO

def load_yolo(model_name="yolov8n.pt"):
    """
    Load YOLOv8 model
    """
    model = YOLO(model_name)
    return model
