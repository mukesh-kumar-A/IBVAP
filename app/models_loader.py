import os
import cv2
from ultralytics import YOLO

class ModelRegistry:
    def __init__(self):
        # Pointing to custom trained weights with base YOLO fallback
        custom_weights = "models/yolov8/self_trained_defense/weights/best.pt"
        if not os.path.exists(custom_weights):
            custom_weights = "yolov8n.pt"
            
        self.yolo_model = YOLO(custom_weights)
        
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_detector = None

model_registry = ModelRegistry()