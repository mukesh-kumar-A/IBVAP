import os
import cv2
from ultralytics import YOLO
from app.config import BASE_DIR, load_system_config

try:
    import torch
except ImportError:
    torch = None


class ModelRegistry:
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        sys_cfg = load_system_config()
        hw_cfg = sys_cfg.get("hardware_optimization", {})

        self.onnx_threads = int(hw_cfg.get("onnx_threads", 2))
        self.use_onnx_yolo = bool(hw_cfg.get("use_onnx_yolo", False))

        # Apply CPU thread limit globally to avoid multi-worker thread starvation
        if torch is not None:
            try:
                torch.set_num_threads(self.onnx_threads)
            except Exception:
                pass

        # Candidate paths for models
        self.default_model_path = os.path.join(self.BASE_DIR, "yolov8n.pt")
        self.onnx_model_path = os.path.join(self.BASE_DIR, "yolov8n.onnx")

        # Check candidate locations for self-trained defense model if present
        custom_candidates = [
            os.path.join(self.BASE_DIR, "models", "yolov8", "self_trained_defense", "weights", "best.pt"),
            os.path.join(self.BASE_DIR, "runs", "detect", "models", "yolov8", "self_trained_defense", "weights", "best.pt")
        ]

        self.custom_weights_found = None
        for candidate in custom_candidates:
            if candidate and os.path.exists(candidate):
                self.custom_weights_found = candidate
                break

        # Select weights path: custom if found, otherwise ONNX (if enabled and exists) or COCO yolov8n.pt
        if self.custom_weights_found:
            self.active_weights_path = self.custom_weights_found
            self.using_custom_model = True
            print(f"[IBVAP] Found custom defense weights: {self.active_weights_path}")
        elif self.use_onnx_yolo and os.path.exists(self.onnx_model_path):
            self.active_weights_path = self.onnx_model_path
            self.using_custom_model = False
            print(f"[IBVAP] Using ONNX YOLO model: {self.active_weights_path}")
        else:
            if not os.path.exists(self.default_model_path):
                self.default_model_path = "yolov8n.pt"
            self.active_weights_path = self.default_model_path
            self.using_custom_model = False
            print(f"[IBVAP] Using default YOLOv8 model: {self.active_weights_path}")

        # Primary model instance for introspection
        self.yolo_model = YOLO(self.active_weights_path)
        print(f"[IBVAP] Model classes initialized: {len(self.yolo_model.names)} detected")

    def create_isolated_model(self):
        """
        Creates an independent YOLO model instance for each camera worker.
        Guarantees ByteTrack Kalman filter and trajectory states remain isolated per stream.
        """
        return YOLO(self.active_weights_path)


model_registry = ModelRegistry()