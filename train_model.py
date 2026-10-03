import torch
from ultralytics import YOLO

def start_training():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on: {device}")

    model = YOLO("yolov8n.pt")  # Base YOLOv8

    model.train(
        data="models/yolov8/data.yaml",
        epochs=30,             # Fast demo training
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project="models/yolov8",
        name="self_trained_defense"
    )
    print("\n[✓] Training Complete! Model saved at: models/yolov8/self_trained_defense/weights/best.pt")

if __name__ == "__main__":
    start_training()