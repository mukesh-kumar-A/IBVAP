import cv2
import numpy as np
from app.config import CLASS_THRESHOLDS
from app.models_loader import model_registry

# 1. CIELAB Night Vision Enhancement (CLAHE + Retinex Auto-Gain)
def apply_clahe_night_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray) + 1e-3
    gain = np.clip(110.0 / mean_val, 1.2, 14.0)
    boosted = cv2.convertScaleAbs(frame, alpha=gain, beta=30)

    gamma = 0.35
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_frame = cv2.LUT(boosted, table)

    lab = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)

# 2. Multi-Mode Optical Anti-Tampering Engine (<50ms trigger)
def check_lens_tampering(frame, night_mode):
    if frame is None or frame.size == 0:
        return True, "SIGNAL_LOST"
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luminance = float(np.mean(gray))
    std_luminance = float(np.std(gray))

    # Case A: Lens Covered / Blackout
    if mean_luminance < 14.0 and laplacian_var < 18.0:
        return True, "CAMERA LENS COVERED / BLACKOUT"

    # Case B: Blinding by Laser / Flash
    if mean_luminance > 240.0 and std_luminance < 12.0:
        return True, "CAMERA BLINDED BY LASER / FLASH"

    # Case C: Extreme Defocus / Occlusion
    if laplacian_var < 8.0 and not night_mode:
        return True, "CAMERA DEFOCUSED / SPRAY COVERED"

    return False, "CLEAR"

# 3. Object Detection Pipeline
def run_object_detection(frame):
    target_classes = list(CLASS_THRESHOLDS.keys())
    results = model_registry.yolo_model.predict(
        frame, imgsz=480, conf=0.25, classes=target_classes, verbose=False
    )
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        b = results[0].boxes.xyxy.cpu().numpy().astype(int)
        c = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        for idx, (box, conf, cl) in enumerate(zip(b, c, cls)):
            if conf < CLASS_THRESHOLDS.get(cl, 0.35):
                continue
            track_id = idx + 1
            if cl == 0: label = "PERSON"
            elif cl == 2: label = "CAR"
            elif cl == 3: label = "MOTORCYCLE"
            elif cl == 5: label = "BUS"
            elif cl == 7: label = "TRUCK"
            elif cl == 67: label = "CELL PHONE"
            elif cl in [14, 15, 16, 17, 18, 19]: label = "WILDLIFE"
            else: label = model_registry.yolo_model.names[cl].upper()
            
            detections.append((box, conf, label, track_id, cl))
    return detections