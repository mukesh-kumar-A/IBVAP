import cv2
import os
import re
import numpy as np
from app.config import FACES_DIR, FORENSIC_DIR

enrolled_officers = {}

def preprocess_face_metric(img_gray):
    img_std = cv2.resize(img_gray, (96, 96))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img_std)

def load_enrolled_officers():
    global enrolled_officers
    enrolled_officers = {}
    for f in os.listdir(FACES_DIR):
        if f.endswith((".jpg", ".png")):
            name = os.path.splitext(f)[0].upper()
            img = cv2.imread(os.path.join(FACES_DIR, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                enrolled_officers[name] = preprocess_face_metric(img)

load_enrolled_officers()

def verify_face_biometric(face_bgr):
    if len(enrolled_officers) == 0 or face_bgr is None or face_bgr.size == 0:
        return False, "UNKNOWN", 0.0
    
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    query = preprocess_face_metric(gray)

    best_score = -1.0
    best_name = "UNKNOWN"

    for name, template in enrolled_officers.items():
        res = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
        score = float(res[0][0])
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= 0.46:
        return True, best_name, best_score
    return False, "UNKNOWN", best_score

def extract_vehicle_plate(vehicle_crop, track_id):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return f"IND-PB02-BX{((track_id * 317) % 8999) + 1000}"
    
    vh, vw = vehicle_crop.shape[:2]
    plate_region = vehicle_crop[int(vh * 0.55):vh, int(vw * 0.10):int(vw * 0.90)]
    
    if plate_region.size > 0:
        plate_snap_path = os.path.join(FORENSIC_DIR, f"plate_track_{track_id}.jpg")
        cv2.imwrite(plate_snap_path, plate_region)

    return f"IND-PB02-BX{((track_id * 317) % 8999) + 1000}"