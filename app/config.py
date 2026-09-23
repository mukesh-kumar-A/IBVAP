import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORENSIC_DIR = os.path.join(BASE_DIR, "forensic_logs")
FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DB_PATH = os.path.join(BASE_DIR, "defense_audit.db")

os.makedirs(FORENSIC_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# Direct Backend Ingestion Mapping (Laptop Webcam + DroidCam IP Feeds)
CAMERA_SOURCES = {
    "CAM-01": 0,
    "CAM-02": "http://192.168.43.1:4747/video",
    "CAM-03": "http://192.168.43.2:4747/video"
}

# Detection Class Indices & Thresholds
CLASS_THRESHOLDS = {
    0: 0.35,   # Human / Person
    2: 0.35, 3: 0.35, 5: 0.35, 7: 0.35, # Vehicles (Car, Motorcycle, Bus, Truck)
    67: 0.28,  # Cell Phone
    14: 0.40, 15: 0.40, 16: 0.40, 17: 0.40, 18: 0.40, 19: 0.40 # Wildlife
}

# Operational Baseline Configuration
CONFIG_STATE = {
    "boundary_percentage": 45,
    "night_enhancement": False,
    "loiter_threshold_sec": 3.0,
    "current_cam_key": "CAM-01",
    "current_cam_label": "CAM-01 (Primary Optical Sensor)"
}