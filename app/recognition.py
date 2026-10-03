import cv2
import os
import re
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine

from app.config import BASE_DIR, FACES_DIR, FORENSIC_DIR, load_thresholds_config
from app.database import (
    db_save_face, db_get_all_faces, db_delete_face,
    db_add_plate_watchlist, db_get_plate_watchlist, db_delete_plate_watchlist
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False)
except Exception as _e:
    print(f"[ANPR] EasyOCR reader warning: {_e}")
    _ocr_reader = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# 1. REAL FACE DETECTION (OpenCV YuNet with Haar Fallback)
# ============================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_MODELS_DIR = os.path.join(MODELS_DIR, "face")
PLATE_MODELS_DIR = os.path.join(MODELS_DIR, "plate")
os.makedirs(FACE_MODELS_DIR, exist_ok=True)
os.makedirs(PLATE_MODELS_DIR, exist_ok=True)

YUNET_PATH = os.path.join(FACE_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
ARCFACE_PATH = os.path.join(FACE_MODELS_DIR, "w600k_mbf.onnx")
PLATE_YOLO_PATH = os.path.join(PLATE_MODELS_DIR, "best.pt")


class FaceDetectorYuNet:
    def __init__(self, model_path=YUNET_PATH):
        self.yunet = None
        self.haar = None
        self.engine_name = "NONE"

        if os.path.exists(model_path) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.60,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU
                )
                self.engine_name = "YUNET_DEEP_DETECTOR"
                print(f"[RECOGNITION] OpenCV YuNet face detector initialized from {model_path}")
            except Exception as e:
                print(f"[RECOGNITION] YuNet initialization error: {e}")
                self.yunet = None

        if self.yunet is None:
            try:
                cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.haar = cv2.CascadeClassifier(cascade_file)
                self.engine_name = "HAAR_CASCADE_FALLBACK"
                print("[RECOGNITION] Operating with OpenCV Haar Cascade fallback detector.")
            except Exception:
                self.haar = None
                self.engine_name = "MORPHOLOGICAL_FALLBACK"

    def detect_face(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return None, None

        h, w = bgr_image.shape[:2]
        if h < 20 or w < 20:
            return None, None

        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, faces = self.yunet.detect(bgr_image)
                if faces is not None and len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[14])
                    fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                    mx1 = max(0, fx - int(fw * 0.10))
                    my1 = max(0, fy - int(fh * 0.10))
                    mx2 = min(w, fx + fw + int(fw * 0.10))
                    my2 = min(h, fy + fh + int(fh * 0.10))
                    face_crop = bgr_image[my1:my2, mx1:mx2]
                    if face_crop.size > 0:
                        return face_crop, (mx1, my1, mx2, my2)
                return None, None
            except Exception:
                pass

        if self.haar is not None:
            try:
                gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(28, 28))
                if len(faces) > 0:
                    fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    face_crop = bgr_image[fy:fy + fh, fx:fx + fw]
                    return face_crop, (fx, fy, fx + fw, fy + fh)
            except Exception:
                pass

        return None, None


face_detector = FaceDetectorYuNet()


# ============================================================
# 2. FACIAL EMBEDDING & DUAL-LIST MATCHER (ArcFace ONNX)
# ============================================================

class ArcFaceRecognizer:
    def __init__(self, model_path=ARCFACE_PATH):
        self.session = None
        self.input_name = None
        self.using_fallback = False

        if ort is not None and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.using_fallback = False
                print(f"[RECOGNITION] FACE ENGINE: ARCFACE ({model_path})")
            except Exception as e:
                print(f"[RECOGNITION] ArcFace load error: {e}. Activating fallback embedding.")
                self.session = None
                self.using_fallback = True
        else:
            self.session = None
            self.using_fallback = True

        if self.using_fallback:
            print("[RECOGNITION] FACE ENGINE: FALLBACK (NOT DEEP LEARNING - Sobel Gradient 512-D)")

        self.registry = {}
        self.reload_biometric_registry()

    @property
    def status_string(self):
        return "ARCFACE" if not self.using_fallback else "FALLBACK (NOT DEEP LEARNING)"

    def extract_face_embedding(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        if not self.using_fallback and self.session is not None:
            try:
                resized = cv2.resize(face_bgr, (112, 112))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                normalized = (rgb - 127.5) / 128.0
                tensor = np.transpose(normalized, (2, 0, 1))
                tensor = np.expand_dims(tensor, axis=0)

                outputs = self.session.run(None, {self.input_name: tensor})
                embedding = outputs[0][0].astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                return embedding
            except Exception as e:
                print(f"[RECOGNITION] ArcFace inference error: {e}")

        # Fallback: Classical normalized 512-D spatial gradient embedding
        aligned = cv2.resize(face_bgr, (128, 128))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        vec = []
        for r in range(0, 128, 16):
            for c in range(0, 128, 16):
                sub_mag = mag[r:r + 16, c:c + 16]
                sub_ang = ang[r:r + 16, c:c + 16]
                hist, _ = np.histogram(sub_ang, bins=8, range=(0, 360), weights=sub_mag)
                vec.extend(hist)
        emb = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

    def reload_biometric_registry(self):
        self.registry.clear()
        records = db_get_all_faces()
        for name, category, _, _, emb_bytes in records:
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                self.registry[name.upper()] = {
                    "category": category.upper(),
                    "embedding": emb
                }

    def match_face_embedding(self, query_embedding):
        if query_embedding is None or len(self.registry) == 0:
            return False, "UNKNOWN", "UNKNOWN", 0.0

        thresholds = load_thresholds_config().get("recognition", {})
        threshold = thresholds.get("face_match_cosine_threshold", 0.52)

        best_score = -1.0
        best_name = "UNKNOWN"
        best_category = "UNKNOWN"

        for name, data in self.registry.items():
            target_emb = data["embedding"]
            dist = cosine(query_embedding, target_emb)
            sim = 1.0 - dist if not np.isnan(dist) else 0.0

            if sim > best_score:
                best_score = sim
                best_name = name
                best_category = data["category"]

        if best_score >= threshold:
            return True, best_name, best_category, round(float(best_score), 3)

        return False, "UNKNOWN", "UNKNOWN", round(float(best_score), 3)

    def enrol_face(self, name: str, category: str, face_bgr):
        clean_name = name.strip().upper()
        clean_cat = category.strip().upper()
        if clean_cat not in ["WHITELIST", "WATCHLIST"]:
            clean_cat = "WHITELIST"

        embedding = self.extract_face_embedding(face_bgr)
        if embedding is None:
            return False, "Could not extract facial embedding from provided image."

        save_path = os.path.join(FACES_DIR, f"{clean_cat}_{clean_name}.jpg")
        cv2.imwrite(save_path, face_bgr)

        from datetime import datetime
        enrolled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db_save_face(clean_name, clean_cat, enrolled_at, save_path, embedding.tobytes())
        self.registry[clean_name] = {
            "category": clean_cat,
            "embedding": embedding
        }
        return True, f"Subject {clean_name} enrolled successfully into {clean_cat}."

    def delete_face(self, category: str, name: str):
        success = db_delete_face(category, name)
        if success:
            self.registry.pop(name.upper(), None)
        return success


face_recognizer = ArcFaceRecognizer()


# ============================================================
# 3. INDIAN LICENSE PLATE NORMALIZER & ANPR PIPELINE
# ============================================================

INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$')
BHARAT_SERIES_REGEX = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')

CHAR_TO_NUM = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0'}
NUM_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

_plate_yolo_model = None
if os.path.exists(PLATE_YOLO_PATH) and YOLO is not None:
    try:
        _plate_yolo_model = YOLO(PLATE_YOLO_PATH)
        print(f"[ANPR] Dedicated YOLO Plate Detector loaded from {PLATE_YOLO_PATH}")
    except Exception as e:
        print(f"[ANPR] Could not load YOLO plate detector: {e}")
        _plate_yolo_model = None


def normalize_indian_plate_ocr(raw_text: str):
    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    if len(clean) < 7 or len(clean) > 11:
        return clean

    chars = list(clean)

    # First 2 must be alphabetic
    for i in range(min(2, len(chars))):
        if chars[i] in NUM_TO_CHAR:
            chars[i] = NUM_TO_CHAR[chars[i]]

    # Next 2 must be numeric
    for i in range(2, min(4, len(chars))):
        if chars[i] in CHAR_TO_NUM:
            chars[i] = CHAR_TO_NUM[chars[i]]

    # Last 4 must be numeric
    for i in range(max(4, len(chars) - 4), len(chars)):
        if chars[i] in CHAR_TO_NUM:
            chars[i] = CHAR_TO_NUM[chars[i]]

    return "".join(chars)


def is_valid_indian_plate(plate_str: str):
    if not plate_str or len(plate_str) < 8 or len(plate_str) > 11:
        return False
    return bool(INDIAN_PLATE_REGEX.match(plate_str) or BHARAT_SERIES_REGEX.match(plate_str))


def detect_plate_region(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    vh, vw = vehicle_crop.shape[:2]

    # Mode 1: Dedicated trained YOLO plate detector if available
    if _plate_yolo_model is not None:
        try:
            res = _plate_yolo_model(vehicle_crop, conf=0.35, verbose=False)
            if res and len(res) > 0 and len(res[0].boxes) > 0:
                best = max(res[0].boxes, key=lambda b: float(b.conf[0]))
                bx = best.xyxy[0].cpu().numpy().astype(int)
                crop = vehicle_crop[max(0, bx[1]):min(vh, bx[3]), max(0, bx[0]):min(vw, bx[2])]
                if crop.size > 0:
                    return crop
        except Exception:
            pass

    # Mode 2: Classical contour-based localization
    search_region = vehicle_crop[int(vh * 0.35):vh, :]
    if search_region.size == 0:
        return None

    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)

    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / max(1, h)
        area = w * h
        if 2.2 <= aspect <= 5.5 and 800 <= area <= (vh * vw * 0.45):
            candidates.append((area, search_region[y:y + h, x:x + w]))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    return vehicle_crop[int(vh * 0.60):vh, int(vw * 0.15):int(vw * 0.85)]


class ANPRVotingTracker:
    def __init__(self):
        self.track_readings = defaultdict(list)
        self.confirmed_plates = {}

    def add_reading(self, track_id: int, plate_text: str, confidence: float):
        if not plate_text or plate_text == "UNREADABLE":
            return

        self.track_readings[track_id].append((plate_text, confidence))
        if len(self.track_readings[track_id]) > 10:
            self.track_readings[track_id].pop(0)

        candidates = [p for p, c in self.track_readings[track_id] if c >= 0.35]
        if candidates:
            counts = Counter(candidates)
            most_common_plate, freq = counts.most_common(1)[0]
            if freq >= 2 or (len(candidates) == 1 and self.track_readings[track_id][-1][1] >= 0.60):
                self.confirmed_plates[track_id] = most_common_plate

    def get_plate(self, track_id: int):
        return self.confirmed_plates.get(track_id, "UNREADABLE")

    def prune(self, active_track_ids: set):
        for tid in list(self.track_readings.keys()):
            if tid not in active_track_ids:
                self.track_readings.pop(tid, None)
                self.confirmed_plates.pop(tid, None)


anpr_voting_tracker = ANPRVotingTracker()


def extract_vehicle_plate(vehicle_crop, track_id: int):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "UNREADABLE"

    if _ocr_reader is None:
        return "OCR_UNAVAILABLE"

    plate_crop = detect_plate_region(vehicle_crop)
    if plate_crop is None or plate_crop.size == 0:
        return "UNREADABLE"

    ph, pw = plate_crop.shape[:2]
    if ph < 50:
        scale = 60.0 / ph
        plate_crop = cv2.resize(plate_crop, (int(pw * scale), 60), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    try:
        results = _ocr_reader.readtext(equalized)
    except Exception:
        return "UNREADABLE"

    if not results:
        return "UNREADABLE"

    best_match = max(results, key=lambda r: r[2])
    raw_ocr_text = best_match[1]
    confidence = float(best_match[2])

    thresholds = load_thresholds_config().get("recognition", {})
    min_conf = thresholds.get("anpr_min_ocr_confidence", 0.35)

    if confidence < min_conf:
        return "UNREADABLE"

    normalized_plate = normalize_indian_plate_ocr(raw_ocr_text)

    if not is_valid_indian_plate(normalized_plate):
        return "UNREADABLE"

    anpr_voting_tracker.add_reading(track_id, normalized_plate, confidence)

    snap_path = os.path.join(FORENSIC_DIR, f"plate_{track_id}_{normalized_plate}.jpg")
    cv2.imwrite(snap_path, plate_crop)

    return anpr_voting_tracker.get_plate(track_id)


def check_plate_watchlist(plate_number: str):
    if not plate_number or plate_number in ["UNREADABLE", "OCR_UNAVAILABLE"]:
        return False, ""

    watchlist = db_get_plate_watchlist()
    clean_target = re.sub(r'[^A-Za-z0-9]', '', plate_number).upper()

    for item in watchlist:
        clean_wl = re.sub(r'[^A-Za-z0-9]', '', item["plate_number"]).upper()
        if clean_target == clean_wl:
            return True, item["reason"]

    return False, ""