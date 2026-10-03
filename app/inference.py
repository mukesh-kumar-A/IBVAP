import cv2
import math
import time
import numpy as np


# ============================================================
# 1. DYNAMIC CLASS RESOLUTION BY NAME
# ============================================================

def resolve_detection_class(raw_name: str):
    name = str(raw_name).lower().strip()

    if name in ["person", "human", "people"]:
        return "PERSON", "person"

    if name in ["car", "automobile"]:
        return "CAR", "vehicle"
    if name in ["motorcycle", "motorbike"]:
        return "MOTORCYCLE", "vehicle"
    if name in ["bus"]:
        return "BUS", "vehicle"
    if name in ["truck"]:
        return "TRUCK", "vehicle"
    if name in ["bicycle", "bike"]:
        return "BICYCLE", "vehicle"
    if name in ["van"]:
        return "VAN", "vehicle"
    if name in ["vehicle"]:
        return "VEHICLE", "vehicle"

    if name in ["phone", "cell phone", "cellphone", "mobile", "mobile phone"]:
        return "CELL PHONE", "phone"

    if name in ["backpack", "handbag", "suitcase", "bag", "package"]:
        return "BAGGAGE", "baggage"

    if name in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
        return "WILDLIFE", "wildlife"

    return name.upper(), "other"


# ============================================================
# 2. DAY/NIGHT DISCRIMINATOR & NIGHT VISION ENHANCEMENT
# ============================================================

def detect_day_night_mode(frame, sat_threshold=32.0, bright_threshold=50.0):
    """
    Evaluates color saturation (S channel) and luminance (V channel) in HSV space.
    Monochrome IR feeds have near-zero saturation; low ambient light has low brightness.
    Returns: (is_night: bool, mean_brightness: float, mean_saturation: float)
    """
    if frame is None or frame.size == 0:
        return False, 0.0, 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_bright = float(np.mean(hsv[:, :, 2]))

    is_night = (mean_sat < sat_threshold) or (mean_bright < bright_threshold)
    return is_night, mean_bright, mean_sat


def apply_clahe_night_vision(frame):
    """
    Contrast-Limited Adaptive Histogram Equalization with gamma correction
    and unsharp masking for enhanced target visibility in dim environments.
    """
    if frame is None or frame.size == 0:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray) + 1e-3
    gain = np.clip(115.0 / mean_val, 1.2, 14.0)

    boosted = cv2.convertScaleAbs(frame, alpha=gain, beta=28)

    gamma = 0.38
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    gamma_frame = cv2.LUT(boosted, table)
    lab = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(enhanced, -1, kernel)


# ============================================================
# 3. NIGHT MOTION SUBTRACTOR WITH NOISE FILTERING
# ============================================================

class NightMotionDetector:
    def __init__(self, history=50, var_threshold=25, min_area=600):
        self.min_area = min_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect_motion(self, frame):
        if frame is None or frame.size == 0:
            return False, [], 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        fg_mask = self.subtractor.apply(blurred)

        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        total_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                motion_boxes.append([x, y, x + w, y + h])
                total_area += area

        return (len(motion_boxes) > 0), motion_boxes, total_area


# ============================================================
# 4. CAMERA ANTI-TAMPERING DETECTION ENGINE
# ============================================================

_shift_state = {}

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

_SHIFT_MIN_POINTS = 15
_HANN = cv2.createHanningWindow((160, 120), cv2.CV_32F)

_ANGLE_SHIFT_PX = 14.0
_ANGLE_ROT_DEG = 4.0
_JOLT_SHIFT_PX = 16.0
_ANGLE_FRAMES = 8
_JOLT_FRAMES = 2
_REBASELINE_SEC = 20.0


def _phase_guess(a_gray, b_gray):
    a = cv2.resize(a_gray, (160, 120)).astype(np.float32)
    b = cv2.resize(b_gray, (160, 120)).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, _HANN)
    if resp > 0.05:
        return dx * 2.0, dy * 2.0
    return 0.0, 0.0


def _global_motion(prev_gray, prev_pts, cur_gray):
    if prev_pts is None or len(prev_pts) < _SHIFT_MIN_POINTS:
        return None

    gx, gy = _phase_guess(prev_gray, cur_gray)
    guess = prev_pts + np.array([gx, gy], dtype=np.float32).reshape(1, 1, 2)

    cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_pts, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK_PARAMS
    )
    if cur_pts is None:
        return None

    good = status.reshape(-1) == 1
    if int(good.sum()) < _SHIFT_MIN_POINTS:
        return None

    p0 = prev_pts[good]
    p1 = cur_pts[good]
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None:
        return None

    inliers = int(inl.sum())
    if inliers < _SHIFT_MIN_POINTS or inliers < 0.5 * len(p0):
        return None

    trans = math.hypot(float(M[0, 2]), float(M[1, 2]))
    rot = abs(math.degrees(math.atan2(float(M[1, 0]), float(M[0, 0]))))
    return trans, rot, inliers


def _find_points(gray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.005, minDistance=6)


def _detect_camera_shift(gray, cam_id):
    now = time.time()
    small = cv2.resize(gray, (320, 240))
    st = _shift_state.get(cam_id)

    if st is None:
        _shift_state[cam_id] = {
            "ref_gray": small,
            "ref_pts": _find_points(small),
            "prev_gray": small,
            "prev_pts": _find_points(small),
            "angle_count": 0,
            "jolt_count": 0,
            "shifted_since": None,
            "ref_time": now
        }
        return False, "CLEAR"

    jolt = _global_motion(st["prev_gray"], st["prev_pts"], small)
    st["prev_gray"] = small
    st["prev_pts"] = _find_points(small)

    if jolt is not None and jolt[0] > _JOLT_SHIFT_PX:
        st["jolt_count"] += 1
    else:
        st["jolt_count"] = 0

    if st["jolt_count"] >= _JOLT_FRAMES:
        return True, "CAMERA SHAKEN / SUDDEN MOVEMENT"

    mot = _global_motion(st["ref_gray"], st["ref_pts"], small)
    if mot is not None and (mot[0] > _ANGLE_SHIFT_PX or mot[1] > _ANGLE_ROT_DEG):
        st["angle_count"] += 1
    else:
        st["angle_count"] = 0

    if st["angle_count"] >= _ANGLE_FRAMES:
        if st["shifted_since"] is None:
            st["shifted_since"] = now
        if now - st["shifted_since"] > _REBASELINE_SEC:
            st.update(
                ref_gray=small,
                ref_pts=_find_points(small),
                angle_count=0,
                shifted_since=None,
                ref_time=now
            )
            return False, "CLEAR"
        return True, "CAMERA ANGLE CHANGED / REPOSITIONED"

    if st["angle_count"] == 0:
        st["shifted_since"] = None
        if now - st["ref_time"] > 5.0:
            st.update(
                ref_gray=small,
                ref_pts=_find_points(small),
                ref_time=now
            )

    return False, "CLEAR"


def check_lens_tampering(frame, night_mode, cam_id="default"):
    """
    Multi-factor lens anti-tamper heuristics:
      1. Blackout / Hand covering lens: very low luminance + minimal texture.
      2. Direct Laser / Blinding flashlight: saturated luminance + low deviation.
      3. Defocus / Spray paint: loss of high-frequency edge gradients (Laplacian variance).
      4. Camera displacement / tilt: optical flow disparity.
    """
    if frame is None or frame.size == 0:
        return True, "SIGNAL_LOST"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luminance = float(np.mean(gray))
    std_luminance = float(np.std(gray))

    # Hand covering camera / Black cloth covering lens
    if mean_luminance < 28.0 and laplacian_var < 35.0:
        return True, "CAMERA LENS COVERED / BLACKOUT"

    # Laser blinding / Intense focused torch
    if mean_luminance > 238.0 and std_luminance < 14.0:
        return True, "CAMERA BLINDED BY LASER / FLASH"

    # Spray paint / Vaseline defocus covering lens
    if laplacian_var < 9.0 and not night_mode and mean_luminance > 30.0:
        return True, "CAMERA DEFOCUSED / SPRAY COVERED"

    # Camera jolt or mechanical angle movement
    shifted, shift_reason = _detect_camera_shift(gray, cam_id)
    if shifted:
        return True, shift_reason

    return False, "CLEAR"


# ============================================================
# 5. BYTETRACK OBJECT TRACKING PIPELINE (WITH SCALING SUPPORT)
# ============================================================

def run_tracked_detection(yolo_model, frame, target_size=None):
    if frame is None or frame.size == 0:
        return []

    orig_h, orig_w = frame.shape[:2]

    if target_size is not None and (target_size[0] > 0 and target_size[1] > 0):
        tw, th = target_size
        scale_x = orig_w / float(tw)
        scale_y = orig_h / float(th)
        infer_frame = cv2.resize(frame, (tw, th))
    else:
        scale_x = 1.0
        scale_y = 1.0
        infer_frame = frame

    try:
        results = yolo_model.track(
            source=infer_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )
    except Exception as e:
        print(f"[IBVAP INFERENCE] ByteTrack execution error: {e}")
        return []

    detections = []
    if not results or len(results) == 0 or results[0].boxes is None:
        return detections

    boxes_obj = results[0].boxes
    if len(boxes_obj) == 0:
        return detections

    model_names = yolo_model.names
    coords = boxes_obj.xyxy.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    class_ids = boxes_obj.cls.cpu().numpy().astype(int)

    track_ids = None
    if boxes_obj.id is not None:
        track_ids = boxes_obj.id.cpu().numpy().astype(int)

    for idx, (box, conf, class_id) in enumerate(zip(coords, confs, class_ids)):
        conf_val = float(conf)
        class_id_val = int(class_id)

        x1 = int(round(box[0] * scale_x))
        y1 = int(round(box[1] * scale_y))
        x2 = int(round(box[2] * scale_x))
        y2 = int(round(box[3] * scale_y))

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(x1 + 1, min(orig_w, x2))
        y2 = max(y1 + 1, min(orig_h, y2))

        if isinstance(model_names, dict):
            raw_name = model_names.get(class_id_val, str(class_id_val))
        else:
            raw_name = model_names[class_id_val] if class_id_val < len(model_names) else str(class_id_val)

        display_label, category_group = resolve_detection_class(raw_name)

        if track_ids is not None and idx < len(track_ids):
            assigned_id = int(track_ids[idx])
        else:
            assigned_id = idx + 1

        detections.append({
            "box": [x1, y1, x2, y2],
            "conf": conf_val,
            "label": display_label,
            "category": category_group,
            "track_id": assigned_id,
            "class_id": class_id_val,
            "raw_name": str(raw_name).lower().strip()
        })

    return detections