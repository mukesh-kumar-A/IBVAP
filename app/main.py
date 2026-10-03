import cv2
import time
import os
import io
import csv
import json
import base64
import hashlib
import secrets
import threading
import asyncio
import urllib.request
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from app.config import (
    CONFIG_STATE, FORENSIC_DIR, FACES_DIR,
    load_camera_configs, save_camera_configs,
    load_zones_config, save_zones_config,
    load_thresholds_config, save_thresholds_config,
    load_system_config
)
from app.models_loader import model_registry
from app.inference import (
    apply_clahe_night_vision, check_lens_tampering,
    run_tracked_detection, detect_day_night_mode, NightMotionDetector
)
from app.recognition import (
    face_detector, face_recognizer,
    extract_vehicle_plate, check_plate_watchlist,
    anpr_voting_tracker, db_get_plate_watchlist,
    db_add_plate_watchlist, db_delete_plate_watchlist
)
from app.tracker import TrackTrajectoryManager, box_intersects_zone, check_direction_constraint
from app.analytics import compute_threat_level, compute_risk_score
from app.database import (
    log_threat_event, update_hitl_status, verify_chain_integrity,
    db_get_all_faces, query_filtered_events
)
from app.websocket_manager import ws_manager
from app.alerts import alert_dispatcher

app = FastAPI(
    title="IBVAP 2.0 Sovereign Defense Enterprise Hub",
    description="Tactical Command and Control API for automated border surveillance, forensic auditing, and C2 integration.",
    version="2.4.0"
)

# ----------------- Dynamic Credentials & Salt Management -----------------
TOKEN_EXPIRY_HOURS = int(load_system_config().get("system", {}).get("token_expiry_hours", 12))

_admin_env = os.getenv("IBVAP_ADMIN_PASSWORD")
_ops_env = os.getenv("IBVAP_OPERATOR_PASSWORD")
_admin_salt_env = os.getenv("IBVAP_ADMIN_SALT")
_ops_salt_env = os.getenv("IBVAP_OPERATOR_SALT")

if not _admin_env:
    ADMIN_RAW_PASS = secrets.token_urlsafe(12)
    print("\n" + "!" * 70)
    print(f" [SECURITY] No IBVAP_ADMIN_PASSWORD set. Generated random admin secret:")
    print(f"   Username: admin")
    print(f"   Password: {ADMIN_RAW_PASS}")
    print("!" * 70 + "\n")
else:
    ADMIN_RAW_PASS = _admin_env

if not _ops_env:
    OPS_RAW_PASS = secrets.token_urlsafe(12)
    print(f" [SECURITY] Generated random operator secret: {OPS_RAW_PASS}")
else:
    OPS_RAW_PASS = _ops_env

ADMIN_SALT = _admin_salt_env if _admin_salt_env else secrets.token_hex(8)
OPS_SALT = _ops_salt_env if _ops_salt_env else secrets.token_hex(8)


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()


USER_DATABASE = {
    "admin": {
        "role": "ADMIN",
        "salt": ADMIN_SALT,
        "password_hash": hash_password(ADMIN_RAW_PASS, ADMIN_SALT)
    },
    "operator": {
        "role": "OPERATOR",
        "salt": OPS_SALT,
        "password_hash": hash_password(OPS_RAW_PASS, OPS_SALT)
    }
}

ACTIVE_SESSIONS = {}
FAILED_LOGINS = {}

_c2_env = os.getenv("IBVAP_C2_API_KEYS") or os.getenv("IBVAP_C2_API_KEY")
if _c2_env:
    C2_API_KEYS = [k.strip() for k in _c2_env.split(",") if k.strip()]
else:
    yaml_keys = load_system_config().get("security", {}).get("c2_api_keys", [])
    if yaml_keys:
        C2_API_KEYS = yaml_keys
    else:
        generated_c2_key = f"C2-{secrets.token_hex(16).upper()}"
        C2_API_KEYS = [generated_c2_key]
        print(f" [SECURITY] Generated dynamic C2 API Key: {generated_c2_key}\n")

# ----------------- Security Dependencies -----------------
bearer_scheme = HTTPBearer(auto_error=False)
c2_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class LoginPayload(BaseModel):
    username: str = Field(..., description="Administrative or operator username")
    password: str = Field(..., description="Plaintext secret credential")


@app.post("/api/login", tags=["Authentication"], summary="User Session Authentication")
def login(payload: LoginPayload, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()

    lockout_info = FAILED_LOGINS.get(client_ip)
    if lockout_info and lockout_info["lockout_until"] > now:
        remaining = int(lockout_info["lockout_until"] - now)
        raise HTTPException(status_code=429, detail=f"Too many failed login attempts. Locked out for {remaining} seconds.")

    user = USER_DATABASE.get(payload.username.lower())
    if not user:
        _record_login_failure(client_ip, now)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    check_hash = hash_password(payload.password, user["salt"])
    if check_hash != user["password_hash"]:
        _record_login_failure(client_ip, now)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    FAILED_LOGINS.pop(client_ip, None)

    token = secrets.token_hex(24)
    expires_at = now + (TOKEN_EXPIRY_HOURS * 3600)
    ACTIVE_SESSIONS[token] = {
        "role": user["role"],
        "expires_at": expires_at
    }
    return {
        "status": "success",
        "token": token,
        "role": user["role"],
        "expires_in_hours": TOKEN_EXPIRY_HOURS
    }


def _record_login_failure(client_ip: str, now: float):
    rec = FAILED_LOGINS.setdefault(client_ip, {"count": 0, "lockout_until": 0.0})
    rec["count"] += 1
    if rec["count"] >= 5:
        rec["lockout_until"] = now + 300.0


def authenticate_user(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> dict:
    if not auth or not auth.credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer token. Use Authorization: Bearer <token>")

    token = auth.credentials
    session = ACTIVE_SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or terminated session token")

    if time.time() > session["expires_at"]:
        ACTIVE_SESSIONS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired. Please re-authenticate.")

    return session


def authenticate_download_stream(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> dict:
    token = None
    if auth and auth.credentials:
        token = auth.credentials
    else:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Authentication required for stream/download.")

    session = ACTIVE_SESSIONS.get(token)
    if not session or time.time() > session["expires_at"]:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    return session


def require_role(session: dict, required_role: str):
    role = session.get("role")
    if role != required_role:
        raise HTTPException(status_code=403, detail=f"Access denied — {required_role} role required")
    return role


def authenticate_c2_api_key(
    request: Request,
    api_key: Optional[str] = Security(c2_api_key_header),
    auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
):
    if api_key and api_key in C2_API_KEYS:
        return {"type": "C2_SYSTEM", "identity": "COMMAND_HEADQUARTERS"}

    token = None
    if auth and auth.credentials:
        token = auth.credentials
    else:
        token = request.query_params.get("token")

    if token and token in ACTIVE_SESSIONS:
        if time.time() <= ACTIVE_SESSIONS[token]["expires_at"]:
            return {"type": "USER", "role": ACTIVE_SESSIONS[token]["role"]}

    raise HTTPException(status_code=401, detail="Invalid or missing C2 credentials. Provide valid X-API-Key header.")


@app.get("/api/whoami", tags=["Authentication"], summary="Validate Current Session Token")
def whoami(session: dict = Depends(authenticate_user)):
    return {"status": "success", "role": session["role"], "valid": True}


# ----------------- Global Telemetry State -----------------
hw_settings = load_system_config().get("hardware_optimization", {})

CONFIG_STATE["night_mode"] = "AUTO"  # Options: AUTO, ON, OFF

SYSTEM_STATE = {
    "is_breached": False,
    "is_loitering": False,
    "is_tampered": False,
    "is_night": False,
    "night_enhancement_active": False,
    "night_mode_setting": "AUTO",
    "boundary_percentage": CONFIG_STATE.get("boundary_percentage", 45),
    "night_motion_alert": False,
    "tamper_reason": "CLEAR",
    "tampered_cam": "",
    "intruder_count": 0,
    "warning_count": 0,
    "total_events_count": 0,
    "active_targets": [],
    "fps": 0.0,
    "latency_ms": 0.0,
    "frs_target": "NO FACE ACQUIRED",
    "anpr_target": "NO VEHICLE DETECTED",
    "face_engine_status": face_recognizer.status_string,
    "hardware_optimization": {
        "frame_skip": hw_settings.get("frame_skip", 2),
        "inference_width": hw_settings.get("inference_width", 480),
        "inference_height": hw_settings.get("inference_height", 360),
        "use_onnx_yolo": hw_settings.get("use_onnx_yolo", False),
        "onnx_threads": hw_settings.get("onnx_threads", 2)
    },
    "interception_vector": {
        "velocity_kmh": 0.0,
        "speed_status": "CALIBRATED" if load_camera_configs()[0].get("pixels_per_meter", 0) > 0 else "ESTIMATED (uncalibrated)",
        "heading": "STATIONARY",
        "eta_bop_seconds": "N/A",
        "threat_level": "CODE GREEN",
        "risk_score": 0,
        "risk_band": "LOW"
    },
    "forensic_records": [],
    "camera_registry": [],
    "qrt_drone": {
        "status": "STANDBY",
        "simulation_mode": True,
        "disclaimer": "SIMULATED WORKFLOW — NO PHYSICAL DRONE INTEGRATED",
        "drone_id": "UAV-GARUDA-09-SIMULATED",
        "battery": 98,
        "payload": "OPTICAL-ZOOM / THERMAL POD (SIMULATED)",
        "target_sector": "PERIMETER SECURE",
        "eta_seconds": 0,
        "dispatched_at": None,
        "coordinates": "31°38'12\"N, 74°52'31\"E"
    }
}


# ----------------- Multi-Camera Ingestion Stream -----------------
class MultiCameraStream:
    def __init__(self, cam_id, src):
        self.cam_id = cam_id
        self.src = str(src).strip()
        self.cap = None
        self.frame = None
        self.status = False
        self.running = True
        self.lock = threading.Lock()

        self.is_empty = (len(self.src) == 0)
        self.is_network = not self.src.isdigit() if not self.is_empty else False
        self.is_rtsp = self.src.lower().startswith(("rtsp://", "rtmp://")) if not self.is_empty else False

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open_usb(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            actual_src = int(self.src)
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(actual_src, backend)
            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
        except Exception:
            pass
        return False

    def _open_rtsp(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
        except Exception:
            pass
        return False

    def update(self):
        if self.is_empty:
            while self.running:
                with self.lock:
                    self.status = False
                    self.frame = None
                time.sleep(2.0)
            return

        if self.is_rtsp:
            reconnect_timer = 0.0
            self._open_rtsp()
            while self.running:
                if self.cap is not None and self.cap.isOpened():
                    status, frame = self.cap.read()
                    if status and frame is not None and frame.size > 0:
                        with self.lock:
                            self.status = True
                            self.frame = frame
                    else:
                        with self.lock:
                            self.status = False
                        if time.time() - reconnect_timer > 4.0:
                            reconnect_timer = time.time()
                            self._open_rtsp()
                        time.sleep(0.05)
                else:
                    if time.time() - reconnect_timer > 4.0:
                        reconnect_timer = time.time()
                        self._open_rtsp()
                    time.sleep(0.3)
                time.sleep(0.01)
            return

        if not self.is_network:
            self._open_usb()
            reconnect_timer = 0.0
            while self.running:
                if self.cap is not None and self.cap.isOpened():
                    status, frame = self.cap.read()
                    if status and frame is not None and frame.size > 0:
                        with self.lock:
                            self.status = True
                            self.frame = frame
                    else:
                        with self.lock:
                            self.status = False
                        time.sleep(0.04)
                else:
                    if time.time() - reconnect_timer > 3.0:
                        reconnect_timer = time.time()
                        self._open_usb()
                    time.sleep(0.3)
                time.sleep(0.015)
            return

        src_url = self.src
        if not src_url.startswith(("http://", "https://", "rtsp://")):
            src_url = "http://" + src_url

        endpoints = [src_url]
        while self.running:
            connected = False
            for target_url in endpoints:
                try:
                    req = urllib.request.Request(
                        target_url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    stream = urllib.request.urlopen(req, timeout=3.0)
                    stream_bytes = b''
                    connected = True

                    while self.running:
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        stream_bytes += chunk

                        a = stream_bytes.find(b'\xff\xd8')
                        b = stream_bytes.find(b'\xff\xd9')

                        if a != -1 and b != -1:
                            jpg = stream_bytes[a:b+2]
                            stream_bytes = stream_bytes[b+2:]
                            decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is not None and decoded.size > 0:
                                with self.lock:
                                    self.status = True
                                    self.frame = decoded
                    stream.close()
                except Exception:
                    time.sleep(0.2)

                if connected:
                    break

            with self.lock:
                self.status = False
            time.sleep(3.0)

    def read(self):
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass


# ----------------- Video Clip Recorder Helpers -----------------
CLIP_PRE_SEC = 3.0
CLIP_POST_SEC = 4.0
CLIP_MAX_SEC = 20.0
TAMPER_RELOG_SEC = 20.0


def _write_clip_file(path, frames, fps):
    try:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    except Exception as e:
        print(f"[IBVAP CLIP] Failed to write {path}: {e}")


# ----------------- Dedicated Continuous Camera Background Worker -----------------
class CameraWorker:
    def __init__(self, cam_config: dict):
        self.config = dict(cam_config)
        self.cam_id = self.config["id"]
        self.stream = MultiCameraStream(self.cam_id, self.config.get("source", ""))
        self.model = model_registry.create_isolated_model()
        self.trajectory_manager = TrackTrajectoryManager()
        self.night_motion_detector = NightMotionDetector()

        self.running = True
        self.lock = threading.Lock()

        self.latest_jpeg = None
        self.fps = 0.0
        self.latency_ms = 0.0

        hw = load_system_config().get("hardware_optimization", {})
        self.frame_skip = max(1, int(hw.get("frame_skip", 2)))
        self.infer_size = (int(hw.get("inference_width", 480)), int(hw.get("inference_height", 360)))
        self.frame_counter = 0
        self.cached_detections = []

        # Operational States
        self.is_tampered = False
        self.tamper_reason = "CLEAR"
        self.is_night = False
        self.clahe_active = False
        self.night_motion_alert = False
        self.current_breach = False
        self.current_loitering = False
        self.detected_targets = []
        self.loiter_targets = []
        self.active_suspicious_events = []
        self.max_speed = 0.0
        self.lead_heading = "STATIONARY"
        self.lead_eta = "N/A"

        self.frame_buffer = deque(maxlen=150)
        self.active_clip = None
        self.loitering_trackers = {}
        self.processed_plates_map = {}
        self.last_snapshot_time = 0.0
        self.last_tamper_log = None
        self.last_night_motion_log = 0.0

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_config(self, new_config: dict):
        with self.lock:
            self.config.update(new_config)

    def _buffer_frame(self, frame, t):
        f = frame.copy()
        self.frame_buffer.append((t, f))
        while self.frame_buffer and (t - self.frame_buffer[0][0]) > CLIP_PRE_SEC:
            self.frame_buffer.popleft()

        if self.active_clip:
            self.active_clip["frames"].append((t, f))
            if t >= self.active_clip["end_time"] or (t - self.active_clip["started"]) >= CLIP_MAX_SEC:
                self._finish_clip()

    def _start_or_extend_clip(self, rec_id, t):
        if self.active_clip:
            self.active_clip["end_time"] = t + CLIP_POST_SEC
            return self.active_clip["rec_id"]
        self.active_clip = {
            "rec_id": rec_id,
            "started": t,
            "end_time": t + CLIP_POST_SEC,
            "frames": list(self.frame_buffer),
        }
        return rec_id

    def _finish_clip(self):
        clip = self.active_clip
        self.active_clip = None
        if not clip or len(clip["frames"]) < 2:
            return
        times = [t for t, _ in clip["frames"]]
        frames = [f for _, f in clip["frames"]]
        span = max(times[-1] - times[0], 0.1)
        fps = max(5.0, min(30.0, len(frames) / span))
        path = os.path.join(FORENSIC_DIR, f"clip_{self.cam_id}_{clip['rec_id']}.mp4")
        threading.Thread(target=_write_clip_file, args=(path, frames, fps), daemon=True).start()

    def _worker_loop(self):
        prev_time = time.perf_counter()
        fps_deque = deque(maxlen=10)

        while self.running:
            success, raw_frame = self.stream.read()
            curr_wall_time = time.time()

            if not success or raw_frame is None:
                with self.lock:
                    self.is_tampered = False
                    self.tamper_reason = "CLEAR"
                    self.fps = 0.0
                    self.latency_ms = 0.0

                fallback = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(fallback, (20, 20), (620, 460), (30, 40, 55), 2)
                status_label = "STANDBY" if not self.config.get("source") else "RECONNECTING"
                cv2.putText(fallback, f"[{self.cam_id}] {status_label}", (130, 230),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 165, 255), 1)
                cv2.putText(fallback, f"Source: {self.config.get('source', 'None')}", (130, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 160), 1)
                _, buf = cv2.imencode('.jpg', fallback)
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
                time.sleep(0.1)
                continue

            t_start = time.perf_counter()

            if self.config.get("mirror", False):
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame

            h, w, _ = frame.shape

            zones_map = load_zones_config()
            cam_zones = zones_map.get(self.cam_id, [])
            thresholds = load_thresholds_config()
            dn_cfg = thresholds.get("day_night", {})
            ppm = float(self.config.get("pixels_per_meter", 0.0))

            # 1. Day / Night Automatic Classification & 3-way toggle
            is_night_detected, mean_bright, mean_sat = detect_day_night_mode(
                frame,
                sat_threshold=dn_cfg.get("saturation_threshold", 32.0),
                bright_threshold=dn_cfg.get("brightness_threshold", 50.0)
            )
            self.is_night = is_night_detected

            mode_pref = CONFIG_STATE.get("night_mode", "AUTO")
            if mode_pref == "ON":
                clahe_enabled = True
            elif mode_pref == "OFF":
                clahe_enabled = False
            else:  # AUTO
                clahe_enabled = is_night_detected

            self.clahe_active = clahe_enabled
            if clahe_enabled:
                display_frame = apply_clahe_night_vision(frame)
            else:
                display_frame = frame.copy()

            # 2. Camera Anti-Tamper Check (High Sensitivity)
            is_tampered, tamper_reason = check_lens_tampering(frame, clahe_enabled, self.cam_id)

            # 3. Night-Time Motion Subtraction
            night_motion_alert = False
            motion_boxes = []
            if self.is_night:
                has_motion, m_boxes, _ = self.night_motion_detector.detect_motion(frame)
                if has_motion:
                    for mb in m_boxes:
                        for z in cam_zones:
                            if z.get("type") == "RESTRICTED" and box_intersects_zone(mb, z):
                                night_motion_alert = True
                                motion_boxes.append(mb)
                                break

            # 4. YOLO Object Detection with Hardware Frame-Skip Optimization
            self.frame_counter += 1
            if (self.frame_counter % self.frame_skip) == 0 or not self.cached_detections:
                tracked_detections = run_tracked_detection(self.model, display_frame, target_size=self.infer_size)
                self.cached_detections = tracked_detections
            else:
                tracked_detections = self.cached_detections

            t_infer_end = time.perf_counter()
            measured_lat = round((t_infer_end - t_start) * 1000.0, 1)

            # Draw Configured Zones Overlays
            for z in cam_zones:
                z_type = z.get("type", "WATCH")
                geom = z.get("geometry_type", "POLYGON").upper()
                coords = z.get("coordinates", [])
                z_color = (0, 0, 230) if z_type == "RESTRICTED" else (0, 165, 255)

                if geom == "LINE" and len(coords) >= 2:
                    p1 = tuple(coords[0])
                    p2 = tuple(coords[1])
                    cv2.line(display_frame, p1, p2, z_color, 3)
                    cv2.putText(display_frame, f"[{z_type}] {z.get('name', 'ZONE')}",
                                (p1[0] + 10, max(20, p1[1] - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.45, z_color, 1)
                elif geom == "POLYGON" and len(coords) >= 3:
                    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                    poly_overlay = display_frame.copy()
                    fill_color = (0, 0, 140) if z_type == "RESTRICTED" else (0, 120, 0)
                    cv2.fillPoly(poly_overlay, [pts], fill_color)
                    cv2.addWeighted(poly_overlay, 0.20, display_frame, 0.80, 0, display_frame)
                    cv2.polylines(display_frame, [pts], True, z_color, 2)
                    cv2.putText(display_frame, f"[{z_type}] {z.get('name', 'ZONE')}",
                                (coords[0][0] + 10, max(20, coords[0][1] + 20)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.42, z_color, 1)

            if night_motion_alert:
                for (mx1, my1, mx2, my2) in motion_boxes:
                    cv2.rectangle(display_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                    cv2.putText(display_frame, "NIGHT MOVEMENT DETECTED", (mx1, max(20, my1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), 1)

            current_breach = False
            current_loitering = False
            detected_targets = []
            loiter_targets = []
            active_ids = []
            has_person = False
            max_speed = 0.0
            lead_heading = "STATIONARY"
            lead_eta = "N/A"
            face_watchlist_alerts = []
            plate_watchlist_alerts = []

            # 5. Process Targets & Recognition Pipeline
            for det in tracked_detections:
                x1, y1, x2, y2 = det["box"]
                track_id = det["track_id"]
                label = det["label"]
                category = det["category"]
                active_ids.append(track_id)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                speed_kmh, heading_str, eta_sec, v_dx, v_dy = self.trajectory_manager.update_track(
                    track_id, [x1, y1, x2, y2], curr_wall_time, 216, label=label, category=category, pixels_per_meter=ppm
                )

                is_safe_officer = False
                officer_name = "UNKNOWN"
                is_suspect_face = False
                suspect_name = ""

                # Real Face Detection (YuNet) & ArcFace Recognition
                if category == "person":
                    has_person = True
                    person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    face_crop, face_coords = face_detector.detect_face(person_crop)

                    if face_crop is not None and face_crop.size > 0:
                        emb = face_recognizer.extract_face_embedding(face_crop)
                        matched, subject_name, matched_cat, match_score = face_recognizer.match_face_embedding(emb)

                        if matched:
                            if matched_cat == "WHITELIST":
                                is_safe_officer = True
                                officer_name = subject_name
                                SYSTEM_STATE["frs_target"] = f"OFFICER: {officer_name} (SIM {match_score})"
                            elif matched_cat == "WATCHLIST":
                                if face_recognizer.using_fallback:
                                    SYSTEM_STATE["frs_target"] = f"POI #{track_id}: {subject_name} [LOW CONFIDENCE FALLBACK]"
                                else:
                                    is_suspect_face = True
                                    suspect_name = subject_name
                                    SYSTEM_STATE["frs_target"] = f"🚨 SUSPECT: {suspect_name} (SIM {match_score})"
                                    face_watchlist_alerts.append(f"WATCHLIST SUSPECT IDENTIFIED: {suspect_name}")
                        else:
                            SYSTEM_STATE["frs_target"] = f"POI #{track_id} (UNVERIFIED)"

                        if face_coords is not None:
                            fx1, fy1, fx2, fy2 = face_coords
                            cv2.rectangle(display_frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2),
                                          (0, 255, 0) if is_safe_officer else ((0, 0, 255) if is_suspect_face else (0, 255, 255)), 1)

                # ANPR & Plate Watchlist
                is_watchlist_vehicle = False

                if category == "vehicle":
                    vehicle_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    plate_text = extract_vehicle_plate(vehicle_crop, track_id)
                    self.processed_plates_map[track_id] = plate_text
                    SYSTEM_STATE["anpr_target"] = plate_text

                    if plate_text not in ["UNREADABLE", "OCR_UNAVAILABLE"]:
                        is_on_wl, wl_reason = check_plate_watchlist(plate_text)
                        if is_on_wl:
                            is_watchlist_vehicle = True
                            plate_watchlist_alerts.append(f"WATCHLIST VEHICLE DETECTED: {plate_text} ({wl_reason})")

                # Velocity Vector Arrow
                if abs(v_dx) > 0 or abs(v_dy) > 0:
                    cv2.arrowedLine(display_frame, (center_x, center_y),
                                    (int(center_x + v_dx * 1.5), int(center_y + v_dy * 1.5)),
                                    (0, 255, 255), 2, tipLength=0.3)

                # Whitelisted Officer Suppression
                if is_safe_officer:
                    self.loitering_trackers.pop(track_id, None)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(display_frame, f"OFFICER: {officer_name}",
                                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 2)
                    continue

                if category == "wildlife":
                    continue

                is_target_interest = category in ["person", "vehicle", "phone"]
                if is_target_interest and speed_kmh >= max_speed:
                    max_speed, lead_heading, lead_eta = speed_kmh, heading_str, eta_sec

                # Zone Intersection Check
                in_restricted_zone = False
                in_watch_zone = False

                for z in cam_zones:
                    if box_intersects_zone([x1, y1, x2, y2], z):
                        track_hist = self.trajectory_manager.tracks.get(track_id, [])
                        if check_direction_constraint(track_hist, z):
                            if z.get("type") == "RESTRICTED":
                                in_restricted_zone = True
                            elif z.get("type") == "WATCH":
                                in_watch_zone = True

                if is_suspect_face or is_watchlist_vehicle:
                    current_breach = True
                    box_color = (0, 0, 255)
                    alert_tag = f"WATCHLIST SUSPECT: {suspect_name}" if is_suspect_face else f"HOTLIST PLATE: {self.processed_plates_map.get(track_id)}"
                    detected_targets.append(alert_tag)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(display_frame, alert_tag, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, box_color, 2)

                elif in_restricted_zone:
                    self.loitering_trackers.pop(track_id, None)
                    current_breach = True
                    label_text = f"RESTRICTED BREACH: {label} #{track_id}"
                    box_color = (0, 0, 255)
                    detected_targets.append(f"BREACH: {label} #{track_id}")

                    cv2.circle(display_frame, (center_x, y1), 6, (0, 0, 255), -1)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(display_frame, label_text, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.50, box_color, 2)

                    if category == "vehicle":
                        cur_plate = self.processed_plates_map.get(track_id, "PENDING")
                        cv2.rectangle(display_frame, (x1, y2 - 22), (x1 + 175, y2), (255, 255, 255), -1)
                        cv2.putText(display_frame, f"ANPR: {cur_plate}", (x1 + 5, y2 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

                elif in_watch_zone:
                    first_seen = self.loitering_trackers.setdefault(track_id, curr_wall_time)
                    dwell = curr_wall_time - first_seen
                    is_loiter = (dwell >= CONFIG_STATE["loiter_threshold_sec"] and
                                 speed_kmh <= CONFIG_STATE["loiter_max_speed_kmh"])
                    if is_loiter:
                        current_loitering = True
                        loiter_targets.append(f"LOITERING: {label} #{track_id} ({int(dwell)}s)")
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                        cv2.putText(display_frame, f"LOITERING {int(dwell)}s: {label} #{track_id}",
                                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(display_frame, f"Watch: {label} #{track_id}",
                                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                else:
                    self.loitering_trackers.pop(track_id, None)
                    cv2.circle(display_frame, (center_x, y1), 5, (0, 255, 0), -1)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Safe: {label} #{track_id}",
                                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Suspicious Activity Detectors
            suspicious_events = self.trajectory_manager.detect_suspicious_activities(
                active_ids, curr_wall_time, thresholds, cam_zones, pixels_per_meter=ppm
            )
            self.active_suspicious_events = suspicious_events

            for sev in suspicious_events:
                sbx1, sby1, sbx2, sby2 = sev["bbox"]
                sev_color = (0, 0, 255) if sev["level"] == "CODE RED" else (0, 140, 255)
                cv2.rectangle(display_frame, (sbx1, sby1), (sbx2, sby2), sev_color, 2)
                cv2.putText(display_frame, f"SUSPICIOUS: {sev['type']}",
                            (sbx1, max(20, sby1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.45, sev_color, 1)

            # Pruning
            anpr_voting_tracker.prune(set(active_ids))
            for tid in list(self.loitering_trackers.keys()):
                if tid not in active_ids:
                    self.loitering_trackers.pop(tid, None)
            self.trajectory_manager.prune_stale_tracks(curr_wall_time)

            if not has_person:
                SYSTEM_STATE["frs_target"] = "NO FACE ACQUIRED"

            # Tampering Visual Alert Overlay
            if is_tampered:
                tamper_overlay = display_frame.copy()
                cv2.rectangle(tamper_overlay, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(tamper_overlay, 0.50, display_frame, 0.50, 0, display_frame)
                cv2.rectangle(display_frame, (20, int(h/2) - 45), (w - 20, int(h/2) + 45), (0, 0, 255), -1)
                cv2.putText(display_frame, f"CRITICAL: {tamper_reason}",
                            (30, int(h/2) - 5), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display_frame, f"[ OPTICAL SENSOR {self.cam_id} TAMPERED ]",
                            (30, int(h/2) + 25), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1)

            tamper_due = False
            if is_tampered:
                prev_log = self.last_tamper_log
                if prev_log is None or prev_log[0] != tamper_reason or (curr_wall_time - prev_log[1]) > TAMPER_RELOG_SEC:
                    tamper_due = True
            else:
                self.last_tamper_log = None

            night_motion_due = False
            if night_motion_alert and (curr_wall_time - self.last_night_motion_log > thresholds.get("night_motion", {}).get("min_alert_interval_sec", 3.0)):
                night_motion_due = True
                self.last_night_motion_log = curr_wall_time

            has_watchlist_hit = (len(face_watchlist_alerts) > 0 or len(plate_watchlist_alerts) > 0)
            has_suspicious_event = len(suspicious_events) > 0
            trigger_snapshot = (
                has_watchlist_hit or
                current_breach or
                tamper_due or
                current_loitering or
                night_motion_due or
                has_suspicious_event
            )

            # Forensic Audit Vault Logging & Alert Dispatching
            if trigger_snapshot and (curr_wall_time - self.last_snapshot_time > 2.0):
                self.last_snapshot_time = curr_wall_time
                if tamper_due:
                    self.last_tamper_log = (tamper_reason, curr_wall_time)

                rec_id = int(curr_wall_time * 1000)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if tamper_due:
                    level_str, threat_type_str = "CODE RED", f"TAMPERING: {tamper_reason}"
                elif len(face_watchlist_alerts) > 0:
                    level_str, threat_type_str = "CODE RED", face_watchlist_alerts[0]
                elif len(plate_watchlist_alerts) > 0:
                    level_str, threat_type_str = "CODE RED", plate_watchlist_alerts[0]
                elif current_breach:
                    level_str, threat_type_str = "CODE RED", "VIRTUAL FENCE INTRUSION"
                elif night_motion_due:
                    level_str, threat_type_str = "CODE RED", "NIGHT MOVEMENT DETECTED"
                elif has_suspicious_event:
                    level_str = suspicious_events[0]["level"]
                    threat_type_str = suspicious_events[0]["type"]
                else:
                    level_str, threat_type_str = "CODE ORANGE", "SUSPICIOUS LOITERING"

                all_targets = detected_targets + loiter_targets + [se["target"] for se in suspicious_events]
                targets_str = ", ".join(all_targets) if all_targets else "PERIMETER THREAT"

                snap_file = os.path.join(FORENSIC_DIR, f"event_{self.cam_id}_{rec_id}.jpg")
                cv2.imwrite(snap_file, display_frame)

                _, buf = cv2.imencode('.jpg', display_frame)
                b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

                log_threat_event(rec_id, time_str, level_str, threat_type_str, targets_str,
                                 max_speed, lead_heading, lead_eta, self.cam_id, snap_file)

                clip_id = self._start_or_extend_clip(rec_id, curr_wall_time)

                rec_item = {
                    "id": rec_id,
                    "camera_id": self.cam_id,
                    "time": time_str,
                    "type": threat_type_str,
                    "level": level_str,
                    "targets": targets_str,
                    "status": "UNVERIFIED",
                    "image": b64_str,
                    "clip_id": clip_id
                }

                SYSTEM_STATE["forensic_records"].insert(0, rec_item)
                if len(SYSTEM_STATE["forensic_records"]) > 100:
                    SYSTEM_STATE["forensic_records"].pop()

                SYSTEM_STATE["total_events_count"] += 1

                alert_dispatcher.dispatch({
                    "id": rec_id,
                    "camera_id": self.cam_id,
                    "time": time_str,
                    "type": threat_type_str,
                    "level": level_str,
                    "targets": targets_str,
                    "status": "UNVERIFIED",
                    "max_speed_kmh": max_speed,
                    "heading": lead_heading,
                    "snapshot_path": snap_file
                })

            self._buffer_frame(display_frame, curr_wall_time)

            now_perf = time.perf_counter()
            frame_delta = now_perf - prev_time
            prev_time = now_perf
            if frame_delta > 0:
                fps_deque.append(1.0 / frame_delta)
            real_fps = round(sum(fps_deque) / len(fps_deque), 1) if fps_deque else 0.0

            ret, jpeg_buffer = cv2.imencode('.jpg', display_frame)
            if ret:
                with self.lock:
                    self.latest_jpeg = jpeg_buffer.tobytes()
                    self.fps = real_fps
                    self.latency_ms = measured_lat
                    self.is_tampered = bool(is_tampered)
                    self.tamper_reason = str(tamper_reason)
                    self.night_motion_alert = bool(night_motion_alert)
                    self.current_breach = bool(current_breach)
                    self.current_loitering = bool(current_loitering)
                    self.detected_targets = list(detected_targets)
                    self.loiter_targets = list(loiter_targets)
                    self.max_speed = float(max_speed)
                    self.lead_heading = str(lead_heading)
                    self.lead_eta = str(lead_eta)

            time.sleep(0.005)

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_telemetry_snapshot(self):
        with self.lock:
            return {
                "cam_id": self.cam_id,
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "is_night": self.is_night,
                "clahe_active": self.clahe_active,
                "night_motion_alert": self.night_motion_alert,
                "is_tampered": self.is_tampered,
                "tamper_reason": self.tamper_reason,
                "current_breach": self.current_breach,
                "current_loitering": self.current_loitering,
                "suspicious_events": list(self.active_suspicious_events),
                "detected_targets": list(self.detected_targets),
                "loiter_targets": list(self.loiter_targets),
                "max_speed": self.max_speed,
                "lead_heading": self.lead_heading,
                "lead_eta": self.lead_eta,
            }

    def stop(self):
        self.running = False
        self.stream.stop()


# ----------------- Global Camera Workers Registry -----------------
camera_workers = {}


def initialize_camera_workers():
    global camera_workers
    configs = load_camera_configs()
    SYSTEM_STATE["camera_registry"] = configs

    for cfg in configs:
        cid = cfg["id"]
        if cid not in camera_workers:
            camera_workers[cid] = CameraWorker(cfg)


initialize_camera_workers()


# ----------------- Telemetry Aggregator & Broadcast Daemon -----------------
def telemetry_aggregator_daemon():
    while True:
        any_tampered = False
        tampered_cam = ""
        tamper_reason = "CLEAR"
        any_breach = False
        any_loiter = False
        any_night = False
        any_clahe = False
        any_night_motion = False
        active_threat_types = []
        active_threat_flags = {}
        active_targets = []
        top_speed = 0.0
        lead_heading = "STATIONARY"
        lead_eta = "N/A"
        fps_list = []
        lat_list = []

        for cid, worker in list(camera_workers.items()):
            t = worker.get_telemetry_snapshot()
            fps_list.append(t["fps"])
            lat_list.append(t["latency_ms"])

            if t["is_night"]:
                any_night = True
            if t.get("clahe_active", False):
                any_clahe = True
            if t["night_motion_alert"]:
                any_night_motion = True
                active_threat_types.append("NIGHT MOVEMENT DETECTED")
                active_threat_flags["night_movement"] = True

            if t["is_tampered"]:
                any_tampered = True
                tampered_cam = cid
                tamper_reason = t["tamper_reason"]

            if t["current_breach"]:
                any_breach = True
                active_threat_flags["breach"] = True

            if t["current_loitering"]:
                any_loiter = True
                active_threat_flags["loitering"] = True

            for sev in t.get("suspicious_events", []):
                active_threat_types.append(sev["type"])
                if sev["type"] == "RUNNING / EVASIVE MOVEMENT":
                    active_threat_flags["running"] = True
                elif sev["type"] == "GROUP GATHERING DETECTED":
                    active_threat_flags["group_gathering"] = True
                elif sev["type"] == "ABANDONED OBJECT DETECTED":
                    active_threat_flags["abandoned_object"] = True
                elif sev["type"] == "VEHICLE HALTED AT PERIMETER":
                    active_threat_flags["vehicle_halted"] = True
                elif sev["type"] == "WRONG DIRECTION INTRUSION":
                    active_threat_flags["wrong_direction"] = True

            active_targets.extend(t["detected_targets"] + t["loiter_targets"])
            if t["max_speed"] >= top_speed:
                top_speed = t["max_speed"]
                lead_heading = t["lead_heading"]
                lead_eta = t["lead_eta"]

        SYSTEM_STATE["is_tampered"] = any_tampered
        SYSTEM_STATE["tampered_cam"] = tampered_cam
        SYSTEM_STATE["tamper_reason"] = tamper_reason
        SYSTEM_STATE["is_night"] = any_night
        SYSTEM_STATE["night_enhancement_active"] = any_clahe
        SYSTEM_STATE["night_mode_setting"] = CONFIG_STATE.get("night_mode", "AUTO")
        SYSTEM_STATE["boundary_percentage"] = CONFIG_STATE.get("boundary_percentage", 45)
        SYSTEM_STATE["night_motion_alert"] = any_night_motion
        SYSTEM_STATE["is_breached"] = any_breach
        SYSTEM_STATE["is_loitering"] = any_loiter
        SYSTEM_STATE["intruder_count"] = len([tg for tg in active_targets if "BREACH" in tg or "WATCHLIST" in tg])
        SYSTEM_STATE["warning_count"] = len([tg for tg in active_targets if "LOITERING" in tg])
        SYSTEM_STATE["active_targets"] = active_targets

        SYSTEM_STATE["fps"] = round(float(np.mean(fps_list)), 1) if fps_list else 0.0
        SYSTEM_STATE["latency_ms"] = round(float(np.mean(lat_list)), 1) if lat_list else 0.0

        threat_code, threat_desc_str = compute_threat_level(
            any_tampered, False, any_breach, any_loiter, active_threat_types
        )
        risk_score, risk_band = compute_risk_score(
            any_tampered, any_breach, any_loiter, top_speed, active_threat_flags
        )

        calibrated = False
        cams = load_camera_configs()
        if cams and cams[0].get("pixels_per_meter", 0.0) > 0.0:
            calibrated = True

        SYSTEM_STATE["interception_vector"] = {
            "velocity_kmh": top_speed,
            "speed_status": "CALIBRATED" if calibrated else "ESTIMATED (uncalibrated)",
            "heading": lead_heading,
            "eta_bop_seconds": lead_eta,
            "threat_level": f"{threat_desc_str} | RISK {risk_score}/100 ({risk_band})",
            "risk_score": risk_score,
            "risk_band": risk_band
        }

        ws_manager.broadcast_from_thread({
            "type": "TELEMETRY_UPDATE",
            "data": SYSTEM_STATE
        })

        time.sleep(0.25)


@app.on_event("startup")
def startup_event():
    loop = asyncio.get_event_loop()
    ws_manager.set_event_loop(loop)
    threading.Thread(target=telemetry_aggregator_daemon, daemon=True).start()


# ----------------- WebSocket Telemetry Endpoint -----------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "INITIAL_SNAPSHOT",
            "data": SYSTEM_STATE
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ----------------- C2 Command & Control Integration APIs -----------------
class C2EventRecord(BaseModel):
    id: int
    timestamp: str
    threat_level: str
    threat_type: str
    targets: str
    velocity_kmh: float
    heading: str
    eta_bop: str
    camera_id: str
    status: str
    snapshot_path: str
    record_hash: str
    prev_hash: str


class C2EventsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    events: list[C2EventRecord]


@app.get(
    "/api/events",
    response_model=C2EventsResponse,
    tags=["Command & Control"],
    summary="C2 Filtered Threat Query"
)
def get_c2_events(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    camera_id: Optional[str] = None,
    threat_type: Optional[str] = None,
    threat_level: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    c2_auth: dict = Depends(authenticate_c2_api_key)
):
    events, total = query_filtered_events(
        start_time=start_time,
        end_time=end_time,
        camera_id=camera_id,
        threat_type=threat_type,
        threat_level=threat_level,
        status=status,
        page=page,
        page_size=page_size
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "events": events
    }


@app.get(
    "/api/events/export",
    tags=["Command & Control"],
    summary="Export Forensic Audit Logs (CSV)"
)
def export_c2_events_csv(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    camera_id: Optional[str] = None,
    threat_type: Optional[str] = None,
    auth_user: dict = Depends(authenticate_download_stream)
):
    events, _ = query_filtered_events(
        start_time=start_time,
        end_time=end_time,
        camera_id=camera_id,
        threat_type=threat_type,
        page=1,
        page_size=100000
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Event ID", "Timestamp", "Threat Level", "Threat Type", "Targets",
        "Velocity (km/h)", "Heading", "ETA to BOP", "Camera ID", "HITL Status",
        "Record Hash (SHA-256)", "Previous Hash (SHA-256)"
    ])

    for e in events:
        writer.writerow([
            e["id"], e["timestamp"], e["threat_level"], e["threat_type"], e["targets"],
            e["velocity_kmh"], e["heading"], e["eta_bop"], e["camera_id"], e["status"],
            e["record_hash"], e["prev_hash"]
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ibvap_audit_vault_{int(time.time())}.csv"}
    )


# ----------------- Video Feed Streaming Endpoint -----------------
def generate_camera_frames(cam_id: str):
    worker = camera_workers.get(cam_id)
    if not worker:
        return

    while True:
        jpeg_bytes = worker.get_latest_jpeg()
        if jpeg_bytes is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(0.033)


@app.get("/video_feed/{cam_id}", tags=["Surveillance Video"])
def video_feed_cam(cam_id: str, auth_user: dict = Depends(authenticate_download_stream)):
    return StreamingResponse(generate_camera_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed", tags=["Surveillance Video"])
def video_feed_default(auth_user: dict = Depends(authenticate_download_stream)):
    return StreamingResponse(generate_camera_frames("CAM-01"), media_type="multipart/x-mixed-replace; boundary=frame")


# ----------------- Zone Management APIs -----------------
class ZoneItem(BaseModel):
    id: str
    name: str
    type: str
    geometry_type: str
    coordinates: list
    direction: str = "ANY"


class SaveZonesPayload(BaseModel):
    zones: list[ZoneItem]


@app.get("/api/zones/{cam_id}", tags=["Zone Editor"])
def get_zones(cam_id: str, session: dict = Depends(authenticate_user)):
    zones_map = load_zones_config()
    return {"cam_id": cam_id, "zones": zones_map.get(cam_id, [])}


@app.post("/api/zones/{cam_id}", tags=["Zone Editor"])
def save_zones(cam_id: str, payload: SaveZonesPayload, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    zones_map = load_zones_config()
    zones_map[cam_id] = [z.model_dump() for z in payload.zones]
    save_zones_config(zones_map)
    return {"status": "success", "cam_id": cam_id, "zones_count": len(zones_map[cam_id])}


# ----------------- Threshold Management APIs -----------------
class SaveThresholdsPayload(BaseModel):
    thresholds: dict


@app.get("/api/thresholds", tags=["Configuration"])
def get_thresholds(session: dict = Depends(authenticate_user)):
    return {"thresholds": load_thresholds_config()}


@app.post("/api/thresholds", tags=["Configuration"])
def save_thresholds(payload: SaveThresholdsPayload, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    save_thresholds_config(payload.thresholds)
    return {"status": "success", "thresholds": payload.thresholds}


# ----------------- Biometrics Dual-List APIs -----------------
class EnrolFacePayload(BaseModel):
    name: str
    category: str = "WHITELIST"
    image_base64: str = ""


@app.get("/api/biometrics/list", tags=["Biometrics"])
def list_biometrics(session: dict = Depends(authenticate_user)):
    records = db_get_all_faces()
    result = []
    for r in records:
        result.append({
            "name": r[0],
            "category": r[1],
            "enrolled_at": r[2],
            "snapshot_path": r[3]
        })
    return {"faces": result, "face_engine_status": face_recognizer.status_string}


@app.post("/api/biometrics/enrol", tags=["Biometrics"])
def enrol_biometrics(payload: EnrolFacePayload, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    clean_name = payload.name.strip().upper()
    category = payload.category.strip().upper()
    if category not in ["WHITELIST", "WATCHLIST"]:
        raise HTTPException(status_code=400, detail="Category must be WHITELIST or WATCHLIST")

    face_img = None
    if payload.image_base64:
        try:
            raw_data = payload.image_base64
            if "," in raw_data:
                raw_data = raw_data.split(",")[1]
            img_bytes = base64.b64decode(raw_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    if face_img is None:
        worker = camera_workers.get("CAM-01")
        if not worker:
            raise HTTPException(status_code=500, detail="Camera worker CAM-01 not active")
        ok, frame = worker.stream.read()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Camera stream offline")
        detected_face, _ = face_detector.detect_face(frame)
        if detected_face is None:
            raise HTTPException(status_code=400, detail="No face detected in live camera frame. Look directly at camera.")
        face_img = detected_face

    success, msg = face_recognizer.enrol_face(clean_name, category, face_img)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@app.delete("/api/biometrics/delete/{category}/{name}", tags=["Biometrics"])
def delete_biometrics(category: str, name: str, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    success = face_recognizer.delete_face(category, name)
    if not success:
        raise HTTPException(status_code=404, detail="Subject not found in biometric registry")
    return {"status": "deleted", "name": name.upper(), "category": category.upper()}


# ----------------- Plate Watchlist APIs -----------------
class AddPlateWatchlistPayload(BaseModel):
    plate_number: str
    reason: str = "SUSPECT VEHICLE"


@app.get("/api/anpr/watchlist", tags=["ANPR"])
def get_plate_watchlist(session: dict = Depends(authenticate_user)):
    return {"watchlist": db_get_plate_watchlist()}


@app.post("/api/anpr/watchlist", tags=["ANPR"])
def add_plate_watchlist(payload: AddPlateWatchlistPayload, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    plate = payload.plate_number.strip().upper()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_add_plate_watchlist(plate, payload.reason, now_str)
    return {"status": "success", "plate_number": plate}


@app.delete("/api/anpr/watchlist/{plate}", tags=["ANPR"])
def delete_plate_watchlist(plate: str, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    success = db_delete_plate_watchlist(plate)
    if not success:
        raise HTTPException(status_code=404, detail="Plate not found in watchlist")
    return {"status": "deleted", "plate_number": plate.upper()}


# ----------------- QRT Drone Mission Daemon (SIMULATED) -----------------
def qrt_drone_mission_daemon():
    while True:
        if SYSTEM_STATE["qrt_drone"]["status"] in ["SCRAMBLED", "AIRBORNE", "INTERCEPTING"]:
            if SYSTEM_STATE["qrt_drone"]["eta_seconds"] > 0:
                SYSTEM_STATE["qrt_drone"]["eta_seconds"] -= 1
                if SYSTEM_STATE["qrt_drone"]["eta_seconds"] == 18:
                    SYSTEM_STATE["qrt_drone"]["status"] = "AIRBORNE"
                elif SYSTEM_STATE["qrt_drone"]["eta_seconds"] == 5:
                    SYSTEM_STATE["qrt_drone"]["status"] = "INTERCEPTING"
            else:
                SYSTEM_STATE["qrt_drone"]["status"] = "TARGET INTERCEPTED (SIMULATED)"
                time.sleep(5.0)
                SYSTEM_STATE["qrt_drone"]["status"] = "RTB (SIMULATED)"
                time.sleep(4.0)
                SYSTEM_STATE["qrt_drone"]["status"] = "STANDBY"
                SYSTEM_STATE["qrt_drone"]["target_sector"] = "PERIMETER SECURE"
        time.sleep(1.0)


threading.Thread(target=qrt_drone_mission_daemon, daemon=True).start()


class DroneDispatchPayload(BaseModel):
    sector: str = "Sector Forward Post"
    mode: str = "INTERCEPT"


@app.post("/api/drone/scramble", tags=["Tactical Drone Response (Simulated)"])
def scramble_qrt_drone(payload: DroneDispatchPayload, session: dict = Depends(authenticate_user)):
    SYSTEM_STATE["qrt_drone"]["status"] = "SCRAMBLED"
    SYSTEM_STATE["qrt_drone"]["target_sector"] = payload.sector
    SYSTEM_STATE["qrt_drone"]["eta_seconds"] = 25
    SYSTEM_STATE["qrt_drone"]["dispatched_at"] = datetime.now().strftime("%H:%M:%S")
    return {
        "status": "success",
        "simulation_disclaimer": "SIMULATED RESPONSE WORKFLOW — NO HARDWARE DRONE INTEGRATED",
        "message": f"QRT Drone {SYSTEM_STATE['qrt_drone']['drone_id']} Dispatched!",
        "telemetry": SYSTEM_STATE["qrt_drone"]
    }


@app.post("/api/drone/abort", tags=["Tactical Drone Response (Simulated)"])
def abort_qrt_drone(session: dict = Depends(authenticate_user)):
    SYSTEM_STATE["qrt_drone"]["status"] = "RTB"
    SYSTEM_STATE["qrt_drone"]["eta_seconds"] = 0
    return {
        "status": "success",
        "simulation_disclaimer": "SIMULATED RESPONSE WORKFLOW — NO HARDWARE DRONE INTEGRATED",
        "message": "QRT Drone Mission Aborted! Returning to base."
    }


# ----------------- Dynamic Camera Management APIs -----------------
class AddCameraPayload(BaseModel):
    name: str
    location: str
    source: str
    mirror: bool = False
    pixels_per_meter: float = 0.0


@app.post("/api/cameras/add", tags=["Camera Management"])
def add_camera(payload: AddCameraPayload, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")

    configs = load_camera_configs()
    existing_ids = [c["id"] for c in configs]
    new_idx = len(configs) + 1
    new_id = f"CAM-0{new_idx}" if new_idx < 10 else f"CAM-{new_idx}"
    while new_id in existing_ids:
        new_idx += 1
        new_id = f"CAM-0{new_idx}" if new_idx < 10 else f"CAM-{new_idx}"

    new_cam = {
        "id": new_id,
        "name": payload.name,
        "location": payload.location,
        "source": payload.source.strip(),
        "mirror": payload.mirror,
        "pixels_per_meter": float(payload.pixels_per_meter),
        "status": "ONLINE" if payload.source.strip() else "STANDBY"
    }

    configs.append(new_cam)
    save_camera_configs(configs)

    SYSTEM_STATE["camera_registry"] = configs
    camera_workers[new_id] = CameraWorker(new_cam)
    return {"status": "success", "camera": new_cam}


@app.delete("/api/cameras/delete/{cam_id}", tags=["Camera Management"])
def delete_camera(cam_id: str, session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    if cam_id == "CAM-01":
        raise HTTPException(status_code=400, detail="Cannot delete default sensor CAM-01")

    configs = load_camera_configs()
    configs = [c for c in configs if c["id"] != cam_id]
    save_camera_configs(configs)

    SYSTEM_STATE["camera_registry"] = configs

    worker = camera_workers.pop(cam_id, None)
    if worker:
        worker.stop()

    return {"status": "deleted", "cam_id": cam_id}


# ----------------- Real-Time Virtual Fence & Night Mode Control -----------------
@app.get("/set_line/{percentage}", tags=["Configuration"])
def set_line(percentage: int, session: dict = Depends(authenticate_user)):
    clamped_pct = max(15, min(85, percentage))
    CONFIG_STATE["boundary_percentage"] = clamped_pct
    SYSTEM_STATE["boundary_percentage"] = clamped_pct

    target_y = int(480 * (clamped_pct / 100.0))
    zones_map = load_zones_config()

    for cid in zones_map:
        for z in zones_map[cid]:
            if z.get("geometry_type") == "LINE":
                z["coordinates"] = [[0, target_y], [640, target_y]]
            elif z.get("geometry_type") == "POLYGON" and z.get("name") == "Forward Buffer Zone":
                z["coordinates"] = [[0, target_y], [640, target_y], [640, min(470, target_y + 96)], [0, min(470, target_y + 96)]]

    save_zones_config(zones_map)
    return {"status": "updated", "boundary_percentage": clamped_pct, "line_y": target_y}


@app.get("/toggle_night/{mode}", tags=["Configuration"])
def toggle_night(mode: str, session: dict = Depends(authenticate_user)):
    clean_mode = mode.upper().strip()
    if clean_mode in ["TRUE", "1", "ON"]:
        actual_mode = "ON"
    elif clean_mode in ["FALSE", "0", "OFF"]:
        actual_mode = "OFF"
    else:
        actual_mode = "AUTO"

    CONFIG_STATE["night_mode"] = actual_mode
    SYSTEM_STATE["night_mode_setting"] = actual_mode
    return {"status": "updated", "night_mode": actual_mode}


@app.get("/hitl_feedback/{rec_id}/{action}", tags=["Forensics"])
def hitl_feedback(rec_id: int, action: str, session: dict = Depends(authenticate_user)):
    status_str = "CONFIRMED_THREAT" if action == "confirm" else "SUPPRESSED_FALSE_ALARM"
    update_hitl_status(rec_id, status_str)
    for rec in SYSTEM_STATE["forensic_records"]:
        if rec["id"] == rec_id:
            rec["status"] = status_str
            break
    return {"status": "updated"}


@app.get("/api/verify_forensic_chain", tags=["Forensics"])
def verify_forensic_chain_endpoint(session: dict = Depends(authenticate_user)):
    require_role(session, "ADMIN")
    intact, message = verify_chain_integrity()
    return {"chain_intact": intact, "message": message}


@app.get("/forensic_clip/{clip_id}", tags=["Forensics"])
def get_forensic_clip(clip_id: int, auth_user: dict = Depends(authenticate_download_stream)):
    suffix = f"_{clip_id}.mp4"
    for fname in os.listdir(FORENSIC_DIR):
        if fname.startswith("clip_") and fname.endswith(suffix):
            return FileResponse(os.path.join(FORENSIC_DIR, fname), media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Clip not found (still recording or not captured)")


@app.get("/system_telemetry", tags=["Surveillance Video"])
def get_telemetry():
    return SYSTEM_STATE


@app.get("/", tags=["Dashboard"])
def home():
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())