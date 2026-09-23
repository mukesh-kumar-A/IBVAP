import cv2
import time
import os
import base64
import threading
import urllib.request
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from app.config import CONFIG_STATE, FORENSIC_DIR, FACES_DIR
from app.models_loader import model_registry
from app.inference import apply_clahe_night_vision, check_lens_tampering, run_object_detection
from app.recognition import verify_face_biometric, extract_vehicle_plate, load_enrolled_officers
from app.tracker import byte_tracker
from app.analytics import compute_threat_level
from app.database import log_threat_event, update_hitl_status

app = FastAPI(title="IBVAP 2.0 Sovereign Defense Enterprise Hub")

# Default: Only Laptop Web Cam Active on Startup
CAMERA_SOURCES = {
    "CAM-01": 0
}

SYSTEM_STATE = {
    "is_breached": False,
    "is_loitering": False,
    "is_tampered": False,
    "tamper_reason": "CLEAR",
    "tampered_cam": "",
    "intruder_count": 0,
    "warning_count": 0,
    "total_events_count": 0,
    "active_targets": [],
    "fps": 30,
    "latency_ms": 10,
    "frs_target": "NO FACE ACQUIRED",
    "anpr_target": "NO VEHICLE DETECTED",
    "interception_vector": {
        "velocity_kmh": 0.0,
        "heading": "STATIONARY",
        "eta_bop_seconds": "N/A",
        "threat_level": "CODE GREEN"
    },
    "forensic_records": [],
    "camera_registry": [
        {
            "id": "CAM-01",
            "name": "Lap Camera",
            "location": "Sector A - Lap Cam",
            "source": "0",
            "status": "ONLINE"
        }
    ],
    # QRT Drone Telemetry State
    "qrt_drone": {
        "status": "STANDBY",             # STANDBY, SCRAMBLED, AIRBORNE, INTERCEPTING, RTB
        "drone_id": "UAV-GARUDA-09",
        "battery": 98,
        "payload": "OPTICAL-ZOOM / THERMAL POD",
        "target_sector": "PERIMETER SECURE",
        "eta_seconds": 0,
        "dispatched_at": None,
        "coordinates": "31°38'12\"N, 74°52'31\"E"
    }
}

loitering_trackers = {}
processed_plates_map = {}
last_snapshot_times = {}
active_tamper_map = {}

# ----------------- Robust Zero-Latency Stream Engine (DroidCam & USB) -----------------
class MultiCameraStream:
    def __init__(self, cam_id, src):
        self.cam_id = cam_id
        self.src = src
        self.cap = None
        self.frame = None
        self.status = False
        self.running = True
        self.lock = threading.Lock()
        self.is_network = not str(src).strip().isdigit()
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open_usb(self):
        try:
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
            actual_src = int(str(self.src).strip())
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

    def update(self):
        if not self.is_network:
            self._open_usb()
            reconnect_timer = 0
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
                    time.sleep(0.2)
                time.sleep(0.015)
            return

        # DroidCam / Network Stream Handling (Zero-Lag Native MJPEG Engine)
        src_url = str(self.src).strip()
        if not src_url.startswith("http://") and not src_url.startswith("https://") and not src_url.startswith("rtsp://"):
            src_url = "http://" + src_url

        if "4747" in src_url:
            base_url = src_url.split("/")[0] + "//" + src_url.split("/")[2]
            endpoints = [f"{base_url}/mjpegfeed", f"{base_url}/video"]
        else:
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
                    time.sleep(0.1)
                
                if connected:
                    break

            with self.lock:
                self.status = False
            time.sleep(2.0)

    def read(self):
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass

camera_streams = {}

def get_or_create_stream(cam_id, src):
    if cam_id not in camera_streams or not camera_streams[cam_id].running:
        camera_streams[cam_id] = MultiCameraStream(cam_id, src)
    return camera_streams[cam_id]

for cam in SYSTEM_STATE["camera_registry"]:
    get_or_create_stream(cam["id"], cam["source"])

# ----------------- Video Processing Generator -----------------
def generate_camera_frames(cam_id: str):
    global SYSTEM_STATE, loitering_trackers, processed_plates_map, last_snapshot_times, active_tamper_map

    target_cam = next((c for c in SYSTEM_STATE["camera_registry"] if c["id"] == cam_id), None)
    if not target_cam:
        return

    stream = get_or_create_stream(cam_id, target_cam["source"])
    if cam_id not in loitering_trackers:
        loitering_trackers[cam_id] = {}
    if cam_id not in processed_plates_map:
        processed_plates_map[cam_id] = {}
    if cam_id not in last_snapshot_times:
        last_snapshot_times[cam_id] = 0

    prev_time = time.time()
    frame_count = 0
    cached_tracks = []

    while True:
        start_time = time.time()
        success, raw_frame = stream.read()

        if not success or raw_frame is None:
            active_tamper_map[cam_id] = False
            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(fallback, (20, 20), (620, 460), (30, 40, 55), 2)
            cv2.putText(fallback, f"[{cam_id}] CONNECTING / STANDBY", (130, 230),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 165, 255), 1)
            cv2.putText(fallback, f"Target: {target_cam['source']}", (130, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 160), 1)
            ret, buffer = cv2.imencode('.jpg', fallback)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.06)
            continue

        frame = cv2.flip(raw_frame, 1)
        h, w, _ = frame.shape
        line_y = int(h * (CONFIG_STATE["boundary_percentage"] / 100.0))
        frame_count += 1

        is_tampered, tamper_reason = check_lens_tampering(frame, CONFIG_STATE["night_enhancement"])
        active_tamper_map[cam_id] = (is_tampered, tamper_reason)

        if CONFIG_STATE["night_enhancement"]:
            display_frame = apply_clahe_night_vision(frame)
        else:
            display_frame = frame.copy()

        curr_time = time.time()

        if frame_count % 2 == 0:
            raw_dets = run_object_detection(display_frame)
            tracker_input = []
            for (box, conf, label, _, cl) in raw_dets:
                tracker_input.append([box[0], box[1], box[2], box[3], conf, cl, label])
            cached_tracks = byte_tracker.update(tracker_input, curr_time, line_y)

        current_breach = False
        detected_targets = []
        active_ids = set()
        has_person = False

        # Visual Tactical Overlays
        top_overlay = display_frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, line_y), (0, 0, 150), -1)
        cv2.addWeighted(top_overlay, 0.15, display_frame, 0.85, 0, display_frame)
        cv2.putText(display_frame, f"[{cam_id} INFILTRATION SECTOR]", (20, 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)

        cv2.line(display_frame, (0, line_y), (w, line_y), (0, 0, 255), 3)
        cv2.putText(display_frame, f"=== VIRTUAL FENCE ({cam_id}) ===",
                    (20, max(25, line_y - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 0, 255), 2)

        bottom_overlay = display_frame.copy()
        cv2.rectangle(bottom_overlay, (0, line_y), (w, h), (0, 120, 0), -1)
        cv2.addWeighted(bottom_overlay, 0.18, display_frame, 0.82, 0, display_frame)

        # FRS Face Recognition
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        verified_faces = []
        try:
            if model_registry.face_detector is not None:
                detected_faces = model_registry.face_detector.detectMultiScale(
                    gray_full, scaleFactor=1.12, minNeighbors=4, minSize=(45, 45)
                )
                for (fx, fy, fw, fh) in detected_faces:
                    face_roi = frame[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
                    is_safe, name, _ = verify_face_biometric(face_roi)
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 255), 2)
                    if is_safe:
                        cv2.putText(display_frame, f"FRS: SAFE {name}", (fx, max(20, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        verified_faces.append((fx, fy, fw, fh, name))
                    else:
                        cv2.putText(display_frame, "FRS BIO-MATCH: #POI", (fx, max(20, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        except Exception:
            pass

        for (bbox, conf, label, track_id, class_id, speed_kmh, heading_str, eta_sec, v_dx, v_dy) in cached_tracks:
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            active_ids.add(track_id)

            is_safe_officer = False
            officer_name = "UNKNOWN"

            if class_id == 0:
                has_person = True
                for (fx, fy, fw, fh, fname) in verified_faces:
                    face_cx = fx + fw // 2
                    face_cy = fy + fh // 2
                    if (x1 <= face_cx <= x2) and (y1 <= face_cy <= y2):
                        is_safe_officer = True
                        officer_name = fname
                        break

                if not is_safe_officer:
                    ph, pw = (y2 - y1), (x2 - x1)
                    face_crop = frame[max(0, y1):min(h, y1 + int(ph * 0.50)), max(0, x1 + int(pw * 0.10)):min(w, x2 - int(pw * 0.10))]
                    if face_crop.size > 0:
                        is_safe_officer, officer_name, _ = verify_face_biometric(face_crop)

                SYSTEM_STATE["frs_target"] = f"SAFE: {officer_name}" if is_safe_officer else f"POI #{track_id} (UNVERIFIED)"

            if class_id in [2, 3, 5, 7]:
                if track_id not in processed_plates_map[cam_id]:
                    vehicle_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    plate_text = extract_vehicle_plate(vehicle_crop, track_id)
                    processed_plates_map[cam_id][track_id] = plate_text
                SYSTEM_STATE["anpr_target"] = processed_plates_map[cam_id][track_id]

            if abs(v_dx) > 0 or abs(v_dy) > 0:
                cv2.arrowedLine(display_frame, (center_x, center_y),
                                (int(center_x + v_dx * 1.5), int(center_y + v_dy * 1.5)), (0, 255, 255), 2, tipLength=0.3)

            if is_safe_officer:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display_frame, f"OFFICER: {officer_name}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 2)
                continue

            if class_id in [14, 15, 16, 17, 18, 19]:
                continue

            # Strict Any-Part Boundary Crossing Logic
            is_fence_crossed = (y1 <= line_y)

            if is_fence_crossed:
                current_breach = True
                label_text = f"FENCE BREACH: {label} #{track_id}"
                box_color = (0, 0, 255)
                detected_targets.append(f"BREACH: {label} #{track_id}")

                cv2.circle(display_frame, (center_x, max(y1, line_y)), 6, (0, 0, 255), -1)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(display_frame, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.50, box_color, 2)

                # Drone Live Intercept Vector HUD
                if SYSTEM_STATE["qrt_drone"]["status"] in ["SCRAMBLED", "AIRBORNE", "INTERCEPTING"]:
                    cv2.putText(display_frame, f"[QRT DRONE INBOUND: ETA {SYSTEM_STATE['qrt_drone']['eta_seconds']}s]", 
                                (x1, min(h - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                if class_id in [2, 3, 5, 7]:
                    cur_plate = processed_plates_map[cam_id].get(track_id, f"IND-PB02-BX{track_id}00")
                    cv2.rectangle(display_frame, (x1, y2 - 22), (x1 + 175, y2), (255, 255, 255), -1)
                    cv2.putText(display_frame, f"ANPR: {cur_plate}", (x1 + 5, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            else:
                cv2.circle(display_frame, (center_x, y1), 5, (0, 255, 0), -1)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Safe: {label} #{track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if not has_person:
            SYSTEM_STATE["frs_target"] = "NO FACE ACQUIRED"

        # Tampering Alert Overlay
        if is_tampered:
            tamper_overlay = display_frame.copy()
            cv2.rectangle(tamper_overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(tamper_overlay, 0.45, display_frame, 0.55, 0, display_frame)

            cv2.rectangle(display_frame, (30, int(h/2) - 40), (w - 30, int(h/2) + 40), (0, 0, 255), -1)
            cv2.putText(display_frame, f"CRITICAL: {tamper_reason}",
                        (40, int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"[ OPTICAL SENSOR {cam_id} TAMPERED ]",
                        (40, int(h/2) + 28), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1)

        # Snapshot & Audit Logging
        if (current_breach or is_tampered) and (curr_time - last_snapshot_times[cam_id] > 2.0):
            last_snapshot_times[cam_id] = curr_time
            rec_id = int(curr_time * 1000)
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            snap_file = os.path.join(FORENSIC_DIR, f"breach_{cam_id}_{rec_id}.jpg")
            cv2.imwrite(snap_file, display_frame)

            _, buf = cv2.imencode('.jpg', display_frame)
            b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            threat_type_str = f"TAMPERING: {tamper_reason}" if is_tampered else "VIRTUAL FENCE INTRUSION"

            log_threat_event(rec_id, time_str, "CODE RED", threat_type_str,
                             ", ".join(detected_targets) if detected_targets else "PERIMETER THREAT",
                             speed_kmh if 'speed_kmh' in locals() else 0.0,
                             heading_str if 'heading_str' in locals() else "STATIONARY",
                             eta_sec if 'eta_sec' in locals() else "N/A",
                             cam_id, snap_file)

            rec_item = {
                "id": rec_id,
                "camera_id": cam_id,
                "time": time_str,
                "type": threat_type_str,
                "level": "CODE RED",
                "targets": ", ".join(detected_targets) if detected_targets else "PERIMETER THREAT",
                "status": "UNVERIFIED",
                "image": b64_str
            }
            SYSTEM_STATE["forensic_records"].insert(0, rec_item)
            SYSTEM_STATE["total_events_count"] += 1

        any_tampered = any(v[0] for v in active_tamper_map.values() if isinstance(v, tuple))
        t_cam = next((k for k, v in active_tamper_map.items() if isinstance(v, tuple) and v[0]), "")
        t_reason = active_tamper_map[t_cam][1] if t_cam else "CLEAR"

        threat_code, threat_desc_str = compute_threat_level(any_tampered, False, current_breach, False)
        SYSTEM_STATE["interception_vector"] = {
            "velocity_kmh": speed_kmh if 'speed_kmh' in locals() else 0.0,
            "heading": heading_str if 'heading_str' in locals() else "STATIONARY",
            "eta_bop_seconds": eta_sec if 'eta_sec' in locals() else "N/A",
            "threat_level": threat_desc_str
        }
        SYSTEM_STATE["fps"] = min(30, int(1.0 / (curr_time - prev_time + 1e-6)))
        prev_time = curr_time
        SYSTEM_STATE["latency_ms"] = max(8, int((time.time() - start_time) * 1000))
        SYSTEM_STATE["is_breached"] = bool(current_breach)
        SYSTEM_STATE["is_loitering"] = False
        SYSTEM_STATE["is_tampered"] = bool(any_tampered)
        SYSTEM_STATE["tamper_reason"] = str(t_reason)
        SYSTEM_STATE["tampered_cam"] = str(t_cam)
        SYSTEM_STATE["intruder_count"] = int(len(detected_targets))
        SYSTEM_STATE["active_targets"] = detected_targets

        ret, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ----------------- QRT Drone Mission Countdown Daemon -----------------
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
                SYSTEM_STATE["qrt_drone"]["status"] = "TARGET INTERCEPTED"
                time.sleep(5.0)
                SYSTEM_STATE["qrt_drone"]["status"] = "RTB"  # Return to Base
                time.sleep(4.0)
                SYSTEM_STATE["qrt_drone"]["status"] = "STANDBY"
                SYSTEM_STATE["qrt_drone"]["target_sector"] = "PERIMETER SECURE"
        time.sleep(1.0)

threading.Thread(target=qrt_drone_mission_daemon, daemon=True).start()

# ----------------- QRT Drone Dispatch APIs -----------------
class DroneDispatchPayload(BaseModel):
    sector: str = "Sector Forward Post"
    mode: str = "INTERCEPT"

@app.post("/api/drone/scramble")
def scramble_qrt_drone(payload: DroneDispatchPayload):
    SYSTEM_STATE["qrt_drone"]["status"] = "SCRAMBLED"
    SYSTEM_STATE["qrt_drone"]["target_sector"] = payload.sector
    SYSTEM_STATE["qrt_drone"]["eta_seconds"] = 25  # 25 seconds tactical intercept
    SYSTEM_STATE["qrt_drone"]["dispatched_at"] = datetime.now().strftime("%H:%M:%S")
    return {
        "status": "success",
        "message": f"QRT Drone {SYSTEM_STATE['qrt_drone']['drone_id']} Dispatched!",
        "telemetry": SYSTEM_STATE["qrt_drone"]
    }

@app.post("/api/drone/abort")
def abort_qrt_drone():
    SYSTEM_STATE["qrt_drone"]["status"] = "RTB"
    SYSTEM_STATE["qrt_drone"]["eta_seconds"] = 0
    return {"status": "success", "message": "QRT Drone Mission Aborted! Returning to base."}

# ----------------- Dynamic Camera Management APIs -----------------
class AddCameraPayload(BaseModel):
    name: str
    location: str
    source: str

@app.post("/api/cameras/add")
def add_camera(payload: AddCameraPayload):
    new_idx = len(SYSTEM_STATE["camera_registry"]) + 1
    new_id = f"CAM-0{new_idx}" if new_idx < 10 else f"CAM-{new_idx}"

    src = int(payload.source) if payload.source.isdigit() else payload.source
    CAMERA_SOURCES[new_id] = src

    new_cam = {
        "id": new_id,
        "name": payload.name,
        "location": payload.location,
        "source": payload.source,
        "status": "ONLINE"
    }
    SYSTEM_STATE["camera_registry"].append(new_cam)
    get_or_create_stream(new_id, src)
    return {"status": "success", "camera": new_cam}

@app.delete("/api/cameras/delete/{cam_id}")
def delete_camera(cam_id: str):
    if cam_id == "CAM-01":
        raise HTTPException(status_code=400, detail="Cannot delete default sensor CAM-01")

    SYSTEM_STATE["camera_registry"] = [c for c in SYSTEM_STATE["camera_registry"] if c["id"] != cam_id]
    if cam_id in CAMERA_SOURCES:
        del CAMERA_SOURCES[cam_id]
    if cam_id in camera_streams:
        camera_streams[cam_id].stop()
        del camera_streams[cam_id]
    if cam_id in active_tamper_map:
        del active_tamper_map[cam_id]
    return {"status": "deleted", "cam_id": cam_id}

@app.get("/video_feed/{cam_id}")
def video_feed_cam(cam_id: str):
    return StreamingResponse(generate_camera_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed")
def video_feed_default():
    return StreamingResponse(generate_camera_frames("CAM-01"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/set_line/{percentage}")
def set_line(percentage: int):
    CONFIG_STATE["boundary_percentage"] = max(15, min(85, percentage))
    return {"status": "updated", "boundary": CONFIG_STATE["boundary_percentage"]}

@app.get("/toggle_night/{status}")
def toggle_night(status: bool):
    CONFIG_STATE["night_enhancement"] = status
    return {"status": "updated", "night": CONFIG_STATE["night_enhancement"]}

@app.get("/enroll_officer/{name}")
def enroll_officer(name: str):
    clean_name = name.strip().upper()
    stream = get_or_create_stream("CAM-01", 0)
    success, frame = stream.read()
    if not success or frame is None:
        return {"status": "failed", "message": "Camera stream offline!"}

    frame_flip = cv2.flip(frame, 1)
    h, w = frame_flip.shape[:2]
    gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
    full_face = None

    try:
        if model_registry.face_detector is not None:
            faces = model_registry.face_detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(45, 45))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                pad_y = int(fh * 0.15)
                pad_x = int(fw * 0.15)
                full_face = frame_flip[max(0, fy-pad_y):min(h, fy+fh+pad_y), max(0, fx-pad_x):min(w, fx+fw+pad_x)]
    except Exception:
        full_face = None

    if full_face is not None and full_face.size > 0:
        file_path = os.path.join(FACES_DIR, f"{clean_name}.jpg")
        cv2.imwrite(file_path, full_face)
        load_enrolled_officers()
        return {"status": "success", "message": f"Officer {clean_name} Enrolled into Whitelist!"}

    return {"status": "failed", "message": "Face not detected! Look directly into camera."}

@app.get("/hitl_feedback/{rec_id}/{action}")
def hitl_feedback(rec_id: int, action: str):
    status_str = "CONFIRMED_THREAT" if action == "confirm" else "SUPPRESSED_FALSE_ALARM"
    update_hitl_status(rec_id, status_str)
    for rec in SYSTEM_STATE["forensic_records"]:
        if rec["id"] == rec_id:
            rec["status"] = status_str
            break
    return {"status": "updated"}

@app.get("/system_telemetry")
def get_telemetry():
    return SYSTEM_STATE

@app.get("/")
def home():
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())