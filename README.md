# IBVAP — Intelligent Border Video Analytics Platform

> **AI-based video analytics for border surveillance using existing CCTV infrastructure — software only, CPU only.**
> **Smart India Hackathon 2026** · Problem Statement ID: **26187**
> **Theme:** Blockchain & Cybersecurity · **Domain:** AI-Based Intelligent Video Analytics Platform for Border Surveillance
> **Team:** Non-Patchable-Coders (NPC) · **Team ID:** 119573

---

## 1. The Problem

**Problem Statement 26187:** *AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure.*


Border Out Posts (BOPs), check posts and border roads already have CCTV cameras, but conventional CCTV only records and streams video. A human has to watch every feed continuously. Face recognition, ANPR, intrusion detection and tracking usually need dedicated smart cameras or proprietary servers, which are expensive and hard to deploy in remote areas.

## 2. Our Approach

IBVAP is a software layer that sits on top of standard IP / RTSP / USB cameras that are already installed. It reads the video streams, runs AI analytics on an ordinary CPU machine, and turns raw footage into **alerts, evidence and a queryable event log** that a command-and-control (C2) system can consume.

No smart cameras. No GPU required. No per-channel vendor licence.

---

## 2A. Theme Alignment: Blockchain & Cybersecurity

Surveillance evidence is only useful if it can be trusted. IBVAP therefore treats the platform's own security and evidence integrity as core features, not add-ons.

| Theme element | What IBVAP implements | Module |
|---|---|---|
| **Blockchain-style tamper-evident ledger** | Every threat event is stored as a record whose SHA-256 hash covers the *previous* record's hash, forming a hash chain. Editing or deleting any past record breaks every later hash, and `/api/verify_forensic_chain` recomputes the chain and reports the first broken record. | `app/database.py` |
| **Evidence linkage** | Each ledger record references its snapshot / clip so evidence and log entry are tied together. | `app/database.py`, `/forensic_clip/{id}` |
| **Access control** | Admin / Operator roles; infrastructure, biometrics and zone changes are Admin-only. | `app/main.py` |
| **Authentication hardening** | Salted PBKDF2-SHA256 password hashing, expiring bearer tokens, API-key authentication for external C2 systems, login lockout after 5 failures. | `app/main.py` |
| **No default secrets at runtime** | If passwords or API keys are not supplied, random ones are generated at start-up instead of using a fixed default. | `app/main.py` |
| **Privacy by locality** | All inference runs on the local machine; no video is sent to a cloud service. | whole pipeline |
| **Camera tamper detection** | Heuristics flag a blocked or moved camera view, which is a physical-security attack on the surveillance system itself. | `app/main.py` |

**Honest scope:** this is a *hash-chained ledger on a single node*, not a distributed blockchain with consensus. It makes tampering **detectable**, not impossible: someone with full database access could rebuild the whole chain. Anchoring the latest hash to an external, independent store is on the roadmap (section 16).

---

## 3. Problem Statement Coverage

| # | Requirement from the problem statement | Status | Where |
|---|---|---|---|
| 1 | Ingest live streams from standard IP/RTSP CCTV | **Implemented** | `MultiCameraStream`, `CameraWorker` — `app/main.py` |
| 2 | Human detection and tracking | **Implemented** | YOLOv8 + ByteTrack — `app/inference.py`, `app/tracker.py` |
| 3 | Vehicle detection and classification | **Implemented** | `resolve_detection_class` — `app/inference.py` (CAR, MOTORCYCLE, BUS, TRUCK, VAN, BICYCLE) |
| 4 | Face detection | **Implemented** | OpenCV YuNet, Haar cascade fallback — `app/recognition.py` |
| 5 | Facial recognition (software-only) | **Implemented** (ArcFace); **falls back** to a weak classical matcher if the ONNX model is missing | `ArcFaceRecognizer` — `app/recognition.py` |
| 6 | ANPR | **Implemented** (contour-based plate localisation + EasyOCR + Indian plate format validation + multi-frame voting) | `extract_vehicle_plate` — `app/recognition.py` |
| 7 | Virtual fence intrusion detection | **Implemented** (polygons and lines, any-part-of-box crossing, optional INBOUND/OUTBOUND direction) | `box_intersects_zone` — `app/tracker.py` |
| 8 | Suspicious activity detection | **Implemented** (rule-based: running, group gathering, abandoned object, halted vehicle, wrong-direction movement) | `detect_suspicious_activities` — `app/tracker.py` |
| 9 | Night-time movement detection | **Implemented** (HSV day/night detection, CLAHE enhancement, MOG2 motion detection) | `detect_day_night_mode`, `NightMotionDetector` — `app/inference.py` |
| 10 | Real-time alerts and event logging | **Implemented** (WebSocket push to dashboard; webhook / MQTT / SMTP / SMS-webhook channels; hash-chained SQLite event vault) | `app/alerts.py`, `app/websocket_manager.py`, `app/database.py` |
| 11 | Integration with command and control systems | **Implemented** (REST API with API-key auth, filtered event query, CSV export, WebSocket telemetry, outbound webhook/MQTT) | `/api/events`, `/api/events/export`, `/ws/telemetry` |
| 12 | Cost-effective, CPU-only, scalable | **Implemented, benchmark pending** (frame skipping, inference downscaling, per-camera worker threads, Docker packaging) | `CameraWorker`, `configs/config.yaml` |
| — | Quick-reaction drone dispatch (not in the problem statement) | **SIMULATED** | `qrt_drone_mission_daemon` — `app/main.py` |

---

## 4. Key Features

- **Per-camera background workers.** Detection runs continuously whether or not anyone is watching the dashboard. The video endpoint only streams the latest annotated frame.
- **Zone editor.** Draw polygons and tripwires on the live feed from the dashboard; zones are saved to `configs/zones.yaml`.
- **Configurable thresholds** in `configs/thresholds.yaml` (running speed, group size, loiter time, night-motion sensitivity).
- **Two-list face matching.** A *whitelist* (friendly personnel; suppresses intrusion alarms for them) and a *watchlist* (persons of interest). In fallback mode the watchlist alerts are downgraded to LOW CONFIDENCE.
- **Plate watchlist.** Raise an alert when a listed number plate is read.
- **Tamper-evident forensic vault.** Every event is stored with a SHA-256 hash that includes the previous record's hash. A verification endpoint recomputes the chain and reports any modified row.
- **Human-in-the-loop review.** Operators can confirm or suppress alerts.
- **Store-and-forward alerting.** If a webhook/MQTT target is unreachable, alerts are queued in memory and retried.
- **Camera tamper heuristics** (blocked or moved camera view).
- **Role-based access.** Admin and Operator roles.

---

## 5. Architecture

```
 IP / RTSP / USB cameras
          │
          ▼
 CameraWorker (one thread per camera)
   frame skip + downscale
   → YOLOv8 + ByteTrack (people, vehicles)
   → zone / tripwire logic, suspicious-activity rules
   → night mode: HSV detection + CLAHE + MOG2 motion
   → YuNet face detection → ArcFace match (whitelist / watchlist)
   → plate localisation → EasyOCR → format validation → voting
          │
          ▼
 Event pipeline
   hash-chained SQLite vault  +  snapshot / clip evidence
   AlertDispatcher (webhook · MQTT · SMTP · SMS-webhook, retry queue)
   WebSocket push → dashboard
          │
          ▼
 FastAPI  →  Web dashboard   |   C2 REST API (X-API-Key)
```

---

## 6. Quick Start

### Prerequisites
- Python 3.10 or 3.11 (developed on Windows with Python 3.11; the Dockerfile uses 3.10)
- Git
- A webcam, a video file, or an RTSP camera URL

### Install

**Windows (PowerShell)**
```powershell
git clone <https://github.com/mukesh-kumar-A/IBVAP>
cd IBVAP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

**Linux / macOS**
```bash
git clone <https://github.com/mukesh-kumar-A/IBVAP>
cd IBVAP
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set your own passwords and API key (see section 8).

### Download model files

```bash
python download_models.py
```

This fetches the YuNet face detector and the ArcFace (`w600k_mbf.onnx`) model into `models/face/`. If your network blocks the download, the script prints manual instructions. YOLOv8n weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first run.

Model files are **not** stored in this repository because of their size.

### Check the setup, then run

```bash
python check_setup.py        # prints PASS / WARN / FAIL and a READY verdict
python run.py                # starts the server on port 8000
```

Open `http://127.0.0.1:8000` and log in. If you did not set passwords in `.env`, random ones are generated and printed in the console at start-up.

Options: `python run.py --host 127.0.0.1 --port 8080`

### Run the tests

```bash
pytest tests/test_core.py -v
```

The tests cover Indian plate normalisation and validation, line and polygon intersection, crossing direction, hash-chain tamper detection, C2 API-key rejection and config schema loading. They need no camera and no internet.

---

## 7. Docker

```bash
cp .env.example .env
docker compose up --build
```

The dashboard is served on `http://localhost:8000`. The compose file mounts `data/`, `forensic_logs/`, `known_faces/`, `configs/` and `models/` from the host.

> **Status:** the Docker setup is provided but has **not been tested end-to-end** by the team. Place the model files in `./models/face/` on the host before starting, because that folder is mounted over the container's copy.

---

## 8. Configuration

### Environment variables (`.env`)

| Variable | Purpose | If not set |
|---|---|---|
| `IBVAP_ADMIN_PASSWORD` | Admin login password | Random password printed at start-up |
| `IBVAP_OPERATOR_PASSWORD` | Operator login password | Random password printed at start-up |
| `IBVAP_ADMIN_SALT`, `IBVAP_OPERATOR_SALT` | Salts for PBKDF2 password hashing | Random per start |
| `IBVAP_C2_API_KEYS` | Comma-separated API keys for external C2 systems | Random key generated and printed |
| `DB_PATH` | SQLite database location | `data/defense_audit.db` |

> The values in `.env.example` are **placeholders for local demos only**. Always set your own before any real deployment. Never commit `.env`.

### YAML files in `configs/`

| File | Contents |
|---|---|
| `config.yaml` | Frame skip, inference size, ONNX option, alert channels (webhook, MQTT, SMTP, SMS), store-and-forward limits |
| `cameras.yaml` | Camera list: id, name, location, source, mirror flag, `pixels_per_meter` |
| `zones.yaml` | Per-camera polygons and tripwires |
| `thresholds.yaml` | Day/night, night-motion and suspicious-activity thresholds |

For weak hardware, raise `frame_skip` and lower `inference_width` / `inference_height` in `config.yaml`.

---

## 9. Adding a Camera

**From the dashboard (Admin):** click **+ Add Camera**, then enter name, location and source (`0` for a webcam, a video file path, or an RTSP URL such as `rtsp://user:pass@192.168.1.50:554/stream`). Optionally set pixels-per-meter.

**From `configs/cameras.yaml`:**
```yaml
cameras:
  - id: CAM-03
    name: Western Strip Sensor
    location: Sector C - Boundary Post
    source: "rtsp://192.168.1.50:554/live"
    mirror: false
    pixels_per_meter: 42.0
    status: ONLINE
```
Restart the server after editing the file.

---

## 10. Command & Control Integration

### Query events (external system)

```bash
curl -H "X-API-Key: <YOUR_C2_KEY>" \
  "http://127.0.0.1:8000/api/events?page=1&page_size=10"
```

Filters (time range, camera, threat level, type, status) and the CSV export at `/api/events/export` are documented in the interactive API docs at `http://127.0.0.1:8000/docs`.

### Push alerts out

Enable a channel in `configs/config.yaml` (`webhook`, `mqtt`, `email_smtp`, `sms_webhook`). To test locally, point the webhook URL at a listener such as [webhook.site](https://webhook.site) and trigger an intrusion.

### Main endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/login` | Obtain a session token |
| `GET /api/events`, `GET /api/events/export` | C2 event query and CSV export |
| `WS /ws/telemetry` | Live telemetry and alerts |
| `GET /video_feed/{cam_id}` | Annotated MJPEG stream |
| `GET/POST /api/zones/{cam_id}` | Read / save zones |
| `GET/POST /api/thresholds` | Read / save thresholds |
| `/api/biometrics/*`, `/api/anpr/watchlist` | Face and plate lists |
| `POST /api/cameras/add`, `DELETE /api/cameras/delete/{cam_id}` | Camera management |
| `GET /api/verify_forensic_chain` | Verify the hash chain |
| `GET /system_telemetry` | FPS, face-engine status, system state |

---

## 11. Security

- **Authentication:** session tokens sent as `Authorization: Bearer <token>`. The `?token=` query form is accepted only where a browser cannot set headers (video stream, downloads, event query).
- **C2 systems** authenticate with an `X-API-Key` header.
- **Passwords** are stored as salted PBKDF2-SHA256 hashes. Credentials come from environment variables or are generated randomly at first start.
- **Login rate limiting:** 5 failed attempts lock the source IP for 5 minutes.
- **RBAC:** Operators monitor, review alerts and use the (simulated) drone control. Admins manage cameras, zones, thresholds and biometrics.
- **Evidence integrity:** the audit vault is a SHA-256 hash chain, so editing or deleting a past row is **detectable**. It is tamper-*evident*, not tamper-*proof*: someone with full database access could rebuild the entire chain, so for real use the latest hash should be anchored externally.

---

## 12. Performance

We do not publish performance numbers we have not measured. To measure on your own hardware:

```bash
python benchmark.py --source 0 --duration 10
python benchmark.py --source path/to/video.mp4 --duration 30
```

It reports FPS, per-stage latency (detection, tracking, face, ANPR), CPU and memory for 1, 2 and 4 streams.

| Streams | FPS (total) | FPS / camera | CPU % | RAM (MB) |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |

*Not yet measured. Replace with the output of `benchmark.py` and name the machine used.*

`evaluate.py` is provided for precision/recall on a labelled test set. No accuracy figures are claimed.

---

## 13. Cost Comparison (illustrative)

> **This is an estimate based on the assumptions below, not a quotation.** Replace the figures with real vendor quotes.

**Assumptions:** one BOP, 8 camera positions, 3-year life, existing cameras and cabling are reusable.

| Item | Dedicated smart-camera approach | IBVAP (software on existing cameras) |
|---|---|---|
| Cameras | 8 × AI smart camera @ ~₹85,000 = ₹6,80,000 | Existing cameras reused: ₹0 |
| Central server | ~₹2,50,000 | One CPU mini-PC / workstation, ~₹60,000 *(8-stream capacity not yet benchmarked)* |
| Cabling / installation | ~₹1,20,000 | Existing network reused: ₹0 |
| Analytics licences | 8 channels × ~₹15,000 × 3 years = ₹3,60,000 | No licence fee |
| **Total** | **~₹14,10,000** | **~₹60,000 + support and maintenance** |

The software-only column excludes engineering, deployment and maintenance effort, and the compute sizing depends on the benchmark results above.

---

## 14. Limitations (please read)

1. **Not validated on real border footage.** We have not tested in field conditions (fog, dust, snow, long range, IR cameras). Accuracy in those conditions is unknown.
2. **ANPR** localises plates with image contours, not a trained plate detector. It struggles with steep angles, motion blur, dirt, low light and small plates, and returns `UNREADABLE` rather than guessing. A trained YOLO plate model in `models/plate/best.pt` is used automatically if present.
3. **Speed values** (km/h) are derived from pixel motion. Unless `pixels_per_meter` is set per camera, they are labelled *estimated (uncalibrated)*. Running detection depends on this.
4. **Suspicious activity** detection is rule-based with configurable thresholds, not a trained behaviour-recognition model. Expect false alarms until thresholds are tuned per camera.
5. **Face recognition** accuracy depends on face size, angle and lighting; faces on typical wide-angle CCTV at distance are often too small to match. If the ArcFace model file is missing, the system uses a weak classical matcher and says so in the dashboard and API.
6. **Night detection** works on what the camera provides. Without IR illumination, very dark scenes give little signal. MOG2 motion detection can be triggered by wind, shadows or insects.
7. **Alert queue is in memory.** Queued alerts are lost if the server restarts. The event vault itself is persistent.
8. **Drone dispatch is simulated.** It does not control any real drone.
9. **CPU load.** Throughput depends on the machine and the number of cameras; use `frame_skip` and smaller inference size on weak hardware.
10. **Docker packaging is untested** (see section 7). The default `.env.example` passwords are demo placeholders.

---

## 15. Project Structure

```
app/
  main.py               FastAPI app, workers, auth, API, WebSocket
  inference.py          YOLO, ByteTrack, night mode, class mapping
  tracker.py            trajectories, zones, suspicious-activity rules
  recognition.py        face detection/recognition, ANPR
  alerts.py             alert dispatcher (webhook, MQTT, SMTP, SMS)
  websocket_manager.py  live push to the dashboard
  database.py           SQLite hash-chained vault
  analytics.py          threat scoring
  config.py             config loading
  models_loader.py      model loading, CPU options
configs/                config.yaml, cameras.yaml, zones.yaml, thresholds.yaml
templates/index.html    dashboard
tests/test_core.py      offline unit tests
benchmark.py            measure FPS / latency / CPU / memory
evaluate.py             precision / recall on a labelled set
check_setup.py          pre-flight check
download_models.py      fetch face models
run.py                  launcher
Dockerfile, docker-compose.yml
DEMO_SCRIPT.md          4-minute demo walkthrough
```

---

## 16. Roadmap

- Train and bundle a dedicated licence-plate detector for Indian plates
- Field testing on real BOP / border-road footage and publish measured accuracy
- Camera calibration tool for true speed and distance
- Persistent (on-disk) alert queue
- ONNX / OpenVINO optimised inference for low-power edge boxes
- Behaviour models (crawling, carrying objects) beyond rule-based heuristics
- Anchor the vault's latest hash to an external, independent store (separate node, trusted timestamping service or a distributed ledger) so a full chain rebuild is also detectable
- Real drone / PTZ integration (MAVLink / ONVIF)

---

## 17. Demo

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for a 4-minute walkthrough: login, camera add, fence intrusion, night motion, suspicious activity, ANPR, face lists, C2 query, and hash-chain verification.

---

## 18. Team

**Team Non-Patchable-Coders (NPC)** · Team ID **119573**
Smart India Hackathon 2026 · Problem Statement 26187
