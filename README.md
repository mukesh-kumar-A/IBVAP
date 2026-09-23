# IBVAP: Intelligent Border Video Analytics Platform

> **Sovereign Defense AI Platform for Real-Time Perimeter Surveillance & Tactical Threat Interception**

## Executive Summary

**IBVAP (Intelligent Border Video Analytics Platform)** is an autonomous, mission-critical edge video analytics platform engineered for isolated Border Outposts (BOP) and tactical forward defense lines. Designed with zero external cloud dependencies, it fuses edge computer vision, multi-target trajectory estimation, low-light night-vision enhancement, optical sensor anti-tampering heuristics, and autonomous Quick Reaction Team (QRT) drone scramble protocols into a unified command dashboard.

## Core Architectural Highlights

### 1. Zero-Cloud Sovereign Offline Edge Deployment

* Operates completely air-gapped on localized defense workstations without internet connectivity or remote API calls.

* Asynchronous ASGI core built on **FastAPI** and **Uvicorn** ensuring sub-15ms system latency.

### 2. Strict "Any-Part" Virtual Tripwire Logic

* Resolves conventional boundary false alarms by enforcing a microsecond bounding box intersection check (y1<=line_y).

* Instantly registers an alert if any component of an intruder (head, limb, torso) or vehicle (tyre, bumper) touches or breaches the demarcation threshold.

### 3. Non-Blocking Native MJPEG Multi-Camera Socket Engine

* Ingests concurrent video streams across integrated laptop webcams, tactical USB cameras, and mobile/remote DroidCam RTSP/HTTP endpoints.

* Decoupled multi-threaded socket ingestion isolates OpenCV buffer bloat, preventing video stream freeze during high-density multi-camera usage.

### 4. Optical Anti-Tampering Sentinel

* Monitors camera sensor integrity using second-order Laplacian spatial derivatives.

* Instantly detects lens occlusion, spray paint attacks, defocusing, and lens blinding, broadcasting localized alarms with audio synthesis.

### 5. Human-in-the-Loop (HITL) Forensic Audit Vault

* Bridges edge AI predictions with human command discretion via an active learning verification loop.

* Operators can inspect high-resolution incident frames in a lightbox and log decisions (`CONFIRM` vs `SUPPRESS`) into an encrypted local SQLite forensic database.

### 6. Autonomous QRT Drone Interception System

* Computes intruder velocity vectors, azimuth headings, and base-of-patrol (BOP) ETAs in real time.

* Provides a tactical one-click QRT Drone Scramble pipeline that transitions through autonomous mission states (`STANDBY` ➔ `SCRAMBLED` ➔ `AIRBORNE` ➔ `INTERCEPTING` ➔ `RTB`) with a dynamic tactical HUD countdown.

### 7. Dual Night Vision & Biometric Defense Whitelist

* **CIELAB CLAHE Pipeline**: Enhances the $L^*$ luminance channel using 8x8 adaptive tiles, restoring critical contrast in low-light night sectors without color distortion.

* **Biometric FRS Verification**: Rapid localized face template matching whitelisting friendly patrol officers while isolating unverified Persons of Interest (POI).

##  Technology Stack

| **Layer** | **Component / Technology** | 
| **Deep Learning & Vision** | Ultralytics YOLOv8, ByteTrack Multi-Object Tracking, OpenCV | 
| **Edge Image Processing** | CIELAB CLAHE Night Vision, Laplacian Blur/Tamper Operator | 
| **Backend & Concurrency** | FastAPI, Uvicorn (ASGI), Python Threading, SQLite3 | 
| **Tactical Dashboard** | HTML5, Tailwind CSS, Native Web Audio API (AudioContext) | 
| **Protocol Ingestion** | DirectShow, FFmpeg, Raw HTTP/MJPEG Sockets | 

##  Project Structure

```
ibvap2.0/
├── app/
│   ├── analytics.py        # Threat level assessment and interception vectors
│   ├── config.py           # Threshold constants and runtime system states
│   ├── database.py         # SQLite forensic threat audit vault
│   ├── inference.py        # YOLO object detection, CLAHE, and anti-tampering
│   ├── main.py             # FastAPI backend controller & streaming engine
│   ├── models_loader.py    # Model weights registry
│   ├── recognition.py      # Face biometrics (FRS) & license plate OCR (ANPR)
│   ├── tracker.py          # ByteTrack + Kalman filter tracking pipeline
│   └── utils.py            # Coordinate mapping utilities
├── templates/
│   └── index.html          # High-performance tactical surveillance dashboard
├── models/                 # Local YOLOv8 weights (yolov8n.pt)
├── known_faces/            # Enrolled friendly personnel biometrics
├── requirements.txt        # Python dependency manifest
└── README.md               # Technical project documentation

```

##  Quick Start & Deployment

### 1. Clone the Repository

```
git clone https://github.com/mukesh-kumar-A/IBVAP.git
cd IBVAP

```

### 2. Set Up Virtual Environment

```
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```

### 3. Install Required Dependencies

```
pip install -r requirements.txt

```

### 4. Launch the Tactical Command Center

```
python -m uvicorn app.main:app --reload

```

Open your browser and navigate to **`http://127.0.0.1:8000`** to access the live command center.

##  Research References

1. **Object Detection**: Redmon et al., *"You Only Look Once: Unified, Real-Time Object Detection"*, [link](https://arxiv.org/pdf/2408.15857).

2. **Multi-Object Tracking**: Zhang et al., *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"*, ECCV, [link](https://arxiv.org/abs/2110.06864).

3. **Adaptive Contrast Enhancement**: *"OpenCV: CLAHE Module OpenCV Maintainers(Documentation)"*,
[link](https://ieeexplore.ieee.org/document/6607556)

4. **Edge Face Efficient Recognition Model**: *": Efficient CNNs for Accurate Real-time Face Verification"*, [link](https://arxiv.org/pdf/1804.07573)

5. **Lens Tampering Heuristics**: *"IEEE Abstract and Research papers"*, [link](https://ieeexplore.ieee.org/document/6607556)

6. **Human-in-the-Loop AI**: Wu et al., *"A Survey of Human-in-the-loop for Machine Learning"*, FGCS 2022, [link](https://arxiv.org/abs/2108.00941).

7. **ANPR Detection using Deep Learning and OCR**: *"IEEE Research Documentation"*, [link](https://ieeexplore.ieee.org/document/10911385)
##  Authors & Team

* **Team Name:** **Non-Patchable-Coders**

* **Lead Developers & System Architects:** Team Non-Patchable-Coders

* **Event:** Smart India Hackathon (SIH) — Border Security & Defense Video Analytics Domain
