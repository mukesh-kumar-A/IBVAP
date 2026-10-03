# IBVAP: 4-Minute SIH 2026 Hackathon Live Demonstration Protocol

This script provides an exact minute-by-minute walkthrough to demonstrate complete compliance with Problem Statement 26187 ("AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV infrastructure").

---

### PRE-DEMO CHECKLIST (T - 2 Minutes)
1. Ensure your virtual environment is active: `source .venv/bin/activate` or `.venv\Scripts\activate`.
2. Run preflight: `python check_setup.py`. Confirm the verdict is `READY`.
3. Launch server: `python run.py`.
4. Open Chrome at `http://127.0.0.1:8000`. Note the auto-generated or configured credentials in your console.
5. Have a test video file (or webcam 0) ready.

---

### MINUTE 1: AUTHENTICATION, RBAC & SOFTWARE-ONLY INGESTION
- **Action:**
  1. Open the UI. Show the judge the lock overlay: *"The platform runs strict dual-role RBAC."*
  2. Log in with `admin` and your password.
  3. Point out the top role badge (`● ADMIN`) and the **Face Engine** badge.
  4. Explain: *"IBVAP turns standard COTS CCTV cameras into intelligent sensors entirely through software. No specialized AI hardware or GPUs are required."*
  5. Click **+ Add Camera**, type Name `Test Sector Video`, Location `Post 4`, and enter a video file path (e.g. `sample.mp4` or webcam `0`). Click **Add Camera**.
- **What the Judge Sees:**
  - Dynamic camera tile appears immediately in the grid with active live streaming.

---

### MINUTE 2: VIRTUAL TRIPWIRES, NIGHT MOTION & SUSPICIOUS BEHAVIOR
- **Action:**
  1. Switch to the **Zone Editor** tab.
  2. Select `CAM-01`. Click 2 points across the video to draw a virtual tripwire boundary. Set direction to `INBOUND` and click **Save Zones**.
  3. Walk across the line or let the video subject cross the tripwire.
  4. Point out the real-time detection: Bounding boxes change to red (`BREACH: PERSON #1`), kinetic velocity vector arrows display heading and ETA.
  5. Demonstrate Night-Time Motion: Point to CLAHE toggle or HSV auto-switch. Explain: *"At night, HSV saturation drops and MOG2 background subtraction isolates low-light unlit motion even if deep models degrade."*
  6. Demonstrate Suspicious Rules: Sudden velocity triggers `RUNNING / EVASIVE MOVEMENT`; stationary target triggers `ABANDONED OBJECT`.
- **What the Judge Sees:**
  - Real-time threat classification banner elevates to `CODE RED: VIRTUAL FENCE INTRUSION`.
  - Forensic Audit Vault card appears instantly on the dashboard with a snapshot image.

---

### MINUTE 3: BIOMETRIC DUAL-LIST (OFFICER SUPPRESSION) & ANPR
- **Action:**
  1. Switch to the **Biometrics & ANPR** tab.
  2. Show the Face Engine status:
     - If ArcFace ONNX is present: `FACE ENGINE: ARCFACE`.
     - If running without weights: `FACE ENGINE: FALLBACK (NOT DEEP LEARNING)`. Explain: *"We never fake deep metrics. In fallback mode, suspect alerts are downgraded to avoid false panic."*
  3. Enrol the judge or yourself as `OFFICER SHARMA` under `WHITELIST`.
  4. Show the camera: The bounding box turns green (`OFFICER: OFFICER SHARMA`), and breach alarms are suppressed.
  5. Add a plate to the Watchlist: `DL01AB1234` with reason `Stolen Recon Vehicle`.
  6. Present a vehicle image or card with `DL01AB1234`. The OCR extracts the plate, matches the watchlist, and flags `HOTLIST PLATE: DL01AB1234`.
- **What the Judge Sees:**
  - Live suppression for friendly officers and immediate alerts for watchlisted suspects.

---

### MINUTE 4: C2 INTEGRATION, WEBHOOKS & HASH-CHAIN FORENSIC AUDIT
- **Action:**
  1. Open a new terminal tab. Query the military Command & Control (C2) endpoint using curl and the C2 API key:
     ```bash
     curl -H "X-API-Key: <C2_KEY_FROM_STARTUP>" "http://127.0.0.1:8000/api/events?threat_level=CODE%20RED"
     ```
  2. Show the structured JSON response: timestamp, velocity, heading, camera ID, snapshot path, and chained hashes.
  3. Explain the store-and-forward alert dispatcher: *"If border telecom links fail, events queue locally in SQLite and forward automatically upon link restoration."*
  4. On the UI Events tab, click **VERIFY CHAIN INTEGRITY**.
  5. The UI verifies every SHA-256 block mathematically and displays: `✅ CHAIN INTACT — NO TAMPERING DETECTED`.
  6. Finally, click **Scramble (Sim)** on the QRT Drone:
     *"The QRT Drone is explicitly labeled as a SIMULATED mission workflow to illustrate how autonomous interception links with CCTV analytics."*
- **What the Judge Sees:**
  - Courtroom-admissible cryptographic proof of forensic record integrity.
  - Complete compliance with all PS 26187 criteria.

---

### BACKUP & FALLBACK CONTINGENCIES
1. **Camera Hardware Lock / Device Disconnected:**
   - If webcam `0` is locked by another program, open `configs/cameras.yaml`, set `source: ""` on CAM-01, and add a pre-recorded `.mp4` file or synthetic feed via the UI.
2. **ArcFace Weights Absent:**
   - The platform will run cleanly in Sobel gradient fallback mode. The UI clearly labels: `FACE ENGINE: FALLBACK (NOT DEEP LEARNING)`. Highlight this transparency to the judges as an engineering strength.
3. **Network Disconnection:**
   - The platform runs 100% offline. Zero internet access is required.