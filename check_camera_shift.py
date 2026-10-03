"""
Quick webcam check for the camera-movement / angle-change detector.

Run from the project folder:   python check_camera_shift.py
1. For the first 15 s keep the camera STILL and move around in front of it normally
   (sit, wave, walk). No alarm should appear.
2. For the next 15 s turn the laptop screen / camera a little. An alarm should appear.
Send the printed summary if anything looks wrong.
"""
import os
import time
import collections
import cv2

from app.inference import check_lens_tampering

backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, backend)
if not cap.isOpened():
    raise SystemExit("Camera 0 could not be opened. Close other apps using the camera and retry.")

print("PHASE 1 (0-15 s): keep the camera still, move normally in front of it")
counts = {1: collections.Counter(), 2: collections.Counter()}
frames = {1: 0, 2: 0}
start = time.time()
announced = False
while time.time() - start < 30:
    ok, frame = cap.read()
    if not ok:
        continue
    phase = 1 if time.time() - start < 15 else 2
    if phase == 2 and not announced:
        announced = True
        print("PHASE 2 (15-30 s): now turn / tilt the camera a little")
    frame = cv2.flip(frame, 1)
    tampered, reason = check_lens_tampering(frame, False, "DIAG")
    frames[phase] += 1
    counts[phase][reason if tampered else "CLEAR"] += 1
cap.release()

for phase in (1, 2):
    print(f"\nPhase {phase}: {frames[phase]} frames")
    for reason, n in counts[phase].most_common():
        print(f"   {reason:45s} {n:5d} frames ({100.0 * n / max(1, frames[phase]):.0f}%)")
print("\nExpected: phase 1 almost all CLEAR, phase 2 shows ANGLE CHANGED or SHAKEN.")
