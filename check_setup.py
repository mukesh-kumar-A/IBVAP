import os
import sys
import socket
import importlib
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT = []


def record(category, test_name, status, detail):
    REPORT.append({
        "category": category,
        "name": test_name,
        "status": status,
        "detail": detail
    })


def check_python():
    ver = sys.version_info
    target = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver.major == 3 and ver.minor == 10:
        record("Runtime", "Python Version", "PASS", f"Python {target} (Target 3.10)")
    elif ver.major == 3 and ver.minor in (9, 11, 12):
        record("Runtime", "Python Version", "WARN", f"Python {target} (Recommended: 3.10)")
    else:
        record("Runtime", "Python Version", "FAIL", f"Python {target} unsupported")


def check_imports():
    packages = [
        ("cv2", "OpenCV Computer Vision"),
        ("ultralytics", "YOLOv8 Real-Time Object Engine"),
        ("onnxruntime", "ONNX CPU Runtime Engine"),
        ("easyocr", "EasyOCR Optical Character Recognition"),
        ("numpy", "NumPy Numerical Mathematics"),
        ("fastapi", "FastAPI Core Framework"),
        ("uvicorn", "Uvicorn ASGI Web Server"),
        ("pydantic", "Pydantic Schema Serialization"),
        ("scipy", "SciPy Spatial Mathematics"),
        ("yaml", "PyYAML Configuration Parser"),
        ("requests", "Requests HTTP Client"),
        ("paho.mqtt", "Paho MQTT Protocol Client"),
        ("psutil", "PSUtil Process Telemetry"),
    ]
    for mod_name, label in packages:
        try:
            importlib.import_module(mod_name)
            record("Packages", label, "PASS", "Importable")
        except ImportError as e:
            record("Packages", label, "FAIL", f"Missing ({e})")


def check_models():
    yolo_pt = os.path.join(BASE_DIR, "yolov8n.pt")
    yunet_onnx = os.path.join(BASE_DIR, "models", "face", "face_detection_yunet_2023mar.onnx")
    arcface_onnx = os.path.join(BASE_DIR, "models", "face", "w600k_mbf.onnx")
    plate_pt = os.path.join(BASE_DIR, "models", "plate", "best.pt")

    if os.path.exists(yolo_pt):
        record("Models", "YOLOv8 Base Model (yolov8n.pt)", "PASS", "Available")
    else:
        record("Models", "YOLOv8 Base Model (yolov8n.pt)", "WARN", "Absent (Auto-downloaded on first start)")

    if os.path.exists(yunet_onnx):
        record("Models", "YuNet Face Detector (ONNX)", "PASS", "Available")
    else:
        record("Models", "YuNet Face Detector (ONNX)", "WARN", "Absent (Runs Haar cascade fallback)")

    if os.path.exists(arcface_onnx):
        record("Models", "ArcFace 512-D Recognizer (ONNX)", "PASS", "Available")
    else:
        record("Models", "ArcFace 512-D Recognizer (ONNX)", "WARN", "Absent (Runs Sobel fallback - watchlist alerts demoted)")

    if os.path.exists(plate_pt):
        record("Models", "Trained License Plate YOLO", "PASS", "Dedicated YOLO Plate Model Available")
    else:
        record("Models", "Trained License Plate YOLO", "WARN", "Absent (Using contour-based ANPR localization)")


def check_configs_and_dirs():
    configs = ["config.yaml", "cameras.yaml", "zones.yaml", "thresholds.yaml"]
    for c in configs:
        p = os.path.join(BASE_DIR, "configs", c)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
                record("Configuration", f"configs/{c}", "PASS", "Valid YAML")
            except Exception as e:
                record("Configuration", f"configs/{c}", "FAIL", f"Parse error: {e}")
        else:
            record("Configuration", f"configs/{c}", "FAIL", "File missing")

    dirs = ["data", "forensic_logs", "known_faces"]
    for d in dirs:
        p = os.path.join(BASE_DIR, d)
        try:
            os.makedirs(p, exist_ok=True)
            test_file = os.path.join(p, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            record("Filesystem", f"Directory {d}/", "PASS", "Writable")
        except Exception as e:
            record("Filesystem", f"Directory {d}/", "FAIL", f"Permission error: {e}")


def check_port():
    port = 8000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        res = s.connect_ex(("127.0.0.1", port))
        if res == 0:
            record("Network", f"Port {port}", "WARN", f"Port {port} is currently in use")
        else:
            record("Network", f"Port {port}", "PASS", "Available")


def check_cameras():
    cameras_yaml = os.path.join(BASE_DIR, "configs", "cameras.yaml")
    if not os.path.exists(cameras_yaml):
        return

    try:
        import cv2
        with open(cameras_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cams = data.get("cameras", []) if data else []
        for c in cams:
            cid = c.get("id", "CAM")
            src = str(c.get("source", "")).strip()
            if not src:
                record("Sensors", f"Camera {cid}", "WARN", "Source empty (Configured in STANDBY mode)")
                continue

            if src.isdigit():
                cap = cv2.VideoCapture(int(src))
                opened = cap.isOpened()
                cap.release()
                if opened:
                    record("Sensors", f"Camera {cid} (USB {src})", "PASS", "Device opened successfully")
                else:
                    record("Sensors", f"Camera {cid} (USB {src})", "WARN", "Camera index not responding")
            else:
                # Test network URL with 3-second timeout
                try:
                    import urllib.request
                    test_url = src if src.startswith(("http://", "https://")) else f"http://{src}"
                    req = urllib.request.Request(test_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=3.0) as resp:
                        record("Sensors", f"Camera {cid} (Network)", "PASS", f"Stream reachable ({resp.status})")
                except Exception as e:
                    record("Sensors", f"Camera {cid} (Network)", "WARN", f"Stream unreachable ({e})")
    except Exception as e:
        record("Sensors", "Camera Check", "WARN", f"Could not inspect cameras: {e}")


def main():
    print("=" * 80)
    print(" IBVAP PREFLIGHT AUDIT & SYSTEM SELF-CHECK")
    print("=" * 80)

    check_python()
    check_imports()
    check_models()
    check_configs_and_dirs()
    check_port()
    check_cameras()

    print(f"\n{'CATEGORY':<14} | {'CHECK ITEM':<36} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 80)
    has_fail = False
    has_warn = False

    for r in REPORT:
        if r["status"] == "FAIL":
            has_fail = True
        if r["status"] == "WARN":
            has_warn = True
        print(f"{r['category']:<14} | {r['name']:<36} | {r['status']:<6} | {r['detail']}")

    print("=" * 80)
    if has_fail:
        print("VERDICT: NOT READY — Resolve FAIL items before starting the platform.")
        sys.exit(1)
    elif has_warn:
        print("VERDICT: READY (WITH OPERATIONAL WARNINGS) — Platform is runnable.")
        sys.exit(0)
    else:
        print("VERDICT: READY — System fully hardened and ready for production surveillance.")
        sys.exit(0)


if __name__ == "__main__":
    main()