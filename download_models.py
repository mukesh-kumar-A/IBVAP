"""
IBVAP Automated Model Weight Provisioner
Downloads OpenCV YuNet and ArcFace ONNX models with streaming progress.
Uses requests with standard browser headers to avoid HTTP 401/403 blocks.
"""
import os
import sys
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DIR = os.path.join(BASE_DIR, "models", "face")
os.makedirs(FACE_DIR, exist_ok=True)

MODELS = [
    {
        "name": "OpenCV YuNet Face Detector",
        "filename": "face_detection_yunet_2023mar.onnx",
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        "target_path": os.path.join(FACE_DIR, "face_detection_yunet_2023mar.onnx"),
        "fallback_instruction": "Download manually from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
    },
    {
        "name": "ArcFace MobileFaceNet ONNX (w600k_mbf)",
        "filename": "w600k_mbf.onnx",
        "urls": [
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_mbf.onnx?download=true",
            "https://huggingface.co/onnx-community/arcface-w600k-mbf/resolve/main/w600k_mbf.onnx?download=true"
        ],
        "target_path": os.path.join(FACE_DIR, "w600k_mbf.onnx"),
        "fallback_instruction": "Open in browser: https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_mbf.onnx?download=true and save to models/face/w600k_mbf.onnx"
    }
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*"
}


def download_file(urls, target_path, filename):
    for url in urls:
        try:
            print(f"    Connecting to: {url}")
            with requests.get(url, headers=HEADERS, stream=True, timeout=25.0) as resp:
                if resp.status_code == 200:
                    total_size = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(target_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    pct = min(100.0, (downloaded / total_size) * 100.0)
                                    sys.stdout.write(f"\r    Downloading: {pct:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                                    sys.stdout.flush()
                    print(f"\n    [✓] Successfully downloaded: {filename}")
                    return True
                else:
                    print(f"    [!] HTTP {resp.status_code} received from mirror. Trying next...")
        except Exception as e:
            print(f"    [!] Connection error: {e}. Trying next...")

    return False


def download_all():
    print("=" * 70)
    print(" IBVAP DEEP LEARNING MODEL ASSET PROVISIONER")
    print("=" * 70)

    all_success = True
    for m in MODELS:
        print(f"\n[*] Checking: {m['name']}...")
        if os.path.exists(m["target_path"]) and os.path.getsize(m["target_path"]) > 1000:
            sz_mb = os.path.getsize(m["target_path"]) / (1024 * 1024)
            print(f"    [✓] Already present: {m['target_path']} ({sz_mb:.2f} MB)")
            continue

        success = download_file(m["urls"], m["target_path"], m["filename"])
        if not success:
            print(f"    [!] AUTOMATED DOWNLOAD FAILED.")
            print(f"    [!] MANUAL ACTION REQUIRED:")
            print(f"        {m['fallback_instruction']}")
            print(f"        Save to exact path: {m['target_path']}")
            if os.path.exists(m["target_path"]):
                try:
                    os.remove(m["target_path"])
                except Exception:
                    pass
            all_success = False

    print("\n" + "=" * 70)
    if all_success:
        print("[✓] All deep vision models verified and ready for inference.")
    else:
        print("[!] Note: If ArcFace is absent, IBVAP operates in honest FALLBACK mode.")
    print("=" * 70)


if __name__ == "__main__":
    download_all()