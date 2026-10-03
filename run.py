import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="IBVAP Edge Surveillance Node Launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host interface")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/WebSocket port")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload (Warning: Spawns duplicate worker threads)")
    args = parser.parse_args()

    print("=" * 72)
    print("  IBVAP 2.0 : SOVEREIGN BORDER DEFENSE VIDEO ANALYTICS PLATFORM")
    print("  Problem Statement: 26187 | COTS CPU-Optimized Pipeline")
    print("=" * 72)

    # Auto-verify weights
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "yolov8n.pt")
    if not os.path.exists(weights_path):
        print("[*] Downloading official YOLOv8 core engine...")
        try:
            YOLO("yolov8n.pt")
            print("[✓] YOLOv8 base engine ready.")
        except Exception as e:
            print(f"[!] Warning downloading YOLOv8: {e}")

    print(f"[*] Launching IBVAP Surveillance Node on http://{args.host}:{args.port}")
    if args.reload:
        print("[!] NOTICE: Reload flag is active. Background camera workers may duplicate.")

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()