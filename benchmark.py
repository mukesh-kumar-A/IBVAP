import argparse
import os
import sys
import time
import threading
import psutil
import cv2
import numpy as np

from app.models_loader import model_registry
from app.inference import run_tracked_detection
from app.recognition import face_detector, face_recognizer, extract_vehicle_plate
from app.tracker import TrackTrajectoryManager


class StreamBenchmarkWorker(threading.Thread):
    def __init__(self, stream_id: int, shared_frames_buffer: list, lock: threading.Lock, duration_sec: int):
        super().__init__()
        self.stream_id = stream_id
        self.shared_frames_buffer = shared_frames_buffer
        self.lock = lock
        self.duration_sec = duration_sec

        self.frames_processed = 0
        self.det_latencies = []
        self.track_latencies = []
        self.face_latencies = []
        self.anpr_latencies = []

        self.model = model_registry.create_isolated_model()
        self.trajectory_manager = TrackTrajectoryManager()
        self.running = True

    def run(self):
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < self.duration_sec:
            frame = None
            with self.lock:
                if self.shared_frames_buffer:
                    frame = self.shared_frames_buffer[-1].copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # Stage 1: Detection & ByteTrack
            t0 = time.perf_counter()
            tracked_dets = run_tracked_detection(self.model, frame, target_size=(480, 360))
            t_det = (time.perf_counter() - t0) * 1000.0
            self.det_latencies.append(t_det)

            # Stage 2: Trajectory & Velocity
            t1 = time.perf_counter()
            curr_wall = time.time()
            for det in tracked_dets:
                self.trajectory_manager.update_track(
                    det["track_id"], det["box"], curr_wall, 216,
                    label=det["label"], category=det["category"]
                )
            t_track = (time.perf_counter() - t1) * 1000.0
            self.track_latencies.append(t_track)

            # Stage 3: Face Detection & ArcFace
            t2 = time.perf_counter()
            for det in tracked_dets:
                if det["category"] == "person":
                    x1, y1, x2, y2 = det["box"]
                    person_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    fcrop, _ = face_detector.detect_face(person_crop)
                    if fcrop is not None and fcrop.size > 0:
                        emb = face_recognizer.extract_face_embedding(fcrop)
                        face_recognizer.match_face_embedding(emb)
                    break
            t_face = (time.perf_counter() - t2) * 1000.0
            self.face_latencies.append(t_face)

            # Stage 4: ANPR
            t3 = time.perf_counter()
            for det in tracked_dets:
                if det["category"] == "vehicle":
                    x1, y1, x2, y2 = det["box"]
                    veh_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    extract_vehicle_plate(veh_crop, det["track_id"])
                    break
            t_anpr = (time.perf_counter() - t3) * 1000.0
            self.anpr_latencies.append(t_anpr)

            self.frames_processed += 1
            time.sleep(0.005)


def run_concurrency_benchmark(source_path: str, duration_sec: int, stream_counts=(1, 2, 4)):
    print("\n" + "=" * 80)
    print(" IBVAP MULTI-STREAM EDGE PIPELINE PERFORMANCE BENCHMARK (REAL MEASUREMENTS)")
    print(f" Source: {source_path} | Duration: {duration_sec}s per tier")
    print(f" CPU: {psutil.cpu_count(logical=False)} Cores ({psutil.cpu_count(logical=True)} Threads) | RAM: {round(psutil.virtual_memory().total / (1024**3), 1)} GB")
    print("=" * 80 + "\n")

    src = int(source_path) if source_path.isdigit() else source_path
    backend = cv2.CAP_DSHOW if os.name == 'nt' and str(src).isdigit() else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)

    if not cap.isOpened():
        print(f"[!] Warning: Device {source_path} not accessible. Using high-resolution synthetic test feed.")
        cap = None

    shared_buffer = []
    buffer_lock = threading.Lock()
    ingestion_running = True

    def ingestion_worker():
        while ingestion_running:
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    with buffer_lock:
                        shared_buffer.append(frame)
                        if len(shared_buffer) > 5:
                            shared_buffer.pop(0)
                    time.sleep(0.02)
                else:
                    time.sleep(0.05)
            else:
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(test_frame, f"IBVAP BENCHMARK FEED: {time.time():.2f}", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                with buffer_lock:
                    shared_buffer.append(test_frame)
                    if len(shared_buffer) > 5:
                        shared_buffer.pop(0)
                time.sleep(0.03)

    ingest_th = threading.Thread(target=ingestion_worker, daemon=True)
    ingest_th.start()

    for _ in range(50):
        if shared_buffer:
            break
        time.sleep(0.05)

    results_table = []

    for num_streams in stream_counts:
        print(f"[*] Benchmarking {num_streams} concurrent camera stream(s)...")

        process = psutil.Process(os.getpid())
        cpu_samples = []
        mem_samples = []

        workers = [
            StreamBenchmarkWorker(i, shared_buffer, buffer_lock, duration_sec)
            for i in range(num_streams)
        ]

        t_start = time.perf_counter()
        for w in workers:
            w.start()

        while any(w.is_alive() for w in workers):
            cpu_samples.append(psutil.cpu_percent(interval=0.5))
            mem_samples.append(process.memory_info().rss / (1024 * 1024))

        for w in workers:
            w.join()

        actual_duration = time.perf_counter() - t_start
        total_frames = sum(w.frames_processed for w in workers)
        aggregate_fps = round(total_frames / actual_duration, 1) if actual_duration > 0 else 0.0
        fps_per_stream = round(aggregate_fps / num_streams, 1)

        all_det = [l for w in workers for l in w.det_latencies]
        all_trk = [l for w in workers for l in w.track_latencies]
        all_face = [l for w in workers for l in w.face_latencies if l > 0.01]
        all_anpr = [l for w in workers for l in w.anpr_latencies if l > 0.01]

        avg_det = round(float(np.mean(all_det)), 1) if all_det else 0.0
        avg_trk = round(float(np.mean(all_trk)), 1) if all_trk else 0.0
        avg_face = round(float(np.mean(all_face)), 1) if all_face else 0.0
        avg_anpr = round(float(np.mean(all_anpr)), 1) if all_anpr else 0.0
        avg_cpu = round(float(np.mean(cpu_samples)), 1) if cpu_samples else 0.0
        peak_mem = round(float(max(mem_samples)), 1) if mem_samples else 0.0

        results_table.append({
            "streams": num_streams,
            "agg_fps": aggregate_fps,
            "fps_per_stream": fps_per_stream,
            "det_ms": avg_det,
            "trk_ms": avg_trk,
            "face_ms": avg_face,
            "anpr_ms": avg_anpr,
            "cpu_pct": avg_cpu,
            "ram_mb": peak_mem
        })

    ingestion_running = False
    if cap is not None:
        cap.release()

    print("\n### Benchmark Results Markdown Output (Paste into README.md):\n")
    headers = [
        "Streams", "Aggregate FPS", "FPS / Camera",
        "YOLO Det (ms)", "ByteTrack (ms)", "Face ArcFace (ms)",
        "ANPR OCR (ms)", "CPU Load (%)", "Memory RSS (MB)"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in results_table:
        row = [
            str(r["streams"]),
            f"{r['agg_fps']} fps",
            f"{r['fps_per_stream']} fps",
            f"{r['det_ms']} ms",
            f"{r['trk_ms']} ms",
            f"{r['face_ms']} ms",
            f"{r['anpr_ms']} ms",
            f"{r['cpu_pct']}%",
            f"{r['ram_mb']} MB"
        ]
        print("| " + " | ".join(row) + " |")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBVAP Real Multi-Stream Edge Benchmark")
    parser.add_argument("--source", type=str, default="0", help="Video file path, camera index (0), or RTSP URL")
    parser.add_argument("--duration", type=int, default=5, help="Duration to test each stream count tier (seconds)")
    args = parser.parse_args()

    run_concurrency_benchmark(args.source, args.duration)