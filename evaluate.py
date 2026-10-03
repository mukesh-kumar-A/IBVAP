import argparse
import os
import json
import glob
import cv2
import numpy as np

from app.models_loader import model_registry
from app.inference import run_tracked_detection
from app.recognition import face_detector, face_recognizer, extract_vehicle_plate
from app.tracker import box_intersects_zone


def safe_div(num, den):
    return (num / den) if den > 0 else 0.0


def create_sample_ground_truth_dataset(target_dir: str):
    """Generates standard test images with ground-truth sidecar annotations."""
    os.makedirs(target_dir, exist_ok=True)
    img_path = os.path.join(target_dir, "test_perimeter_01.jpg")
    json_path = os.path.join(target_dir, "test_perimeter_01.json")

    # Generate synthetic validation frame
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (100, 100), (250, 350), (200, 200, 200), -1)
    cv2.putText(canvas, "TEST INTRUDER", (110, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(img_path, canvas)

    gt_data = {
        "intrusions": [
            {"box": [100, 100, 250, 350], "label": "PERSON"}
        ],
        "plates": [
            {"box": [300, 200, 480, 270], "text": "DL01AB1234"}
        ],
        "faces": []
    }
    with open(json_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"[✓] Created synthetic ground-truth test suite in: {target_dir}")


def evaluate_dataset(dataset_dir: str):
    print("\n" + "=" * 75)
    print(f" IBVAP ACCURACY EVALUATOR | Dataset: {dataset_dir}")
    print("=" * 75 + "\n")

    if not os.path.exists(dataset_dir):
        print(f"[*] Dataset folder '{dataset_dir}' not found. Automatically generating verification fixtures...")
        create_sample_ground_truth_dataset(dataset_dir)

    img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))

    metrics = {
        "intrusion": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "anpr": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "face": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0}
    }

    dummy_restricted_zone = {
        "type": "RESTRICTED",
        "geometry_type": "LINE",
        "coordinates": [[0, 240], [640, 240]],
        "direction": "ANY"
    }

    yolo = model_registry.create_isolated_model()

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        base_name = os.path.splitext(img_path)[0]
        json_path = f"{base_name}.json"

        gt = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    gt = json.load(f)
            except Exception:
                gt = {}

        # 1. Evaluate Virtual Fence / Intrusion
        if "intrusions" in gt and gt["intrusions"]:
            gt_intrusions = gt["intrusions"]
            metrics["intrusion"]["total_gt"] += len(gt_intrusions)

            pred_dets = run_tracked_detection(yolo, frame)
            pred_intrusions = [
                d for d in pred_dets
                if box_intersects_zone(d["box"], dummy_restricted_zone)
            ]

            matched_preds = set()
            for gt_item in gt_intrusions:
                gt_box = gt_item["box"]
                matched = False
                for p_idx, pred in enumerate(pred_intrusions):
                    if p_idx in matched_preds:
                        continue
                    px1, py1, px2, py2 = pred["box"]
                    gx1, gy1, gx2, gy2 = gt_box
                    xx1, yy1 = max(px1, gx1), max(py1, gy1)
                    xx2, yy2 = min(px2, gx2), min(py2, gy2)
                    w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
                    inter = w * h
                    union = ((px2 - px1) * (py2 - py1)) + ((gx2 - gx1) * (gy2 - gy1)) - inter
                    iou = (inter / union) if union > 0 else 0
                    if iou >= 0.40:
                        matched = True
                        matched_preds.add(p_idx)
                        break

                if matched:
                    metrics["intrusion"]["tp"] += 1
                else:
                    metrics["intrusion"]["fn"] += 1

            metrics["intrusion"]["fp"] += (len(pred_intrusions) - len(matched_preds))

        # 2. Evaluate ANPR Plates
        if "plates" in gt and gt["plates"]:
            gt_plates = gt["plates"]
            metrics["anpr"]["total_gt"] += len(gt_plates)

            for item in gt_plates:
                gt_text = item["text"].upper().replace(" ", "")
                gx1, gy1, gx2, gy2 = item["box"]
                veh_crop = frame[max(0, gy1):max(0, gy2), max(0, gx1):max(0, gx2)]

                read_plate = extract_vehicle_plate(veh_crop, track_id=999)
                clean_read = read_plate.upper().replace(" ", "")

                if clean_read != "UNREADABLE":
                    if clean_read == gt_text:
                        metrics["anpr"]["tp"] += 1
                    else:
                        metrics["anpr"]["fp"] += 1
                else:
                    metrics["anpr"]["fn"] += 1

        # 3. Evaluate Face Recognition Matches
        if "faces" in gt and gt["faces"]:
            gt_faces = gt["faces"]
            metrics["face"]["total_gt"] += len(gt_faces)

            for item in gt_faces:
                gt_name = item["name"].upper()
                fx1, fy1, fx2, fy2 = item["box"]
                person_crop = frame[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]

                fcrop, _ = face_detector.detect_face(person_crop)
                if fcrop is not None and fcrop.size > 0:
                    emb = face_recognizer.extract_face_embedding(fcrop)
                    matched, matched_name, _, _ = face_recognizer.match_face_embedding(emb)
                    if matched:
                        if matched_name.upper() == gt_name:
                            metrics["face"]["tp"] += 1
                        else:
                            metrics["face"]["fp"] += 1
                    else:
                        metrics["face"]["fn"] += 1
                else:
                    metrics["face"]["fn"] += 1

    # Print Table
    print("### Empirical Accuracy Metrics (Ground-Truth Verified):\n")
    headers = ["Task / Module", "GT Ground Truth", "True Pos (TP)", "False Pos (FP)", "False Neg (FN)", "Precision", "Recall", "F1-Score"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for task, data in metrics.items():
        total_gt = data["total_gt"]
        if total_gt == 0:
            print(f"| {task.capitalize():13s} | N/A | N/A | N/A | N/A | N/A | N/A | N/A (No GT data) |")
        else:
            tp, fp, fn = data["tp"], data["fp"], data["fn"]
            prec = safe_div(tp, tp + fp)
            rec = safe_div(tp, tp + fn)
            f1 = safe_div(2 * (prec * rec), prec + rec)
            row = [
                task.capitalize(),
                str(total_gt),
                str(tp),
                str(fp),
                str(fn),
                f"{prec * 100:.1f}%",
                f"{rec * 100:.1f}%",
                f"{f1 * 100:.1f}%"
            ]
            print("| " + " | ".join(row) + " |")

    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBVAP Model Accuracy & Verification Evaluator")
    parser.add_argument("--dataset", type=str, default="data/dataset_processed/splits/val", help="Path to evaluation test folder")
    args = parser.parse_args()

    evaluate_dataset(args.dataset)