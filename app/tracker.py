import math
import numpy as np

class KalmanBoxTracker:
    count = 1
    def __init__(self, bbox):
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.time_since_update = 0
        self.history = []

    def update(self, bbox, curr_time):
        self.bbox = bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy, curr_time))
        if len(self.history) > 15:
            self.history.pop(0)
        self.hits += 1
        self.time_since_update = 0

def calculate_iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    return wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 
                 (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)

class AdvancedByteTracker:
    def __init__(self, max_age=25, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections, curr_time, line_y):
        for trk in self.trackers:
            trk.time_since_update += 1

        if len(detections) > 0 and len(self.trackers) > 0:
            iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = calculate_iou(det[:4], trk.bbox)

            matched_det, matched_trk = set(), set()

            for d in range(len(detections)):
                best_t = np.argmax(iou_matrix[d])
                if iou_matrix[d, best_t] >= self.iou_threshold and best_t not in matched_trk:
                    self.trackers[best_t].update(detections[d][:4], curr_time)
                    matched_det.add(d)
                    matched_trk.add(best_t)

            for d, det in enumerate(detections):
                if d not in matched_det:
                    new_trk = KalmanBoxTracker(det[:4])
                    new_trk.update(det[:4], curr_time)
                    self.trackers.append(new_trk)
        else:
            for det in detections:
                new_trk = KalmanBoxTracker(det[:4])
                new_trk.update(det[:4], curr_time)
                self.trackers.append(new_trk)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        results = []
        for det in detections:
            bbox = det[:4]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            
            best_trk = min(self.trackers, key=lambda t: math.hypot(int((t.bbox[0]+t.bbox[2])/2) - cx, int((t.bbox[1]+t.bbox[3])/2) - cy))
            
            speed_kmh, heading, eta, dx, dy = 0.0, "STATIONARY", "N/A", 0, 0
            if len(best_trk.history) >= 4:
                ox, oy, ot = best_trk.history[0]
                dt = max(curr_time - ot, 1e-5)
                dx, dy = cx - ox, cy - oy
                dist = math.hypot(dx, dy)
                speed_mps = (dist * 0.025) / dt
                speed_kmh = round(speed_mps * 3.6, 1)

                if dist > 10:
                    angle = math.atan2(-dy, dx) * 180.0 / math.pi
                    if -45 <= angle <= 45: heading = "EAST"
                    elif 45 < angle <= 135: heading = "NORTH"
                    elif -135 <= angle < -45: heading = "SOUTH (INTRUSION)"
                    else: heading = "WEST"

                    if speed_mps > 0.2:
                        rem_px = abs(line_y - cy)
                        eta = f"{max(1, int(rem_px / (dist / dt)))}s"

            results.append((bbox, det[4], det[6], best_trk.id, det[5], speed_kmh, heading, eta, dx, dy))

        return results

byte_tracker = AdvancedByteTracker()