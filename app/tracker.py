import math
from collections import deque


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _segments_intersect(A, B, C, D):
    return (_ccw(A, C, D) != _ccw(B, C, D)) and (_ccw(A, B, C) != _ccw(A, B, D))


def point_in_polygon(pt, polygon_pts):
    x, y = pt
    n = len(polygon_pts)
    inside = False
    p1x, p1y = polygon_pts[0]
    for i in range(n + 1):
        p2x, p2y = polygon_pts[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def box_intersects_zone(bbox, zone):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    geom = zone.get("geometry_type", "POLYGON").upper()
    coords = zone.get("coordinates", [])
    if len(coords) < 2:
        return False

    box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    box_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1))
    ]

    if geom == "LINE":
        p1 = coords[0]
        p2 = coords[1]
        for e in box_edges:
            if _segments_intersect(e[0], e[1], p1, p2):
                return True
        line_min_y = min(p1[1], p2[1])
        line_max_y = max(p1[1], p2[1])
        line_min_x = min(p1[0], p2[0])
        line_max_x = max(p1[0], p2[0])
        if (y1 <= line_max_y and y2 >= line_min_y) and (x1 <= line_max_x and x2 >= line_min_x):
            # Check vertical ray intersection with line segment
            denom = (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else 1e-5
            slope = (p2[1] - p1[1]) / denom
            line_y_at_cx = p1[1] + slope * (cx - p1[0])
            if y1 <= line_y_at_cx <= y2:
                return True
        return False

    poly = [(p[0], p[1]) for p in coords]

    if point_in_polygon((cx, cy), poly):
        return True
    for pt in box_corners:
        if point_in_polygon(pt, poly):
            return True

    n = len(poly)
    for i in range(n):
        seg_a = poly[i]
        seg_b = poly[(i + 1) % n]
        for b_edge in box_edges:
            if _segments_intersect(b_edge[0], b_edge[1], seg_a, seg_b):
                return True

    return False


def check_direction_constraint(history, zone):
    direction = str(zone.get("direction", "ANY")).upper()
    if direction == "ANY" or len(history) < 2:
        return True

    ox, oy, _ = history[0]
    cx, cy, _ = history[-1]
    dy = cy - oy

    if direction == "INBOUND":
        return dy >= -2
    elif direction == "OUTBOUND":
        return dy <= 2

    return True


class TrackTrajectoryManager:
    def __init__(self, max_history=15, max_idle_sec=5.0):
        self.max_history = max_history
        self.max_idle_sec = max_idle_sec
        self.tracks = {}
        self.last_seen = {}
        self.static_dwell = {}
        self.metadata = {}

    def update_track(self, track_id: int, bbox: list, curr_time: float, line_y: int, label: str = "", category: str = "", pixels_per_meter: float = 0.0):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        if track_id not in self.tracks:
            self.tracks[track_id] = deque(maxlen=self.max_history)

        self.tracks[track_id].append((cx, cy, curr_time))
        self.last_seen[track_id] = curr_time
        self.metadata[track_id] = {
            "bbox": bbox,
            "label": label,
            "category": category,
            "cx": cx,
            "cy": cy
        }

        speed_kmh = 0.0
        heading = "STATIONARY"
        eta_str = "N/A"
        dx = 0
        dy = 0

        history = self.tracks[track_id]
        if len(history) >= 3:
            ox, oy, ot = history[0]
            dt = max(curr_time - ot, 1e-4)
            dx = cx - ox
            dy = cy - oy
            dist_px = math.hypot(dx, dy)

            # If pixels_per_meter is provided and > 0, compute calibrated speed; else use standard 40 px/m estimate
            effective_ppm = pixels_per_meter if pixels_per_meter > 0 else 40.0
            meters_traveled = dist_px / effective_ppm
            speed_mps = meters_traveled / dt
            speed_kmh = round(speed_mps * 3.6, 1)

            if dist_px > 10:
                angle_deg = math.atan2(-dy, dx) * 180.0 / math.pi
                if -45 <= angle_deg <= 45:
                    heading = "EAST"
                elif 45 < angle_deg <= 135:
                    heading = "NORTH"
                elif -135 <= angle_deg < -45:
                    heading = "SOUTH (INTRUSION)"
                else:
                    heading = "WEST"

                if speed_mps > 0.2:
                    rem_px = abs(line_y - cy)
                    speed_px_sec = dist_px / dt
                    if speed_px_sec > 1.0:
                        calc_eta = int(rem_px / speed_px_sec)
                        eta_str = f"{max(1, calc_eta)}s"

        return speed_kmh, heading, eta_str, dx, dy

    def detect_suspicious_activities(self, active_track_ids: list, curr_time: float, thresholds: dict, active_zones: list, pixels_per_meter: float = 0.0):
        events = []
        susp_cfg = thresholds.get("suspicious_activities", {})
        run_speed_thresh = susp_cfg.get("running_speed_kmh", 9.5)
        group_min = susp_cfg.get("group_gathering_min_persons", 3)
        group_radius = susp_cfg.get("group_gathering_radius_px", 120)
        abandon_dwell_thresh = susp_cfg.get("abandoned_object_min_dwell_sec", 6.0)
        halted_speed_thresh = susp_cfg.get("vehicle_halted_max_speed_kmh", 1.5)
        halted_dwell_thresh = susp_cfg.get("vehicle_halted_min_dwell_sec", 5.0)
        wrong_headings = susp_cfg.get("wrong_direction_headings", ["SOUTH (INTRUSION)"])

        persons = []
        vehicles = []

        effective_ppm = pixels_per_meter if pixels_per_meter > 0 else 40.0

        for tid in active_track_ids:
            meta = self.metadata.get(tid)
            if not meta:
                continue

            history = self.tracks.get(tid)
            speed_kmh = 0.0
            heading = "STATIONARY"
            if history and len(history) >= 3:
                ox, oy, ot = history[0]
                cx, cy, ct = history[-1]
                dt = max(ct - ot, 1e-4)
                dist = math.hypot(cx - ox, cy - oy)
                speed_kmh = round(((dist / effective_ppm) / dt) * 3.6, 1)

                if dist > 10:
                    ang = math.atan2(-(cy - oy), cx - ox) * 180.0 / math.pi
                    if -45 <= ang <= 45: heading = "EAST"
                    elif 45 < ang <= 135: heading = "NORTH"
                    elif -135 <= ang < -45: heading = "SOUTH (INTRUSION)"
                    else: heading = "WEST"

            category = meta["category"]
            bbox = meta["bbox"]

            if category == "person" and speed_kmh >= run_speed_thresh:
                events.append({
                    "type": "RUNNING / EVASIVE MOVEMENT",
                    "level": "CODE ORANGE",
                    "target": f"PERSON #{tid} ({speed_kmh} km/h)",
                    "track_id": tid,
                    "bbox": bbox
                })

            if category in ["person", "vehicle"] and heading in wrong_headings and speed_kmh > 3.0:
                events.append({
                    "type": "WRONG DIRECTION INTRUSION",
                    "level": "CODE ORANGE",
                    "target": f"{meta['label']} #{tid} (HEADING {heading})",
                    "track_id": tid,
                    "bbox": bbox
                })

            if category == "person":
                persons.append((tid, meta["cx"], meta["cy"], bbox))

            if category == "vehicle":
                vehicles.append((tid, speed_kmh, bbox))

            if category in ["phone", "baggage", "other"]:
                dwell_start = self.static_dwell.setdefault(tid, curr_time)
                dwell_dur = curr_time - dwell_start
                if speed_kmh <= 1.0 and dwell_dur >= abandon_dwell_thresh:
                    person_nearby = False
                    for _, px, py, _ in persons:
                        if math.hypot(px - meta["cx"], py - meta["cy"]) < 70:
                            person_nearby = True
                            break
                    if not person_nearby:
                        events.append({
                            "type": "ABANDONED OBJECT DETECTED",
                            "level": "CODE RED",
                            "target": f"STATIC {meta['label']} #{tid} ({int(dwell_dur)}s)",
                            "track_id": tid,
                            "bbox": bbox
                        })
            else:
                self.static_dwell.pop(tid, None)

        if len(persons) >= group_min:
            clustered = set()
            for i in range(len(persons)):
                cluster = [persons[i]]
                for j in range(len(persons)):
                    if i != j:
                        dist = math.hypot(persons[i][1] - persons[j][1], persons[i][2] - persons[j][2])
                        if dist <= group_radius:
                            cluster.append(persons[j])
                if len(cluster) >= group_min:
                    clustered.update([p[0] for p in cluster])

            if len(clustered) >= group_min:
                events.append({
                    "type": "GROUP GATHERING DETECTED",
                    "level": "CODE ORANGE",
                    "target": f"CLUSTER OF {len(clustered)} PERSONS ({list(clustered)})",
                    "track_id": list(clustered)[0],
                    "bbox": persons[0][3]
                })

        for vid, v_speed, v_box in vehicles:
            in_zone = False
            for z in active_zones:
                if box_intersects_zone(v_box, z):
                    in_zone = True
                    break
            if in_zone and v_speed <= halted_speed_thresh:
                v_dwell = curr_time - self.static_dwell.setdefault(vid, curr_time)
                if v_dwell >= halted_dwell_thresh:
                    events.append({
                        "type": "VEHICLE HALTED AT PERIMETER",
                        "level": "CODE ORANGE",
                        "target": f"VEHICLE #{vid} HALTED ({int(v_dwell)}s)",
                        "track_id": vid,
                        "bbox": v_box
                    })
            else:
                self.static_dwell.pop(vid, None)

        return events

    def prune_stale_tracks(self, curr_time: float):
        stale_ids = [
            tid for tid, last_t in self.last_seen.items()
            if (curr_time - last_t) > self.max_idle_sec
        ]
        for tid in stale_ids:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.static_dwell.pop(tid, None)
            self.metadata.pop(tid, None)