import sqlite3
import hashlib
import threading
import os
from app.config import DB_PATH

_db_lock = threading.Lock()


def get_db_connection():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db():
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Tamper-evident forensic audit vault
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_audit_vault (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                threat_level TEXT,
                threat_type TEXT,
                targets TEXT,
                velocity_kmh REAL,
                heading TEXT,
                eta_bop TEXT,
                camera_id TEXT,
                status TEXT DEFAULT 'UNVERIFIED',
                snapshot_path TEXT,
                record_hash TEXT,
                prev_hash TEXT
            )
        ''')

        # Dual-List Biometric Database (WHITELIST & WATCHLIST)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS biometric_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                category TEXT,
                enrolled_at TEXT,
                snapshot_path TEXT,
                embedding_blob BLOB
            )
        ''')

        # Plate Watchlist (Hotlist / Stolen Vehicles)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plate_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE,
                reason TEXT,
                added_at TEXT
            )
        ''')

        # Store-and-Forward Alert Backlog Queue
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_backlog_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                threat_type TEXT,
                threat_level TEXT,
                payload_json TEXT,
                dispatched INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                queued_at TEXT
            )
        ''')

        conn.commit()
        conn.close()


init_db()


def get_last_hash():
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT record_hash FROM threat_audit_vault ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return "0" * 64


def compute_record_hash(prev_hash, timestamp, level, threat_type, targets, cam_id):
    payload = f"{prev_hash}|{timestamp}|{level}|{threat_type}|{targets}|{cam_id}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def log_threat_event(rec_id, timestamp, level, threat_type, targets, velocity, heading, eta, cam_id, snap_path):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT record_hash FROM threat_audit_vault ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        prev_hash = row[0] if (row and row[0]) else ("0" * 64)

        record_hash = compute_record_hash(prev_hash, timestamp, level, threat_type, targets, cam_id)
        cursor.execute('''
            INSERT OR REPLACE INTO threat_audit_vault
            (id, timestamp, threat_level, threat_type, targets, velocity_kmh, heading, eta_bop, camera_id, status, snapshot_path, record_hash, prev_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (rec_id, timestamp, level, threat_type, targets, velocity, heading, eta, cam_id, 'UNVERIFIED', snap_path, record_hash, prev_hash))
        conn.commit()
        conn.close()
        return record_hash


def update_hitl_status(rec_id, status):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE threat_audit_vault SET status = ? WHERE id = ?", (status, rec_id))
        conn.commit()
        conn.close()


def verify_chain_integrity():
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, threat_level, threat_type, targets, camera_id, record_hash, prev_hash FROM threat_audit_vault ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()

    expected_prev = "0" * 64
    for row in rows:
        _, timestamp, level, threat_type, targets, cam_id, stored_hash, prev_hash = row
        if prev_hash != expected_prev:
            return False, f"TAMPERING DETECTED: prev_hash broken at record {stored_hash[:16]}"
        expected_hash = compute_record_hash(prev_hash, timestamp, level, threat_type, targets, cam_id)
        if expected_hash != stored_hash:
            return False, f"TAMPERING DETECTED: record hash mismatch at record {stored_hash[:16]}"
        expected_prev = stored_hash

    return True, "CHAIN INTACT — NO TAMPERING DETECTED"


def query_filtered_events(start_time=None, end_time=None, camera_id=None, threat_type=None, threat_level=None, status=None, page=1, page_size=50):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if threat_type:
            conditions.append("threat_type LIKE ?")
            params.append(f"%{threat_type}%")
        if threat_level:
            conditions.append("threat_level = ?")
            params.append(threat_level)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        cursor.execute(f"SELECT COUNT(*) FROM threat_audit_vault {where_clause}", tuple(params))
        total_records = cursor.fetchone()[0]

        offset = (max(1, page) - 1) * page_size
        query = f"""
            SELECT id, timestamp, threat_level, threat_type, targets, velocity_kmh, heading, eta_bop, camera_id, status, snapshot_path, record_hash, prev_hash
            FROM threat_audit_vault
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, tuple(params + [page_size, offset]))
        rows = cursor.fetchall()
        conn.close()

    events = []
    for r in rows:
        events.append({
            "id": r[0],
            "timestamp": r[1],
            "threat_level": r[2],
            "threat_type": r[3],
            "targets": r[4],
            "velocity_kmh": r[5],
            "heading": r[6],
            "eta_bop": r[7],
            "camera_id": r[8],
            "status": r[9],
            "snapshot_path": r[10],
            "record_hash": r[11],
            "prev_hash": r[12]
        })

    return events, total_records


def db_save_face(name: str, category: str, enrolled_at: str, snapshot_path: str, embedding_bytes: bytes):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO biometric_registry (name, category, enrolled_at, snapshot_path, embedding_blob)
            VALUES (?, ?, ?, ?, ?)
        ''', (name.upper(), category.upper(), enrolled_at, snapshot_path, embedding_bytes))
        conn.commit()
        conn.close()


def db_get_all_faces():
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, category, enrolled_at, snapshot_path, embedding_blob FROM biometric_registry")
        rows = cursor.fetchall()
        conn.close()
    return rows


def db_delete_face(category: str, name: str):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM biometric_registry WHERE category = ? AND name = ?", (category.upper(), name.upper()))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
    return affected > 0


def db_add_plate_watchlist(plate_number: str, reason: str, added_at: str):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO plate_watchlist (plate_number, reason, added_at)
            VALUES (?, ?, ?)
        ''', (plate_number.upper(), reason, added_at))
        conn.commit()
        conn.close()


def db_get_plate_watchlist():
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number, reason, added_at FROM plate_watchlist ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
    return [{"plate_number": r[0], "reason": r[1], "added_at": r[2]} for r in rows]


def db_delete_plate_watchlist(plate_number: str):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM plate_watchlist WHERE plate_number = ?", (plate_number.upper(),))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
    return affected > 0


def db_queue_alert(event_id, threat_type, level, payload_json):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        import time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO alert_backlog_queue (event_id, threat_type, threat_level, payload_json, dispatched, retry_count, queued_at)
            VALUES (?, ?, ?, ?, 0, 0, ?)
        ''', (event_id, threat_type, level, payload_json, now_str))
        conn.commit()
        conn.close()


def db_get_pending_alerts(max_retries=5):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT event_id, payload_json, retry_count FROM alert_backlog_queue
            WHERE dispatched = 0 AND retry_count < ?
            ORDER BY id ASC LIMIT 20
        ''', (max_retries,))
        rows = cursor.fetchall()
        conn.close()
    return rows


def db_mark_alert_dispatched(event_id):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE alert_backlog_queue SET dispatched = 1 WHERE event_id = ?", (event_id,))
        conn.commit()
        conn.close()


def db_increment_alert_retry(event_id):
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE alert_backlog_queue SET retry_count = retry_count + 1 WHERE event_id = ?", (event_id,))
        conn.commit()
        conn.close()