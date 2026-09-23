import sqlite3
from app.config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
            snapshot_path TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS biometric_whitelist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            officer_name TEXT UNIQUE,
            enrolled_at TEXT,
            snapshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_threat_event(rec_id, timestamp, level, threat_type, targets, velocity, heading, eta, cam_id, snap_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO threat_audit_vault 
        (id, timestamp, threat_level, threat_type, targets, velocity_kmh, heading, eta_bop, camera_id, status, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (rec_id, timestamp, level, threat_type, targets, velocity, heading, eta, cam_id, 'UNVERIFIED', snap_path))
    conn.commit()
    conn.close()

def update_hitl_status(rec_id, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE threat_audit_vault SET status = ? WHERE id = ?", (status, rec_id))
    conn.commit()
    conn.close()