"""
IBVAP Core Engine Unit Test Suite
Covers:
  1. Indian License Plate Normalization (O/0, I/1, B/8 confusion matrix resolution)
  2. Indian License Plate Format Validation (Standard state + Bharat Series formats)
  3. Strict Geometric Zone & Line Intersection Detection
  4. Directional Crossing Constraint Filtering (INBOUND / OUTBOUND / ANY)
  5. SHA-256 Tamper-Evident Hash Chain Integrity & Row Tamper Detection
  6. Command & Control API Key Rejection & Enforcement on /api/events
  7. YAML Configuration Parsing for Zones and System Thresholds

Requires NO camera hardware, NO GPU, NO internet connection, and NO external weights.
"""
import os
import sys
import yaml
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.recognition import normalize_indian_plate_ocr, is_valid_indian_plate
from app.tracker import box_intersects_zone, check_direction_constraint
from app.database import compute_record_hash
from app.main import app


# ============================================================
# 1. INDIAN LICENSE PLATE NORMALIZATION & VALIDATION TESTS
# ============================================================

def test_indian_plate_character_swaps():
    """Verify OCR optical confusions are deterministically fixed based on position."""
    # District numbers: 'O' -> '0', 'I' -> '1'
    # Suffix numbers: 'I' -> '1', 'B' -> '8'
    raw_ocr = "DL-OI-AB-I2B4"
    normalized = normalize_indian_plate_ocr(raw_ocr)
    assert normalized == "DL01AB1284"

    # Verify '0' -> 'O' swap in state prefix (e.g., 0D -> OD for Odisha)
    raw_state_ocr = "0D-01-AB-1234"
    assert normalize_indian_plate_ocr(raw_state_ocr) == "OD01AB1234"


def test_indian_plate_format_validations():
    """Verify syntax checks for standard Indian and Bharat (BH) series plates."""
    assert is_valid_indian_plate("DL01AB1234") is True
    assert is_valid_indian_plate("MH12DE1432") is True
    assert is_valid_indian_plate("22BH1234AA") is True
    assert is_valid_indian_plate("KA04M9999") is True
    assert is_valid_indian_plate("INVALID123") is False
    assert is_valid_indian_plate("ABC") is False
    assert is_valid_indian_plate("UNREADABLE") is False


# ============================================================
# 2. GEOMETRIC INTERSECTION & DIRECTIONAL TESTS
# ============================================================

def test_line_boundary_intersection():
    """Verify boundary tripwire detection for boxes intersecting a virtual line."""
    line_zone = {
        "type": "RESTRICTED",
        "geometry_type": "LINE",
        "coordinates": [[0, 200], [640, 200]],
        "direction": "ANY"
    }

    box_crossing = [100, 180, 200, 220]  # Straddles y=200
    box_safe_above = [100, 50, 200, 150] # Above y=200
    box_safe_below = [100, 250, 200, 350] # Below y=200

    assert box_intersects_zone(box_crossing, line_zone) is True
    assert box_intersects_zone(box_safe_above, line_zone) is False
    assert box_intersects_zone(box_safe_below, line_zone) is False


def test_polygon_zone_intersection():
    """Verify restricted polygon intrusion for boxes inside or touching polygon edges."""
    poly_zone = {
        "type": "RESTRICTED",
        "geometry_type": "POLYGON",
        "coordinates": [[100, 100], [300, 100], [300, 300], [100, 300]],
        "direction": "ANY"
    }

    box_inside = [150, 150, 250, 250]
    box_edge_overlap = [50, 50, 150, 150]
    box_outside = [400, 400, 500, 500]

    assert box_intersects_zone(box_inside, poly_zone) is True
    assert box_intersects_zone(box_edge_overlap, poly_zone) is True
    assert box_intersects_zone(box_outside, poly_zone) is False


def test_directional_crossing_filter():
    """Verify INBOUND (downward dy > 0) vs OUTBOUND (upward dy < 0) directional gates."""
    inbound_zone = {"direction": "INBOUND"}
    outbound_zone = {"direction": "OUTBOUND"}

    history_downward = [(100, 100, 1.0), (100, 150, 1.5), (100, 200, 2.0)]
    history_upward = [(100, 200, 1.0), (100, 150, 1.5), (100, 100, 2.0)]

    assert check_direction_constraint(history_downward, inbound_zone) is True
    assert check_direction_constraint(history_downward, outbound_zone) is False
    assert check_direction_constraint(history_upward, inbound_zone) is False
    assert check_direction_constraint(history_upward, outbound_zone) is True


# ============================================================
# 3. CRYPTOGRAPHIC HASH CHAIN INTEGRITY TESTS
# ============================================================

def test_hash_chain_tamper_detection():
    """Verify that altering any historical record invalidates the cryptographic chain."""
    prev_hash_0 = "0" * 64

    t1 = "2026-03-01 10:00:00"
    h1 = compute_record_hash(prev_hash_0, t1, "CODE RED", "FENCE BREACH", "PERSON #1", "CAM-01")

    t2 = "2026-03-01 10:00:05"
    h2 = compute_record_hash(h1, t2, "CODE RED", "WATCHLIST SUSPECT", "SUSPECT A", "CAM-01")

    t3 = "2026-03-01 10:00:10"
    h3 = compute_record_hash(h2, t3, "CODE ORANGE", "LOITERING", "VEHICLE #2", "CAM-01")

    assert compute_record_hash(h2, t3, "CODE ORANGE", "LOITERING", "VEHICLE #2", "CAM-01") == h3

    # Tampering test: alter Row 1
    tampered_h1 = compute_record_hash(prev_hash_0, t1, "CODE GREEN", "FENCE BREACH", "PERSON #1", "CAM-01")
    recomputed_h2 = compute_record_hash(tampered_h1, t2, "CODE RED", "WATCHLIST SUSPECT", "SUSPECT A", "CAM-01")
    assert recomputed_h2 != h2, "Cryptographic audit failed to detect modified history!"


# ============================================================
# 4. COMMAND & CONTROL SECURITY & API KEY REJECTION TESTS
# ============================================================

def test_c2_api_key_rejection():
    """Verify /api/events rejects unauthorized or missing C2 credentials with 401."""
    client = TestClient(app)

    res_no_auth = client.get("/api/events")
    assert res_no_auth.status_code == 401

    res_invalid_key = client.get("/api/events", headers={"X-API-Key": "INVALID-MALICIOUS-KEY"})
    assert res_invalid_key.status_code == 401

    from app.main import C2_API_KEYS
    if C2_API_KEYS:
        valid_key = C2_API_KEYS[0]
        res_valid = client.get("/api/events", headers={"X-API-Key": valid_key})
        assert res_valid.status_code == 200
        assert "events" in res_valid.json()


# ============================================================
# 5. CONFIGURATION SCHEMA & PERSISTENCE TESTS
# ============================================================

def test_configs_and_thresholds_schema():
    """Verify thresholds and system configs parse properly and have required operational keys."""
    from app.config import CONFIG_YAML_PATH, THRESHOLDS_YAML_PATH, ZONES_YAML_PATH

    assert os.path.exists(CONFIG_YAML_PATH)
    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
        sys_cfg = yaml.safe_load(f)
    assert "hardware_optimization" in sys_cfg
    assert "frame_skip" in sys_cfg["hardware_optimization"]

    assert os.path.exists(THRESHOLDS_YAML_PATH)
    with open(THRESHOLDS_YAML_PATH, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)
    assert "suspicious_activities" in thresh_cfg["thresholds"]
    assert "running_speed_kmh" in thresh_cfg["thresholds"]["suspicious_activities"]

    assert os.path.exists(ZONES_YAML_PATH)
    with open(ZONES_YAML_PATH, "r", encoding="utf-8") as f:
        zones_cfg = yaml.safe_load(f)
    assert "zones" in zones_cfg