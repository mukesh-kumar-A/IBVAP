import os
import yaml

# ============================================================
# BASE DIRECTORIES & FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Automatically load .env file into os.environ if present
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip()
                    # Strip any surrounding quotes
                    if len(_v) >= 2 and ((_v[0] == '"' and _v[-1] == '"') or (_v[0] == "'" and _v[-1] == "'")):
                        _v = _v[1:-1]
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception as _e:
        print(f"[IBVAP CONFIG] Notice: Could not read .env: {_e}")

DATA_DIR = os.path.join(BASE_DIR, "data")
FORENSIC_DIR = os.path.join(BASE_DIR, "forensic_logs")
FACES_DIR = os.path.join(BASE_DIR, "known_faces")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "defense_audit.db"))

CONFIG_YAML_PATH = os.path.join(CONFIGS_DIR, "config.yaml")
CAMERAS_YAML_PATH = os.path.join(CONFIGS_DIR, "cameras.yaml")
ZONES_YAML_PATH = os.path.join(CONFIGS_DIR, "zones.yaml")
THRESHOLDS_YAML_PATH = os.path.join(CONFIGS_DIR, "thresholds.yaml")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FORENSIC_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# MASTER SYSTEM CONFIGURATION (configs/config.yaml)
# ============================================================

def load_system_config():
    if not os.path.exists(CONFIG_YAML_PATH):
        return {}
    try:
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[IBVAP CONFIG] Error loading {CONFIG_YAML_PATH}: {e}")
        return {}


SYSTEM_CONFIG = load_system_config()


# ============================================================
# CAMERA REGISTRY PERSISTENCE (configs/cameras.yaml)
# ============================================================

DEFAULT_CAMERA_REGISTRY = [
    {
        "id": "CAM-01",
        "name": "Primary Optical Sensor",
        "location": "Sector A - Main Perimeter",
        "source": "0",
        "mirror": False,
        "pixels_per_meter": 40.0,
        "status": "ONLINE"
    }
]


def load_camera_configs():
    if not os.path.exists(CAMERAS_YAML_PATH):
        save_camera_configs(DEFAULT_CAMERA_REGISTRY)
        return list(DEFAULT_CAMERA_REGISTRY)

    try:
        with open(CAMERAS_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "cameras" in data and len(data["cameras"]) > 0:
                cameras = []
                for c in data["cameras"]:
                    cameras.append({
                        "id": str(c.get("id", "CAM-01")),
                        "name": str(c.get("name", "Camera")),
                        "location": str(c.get("location", "Sector")),
                        "source": str(c.get("source", "")),
                        "mirror": bool(c.get("mirror", False)),
                        "pixels_per_meter": float(c.get("pixels_per_meter", 0.0)),
                        "status": str(c.get("status", "ONLINE"))
                    })
                return cameras
    except Exception as e:
        print(f"[IBVAP CONFIG] Error reading {CAMERAS_YAML_PATH}: {e}")

    save_camera_configs(DEFAULT_CAMERA_REGISTRY)
    return list(DEFAULT_CAMERA_REGISTRY)


def save_camera_configs(cameras_list):
    try:
        clean_list = []
        for c in cameras_list:
            clean_list.append({
                "id": str(c.get("id")),
                "name": str(c.get("name")),
                "location": str(c.get("location")),
                "source": str(c.get("source", "")),
                "mirror": bool(c.get("mirror", False)),
                "pixels_per_meter": float(c.get("pixels_per_meter", 0.0)),
                "status": str(c.get("status", "ONLINE"))
            })
        with open(CAMERAS_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": clean_list}, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"[IBVAP CONFIG] Failed to save {CAMERAS_YAML_PATH}: {e}")
        return False


# ============================================================
# ZONES & THRESHOLDS PERSISTENCE
# ============================================================

def load_zones_config():
    if not os.path.exists(ZONES_YAML_PATH):
        return {}
    try:
        with open(ZONES_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("zones", {}) if data else {}
    except Exception:
        return {}


def save_zones_config(zones_dict):
    try:
        with open(ZONES_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump({"zones": zones_dict}, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception:
        return False


def load_thresholds_config():
    if not os.path.exists(THRESHOLDS_YAML_PATH):
        return {}
    try:
        with open(THRESHOLDS_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("thresholds", {}) if data else {}
    except Exception:
        return {}


def save_thresholds_config(thresh_dict):
    try:
        with open(THRESHOLDS_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump({"thresholds": thresh_dict}, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception:
        return False


# ============================================================
# OPERATIONAL STATE
# ============================================================

CONFIG_STATE = {
    "boundary_percentage": 45,
    "night_enhancement": False,
    "loiter_threshold_sec": 3.0,
    "loiter_max_speed_kmh": 4.0,
    "current_cam_key": "CAM-01",
    "current_cam_label": "CAM-01 (Primary Optical Sensor)"
}