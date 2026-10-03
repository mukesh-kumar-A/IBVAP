# Rule-based threat classification + weighted risk-score fusion

RISK_WEIGHTS = {
    "tamper": 90,
    "watchlist_suspect": 95,
    "watchlist_vehicle": 90,
    "fence_breach": 80,
    "night_movement": 75,
    "abandoned_object": 70,
    "group_gathering": 65,
    "vehicle_halted": 60,
    "wrong_direction": 55,
    "running": 50,
    "loitering": 40
}


def compute_threat_level(
    is_tampered: bool,
    has_critical_weapon: bool,
    is_breached: bool,
    is_loitering: bool,
    active_threat_types: list = None
):
    """
    Computes alert code and human-readable threat banner across all threat signals.
    """
    threats = set(active_threat_types or [])

    if is_tampered or has_critical_weapon:
        return "CODE RED", "CODE RED: OPTICAL TAMPERING DETECTED"

    for t in threats:
        if "WATCHLIST SUSPECT IDENTIFIED" in t:
            return "CODE RED", f"CODE RED: {t}"
        if "WATCHLIST VEHICLE DETECTED" in t:
            return "CODE RED", f"CODE RED: {t}"

    if is_breached or "VIRTUAL FENCE INTRUSION" in threats:
        return "CODE RED", "CODE RED: VIRTUAL FENCE INTRUSION"

    if "NIGHT MOVEMENT DETECTED" in threats:
        return "CODE RED", "CODE RED: NIGHT-TIME MOVEMENT IN RESTRICTED ZONE"

    if "ABANDONED OBJECT DETECTED" in threats:
        return "CODE RED", "CODE RED: UNATTENDED SUSPICIOUS OBJECT DEPOSITED"

    if "GROUP GATHERING DETECTED" in threats:
        return "CODE ORANGE", "CODE ORANGE: TACTICAL GROUP FORMATION NEAR FENCE"

    if "VEHICLE HALTED AT PERIMETER" in threats:
        return "CODE ORANGE", "CODE ORANGE: RECONNAISSANCE VEHICLE STATIONED"

    if "RUNNING / EVASIVE MOVEMENT" in threats:
        return "CODE ORANGE", "CODE ORANGE: HIGH-VELOCITY EVASIVE INFILTRATOR"

    if "WRONG DIRECTION INTRUSION" in threats:
        return "CODE ORANGE", "CODE ORANGE: TARGET PROGRESSING TOWARDS DEFENSE LINE"

    if is_loitering or "SUSPICIOUS LOITERING" in threats:
        return "CODE ORANGE", "CODE ORANGE: SUSPICIOUS PROLONGED LOITERING"

    return "CODE GREEN", "CODE GREEN: ALL SECTORS SECURE"


def compute_risk_score(
    is_tampered: bool,
    is_breached: bool,
    is_loitering: bool,
    max_speed_kmh: float = 0.0,
    active_threat_flags: dict = None
):
    """
    Comprehensive multi-signal risk fusion (0 - 100):
    HIGH >= 70 (CODE RED)
    MEDIUM 30 - 69 (CODE ORANGE)
    LOW < 30 (CODE GREEN)
    """
    flags = active_threat_flags or {}
    score = 0

    if is_tampered:
        score += RISK_WEIGHTS["tamper"]
    if flags.get("watchlist_suspect", False):
        score += RISK_WEIGHTS["watchlist_suspect"]
    if flags.get("watchlist_vehicle", False):
        score += RISK_WEIGHTS["watchlist_vehicle"]
    if is_breached or flags.get("breach", False):
        score += RISK_WEIGHTS["fence_breach"]
    if flags.get("night_movement", False):
        score += RISK_WEIGHTS["night_movement"]
    if flags.get("abandoned_object", False):
        score += RISK_WEIGHTS["abandoned_object"]
    if flags.get("group_gathering", False):
        score += RISK_WEIGHTS["group_gathering"]
    if flags.get("vehicle_halted", False):
        score += RISK_WEIGHTS["vehicle_halted"]
    if flags.get("wrong_direction", False):
        score += RISK_WEIGHTS["wrong_direction"]
    if flags.get("running", False):
        score += RISK_WEIGHTS["running"]
    if is_loitering or flags.get("loitering", False):
        score += RISK_WEIGHTS["loitering"]

    if (is_breached or flags.get("breach", False) or flags.get("running", False)):
        speed_bonus = min(20, int(max(0.0, max_speed_kmh) * 1.5))
        score += speed_bonus

    score = min(100, score)

    if score >= 70:
        band = "HIGH"
    elif score >= 30:
        band = "MEDIUM"
    else:
        band = "LOW"

    return score, band