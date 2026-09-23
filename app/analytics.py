def compute_threat_level(is_tampered, has_critical_weapon, is_breached, is_loitering):
    if is_tampered or has_critical_weapon:
        return "CODE RED", "CODE RED: OPTICAL TAMPERING DETECTED"
    elif is_loitering:
        return "CODE ORANGE", "CODE ORANGE: SUSPICIOUS LOITERING"
    elif is_breached:
        return "CODE RED", "CODE RED: VIRTUAL FENCE INTRUSION"
    return "CODE GREEN", "CODE GREEN: ALL SECTORS SECURE"