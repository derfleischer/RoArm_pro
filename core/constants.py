#!/usr/bin/env python3
"""
RoArm M3 Hardware Constants
Alle Hardware-spezifischen Konstanten und Limits.
Basiert auf Waveshare Wiki Dokumentation.
"""

# ============== SERVO LIMITS (RADIANS) ==============
# Basiert auf Waveshare RoArm-M3 Dokumentation

SERVO_LIMITS = {
    "base": (-3.14, 3.14),      # ±180° (360° Rotation)
    "shoulder": (-1.57, 1.57),  # ±90° (180° Bereich)
    "elbow": (0.0, 3.14),       # 0-180° (nur positive Richtung)
    "wrist": (-1.57, 1.57),     # ±90° (180° Bereich)
    "roll": (-3.14, 3.14),      # ±180° (360° Rotation)
    "hand": (1.08, 3.14)        # 62°-180° (Greifer: offen bis geschlossen)
}

# ============== STANDARD POSITIONEN ==============

# Home Position - Neutrale Ausgangsposition
HOME_POSITION = {
    "base": 0.0,        # Geradeaus
    "shoulder": 0.0,    # Horizontal
    "elbow": 1.57,      # 90° gebeugt
    "wrist": 0.0,       # Level
    "roll": 0.0,        # Neutral
    "hand": 3.14        # Geschlossen (sicher)
}

# Scanner Position - Optimal für Revopoint Mini2
SCANNER_POSITION = {
    "base": 0.0,        # Zentriert
    "shoulder": 0.35,   # 20° nach oben
    "elbow": 1.22,      # 70° gebeugt
    "wrist": -1.57,     # 90° nach unten (kompensiert für horizontalen Scanner)
    "roll": 1.57,       # 90° gedreht für Scanner-Montage
    "hand": 2.5         # Scanner-Griff (fest aber nicht zu stark)
}

# Park Position - Kompakte Position zum Verstauen
PARK_POSITION = {
    "base": 0.0,
    "shoulder": -1.57,  # Nach unten
    "elbow": 3.14,      # Voll eingeklappt
    "wrist": 0.0,
    "roll": 0.0,
    "hand": 3.14        # Geschlossen
}

# ============== COMMAND IDs ==============
# Basiert auf Waveshare JSON Protocol

COMMANDS = {
    "EMERGENCY_STOP": 0,      # Sofortiger Halt
    "STATUS_QUERY": 1,        # Status abfragen
    "POSITION_QUERY": 2,      # Aktuelle Position
    "LED_CONTROL": 51,        # LED Steuerung
    "JOINT_CONTROL": 102,     # Joint-Bewegung
    "TORQUE_CONTROL": 210,    # Torque Ein/Aus
    "SPEED_CONTROL": 211,     # Geschwindigkeitseinstellung
    "CALIBRATION": 300,       # Kalibrierung
    "RESET": 999              # System Reset
}

# ============== MOTION PARAMETERS ==============

# Geschwindigkeitsgrenzen
SPEED_LIMITS = {
    "min": 0.1,         # Minimum Geschwindigkeit
    "max": 2.0,         # Maximum Geschwindigkeit  
    "default": 1.0,     # Standard
    "scan": 0.3,        # Scanning (langsam)
    "teaching": 0.5     # Teaching Mode
}

# Beschleunigungsgrenzen
ACCELERATION_LIMITS = {
    "min": 0.5,
    "max": 5.0,
    "default": 2.0,
    "smooth": 1.0,      # Sanfte Bewegungen
    "quick": 3.0        # Schnelle Bewegungen
}

# Jerk (Ruck) Limits
JERK_LIMITS = {
    "min": 1.0,
    "max": 10.0,
    "default": 5.0,
    "smooth": 2.0,
    "aggressive": 8.0
}

# ============== SCANNER SPEZIFIKATIONEN ==============

# Revopoint Mini2 Scanner
SCANNER_SPECS = {
    "weight": 0.2,              # 200g
    "optimal_distance": 0.15,   # 15cm optimal
    "min_distance": 0.10,       # 10cm minimum
    "max_distance": 0.30,       # 30cm maximum
    "fov_horizontal": 0.698,    # 40° in radians
    "fov_vertical": 0.524,      # 30° in radians
    "settle_time": 0.5,         # Wartezeit für Stabilität
    "scan_speed": 0.3           # Optimale Scan-Geschwindigkeit
}

# Scanner Mount Offset (relativ zum End-Effektor)
SCANNER_MOUNT_OFFSET = {
    "x": 0.0,       # Keine seitliche Verschiebung
    "y": 0.0,       # Keine Vorwärts/Rückwärts Verschiebung
    "z": 0.05       # 5cm über dem Greifer
}

# ============== SCAN PATTERNS DEFAULTS ==============

SCAN_DEFAULTS = {
    "raster": {
        "width": 0.20,          # 20cm Breite
        "height": 0.15,         # 15cm Höhe
        "rows": 10,             # Anzahl Zeilen
        "overlap": 0.2,         # 20% Überlappung
        "speed": 0.3,           # Bewegungsgeschwindigkeit
        "settle_time": 0.5      # Pause pro Position
    },
    "spiral": {
        "radius_start": 0.05,   # Start-Radius 5cm
        "radius_end": 0.15,     # End-Radius 15cm
        "revolutions": 5,       # Anzahl Umdrehungen
        "points_per_rev": 36,   # Punkte pro Umdrehung
        "speed": 0.25,          # Kontinuierliche Bewegung
        "height_range": 0.1     # Höhenvariation 10cm
    },
    "spherical": {
        "radius": 0.15,         # Kugelradius 15cm
        "theta_steps": 12,      # Horizontale Schritte
        "phi_steps": 8,         # Vertikale Schritte
        "speed": 0.3,
        "settle_time": 0.7      # Längere Pause für Neuausrichtung
    },
    "turntable": {
        "steps": 36,            # 10° Schritte
        "radius": 0.15,         # Abstand zum Objekt
        "height_levels": 3,     # Höhenebenen
        "speed": 0.5,
        "settle_time": 1.0      # Pause für Scanner
    }
}

# ============== SAFETY PARAMETERS ==============

SAFETY_LIMITS = {
    "max_velocity": 3.14,           # rad/s
    "max_acceleration": 5.0,        # rad/s²
    "collision_threshold": 0.05,    # 5cm Sicherheitsabstand
    "torque_limit": 0.8,            # 80% max torque
    "temperature_warning": 50,      # °C
    "temperature_critical": 60,     # °C
    "voltage_min": 5.5,             # V
    "voltage_max": 7.0              # V
}

# ============== TEACHING MODE PARAMETERS ==============

TEACHING_DEFAULTS = {
    "sample_rate": 50,              # Hz - Abtastrate
    "position_threshold": 0.01,     # rad - Minimale Änderung zum Aufzeichnen
    "max_waypoints": 1000,          # Maximale Anzahl Wegpunkte
    "compression": True,            # Datenkompression aktiviert
    "interpolation": "cubic",       # Interpolationsmethode
    "file_format": "json",          # Speicherformat
    "auto_optimize": True           # Automatische Pfadoptimierung
}

# ============== TRAJECTORY PROFILES ==============

TRAJECTORY_PROFILES = {
    "linear": {
        "acceleration": 0,          # Konstante Geschwindigkeit
        "jerk": 0
    },
    "trapezoidal": {
        "acceleration": 2.0,        # Rampen
        "jerk": 10.0
    },
    "s_curve": {
        "acceleration": 2.0,
        "jerk": 5.0,
        "smoothness": 0.7          # Glättungsfaktor
    },
    "sinusoidal": {
        "frequency": 1.0,
        "amplitude": 1.0
    },
    "minimum_jerk": {
        "order": 5,                # Polynomordnung
        "boundary_conditions": 3   # Position, Velocity, Acceleration
    }
}

# ============== SERIAL COMMUNICATION ==============

SERIAL_CONFIG = {
    "baudrate": 115200,
    "bytesize": 8,
    "parity": 'N',
    "stopbits": 1,
    "timeout": 2.0,
    "write_timeout": 2.0,
    "read_buffer": 1024,
    "encoding": 'utf-8'
}

# ============== DEFAULT SPEED FACTORS ==============

DEFAULT_SPEED = 1.0

SPEED_PRESETS = {
    "very_slow": 0.2,
    "slow": 0.5,
    "normal": 1.0,
    "fast": 1.5,
    "very_fast": 2.0,
    "scan": 0.3,
    "teaching": 0.5,
    "demo": 0.8
}

# ============== ERROR MESSAGES ==============

ERROR_MESSAGES = {
    "connection_failed": "Failed to connect to RoArm",
    "port_not_found": "Serial port not found",
    "position_limit": "Position exceeds servo limits",
    "emergency_stop": "Emergency stop activated",
    "communication_error": "Serial communication error",
    "invalid_command": "Invalid command format",
    "timeout": "Command timeout",
    "temperature_warning": "Servo temperature high",
    "voltage_warning": "Supply voltage out of range"
}

# ============== SUCCESS MESSAGES ==============

SUCCESS_MESSAGES = {
    "connected": "Successfully connected to RoArm",
    "home_reached": "Home position reached",
    "scan_complete": "Scan pattern completed",
    "teaching_saved": "Teaching sequence saved",
    "calibration_done": "Calibration completed"
}
