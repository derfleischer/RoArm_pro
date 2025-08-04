"""
Default configuration values for RoArm Pro
All hardware limits, positions and commands
"""

import math

# ==================== HARDWARE LIMITS ====================

# Servo limits in radians (from official Waveshare documentation)
SERVO_LIMITS = {
    "base": (-3.14, 3.14),      # 360° full rotation
    "shoulder": (-1.57, 1.57),  # 180° range  
    "elbow": (0.0, 3.14),       # 0-180° range
    "wrist": (-1.57, 1.57),     # 180° range (pitch)
    "roll": (-3.14, 3.14),      # 360° full rotation
    "hand": (1.08, 3.14)        # 62-180° (gripper)
}

# Scanner-safe limits (when Revopoint Mini2 is mounted)
SCANNER_LIMITS = {
    "hand": (2.2, 2.8),         # Safe grip range for scanner
    "shoulder": (-1.0, 1.0),    # Reduced to prevent collisions
    "wrist": (-1.3, 1.3)        # Slightly reduced for cable management
}

# ==================== POSITIONS ====================

# Default home position
HOME_POSITION = {
    "base": 0.0,
    "shoulder": 0.0,
    "elbow": 1.57,      # 90°
    "wrist": -1.57,     # Compensated for level scanner
    "roll": 1.57,       # 90° for scanner orientation
    "hand": 2.5         # Safe scanner grip
}

# Calibration positions
CALIBRATION_POSITIONS = {
    "rest": {
        "base": 0.0,
        "shoulder": -0.5,
        "elbow": 2.0,
        "wrist": 0.0,
        "roll": 0.0,
        "hand": 3.14    # Fully closed
    },
    "scanner_mount": {
        "base": 0.0,
        "shoulder": 0.3,
        "elbow": 1.0,
        "wrist": -1.3,
        "roll": 1.57,
        "hand": 1.3     # Open for mounting
    },
    "scan_start": {
        "base": 0.0,
        "shoulder": 0.0,
        "elbow": 1.2,
        "wrist": -1.2,
        "roll": 1.57,
        "hand": 2.5
    },
    "transport": {
        "base": 0.0,
        "shoulder": 0.0,
        "elbow": 0.5,
        "wrist": 0.0,
        "roll": 0.0,
        "hand": 3.14
    }
}

# ==================== COMMANDS ====================

# Official Waveshare JSON command IDs
COMMANDS = {
    "EMERGENCY_STOP": {"T": 0},
    "STATUS_QUERY": {"T": 1},
    "POSITION_QUERY": {"T": 2},
    "LED_CONTROL": {"T": 51},
    "JOINT_CONTROL": {"T": 102},
    "COORDINATE_CONTROL": {"T": 104},
    "GRIPPER_CONTROL": {"T": 106},
    "LED_ON": {"T": 114, "led": 1},
    "LED_OFF": {"T": 114, "led": 0},
    "TORQUE_CONTROL": {"T": 210},
    "STATUS_QUERY_EXT": {"T": 1051}
}

# ==================== MOTION PARAMETERS ====================

# Default motion parameters
MOTION_DEFAULTS = {
    "default_speed": 1.0,
    "min_duration": 0.5,
    "max_duration": 10.0,
    "trajectory_points": 50,
    "settle_time": 0.1
}

# Speed profiles for different operations
SPEED_PROFILES = {
    "slow": {
        "factor": 0.3,
        "min_duration": 2.0,
        "settle_time": 0.3
    },
    "normal": {
        "factor": 1.0,
        "min_duration": 1.0,
        "settle_time": 0.1
    },
    "fast": {
        "factor": 1.5,
        "min_duration": 0.5,
        "settle_time": 0.05
    },
    "scanner": {
        "factor": 0.5,
        "min_duration": 1.5,
        "settle_time": 0.2
    }
}

# ==================== SCANNER PARAMETERS ====================

# Revopoint Mini2 scanning parameters
SCANNER_DEFAULTS = {
    "scan_distance": 0.15,      # 15cm optimal distance
    "scan_width": 0.2,          # 20cm scan width
    "scan_height": 0.15,        # 15cm scan height
    "step_size": 0.01,          # 1cm steps
    "rotation_speed": 0.5,      # rad/s for continuous scans
    "led_brightness": 128       # 50% brightness
}

# ==================== SAFETY PARAMETERS ====================

# Safety limits
SAFETY_LIMITS = {
    "max_velocity": 2.0,        # rad/s
    "max_acceleration": 4.0,    # rad/s²
    "emergency_decel": 10.0,    # rad/s² for emergency stop
    "collision_threshold": 0.1,  # Distance threshold
    "timeout": 30.0             # Command timeout
}

# ==================== COMMUNICATION ====================

# Serial communication settings
SERIAL_DEFAULTS = {
    "baudrate": 115200,
    "timeout": 2.0,
    "write_timeout": 1.0,
    "command_delay": 0.02,      # 20ms between commands
    "retry_count": 3
}
