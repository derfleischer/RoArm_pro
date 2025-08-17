#!/usr/bin/env python3
"""
RoArm M3 Core Module
Zentrale Komponenten für die Robotersteuerung
"""

# Version Info
__version__ = '3.1.0'
__author__ = 'RoArm Professional Team'

# Controller und Config
from .controller import RoArmController, RoArmConfig

# Serial Communication - Real und Mock
from .serial_comm import SerialManager

# Mock/Simulator Support
try:
    from .mock_serial import MockSerialManager, MockRobotState, create_mock_serial
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False
    MockSerialManager = None
    MockRobotState = None
    create_mock_serial = None

# Constants
from .constants import (
    # Servo Limits
    SERVO_LIMITS,
    
    # Standard Positions
    HOME_POSITION,
    SCANNER_POSITION,
    PARK_POSITION,
    
    # Commands
    COMMANDS,
    
    # Motion Parameters
    SPEED_LIMITS,
    ACCELERATION_LIMITS,
    JERK_LIMITS,
    
    # Scanner Specs
    SCANNER_SPECS,
    SCANNER_MOUNT_OFFSET,
    
    # Scan Defaults
    SCAN_DEFAULTS,
    
    # Safety
    SAFETY_LIMITS,
    
    # Teaching
    TEACHING_DEFAULTS,
    
    # Trajectory Profiles
    TRAJECTORY_PROFILES,
    
    # Serial Config
    SERIAL_CONFIG,
    
    # Speed Presets
    DEFAULT_SPEED,
    SPEED_PRESETS,
    
    # Messages
    ERROR_MESSAGES,
    SUCCESS_MESSAGES
)

# Export all public components
__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Controller
    'RoArmController',
    'RoArmConfig',
    
    # Serial
    'SerialManager',
    
    # Mock/Simulator
    'MockSerialManager',
    'MockRobotState',
    'create_mock_serial',
    'MOCK_AVAILABLE',
    
    # Constants
    'SERVO_LIMITS',
    'HOME_POSITION',
    'SCANNER_POSITION',
    'PARK_POSITION',
    'COMMANDS',
    'SPEED_LIMITS',
    'ACCELERATION_LIMITS',
    'JERK_LIMITS',
    'SCANNER_SPECS',
    'SCANNER_MOUNT_OFFSET',
    'SCAN_DEFAULTS',
    'SAFETY_LIMITS',
    'TEACHING_DEFAULTS',
    'TRAJECTORY_PROFILES',
    'SERIAL_CONFIG',
    'DEFAULT_SPEED',
    'SPEED_PRESETS',
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES'
]

def get_serial_manager(simulate: bool = False, *args, **kwargs):
    """
    Factory function to get appropriate serial manager.
    
    Args:
        simulate: If True, returns MockSerialManager, else SerialManager
        *args, **kwargs: Arguments passed to the manager
        
    Returns:
        SerialManager or MockSerialManager instance
    """
    if simulate:
        if MOCK_AVAILABLE:
            return MockSerialManager(*args, **kwargs)
        else:
            raise ImportError("MockSerialManager not available - check mock_serial.py")
    else:
        return SerialManager(*args, **kwargs)

def is_simulator_available() -> bool:
    """
    Check if simulator mode is available.
    
    Returns:
        True if MockSerialManager can be imported
    """
    return MOCK_AVAILABLE

# Module initialization message
def _init_message():
    """Print initialization message if debug is enabled."""
    import os
    if os.environ.get('ROARM_DEBUG'):
        print(f"RoArm Core Module v{__version__} loaded")
        print(f"  Simulator: {'Available' if MOCK_AVAILABLE else 'Not available'}")

# Run init if debug
_init_message()
