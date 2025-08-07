"""
RoArm M3 Core Module
Basis-Controller und Hardware-Kommunikation
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .constants import (
    SERVO_LIMITS,
    HOME_POSITION,
    SCANNER_POSITION,
    PARK_POSITION,
    COMMANDS,
    SPEED_LIMITS,
    SCANNER_SPECS,
    TRAJECTORY_PROFILES,
    DEFAULT_SPEED
)

from .serial_comm import SerialManager

try:
    from enhanced.controller import EnhancedController as RoArmController
    from enhanced.controller import EnhancedConfig as RoArmConfig
    ENHANCED_CONTROLLER = True
except ImportError:
    from .controller import RoArmController, RoArmConfig
    ENHANCED_CONTROLLER = False

__all__ = [
    'RoArmController',
    'RoArmConfig',
    'SerialManager',
    'SERVO_LIMITS',
    'HOME_POSITION',
    'SCANNER_POSITION',
    'PARK_POSITION',
    'COMMANDS',
    'SPEED_LIMITS',
    'SCANNER_SPECS',
    'TRAJECTORY_PROFILES',
    'DEFAULT_SPEED',
    'ENHANCED_CONTROLLER'
]
