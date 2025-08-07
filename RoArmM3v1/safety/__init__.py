"""
RoArm M3 Safety System
Emergency stop, graceful shutdown, and monitoring
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .safety_system import (
    SafetySystem,
    SafetyState,
    ShutdownReason,
    SafetyEvent,
    SystemState
)

ENHANCED_SAFETY = False

__all__ = [
    'SafetySystem',
    'SafetyState',
    'ShutdownReason',
    'SafetyEvent',
    'SystemState',
    'ENHANCED_SAFETY'
]
