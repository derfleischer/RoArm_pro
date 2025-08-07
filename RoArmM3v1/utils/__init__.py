#!/usr/bin/env python3
"""
RoArm M3 Utilities Package
Hilfsfunktionen und Tools für das RoArm System.
"""

from .logger import setup_logger, get_logger
from .terminal import TerminalController
from .safety import SafetyMonitor

# NEU: Debug Mode statt debug_tool
from .debug_mode import (
    MockController,
    MockSerialManager,
    SimulationMode,
    DebugMenu,
    run_debug_session
)

__all__ = [
    'setup_logger',
    'get_logger',
    'TerminalController',
    'SafetyMonitor',
    'MockController',
    'MockSerialManager', 
    'SimulationMode',
    'DebugMenu',
    'run_debug_session'
]

__version__ = '2.1.0'
__author__ = 'RoArm Professional Team'
