"""
RoArm M3 Utilities
Logging, terminal control, and helper functions
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .logger import setup_logger, get_logger
from .terminal import TerminalController

# Optional utilities
try:
    from .safety import SafetyMonitor
    SAFETY_MONITOR = True
except ImportError:
    SafetyMonitor = None
    SAFETY_MONITOR = False

try:
    from .debug_mode import MockController, DebugMode
    DEBUG_MODE = True
except ImportError:
    MockController = None
    DebugMode = None
    DEBUG_MODE = False

__all__ = [
    'setup_logger',
    'get_logger',
    'TerminalController',
    'SAFETY_MONITOR',
    'DEBUG_MODE'
]

if SAFETY_MONITOR:
    __all__.append('SafetyMonitor')
if DEBUG_MODE:
    __all__.extend(['MockController', 'DebugMode'])
