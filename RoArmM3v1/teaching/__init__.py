"""
RoArm M3 Teaching System
Advanced recording and playback
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .recorder import (
    TeachingRecorder,
    TeachingWaypoint,
    TeachingSequence,
    RecordingMode
)

ENHANCED_TEACHING = False

try:
    from enhanced.teaching import SmartTeaching
    ENHANCED_TEACHING = True
except ImportError:
    pass

__all__ = [
    'TeachingRecorder',
    'TeachingWaypoint',
    'TeachingSequence',
    'RecordingMode',
    'ENHANCED_TEACHING'
]
