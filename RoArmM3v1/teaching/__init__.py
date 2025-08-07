# ============================================
# teaching/__init__.py
# ============================================
"""
RoArm M3 Teaching Module
Record and replay movement sequences
"""

from .recorder import (
    TeachingRecorder,
    TeachingWaypoint,
    TeachingSequence,
    RecordingMode
)

__all__ = [
    'TeachingRecorder',
    'TeachingWaypoint',
    'TeachingSequence',
    'RecordingMode'
]
