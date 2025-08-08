"""
RoArm M3 Motion Control
Trajectory generation and motion planning
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .trajectory import (
    TrajectoryGenerator,
    TrajectoryType,
    TrajectoryPoint
)

ENHANCED_MOTION = False

try:
    from enhanced.motion import AdvancedTrajectoryGenerator
    TrajectoryGenerator = AdvancedTrajectoryGenerator
    ENHANCED_MOTION = True
except ImportError:
    pass

__all__ = [
    'TrajectoryGenerator',
    'TrajectoryType',
    'TrajectoryPoint',
    'ENHANCED_MOTION'
]
