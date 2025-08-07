# ============================================
# motion/__init__.py
# ============================================
"""
RoArm M3 Motion Module
Trajectory generation and motion planning
"""

from .trajectory import TrajectoryGenerator, TrajectoryType, TrajectoryPoint

__all__ = [
    'TrajectoryGenerator',
    'TrajectoryType',
    'TrajectoryPoint'
]

