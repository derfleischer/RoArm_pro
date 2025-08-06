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

# ============================================
# patterns/__init__.py
# ============================================
"""
RoArm M3 Scan Patterns Module
Predefined scanning patterns for 3D scanning
"""

from .scan_patterns import (
    ScanPattern,
    RasterScanPattern,
    SpiralScanPattern,
    SphericalScanPattern,
    TurntableScanPattern,
    CobwebScanPattern,
    ScanPoint
)

__all__ = [
    'ScanPattern',
    'RasterScanPattern',
    'SpiralScanPattern',
    'SphericalScanPattern',
    'TurntableScanPattern',
    'CobwebScanPattern',
    'ScanPoint'
]
