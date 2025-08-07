"""
RoArm M3 Scan Patterns
Optimized patterns for 3D scanning
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .scan_patterns import (
    ScanPattern,
    ScanPoint,
    RasterScanPattern,
    SpiralScanPattern,
    SphericalScanPattern,
    TurntableScanPattern,
    CobwebScanPattern
)

ENHANCED_PATTERNS = False

__all__ = [
    'ScanPattern',
    'ScanPoint',
    'RasterScanPattern',
    'SpiralScanPattern',
    'SphericalScanPattern',
    'TurntableScanPattern',
    'CobwebScanPattern',
    'ENHANCED_PATTERNS'
]
