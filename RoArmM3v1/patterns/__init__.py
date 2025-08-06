#!/usr/bin/env python3
"""
RoArm M3 Scan Patterns Module
Predefined scanning patterns for 3D scanning with Revopoint Mini2
"""

from .scan_patterns import (
    ScanPattern,
    ScanPoint,
    RasterScanPattern,
    SpiralScanPattern,
    SphericalScanPattern,
    TurntableScanPattern,
    AdaptiveScanPattern,
    CobwebScanPattern
)

__all__ = [
    'ScanPattern',
    'ScanPoint',
    'RasterScanPattern',
    'SpiralScanPattern',
    'SphericalScanPattern',
    'TurntableScanPattern',
    'AdaptiveScanPattern',
    'CobwebScanPattern'
]

__version__ = '2.0.0'
