# ============================================
# calibration/__init__.py
# ============================================
"""
RoArm M3 Calibration Suite
Professionelle Kalibrierung für präzise Bewegungen und Scanner-Ausrichtung.
"""

from .calibration_suite import (
    CalibrationSuite,
    CalibrationType,
    CalibrationPoint,
    JointCalibration,
    ScannerCalibration,
    SystemCalibration
)

__all__ = [
    'CalibrationSuite',
    'CalibrationType',
    'CalibrationPoint',
    'JointCalibration',
    'ScannerCalibration',
    'SystemCalibration'
]

__version__ = '2.0.0'
__author__ = 'RoArm Professional Team'
