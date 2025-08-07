"""
RoArm M3 Calibration Suite
Professionelle Kalibrierung für präzise Bewegungen
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .calibration_suite import (
    CalibrationSuite,
    CalibrationType,
    CalibrationPoint,
    JointCalibration,
    ScannerCalibration,
    SystemCalibration
)

ENHANCED_CALIBRATION = False

try:
    from enhanced.calibration import AutoCalibration, VisionCalibration
    ENHANCED_CALIBRATION = True
except ImportError:
    pass

__all__ = [
    'CalibrationSuite',
    'CalibrationType',
    'CalibrationPoint',
    'JointCalibration',
    'ScannerCalibration',
    'SystemCalibration',
    'ENHANCED_CALIBRATION'
]
