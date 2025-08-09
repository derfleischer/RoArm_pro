# ============================================
# calibration/__init__.py
# ============================================
"""
RoArm M3 Calibration Suite
Professionelle Kalibrierung für präzise Bewegungen und Scanner-Ausrichtung.
"""

# Versuche den Import - wenn die neue Version da ist, nutze sie
try:
    from calibration.calibration_suite import (
        SafeCalibrationSuite as CalibrationSuite,
        CalibrationType,
        CalibrationPoint,
        JointCalibration,
        ScannerCalibration,
        SystemCalibration
    )
except ImportError:
    # Fallback für die alte Version
    from calibration.calibration_suite import *
    # Wenn SafeCalibrationSuite existiert, nutze es als CalibrationSuite
    if 'SafeCalibrationSuite' in locals():
        CalibrationSuite = SafeCalibrationSuite

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
