"""
RoArm Pro - Professional Robot Arm Controller
============================================

A modular, professional control system for the Waveshare RoArm M3
with focus on scanner operations and macOS optimization.
"""

__version__ = "1.0.0"
__author__ = "RoArm Pro Team"

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("RoArm Pro requires Python 3.8 or later")

# Convenience imports
from .control.controller import RoArmController
from .ui.app import RoArmApp

__all__ = ['RoArmController', 'RoArmApp']
