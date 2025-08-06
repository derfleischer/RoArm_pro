# ============================================
# utils/safety.py
# ============================================
#!/usr/bin/env python3
"""
Safety monitoring für RoArm M3.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Überwacht Sicherheitsgrenzen."""
    
    def __init__(self, servo_limits: Dict[str, tuple]):
        self.servo_limits = servo_limits
        
    def validate_positions(self, positions: Dict[str, float]) -> bool:
        """
        Validiert Positionen gegen Servo-Limits.
        
        Args:
            positions: Joint-Positionen
            
        Returns:
            True wenn alle Positionen gültig
        """
        for joint, pos in positions.items():
            if joint in self.servo_limits:
                min_val, max_val = self.servo_limits[joint]
                if pos < min_val or pos > max_val:
                    logger.error(f"Position {joint}={pos:.3f} outside limits [{min_val:.3f}, {max_val:.3f}]")
                    return False
        return True
    
    def clamp_positions(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Begrenzt Positionen auf sichere Werte."""
        clamped = {}
        for joint, pos in positions.items():
            if joint in self.servo_limits:
                min_val, max_val = self.servo_limits[joint]
                clamped[joint] = max(min_val, min(max_val, pos))
            else:
                clamped[joint] = pos
        return clamped
