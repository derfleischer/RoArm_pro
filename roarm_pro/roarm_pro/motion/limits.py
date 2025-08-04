"""
Joint limits validation and clamping
Ensures all movements stay within safe ranges
"""

import logging
from typing import Dict, Tuple, Optional
from ..config import SERVO_LIMITS, SCANNER_LIMITS, SAFETY_LIMITS

logger = logging.getLogger(__name__)

class JointLimits:
    """Manage and validate joint limits"""
    
    def __init__(self, scanner_mounted: bool = False):
        """
        Initialize joint limits
        Args:
            scanner_mounted: Use scanner-safe limits if True
        """
        self.scanner_mounted = scanner_mounted
        self._limits = SERVO_LIMITS.copy()
        
        # Apply scanner limits if mounted
        if scanner_mounted:
            self._apply_scanner_limits()
    
    def _apply_scanner_limits(self):
        """Apply restricted limits when scanner is mounted"""
        for joint, limits in SCANNER_LIMITS.items():
            self._limits[joint] = limits
        logger.info("📷 Scanner limits applied")
    
    def set_scanner_mounted(self, mounted: bool):
        """Update scanner mounted state"""
        self.scanner_mounted = mounted
        self._limits = SERVO_LIMITS.copy()
        
        if mounted:
            self._apply_scanner_limits()
    
    def get_limits(self, joint: str) -> Tuple[float, float]:
        """Get limits for specific joint"""
        return self._limits.get(joint, (-3.14, 3.14))
    
    def validate(self, **joint_positions) -> Dict[str, float]:
        """
        Validate and clamp joint positions
        
        Args:
            **joint_positions: Joint positions to validate
            
        Returns:
            Dict of clamped positions
        """
        validated = {}
        
        for joint, value in joint_positions.items():
            if value is None:
                continue
                
            if joint in self._limits:
                min_val, max_val = self._limits[joint]
                
                if value < min_val or value > max_val:
                    # Clamp to limits
                    clamped = max(min_val, min(max_val, value))
                    logger.warning(
                        f"⚠️ {joint}={value:.3f} clamped to {clamped:.3f} "
                        f"(limits: {min_val:.3f} to {max_val:.3f})"
                    )
                    validated[joint] = clamped
                else:
                    validated[joint] = value
            else:
                # Unknown joint, pass through
                logger.debug(f"Unknown joint: {joint}")
                validated[joint] = value
        
        return validated
    
    def check_velocity(self, current: Dict[str, float], 
                      target: Dict[str, float], 
                      duration: float) -> bool:
        """
        Check if movement velocity is within limits
        
        Args:
            current: Current positions
            target: Target positions
            duration: Movement duration
            
        Returns:
            True if velocities are safe
        """
        max_velocity = SAFETY_LIMITS["max_velocity"]
        
        for joint in current:
            if joint in target:
                distance = abs(target[joint] - current[joint])
                velocity = distance / duration
                
                if velocity > max_velocity:
                    logger.warning(
                        f"⚠️ {joint} velocity {velocity:.2f} rad/s exceeds "
                        f"limit {max_velocity:.2f} rad/s"
                    )
                    return False
        
        return True
    
    def suggest_duration(self, current: Dict[str, float], 
                        target: Dict[str, float]) -> float:
        """
        Suggest safe duration for movement
        
        Args:
            current: Current positions
            target: Target positions
            
        Returns:
            Suggested duration in seconds
        """
        max_velocity = SAFETY_LIMITS["max_velocity"]
        max_distance = 0.0
        
        for joint in current:
            if joint in target:
                distance = abs(target[joint] - current[joint])
                max_distance = max(max_distance, distance)
        
        # Calculate minimum duration based on max velocity
        min_duration = max_distance / max_velocity
        
        # Add safety margin
        suggested = min_duration * 1.2
        
        # Ensure minimum duration
        return max(suggested, 0.5)
    
    def is_position_safe(self, positions: Dict[str, float]) -> bool:
        """
        Check if position is within all limits
        
        Args:
            positions: Joint positions to check
            
        Returns:
            True if all positions are within limits
        """
        for joint, value in positions.items():
            if joint in self._limits:
                min_val, max_val = self._limits[joint]
                if value < min_val or value > max_val:
                    return False
        return True
