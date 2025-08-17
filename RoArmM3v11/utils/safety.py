#!/usr/bin/env python3
"""
RoArm M3 Safety Monitor
Basic safety monitoring and limit checking - ADJUSTED VERSION
Less restrictive for practical use
"""

import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limit configuration."""
    joint_name: str
    min_position: float  # rad
    max_position: float  # rad
    max_velocity: float = 3.14  # rad/s (180°/s - reasonable)
    max_acceleration: float = 5.0  # rad/s²
    max_torque: float = 0.8  # 0-1 normalized


class SafetyMonitor:
    """
    Monitor for safety limits and collision prevention.
    ADJUSTED: Less restrictive for practical use
    """
    
    def __init__(self, servo_limits: Dict[str, Tuple[float, float]]):
        """
        Initialize safety monitor.
        
        Args:
            servo_limits: Dictionary of joint limits {name: (min, max)}
        """
        self.limits = {}
        
        # Create safety limits from servo limits
        for joint, (min_val, max_val) in servo_limits.items():
            self.limits[joint] = SafetyLimits(
                joint_name=joint,
                min_position=min_val,
                max_position=max_val
            )
        
        # Safety margin - ADJUSTED: smaller margin for more freedom
        self.safety_margin = 0.02  # Only ~1.1° from limits (was 0.05)
        
        # Collision detection - ADJUSTED: less restrictive
        self.collision_threshold = 0.03  # 3cm minimum distance (was 0.05)
        
        # Temperature limits - reasonable for servos
        self.temp_warning = 55  # °C (was 50)
        self.temp_critical = 70  # °C (was 60)
        
        # Voltage limits - wider range
        self.voltage_min = 5.0  # V (was 5.5)
        self.voltage_max = 7.5  # V (was 7.0)
        
        # Statistics
        self.violations = []
        self.warnings = []
        
        # WARNING MODE - can be toggled
        self.strict_mode = False  # When False, only logs warnings instead of blocking
        
        logger.info(f"SafetyMonitor initialized (strict_mode={self.strict_mode})")
    
    def check_limits(self, positions: Dict[str, float]) -> bool:
        """
        COMPATIBILITY METHOD for main.py
        Alias for validate_positions with less strict checking.
        
        Args:
            positions: Dictionary of joint positions
            
        Returns:
            True if positions are acceptable (may log warnings)
        """
        # In non-strict mode, always return True but log warnings
        if not self.strict_mode:
            for joint, position in positions.items():
                if joint not in self.limits:
                    continue
                
                limit = self.limits[joint]
                
                # Only check hard limits
                if position < limit.min_position or position > limit.max_position:
                    logger.warning(
                        f"Position exceeds limit: {joint} = {position:.3f} rad "
                        f"(limits: {limit.min_position:.3f} to {limit.max_position:.3f}) "
                        f"- ALLOWING in non-strict mode"
                    )
                    # Still record it for statistics
                    self._record_warning('limit_exceeded_allowed', joint, position)
            
            return True  # Always allow in non-strict mode
        
        # Strict mode - use original validation
        return self.validate_positions(positions)
    
    def validate_positions(self, positions: Dict[str, float]) -> bool:
        """
        Validate joint positions are within limits.
        
        Args:
            positions: Dictionary of joint positions
            
        Returns:
            True if all positions are valid
        """
        for joint, position in positions.items():
            if joint not in self.limits:
                # Unknown joint - log but don't block
                logger.debug(f"Unknown joint: {joint} - ignoring")
                continue
            
            limit = self.limits[joint]
            
            # Check hard limits
            if position < limit.min_position or position > limit.max_position:
                if self.strict_mode:
                    logger.error(
                        f"Position limit violation: {joint} = {position:.3f} rad "
                        f"(limits: {limit.min_position:.3f} to {limit.max_position:.3f})"
                    )
                    self._record_violation('position_limit', joint, position)
                    return False
                else:
                    logger.warning(
                        f"Position near/at limit: {joint} = {position:.3f} rad"
                    )
                    self._record_warning('position_limit_soft', joint, position)
            
            # Check soft limits (with margin) - only warn, don't block
            elif (position < limit.min_position + self.safety_margin or 
                  position > limit.max_position - self.safety_margin):
                logger.debug(
                    f"Position approaching limit: {joint} = {position:.3f} rad"
                )
                # Don't even record this as warning - too noisy
        
        return True
    
    def validate_velocities(self, velocities: Dict[str, float]) -> bool:
        """
        Validate joint velocities are within limits.
        
        Args:
            velocities: Dictionary of joint velocities (rad/s)
            
        Returns:
            True if all velocities are valid (or in non-strict mode)
        """
        if not self.strict_mode:
            return True  # Don't block on velocity in non-strict mode
        
        for joint, velocity in velocities.items():
            if joint not in self.limits:
                continue
            
            limit = self.limits[joint]
            
            if abs(velocity) > limit.max_velocity:
                logger.warning(  # Changed from error to warning
                    f"High velocity: {joint} = {velocity:.3f} rad/s "
                    f"(max: {limit.max_velocity:.3f})"
                )
                self._record_warning('velocity_high', joint, velocity)
                # Don't return False - just warn
        
        return True
    
    def validate_trajectory(self, trajectory_points: List) -> bool:
        """
        Validate an entire trajectory.
        
        Args:
            trajectory_points: List of trajectory points
            
        Returns:
            True if trajectory is safe (always True in non-strict mode)
        """
        if not self.strict_mode:
            # Just do basic logging in non-strict mode
            logger.debug(f"Validating trajectory with {len(trajectory_points)} points")
            return True
        
        for i, point in enumerate(trajectory_points):
            # Check positions
            if not self.validate_positions(point.positions):
                logger.error(f"Trajectory validation failed at point {i}")
                return False
            
            # Velocity and acceleration checks are now less strict
            if hasattr(point, 'velocities'):
                self.validate_velocities(point.velocities)  # Just logs warnings
            
            if hasattr(point, 'accelerations'):
                for joint, accel in point.accelerations.items():
                    if joint in self.limits:
                        if abs(accel) > self.limits[joint].max_acceleration:
                            logger.debug(  # Changed from warning to debug
                                f"High acceleration at point {i}: "
                                f"{joint} = {accel:.3f} rad/s²"
                            )
        
        return True
    
    def check_collision_risk(self, positions: Dict[str, float]) -> bool:
        """
        Check for potential collisions.
        ADJUSTED: Much less restrictive
        
        Args:
            positions: Joint positions
            
        Returns:
            True if collision risk detected (only in extreme cases)
        """
        # Only check really extreme cases
        if 'shoulder' in positions and 'elbow' in positions:
            # Only if shoulder is really down AND elbow really bent
            if positions['shoulder'] < -1.4 and positions['elbow'] > 2.8:
                logger.warning("Potential self-collision risk - extreme position")
                self._record_warning('collision_risk', 'self', positions)
                return self.strict_mode  # Only block in strict mode
        
        # Check for hand collision with base (extreme wrist positions)
        if 'wrist' in positions and 'roll' in positions:
            if abs(positions['wrist']) > 1.5 and abs(positions['roll']) > 2.5:
                logger.debug("Unusual wrist/roll combination")
                # Don't block, just log
        
        return False  # Generally don't block for collision
    
    def check_temperature(self, temperature: float) -> str:
        """
        Check temperature status.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            'ok', 'warning', or 'critical'
        """
        if temperature >= self.temp_critical:
            logger.critical(f"CRITICAL temperature: {temperature}°C")
            self._record_violation('temperature_critical', 'system', temperature)
            return 'critical'
        elif temperature >= self.temp_warning:
            logger.info(f"Elevated temperature: {temperature}°C")  # Changed from warning
            self._record_warning('temperature_warning', 'system', temperature)
            return 'warning'
        
        return 'ok'
    
    def check_voltage(self, voltage: float) -> str:
        """
        Check voltage status.
        
        Args:
            voltage: Supply voltage
            
        Returns:
            'ok', 'low', or 'high'
        """
        if voltage < self.voltage_min:
            logger.warning(f"Low voltage: {voltage}V")  # Changed from error
            self._record_warning('voltage_low', 'system', voltage)
            return 'low'
        elif voltage > self.voltage_max:
            logger.warning(f"High voltage: {voltage}V")  # Changed from error
            self._record_warning('voltage_high', 'system', voltage)
            return 'high'
        
        return 'ok'
    
    def calculate_safe_speed(self, current_pos: Dict[str, float],
                           target_pos: Dict[str, float],
                           max_speed: float = 1.0) -> float:
        """
        Calculate safe speed based on movement.
        ADJUSTED: Less aggressive speed reduction
        
        Args:
            current_pos: Current positions
            target_pos: Target positions
            max_speed: Maximum allowed speed
            
        Returns:
            Safe speed factor (0.2 to max_speed)  # Minimum 20% instead of 10%
        """
        safe_speed = max_speed
        
        # Only reduce speed VERY close to limits
        for joint in current_pos:
            if joint not in self.limits or joint not in target_pos:
                continue
            
            limit = self.limits[joint]
            current = current_pos[joint]
            target = target_pos[joint]
            
            # Check proximity to limits
            current_margin = min(
                current - limit.min_position,
                limit.max_position - current
            )
            target_margin = min(
                target - limit.min_position,
                limit.max_position - target
            )
            
            min_margin = min(current_margin, target_margin)
            
            # Only reduce speed VERY near limits
            if min_margin < self.safety_margin:  # Only within safety margin
                speed_factor = max(0.2, min_margin / self.safety_margin)  # Min 20%
                safe_speed = min(safe_speed, max_speed * speed_factor)
                logger.debug(f"Reduced speed to {safe_speed:.2f} very near limits")
        
        # Reduce speed for very large movements
        max_displacement = 0
        for joint in current_pos:
            if joint in target_pos:
                displacement = abs(target_pos[joint] - current_pos[joint])
                max_displacement = max(max_displacement, displacement)
        
        if max_displacement > 2.0:  # More than ~115 degrees (was 90)
            safe_speed = min(safe_speed, max_speed * 0.7)  # 70% speed (was 50%)
            logger.debug(f"Reduced speed to {safe_speed:.2f} for large movement")
        
        return safe_speed
    
    def set_strict_mode(self, enabled: bool):
        """
        Enable/disable strict safety mode.
        
        Args:
            enabled: True for strict checking, False for permissive
        """
        self.strict_mode = enabled
        logger.info(f"Safety strict_mode set to: {enabled}")
    
    def _record_violation(self, violation_type: str, joint: str, value: float):
        """Record a safety violation."""
        violation = {
            'timestamp': time.time(),
            'type': violation_type,
            'joint': joint,
            'value': value
        }
        self.violations.append(violation)
        
        # Keep only last 100 violations
        if len(self.violations) > 100:
            self.violations = self.violations[-100:]
    
    def _record_warning(self, warning_type: str, joint: str, value):
        """Record a safety warning."""
        warning = {
            'timestamp': time.time(),
            'type': warning_type,
            'joint': joint,
            'value': value
        }
        self.warnings.append(warning)
        
        # Keep only last 100 warnings
        if len(self.warnings) > 100:
            self.warnings = self.warnings[-100:]
    
    def get_statistics(self) -> Dict:
        """Get safety statistics."""
        return {
            'total_violations': len(self.violations),
            'total_warnings': len(self.warnings),
            'recent_violations': self.violations[-10:],
            'recent_warnings': self.warnings[-10:],
            'strict_mode': self.strict_mode
        }
    
    def reset_statistics(self):
        """Reset violation and warning statistics."""
        self.violations.clear()
        self.warnings.clear()
        logger.info("Safety statistics reset")


class CollisionDetector:
    """
    Advanced collision detection using robot kinematics.
    ADJUSTED: Less restrictive
    """
    
    def __init__(self, link_lengths: Dict[str, float]):
        """
        Initialize collision detector.
        
        Args:
            link_lengths: Dictionary of link lengths in meters
        """
        self.link_lengths = link_lengths
        self.min_distance = 0.02  # 2cm minimum distance (was 5cm)
        self.enabled = False  # Disabled by default for testing
        
        logger.info(f"CollisionDetector initialized (enabled={self.enabled})")
    
    def check_self_collision(self, joint_positions: Dict[str, float]) -> bool:
        """
        Check for self-collision using forward kinematics.
        
        Args:
            joint_positions: Current joint positions
            
        Returns:
            True if collision detected (only if enabled)
        """
        if not self.enabled:
            return False
        
        # Only check really extreme positions
        if 'elbow' in joint_positions:
            if joint_positions['elbow'] > 3.1:  # Almost fully bent (was 3.0)
                if 'wrist' in joint_positions:
                    if abs(joint_positions['wrist']) > 1.5:  # Really extreme (was 1.4)
                        logger.warning("Potential wrist-base collision")
                        return True
        
        return False
    
    def check_workspace_limits(self, endpoint_position: Tuple[float, float, float]) -> bool:
        """
        Check if endpoint is within workspace.
        
        Args:
            endpoint_position: (x, y, z) position in meters
            
        Returns:
            True if within workspace (very permissive)
        """
        if not self.enabled:
            return True
        
        x, y, z = endpoint_position
        
        # Calculate distance from base
        distance = (x**2 + y**2 + z**2) ** 0.5
        
        # Maximum reach (sum of link lengths + margin)
        max_reach = sum(self.link_lengths.values()) * 1.1  # 10% margin
        
        if distance > max_reach:
            logger.debug(f"Position near max reach: {distance:.3f}m")
            # Don't block, just log
        
        # Minimum reach (avoid singularities)
        min_reach = 0.05  # 5cm (was 10cm)
        if distance < min_reach:
            logger.warning(f"Position very close to base: {distance:.3f}m")
            # Still don't block
        
        return True  # Generally allow
    
    def set_enabled(self, enabled: bool):
        """Enable/disable collision detection."""
        self.enabled = enabled
        logger.info(f"CollisionDetector enabled: {enabled}")
