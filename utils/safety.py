#!/usr/bin/env python3
"""
RoArm M3 Safety Monitor
Basic safety monitoring and limit checking
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
    max_velocity: float = 3.14  # rad/s
    max_acceleration: float = 5.0  # rad/s²
    max_torque: float = 0.8  # 0-1 normalized


class SafetyMonitor:
    """
    Monitor for safety limits and collision prevention.
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
        
        # Safety margin (radians from limits)
        self.safety_margin = 0.05
        
        # Collision detection
        self.collision_threshold = 0.05  # 5cm minimum distance
        
        # Temperature limits
        self.temp_warning = 50  # °C
        self.temp_critical = 60  # °C
        
        # Voltage limits
        self.voltage_min = 5.5  # V
        self.voltage_max = 7.0  # V
        
        # Statistics
        self.violations = []
        self.warnings = []
        
        logger.info("SafetyMonitor initialized")
    
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
                logger.warning(f"Unknown joint: {joint}")
                continue
            
            limit = self.limits[joint]
            
            # Check hard limits
            if position < limit.min_position or position > limit.max_position:
                logger.error(
                    f"Position limit violation: {joint} = {position:.3f} rad "
                    f"(limits: {limit.min_position:.3f} to {limit.max_position:.3f})"
                )
                self._record_violation('position_limit', joint, position)
                return False
            
            # Check soft limits (with margin)
            if (position < limit.min_position + self.safety_margin or 
                position > limit.max_position - self.safety_margin):
                logger.warning(
                    f"Position near limit: {joint} = {position:.3f} rad"
                )
                self._record_warning('near_limit', joint, position)
        
        return True
    
    def validate_velocities(self, velocities: Dict[str, float]) -> bool:
        """
        Validate joint velocities are within limits.
        
        Args:
            velocities: Dictionary of joint velocities (rad/s)
            
        Returns:
            True if all velocities are valid
        """
        for joint, velocity in velocities.items():
            if joint not in self.limits:
                continue
            
            limit = self.limits[joint]
            
            if abs(velocity) > limit.max_velocity:
                logger.error(
                    f"Velocity limit violation: {joint} = {velocity:.3f} rad/s "
                    f"(max: {limit.max_velocity:.3f})"
                )
                self._record_violation('velocity_limit', joint, velocity)
                return False
        
        return True
    
    def validate_trajectory(self, trajectory_points: List) -> bool:
        """
        Validate an entire trajectory.
        
        Args:
            trajectory_points: List of trajectory points
            
        Returns:
            True if trajectory is safe
        """
        for i, point in enumerate(trajectory_points):
            # Check positions
            if not self.validate_positions(point.positions):
                logger.error(f"Trajectory validation failed at point {i}")
                return False
            
            # Check velocities
            if hasattr(point, 'velocities'):
                if not self.validate_velocities(point.velocities):
                    logger.error(f"Trajectory velocity validation failed at point {i}")
                    return False
            
            # Check accelerations
            if hasattr(point, 'accelerations'):
                for joint, accel in point.accelerations.items():
                    if joint in self.limits:
                        if abs(accel) > self.limits[joint].max_acceleration:
                            logger.warning(
                                f"High acceleration at point {i}: "
                                f"{joint} = {accel:.3f} rad/s²"
                            )
        
        return True
    
    def check_collision_risk(self, positions: Dict[str, float]) -> bool:
        """
        Check for potential collisions.
        
        Args:
            positions: Joint positions
            
        Returns:
            True if collision risk detected
        """
        # Simple collision check based on joint configurations
        # This would need actual kinematics for real collision detection
        
        # Example: Check if elbow is too bent when shoulder is down
        if 'shoulder' in positions and 'elbow' in positions:
            if positions['shoulder'] < -1.0 and positions['elbow'] > 2.5:
                logger.warning("Potential self-collision risk detected")
                self._record_warning('collision_risk', 'self', positions)
                return True
        
        return False
    
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
            logger.warning(f"High temperature: {temperature}°C")
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
            logger.error(f"Low voltage: {voltage}V")
            self._record_violation('voltage_low', 'system', voltage)
            return 'low'
        elif voltage > self.voltage_max:
            logger.error(f"High voltage: {voltage}V")
            self._record_violation('voltage_high', 'system', voltage)
            return 'high'
        
        return 'ok'
    
    def calculate_safe_speed(self, current_pos: Dict[str, float],
                           target_pos: Dict[str, float],
                           max_speed: float = 1.0) -> float:
        """
        Calculate safe speed based on movement.
        
        Args:
            current_pos: Current positions
            target_pos: Target positions
            max_speed: Maximum allowed speed
            
        Returns:
            Safe speed factor (0.1 to max_speed)
        """
        safe_speed = max_speed
        
        # Reduce speed near limits
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
            
            # Reduce speed near limits
            if min_margin < self.safety_margin * 2:
                speed_factor = max(0.1, min_margin / (self.safety_margin * 2))
                safe_speed = min(safe_speed, max_speed * speed_factor)
                logger.debug(f"Reduced speed to {safe_speed:.2f} near limits")
        
        # Reduce speed for large movements
        max_displacement = 0
        for joint in current_pos:
            if joint in target_pos:
                displacement = abs(target_pos[joint] - current_pos[joint])
                max_displacement = max(max_displacement, displacement)
        
        if max_displacement > 1.57:  # More than 90 degrees
            safe_speed = min(safe_speed, max_speed * 0.5)
            logger.debug(f"Reduced speed to {safe_speed:.2f} for large movement")
        
        return safe_speed
    
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
            'recent_warnings': self.warnings[-10:]
        }
    
    def reset_statistics(self):
        """Reset violation and warning statistics."""
        self.violations.clear()
        self.warnings.clear()
        logger.info("Safety statistics reset")


class CollisionDetector:
    """
    Advanced collision detection using robot kinematics.
    """
    
    def __init__(self, link_lengths: Dict[str, float]):
        """
        Initialize collision detector.
        
        Args:
            link_lengths: Dictionary of link lengths in meters
        """
        self.link_lengths = link_lengths
        self.min_distance = 0.05  # 5cm minimum distance
        
        logger.info("CollisionDetector initialized")
    
    def check_self_collision(self, joint_positions: Dict[str, float]) -> bool:
        """
        Check for self-collision using forward kinematics.
        
        Args:
            joint_positions: Current joint positions
            
        Returns:
            True if collision detected
        """
        # This would require full forward kinematics implementation
        # Simplified check for now
        
        # Example: Check extreme positions
        if 'elbow' in joint_positions:
            if joint_positions['elbow'] > 3.0:  # Very bent
                if 'wrist' in joint_positions:
                    if abs(joint_positions['wrist']) > 1.4:
                        logger.warning("Potential wrist-base collision")
                        return True
        
        return False
    
    def check_workspace_limits(self, endpoint_position: Tuple[float, float, float]) -> bool:
        """
        Check if endpoint is within workspace.
        
        Args:
            endpoint_position: (x, y, z) position in meters
            
        Returns:
            True if within workspace
        """
        x, y, z = endpoint_position
        
        # Calculate distance from base
        distance = (x**2 + y**2 + z**2) ** 0.5
        
        # Maximum reach (sum of link lengths)
        max_reach = sum(self.link_lengths.values())
        
        if distance > max_reach:
            logger.warning(f"Position out of reach: {distance:.3f}m > {max_reach:.3f}m")
            return False
        
        # Minimum reach (avoid singularities)
        min_reach = 0.1  # 10cm
        if distance < min_reach:
            logger.warning(f"Position too close: {distance:.3f}m < {min_reach:.3f}m")
            return False
        
        return True
