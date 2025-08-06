#!/usr/bin/env python3
"""
RoArm M3 Trajectory Generation
Smooth trajectory planning with multiple profiles
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

import logging
logger = logging.getLogger(__name__)


class TrajectoryType(Enum):
    """Available trajectory types."""
    LINEAR = "linear"              # Constant velocity
    TRAPEZOIDAL = "trapezoidal"   # Trapezoidal velocity profile
    S_CURVE = "s_curve"            # S-curve velocity profile
    SINUSOIDAL = "sinusoidal"     # Sinusoidal motion
    MINIMUM_JERK = "minimum_jerk"  # Minimum jerk trajectory


@dataclass
class TrajectoryPoint:
    """A single point in a trajectory."""
    time: float                        # Time since start (s)
    positions: Dict[str, float]        # Joint positions (rad)
    velocities: Dict[str, float]       # Joint velocities (rad/s)
    accelerations: Dict[str, float]    # Joint accelerations (rad/s²)
    time_delta: float = 0.0            # Time since last point


class TrajectoryGenerator:
    """
    Generates smooth trajectories for robot motion.
    Supports multiple trajectory profiles.
    """
    
    def __init__(self, sample_rate: float = 50.0):
        """
        Initialize trajectory generator.
        
        Args:
            sample_rate: Trajectory sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.sample_time = 1.0 / sample_rate
        
        # Default parameters
        self.default_velocity = 1.0      # rad/s
        self.default_acceleration = 2.0  # rad/s²
        self.default_jerk = 5.0          # rad/s³
        
        logger.info(f"TrajectoryGenerator initialized at {sample_rate} Hz")
    
    def generate(self, start: Dict[str, float],
                end: Dict[str, float],
                speed: float = 1.0,
                trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                acceleration: Optional[float] = None,
                jerk: Optional[float] = None) -> List[TrajectoryPoint]:
        """
        Generate trajectory from start to end position.
        
        Args:
            start: Starting joint positions
            end: Target joint positions
            speed: Speed factor (0.1 to 2.0)
            trajectory_type: Type of trajectory profile
            acceleration: Max acceleration (rad/s²)
            jerk: Max jerk (rad/s³)
            
        Returns:
            List of trajectory points
        """
        # Validate inputs
        if not self._validate_positions(start, end):
            logger.error("Invalid positions")
            return []
        
        # Use defaults if not specified
        acceleration = acceleration or self.default_acceleration * speed
        jerk = jerk or self.default_jerk * speed
        
        # Generate based on type
        if trajectory_type == TrajectoryType.LINEAR:
            return self._generate_linear(start, end, speed)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            return self._generate_trapezoidal(start, end, speed, acceleration)
        elif trajectory_type == TrajectoryType.S_CURVE:
            return self._generate_s_curve(start, end, speed, acceleration, jerk)
        elif trajectory_type == TrajectoryType.SINUSOIDAL:
            return self._generate_sinusoidal(start, end, speed)
        elif trajectory_type == TrajectoryType.MINIMUM_JERK:
            return self._generate_minimum_jerk(start, end, speed)
        else:
            logger.warning(f"Unknown trajectory type: {trajectory_type}, using linear")
            return self._generate_linear(start, end, speed)
    
    def _validate_positions(self, start: Dict[str, float], 
                           end: Dict[str, float]) -> bool:
        """Validate position dictionaries."""
        if not start or not end:
            return False
        
        # Check keys match
        if set(start.keys()) != set(end.keys()):
            logger.error("Start and end positions have different joints")
            return False
        
        return True
    
    def _calculate_duration(self, start: Dict[str, float],
                          end: Dict[str, float],
                          max_velocity: float) -> float:
        """Calculate movement duration based on largest displacement."""
        max_displacement = 0.0
        
        for joint in start:
            displacement = abs(end[joint] - start[joint])
            max_displacement = max(max_displacement, displacement)
        
        if max_displacement == 0:
            return 0.0
        
        # Duration based on max velocity
        duration = max_displacement / max_velocity
        
        # Minimum duration for safety
        return max(0.5, duration)
    
    def _generate_linear(self, start: Dict[str, float],
                        end: Dict[str, float],
                        speed: float) -> List[TrajectoryPoint]:
        """Generate linear (constant velocity) trajectory."""
        points = []
        
        # Calculate duration
        max_velocity = self.default_velocity * speed
        duration = self._calculate_duration(start, end, max_velocity)
        
        # Number of points
        num_points = max(2, int(duration * self.sample_rate))
        
        # Generate points
        for i in range(num_points + 1):
            t = i / num_points
            time = t * duration
            
            # Linear interpolation
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                positions[joint] = start[joint] + t * (end[joint] - start[joint])
                
                if i == 0 or i == num_points:
                    velocities[joint] = 0.0
                else:
                    velocities[joint] = (end[joint] - start[joint]) / duration
                
                accelerations[joint] = 0.0
            
            # Calculate time delta
            time_delta = self.sample_time if i > 0 else 0.0
            
            points.append(TrajectoryPoint(
                time=time,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=time_delta
            ))
        
        logger.debug(f"Generated {len(points)} linear trajectory points")
        return points
    
    def _generate_trapezoidal(self, start: Dict[str, float],
                             end: Dict[str, float],
                             speed: float,
                             acceleration: float) -> List[TrajectoryPoint]:
        """Generate trapezoidal velocity profile trajectory."""
        points = []
        
        # Calculate parameters for each joint
        joint_params = {}
        max_duration = 0.0
        
        for joint in start:
            displacement = end[joint] - start[joint]
            
            if abs(displacement) < 0.001:
                joint_params[joint] = {
                    'displacement': 0,
                    't_accel': 0,
                    't_cruise': 0,
                    't_total': 0,
                    'v_max': 0
                }
                continue
            
            # Maximum velocity for this joint
            v_max = self.default_velocity * speed
            
            # Time to accelerate to max velocity
            t_accel = v_max / acceleration
            
            # Distance covered during acceleration
            d_accel = 0.5 * acceleration * t_accel**2
            
            # Check if we can reach max velocity
            if 2 * d_accel > abs(displacement):
                # Triangle profile (no cruise)
                t_accel = math.sqrt(abs(displacement) / acceleration)
                t_cruise = 0
                v_max = acceleration * t_accel
            else:
                # Trapezoidal profile
                t_cruise = (abs(displacement) - 2 * d_accel) / v_max
            
            t_total = 2 * t_accel + t_cruise
            max_duration = max(max_duration, t_total)
            
            joint_params[joint] = {
                'displacement': displacement,
                't_accel': t_accel,
                't_cruise': t_cruise,
                't_total': t_total,
                'v_max': v_max * np.sign(displacement)
            }
        
        # Generate points
        num_points = max(3, int(max_duration * self.sample_rate))
        
        for i in range(num_points + 1):
            time = i * max_duration / num_points
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                params = joint_params[joint]
                
                if params['displacement'] == 0:
                    positions[joint] = start[joint]
                    velocities[joint] = 0
                    accelerations[joint] = 0
                    continue
                
                # Determine phase
                if time <= params['t_accel']:
                    # Acceleration phase
                    positions[joint] = start[joint] + 0.5 * acceleration * time**2 * np.sign(params['displacement'])
                    velocities[joint] = acceleration * time * np.sign(params['displacement'])
                    accelerations[joint] = acceleration * np.sign(params['displacement'])
                
                elif time <= params['t_accel'] + params['t_cruise']:
                    # Cruise phase
                    t_cruise = time - params['t_accel']
                    d_accel = 0.5 * acceleration * params['t_accel']**2
                    positions[joint] = start[joint] + (d_accel + params['v_max'] * t_cruise) * np.sign(params['displacement'])
                    velocities[joint] = params['v_max']
                    accelerations[joint] = 0
                
                elif time <= params['t_total']:
                    # Deceleration phase
                    t_decel = time - params['t_accel'] - params['t_cruise']
                    d_before = 0.5 * acceleration * params['t_accel']**2 + params['v_max'] * params['t_cruise']
                    positions[joint] = start[joint] + (d_before + params['v_max'] * t_decel - 0.5 * acceleration * t_decel**2) * np.sign(params['displacement'])
                    velocities[joint] = (params['v_max'] - acceleration * t_decel) * np.sign(params['displacement'])
                    accelerations[joint] = -acceleration * np.sign(params['displacement'])
                
                else:
                    # End position
                    positions[joint] = end[joint]
                    velocities[joint] = 0
                    accelerations[joint] = 0
            
            # Calculate time delta
            time_delta = self.sample_time if i > 0 else 0.0
            
            points.append(TrajectoryPoint(
                time=time,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=time_delta
            ))
        
        logger.debug(f"Generated {len(points)} trapezoidal trajectory points")
        return points
    
    def _generate_s_curve(self, start: Dict[str, float],
                         end: Dict[str, float],
                         speed: float,
                         acceleration: float,
                         jerk: float) -> List[TrajectoryPoint]:
        """Generate S-curve trajectory with jerk limiting."""
        # For simplicity, use smoothed trapezoidal
        trap_points = self._generate_trapezoidal(start, end, speed, acceleration)
        
        if len(trap_points) < 3:
            return trap_points
        
        # Apply smoothing filter for S-curve effect
        window_size = min(5, len(trap_points) // 4)
        if window_size < 3:
            return trap_points
        
        smoothed_points = []
        
        for i, point in enumerate(trap_points):
            if i < window_size or i >= len(trap_points) - window_size:
                smoothed_points.append(point)
            else:
                # Average positions in window
                avg_positions = {}
                for joint in point.positions:
                    values = []
                    for j in range(i - window_size//2, i + window_size//2 + 1):
                        values.append(trap_points[j].positions[joint])
                    avg_positions[joint] = np.mean(values)
                
                smoothed_point = TrajectoryPoint(
                    time=point.time,
                    positions=avg_positions,
                    velocities=point.velocities,
                    accelerations=point.accelerations,
                    time_delta=point.time_delta
                )
                smoothed_points.append(smoothed_point)
        
        logger.debug(f"Generated {len(smoothed_points)} S-curve trajectory points")
        return smoothed_points
    
    def _generate_sinusoidal(self, start: Dict[str, float],
                            end: Dict[str, float],
                            speed: float) -> List[TrajectoryPoint]:
        """Generate sinusoidal trajectory."""
        points = []
        
        # Calculate duration
        max_velocity = self.default_velocity * speed
        duration = self._calculate_duration(start, end, max_velocity)
        
        # Number of points
        num_points = max(10, int(duration * self.sample_rate))
        
        # Generate points
        for i in range(num_points + 1):
            t = i / num_points
            time = t * duration
            
            # Sinusoidal interpolation
            s = 0.5 * (1 - np.cos(np.pi * t))
            s_dot = 0.5 * np.pi * np.sin(np.pi * t) / duration
            s_ddot = 0.5 * np.pi**2 * np.cos(np.pi * t) / duration**2
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                displacement = end[joint] - start[joint]
                positions[joint] = start[joint] + s * displacement
                velocities[joint] = s_dot * displacement
                accelerations[joint] = s_ddot * displacement
            
            # Calculate time delta
            time_delta = self.sample_time if i > 0 else 0.0
            
            points.append(TrajectoryPoint(
                time=time,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=time_delta
            ))
        
        logger.debug(f"Generated {len(points)} sinusoidal trajectory points")
        return points
    
    def _generate_minimum_jerk(self, start: Dict[str, float],
                              end: Dict[str, float],
                              speed: float) -> List[TrajectoryPoint]:
        """Generate minimum jerk trajectory using 5th order polynomial."""
        points = []
        
        # Calculate duration
        max_velocity = self.default_velocity * speed
        duration = self._calculate_duration(start, end, max_velocity)
        
        # Number of points
        num_points = max(10, int(duration * self.sample_rate))
        
        # Generate points
        for i in range(num_points + 1):
            tau = i / num_points  # Normalized time [0, 1]
            time = tau * duration
            
            # 5th order polynomial coefficients for minimum jerk
            # s(t) = 10*t^3 - 15*t^4 + 6*t^5
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                displacement = end[joint] - start[joint]
                positions[joint] = start[joint] + s * displacement
                velocities[joint] = s_dot * displacement
                accelerations[joint] = s_ddot * displacement
            
            # Calculate time delta
            time_delta = self.sample_time if i > 0 else 0.0
            
            points.append(TrajectoryPoint(
                time=time,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=time_delta
            ))
        
        logger.debug(f"Generated {len(points)} minimum jerk trajectory points")
        return points
