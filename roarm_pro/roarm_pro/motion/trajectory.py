"""
Trajectory generation with various motion profiles
Optimized for smooth scanner movements
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class TrajectoryType(Enum):
    """Available trajectory types"""
    LINEAR = "linear"
    S_CURVE = "s_curve"
    TRAPEZOIDAL = "trapezoidal"
    MINIMUM_JERK = "minimum_jerk"
    CUBIC_SPLINE = "cubic_spline"

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    positions: Dict[str, float]  # Joint positions in radians
    velocities: Dict[str, float] = None  # Joint velocities
    accelerations: Dict[str, float] = None  # Joint accelerations
    time: float = 0.0  # Time from start in seconds

class TrajectoryGenerator:
    """Generate smooth trajectories for robot motion"""
    
    def __init__(self, sample_rate: int = 50):
        """
        Initialize trajectory generator
        Args:
            sample_rate: Points per second (Hz)
        """
        self.sample_rate = sample_rate
        
    def generate(self,
                start_pos: Dict[str, float],
                end_pos: Dict[str, float],
                duration: float,
                trajectory_type: TrajectoryType = TrajectoryType.S_CURVE) -> List[TrajectoryPoint]:
        """
        Generate trajectory between two positions
        
        Args:
            start_pos: Starting joint positions
            end_pos: Target joint positions
            duration: Movement duration in seconds
            trajectory_type: Type of trajectory profile
            
        Returns:
            List of trajectory points
        """
        # Calculate number of points
        num_points = max(int(duration * self.sample_rate), 2)
        
        # Generate time vector
        time_vec = np.linspace(0, duration, num_points)
        
        # Generate normalized trajectory (0 to 1)
        if trajectory_type == TrajectoryType.LINEAR:
            s_vec = self._linear_profile(num_points)
        elif trajectory_type == TrajectoryType.S_CURVE:
            s_vec = self._s_curve_profile(num_points)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            s_vec = self._trapezoidal_profile(num_points)
        elif trajectory_type == TrajectoryType.MINIMUM_JERK:
            s_vec = self._minimum_jerk_profile(num_points)
        else:
            s_vec = self._s_curve_profile(num_points)  # Default
        
        # Generate trajectory points
        points = []
        
        for i, (t, s) in enumerate(zip(time_vec, s_vec)):
            # Interpolate positions
            positions = {}
            for joint in start_pos:
                if joint in end_pos:
                    positions[joint] = start_pos[joint] + s * (end_pos[joint] - start_pos[joint])
                else:
                    positions[joint] = start_pos[joint]
            
            # Calculate velocities (finite difference)
            velocities = {}
            if i > 0 and i < num_points - 1:
                dt = time_vec[i+1] - time_vec[i-1]
                ds = s_vec[i+1] - s_vec[i-1]
                
                for joint in start_pos:
                    if joint in end_pos:
                        velocities[joint] = (end_pos[joint] - start_pos[joint]) * ds / dt
                    else:
                        velocities[joint] = 0.0
            
            points.append(TrajectoryPoint(
                positions=positions,
                velocities=velocities,
                time=t
            ))
        
        return points
    
    def _linear_profile(self, num_points: int) -> np.ndarray:
        """Linear trajectory profile"""
        return np.linspace(0, 1, num_points)
    
    def _s_curve_profile(self, num_points: int) -> np.ndarray:
        """S-curve (smooth) trajectory profile"""
        t = np.linspace(0, 1, num_points)
        # Classic S-curve: 3t² - 2t³
        return 3 * t**2 - 2 * t**3
    
    def _trapezoidal_profile(self, num_points: int, acc_time: float = 0.2) -> np.ndarray:
        """Trapezoidal velocity profile"""
        t = np.linspace(0, 1, num_points)
        s = np.zeros_like(t)
        
        # Acceleration phase
        acc_mask = t <= acc_time
        s[acc_mask] = 0.5 * t[acc_mask]**2 / acc_time
        
        # Constant velocity phase
        const_mask = (t > acc_time) & (t < 1 - acc_time)
        s[const_mask] = t[const_mask] - 0.5 * acc_time
        
        # Deceleration phase
        dec_mask = t >= 1 - acc_time
        t_dec = t[dec_mask] - (1 - acc_time)
        s[dec_mask] = 1 - 0.5 * (acc_time - t_dec)**2 / acc_time
        
        return s
    
    def _minimum_jerk_profile(self, num_points: int) -> np.ndarray:
        """Minimum jerk trajectory (very smooth)"""
        t = np.linspace(0, 1, num_points)
        # 7th order polynomial for minimum jerk
        return 35*t**4 - 84*t**5 + 70*t**6 - 20*t**7
    
    def blend_trajectories(self,
                          traj1: List[TrajectoryPoint],
                          traj2: List[TrajectoryPoint],
                          blend_duration: float = 0.5) -> List[TrajectoryPoint]:
        """
        Blend two trajectories smoothly
        Useful for continuous motion
        """
        # TODO: Implement trajectory blending
        pass
    
    def interpolate_waypoints(self,
                            waypoints: List[Dict[str, float]],
                            durations: List[float],
                            trajectory_type: TrajectoryType = TrajectoryType.S_CURVE) -> List[TrajectoryPoint]:
        """
        Generate trajectory through multiple waypoints
        
        Args:
            waypoints: List of joint positions
            durations: Time between waypoints
            trajectory_type: Profile type
            
        Returns:
            Complete trajectory through all waypoints
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
            
        if len(durations) != len(waypoints) - 1:
            raise ValueError("Number of durations must be len(waypoints) - 1")
        
        complete_trajectory = []
        
        for i in range(len(waypoints) - 1):
            segment = self.generate(
                waypoints[i],
                waypoints[i + 1],
                durations[i],
                trajectory_type
            )
            
            # Adjust time stamps
            if complete_trajectory:
                time_offset = complete_trajectory[-1].time
                for point in segment[1:]:  # Skip first point (duplicate)
                    point.time += time_offset
                    complete_trajectory.append(point)
            else:
                complete_trajectory.extend(segment)
        
        return complete_trajectory
