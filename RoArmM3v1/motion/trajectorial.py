# ============================================
# motion/trajectory.py
# ============================================
#!/usr/bin/env python3
"""
Trajectory Generator für RoArm M3
Verschiedene Bewegungsprofile für smooth motion.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import math


class TrajectoryType(Enum):
    """Verfügbare Trajectory-Typen."""
    LINEAR = "linear"
    TRAPEZOIDAL = "trapezoidal"
    S_CURVE = "s_curve"
    SINUSOIDAL = "sinusoidal"
    QUINTIC = "quintic"


@dataclass
class TrajectoryPoint:
    """Ein Punkt in der Trajectory."""
    positions: Dict[str, float]
    velocities: Dict[str, float] = None
    time_delta: float = 0.0


class TrajectoryGenerator:
    """Generiert smooth trajectories zwischen Positionen."""
    
    def generate(self, start: Dict[str, float], 
                end: Dict[str, float],
                speed: float = 1.0,
                trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                num_points: int = 50) -> List[TrajectoryPoint]:
        """
        Generiert Trajectory-Punkte.
        
        Args:
            start: Start-Positionen
            end: Ziel-Positionen
            speed: Geschwindigkeitsfaktor
            trajectory_type: Bewegungsprofil
            num_points: Anzahl Zwischenpunkte
            
        Returns:
            Liste von Trajectory-Punkten
        """
        points = []
        
        # Berechne Gesamtzeit basierend auf größter Bewegung
        max_delta = 0
        for joint in start:
            if joint in end:
                delta = abs(end[joint] - start[joint])
                max_delta = max(max_delta, delta)
        
        # Basis-Zeit (kann durch speed angepasst werden)
        total_time = max_delta / speed
        
        if total_time == 0:
            # Keine Bewegung nötig
            return [TrajectoryPoint(positions=end.copy())]
        
        # Generiere Zeitpunkte
        times = np.linspace(0, total_time, num_points)
        
        # Generiere Punkte basierend auf Profil
        if trajectory_type == TrajectoryType.LINEAR:
            points = self._linear_trajectory(start, end, times)
        elif trajectory_type == TrajectoryType.S_CURVE:
            points = self._s_curve_trajectory(start, end, times)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            points = self._trapezoidal_trajectory(start, end, times)
        elif trajectory_type == TrajectoryType.SINUSOIDAL:
            points = self._sinusoidal_trajectory(start, end, times)
        else:
            points = self._linear_trajectory(start, end, times)
        
        return points
    
    def _linear_trajectory(self, start: Dict[str, float], 
                          end: Dict[str, float], 
                          times: np.ndarray) -> List[TrajectoryPoint]:
        """Lineare Interpolation."""
        points = []
        
        for i, t in enumerate(times):
            s = t / times[-1]  # Normalized time (0 to 1)
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s * (end[joint] - start[joint])
            
            time_delta = times[i] - times[i-1] if i > 0 else 0
            points.append(TrajectoryPoint(positions=positions, time_delta=time_delta))
        
        return points
    
    def _s_curve_trajectory(self, start: Dict[str, float], 
                           end: Dict[str, float], 
                           times: np.ndarray) -> List[TrajectoryPoint]:
        """S-Kurven Profil (smooth acceleration)."""
        points = []
        
        for i, t in enumerate(times):
            # S-curve formula (sigmoid-like)
            s_norm = t / times[-1]
            s = s_norm**2 * (3 - 2 * s_norm)  # Smooth step
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s * (end[joint] - start[joint])
            
            time_delta = times[i] - times[i-1] if i > 0 else 0
            points.append(TrajectoryPoint(positions=positions, time_delta=time_delta))
        
        return points
    
    def _trapezoidal_trajectory(self, start: Dict[str, float], 
                               end: Dict[str, float], 
                               times: np.ndarray) -> List[TrajectoryPoint]:
        """Trapez-Profil (constant velocity with ramps)."""
        points = []
        
        # 20% acceleration, 60% constant, 20% deceleration
        accel_time = 0.2
        
        for i, t in enumerate(times):
            s_norm = t / times[-1]
            
            if s_norm < accel_time:
                # Acceleration phase
                s = 0.5 * (s_norm / accel_time) ** 2 * accel_time
            elif s_norm < 1 - accel_time:
                # Constant velocity phase
                s = 0.5 * accel_time + (s_norm - accel_time)
            else:
                # Deceleration phase
                decel_s = (s_norm - (1 - accel_time)) / accel_time
                s = 1 - 0.5 * accel_time * (1 - decel_s) ** 2
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s * (end[joint] - start[joint])
            
            time_delta = times[i] - times[i-1] if i > 0 else 0
            points.append(TrajectoryPoint(positions=positions, time_delta=time_delta))
        
        return points
    
    def _sinusoidal_trajectory(self, start: Dict[str, float], 
                              end: Dict[str, float], 
                              times: np.ndarray) -> List[TrajectoryPoint]:
        """Sinusförmiges Profil."""
        points = []
        
        for i, t in enumerate(times):
            s_norm = t / times[-1]
            s = 0.5 * (1 - math.cos(math.pi * s_norm))
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s * (end[joint] - start[joint])
            
            time_delta = times[i] - times[i-1] if i > 0 else 0
            points.append(TrajectoryPoint(positions=positions, time_delta=time_delta))
        
        return points
