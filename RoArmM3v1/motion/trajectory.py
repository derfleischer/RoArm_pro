#!/usr/bin/env python3
"""
RoArm M3 Trajectory Generation
Verschiedene Bewegungsprofile für sanfte und präzise Bewegungen.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

from utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryType(Enum):
    """Verfügbare Trajectory-Typen."""
    LINEAR = "linear"              # Konstante Geschwindigkeit
    TRAPEZOIDAL = "trapezoidal"    # Trapez-Profil
    S_CURVE = "s_curve"            # S-Kurve (sanft)
    SINUSOIDAL = "sinusoidal"      # Sinusförmig
    MINIMUM_JERK = "minimum_jerk"  # Minimum Jerk (5. Ordnung)
    CUBIC = "cubic"                # Kubische Interpolation
    QUINTIC = "quintic"            # Quintische Interpolation


@dataclass
class TrajectoryPoint:
    """Ein Punkt auf der Trajectory."""
    time: float                    # Zeit seit Start (s)
    positions: Dict[str, float]    # Joint-Positionen (rad)
    velocities: Dict[str, float]   # Geschwindigkeiten (rad/s)
    accelerations: Dict[str, float]  # Beschleunigungen (rad/s²)
    time_delta: float = 0.01       # Zeit bis zum nächsten Punkt


class TrajectoryGenerator:
    """
    Generiert Bewegungstrajektorien für verschiedene Profile.
    """
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialisiert den Trajectory Generator.
        
        Args:
            sample_rate: Abtastrate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        logger.debug(f"Trajectory Generator initialized (rate: {sample_rate} Hz)")
    
    def generate(self, start: Dict[str, float], 
                end: Dict[str, float],
                speed: float = 1.0,
                trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                acceleration: float = 2.0,
                jerk: float = 5.0) -> List[TrajectoryPoint]:
        """
        Generiert eine Trajectory von Start zu End.
        
        Args:
            start: Start-Positionen
            end: End-Positionen
            speed: Geschwindigkeitsfaktor (0.1-2.0)
            trajectory_type: Bewegungsprofil
            acceleration: Max Beschleunigung (rad/s²)
            jerk: Max Jerk (rad/s³)
            
        Returns:
            Liste von Trajectory-Punkten
        """
        # Berechne maximale Bewegung
        max_delta = 0.0
        deltas = {}
        
        for joint in start:
            if joint in end:
                delta = abs(end[joint] - start[joint])
                deltas[joint] = end[joint] - start[joint]
                max_delta = max(max_delta, delta)
        
        if max_delta < 0.001:  # Keine Bewegung nötig
            return [TrajectoryPoint(
                time=0.0,
                positions=end.copy(),
                velocities={j: 0.0 for j in end},
                accelerations={j: 0.0 for j in end}
            )]
        
        # Berechne Dauer basierend auf max_delta und speed
        duration = max_delta / (speed * 1.0)  # 1.0 rad/s Basis-Geschwindigkeit
        duration = max(0.5, duration)  # Minimum 0.5s
        
        # Generiere Profil
        if trajectory_type == TrajectoryType.LINEAR:
            return self._generate_linear(start, end, duration)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            return self._generate_trapezoidal(start, end, duration, acceleration)
        elif trajectory_type == TrajectoryType.S_CURVE:
            return self._generate_s_curve(start, end, duration, acceleration, jerk)
        elif trajectory_type == TrajectoryType.SINUSOIDAL:
            return self._generate_sinusoidal(start, end, duration)
        elif trajectory_type == TrajectoryType.MINIMUM_JERK:
            return self._generate_minimum_jerk(start, end, duration)
        elif trajectory_type == TrajectoryType.CUBIC:
            return self._generate_cubic(start, end, duration)
        elif trajectory_type == TrajectoryType.QUINTIC:
            return self._generate_quintic(start, end, duration)
        else:
            # Fallback zu S-Curve
            return self._generate_s_curve(start, end, duration, acceleration, jerk)
    
    def _generate_linear(self, start: Dict[str, float], 
                        end: Dict[str, float],
                        duration: float) -> List[TrajectoryPoint]:
        """
        Generiert lineare Trajectory (konstante Geschwindigkeit).
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # Lineare Interpolation
            s = t / duration
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s * (end[joint] - start[joint])
                    velocities[joint] = (end[joint] - start[joint]) / duration
                    accelerations[joint] = 0.0
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_trapezoidal(self, start: Dict[str, float],
                             end: Dict[str, float],
                             duration: float,
                             acceleration: float) -> List[TrajectoryPoint]:
        """
        Generiert trapezförmige Trajectory.
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        # Berechne Rampen-Zeit
        t_ramp = min(duration / 3, 1.0 / acceleration)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # Trapez-Profil
            if t < t_ramp:
                # Beschleunigungsphase
                s = 0.5 * acceleration * t * t / (duration * acceleration * t_ramp)
            elif t < duration - t_ramp:
                # Konstante Geschwindigkeit
                s = (t - 0.5 * t_ramp) / (duration - t_ramp)
            else:
                # Bremsphase
                t_brake = duration - t
                s = 1.0 - 0.5 * acceleration * t_brake * t_brake / (duration * acceleration * t_ramp)
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s * delta
                    
                    # Geschwindigkeit
                    if t < t_ramp:
                        velocities[joint] = delta * t / (t_ramp * duration)
                    elif t < duration - t_ramp:
                        velocities[joint] = delta / (duration - t_ramp)
                    else:
                        velocities[joint] = delta * (duration - t) / (t_ramp * duration)
                    
                    # Beschleunigung
                    if t < t_ramp:
                        accelerations[joint] = delta / (t_ramp * t_ramp)
                    elif t < duration - t_ramp:
                        accelerations[joint] = 0.0
                    else:
                        accelerations[joint] = -delta / (t_ramp * t_ramp)
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_s_curve(self, start: Dict[str, float],
                         end: Dict[str, float],
                         duration: float,
                         acceleration: float,
                         jerk: float) -> List[TrajectoryPoint]:
        """
        Generiert S-Kurven Trajectory (sanfte Übergänge).
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # S-Kurve mit sigmoid function
            tau = t / duration
            s = self._sigmoid(tau)
            s_dot = self._sigmoid_derivative(tau) / duration
            s_ddot = self._sigmoid_second_derivative(tau) / (duration * duration)
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s * delta
                    velocities[joint] = s_dot * delta
                    accelerations[joint] = s_ddot * delta
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_sinusoidal(self, start: Dict[str, float],
                            end: Dict[str, float],
                            duration: float) -> List[TrajectoryPoint]:
        """
        Generiert sinusförmige Trajectory.
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # Sinusförmiges Profil
            tau = t / duration
            s = 0.5 * (1 - math.cos(math.pi * tau))
            s_dot = 0.5 * math.pi * math.sin(math.pi * tau) / duration
            s_ddot = 0.5 * math.pi * math.pi * math.cos(math.pi * tau) / (duration * duration)
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s * delta
                    velocities[joint] = s_dot * delta
                    accelerations[joint] = s_ddot * delta
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_minimum_jerk(self, start: Dict[str, float],
                               end: Dict[str, float],
                               duration: float) -> List[TrajectoryPoint]:
        """
        Generiert Minimum-Jerk Trajectory (5. Ordnung Polynom).
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # Minimum jerk polynomial (5th order)
            tau = t / duration
            tau3 = tau * tau * tau
            tau4 = tau3 * tau
            tau5 = tau4 * tau
            
            s = 10 * tau3 - 15 * tau4 + 6 * tau5
            s_dot = (30 * tau * tau - 60 * tau3 + 30 * tau4) / duration
            s_ddot = (60 * tau - 180 * tau * tau + 120 * tau3) / (duration * duration)
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s * delta
                    velocities[joint] = s_dot * delta
                    accelerations[joint] = s_ddot * delta
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_cubic(self, start: Dict[str, float],
                       end: Dict[str, float],
                       duration: float) -> List[TrajectoryPoint]:
        """
        Generiert kubische Trajectory (3. Ordnung Polynom).
        """
        points = []
        num_points = int(duration * self.sample_rate)
        
        for i in range(num_points + 1):
            t = i * self.dt
            if t > duration:
                t = duration
            
            # Cubic polynomial
            tau = t / duration
            tau2 = tau * tau
            tau3 = tau2 * tau
            
            s = 3 * tau2 - 2 * tau3
            s_dot = (6 * tau - 6 * tau2) / duration
            s_ddot = (6 - 12 * tau) / (duration * duration)
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s * delta
                    velocities[joint] = s_dot * delta
                    accelerations[joint] = s_ddot * delta
            
            points.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time_delta=self.dt
            ))
        
        return points
    
    def _generate_quintic(self, start: Dict[str, float],
                         end: Dict[str, float],
                         duration: float) -> List[TrajectoryPoint]:
        """
        Generiert quintische Trajectory (5. Ordnung Polynom).
        """
        return self._generate_minimum_jerk(start, end, duration)
    
    # ============== HELPER FUNCTIONS ==============
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid-Funktion für S-Kurve."""
        # Scaled sigmoid für sanften Übergang
        k = 10.0  # Steilheit
        return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
    
    def _sigmoid_derivative(self, x: float) -> float:
        """Erste Ableitung der Sigmoid-Funktion."""
        k = 10.0
        sig = self._sigmoid(x)
        return k * sig * (1 - sig)
    
    def _sigmoid_second_derivative(self, x: float) -> float:
        """Zweite Ableitung der Sigmoid-Funktion."""
        k = 10.0
        sig = self._sigmoid(x)
        return k * k * sig * (1 - sig) * (1 - 2 * sig)
