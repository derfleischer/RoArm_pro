#!/usr/bin/env python3
"""
RoArm M3 Trajectory Generation
Erzeugt sanfte Bewegungsprofile für präzise Steuerung.
"""

import math
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryType(Enum):
    """Verfügbare Trajektorien-Typen."""
    LINEAR = "linear"
    TRAPEZOIDAL = "trapezoidal"
    S_CURVE = "s_curve"
    SINUSOIDAL = "sinusoidal"
    QUINTIC = "quintic"
    MINIMUM_JERK = "minimum_jerk"


@dataclass
class TrajectoryPoint:
    """Ein Punkt in der Trajektorie."""
    positions: Dict[str, float]
    velocities: Optional[Dict[str, float]] = None
    accelerations: Optional[Dict[str, float]] = None
    time: float = 0.0
    time_delta: float = 0.0  # Zeit seit letztem Punkt


class TrajectoryGenerator:
    """
    Trajektorien-Generator für sanfte Bewegungen.
    FIXED VERSION - funktioniert mit der Controller-Architektur.
    """
    
    def __init__(self, max_velocity: float = 2.0, max_acceleration: float = 3.0, max_jerk: float = 5.0):
        """
        Initialisiert den Trajektorien-Generator.
        
        Args:
            max_velocity: Maximale Geschwindigkeit (rad/s)
            max_acceleration: Maximale Beschleunigung (rad/s²)
            max_jerk: Maximaler Ruck (rad/s³)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        
        logger.debug(f"TrajectoryGenerator initialized: v_max={max_velocity}, a_max={max_acceleration}")
    
    def generate(self, start: Dict[str, float], end: Dict[str, float],
                speed: float = 1.0, trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                num_points: int = 20) -> List[TrajectoryPoint]:
        """
        Generiert Trajektorie zwischen Start- und Endposition.
        
        Args:
            start: Start-Positionen
            end: End-Positionen
            speed: Geschwindigkeitsfaktor (0.1 bis 2.0)
            trajectory_type: Art der Trajektorie
            num_points: Anzahl der Zwischenpunkte
            
        Returns:
            Liste von TrajectoryPoints
        """
        # Sicherheitsprüfungen
        if not start or not end:
            logger.error("Start or end positions empty")
            return []
        
        # Berechne maximale Bewegung
        max_delta = 0.0
        active_joints = {}
        
        for joint in start:
            if joint in end:
                delta = abs(end[joint] - start[joint])
                if delta > 0.001:  # Nur Joints mit merklicher Bewegung
                    active_joints[joint] = delta
                    max_delta = max(max_delta, delta)
        
        if max_delta < 0.001:
            logger.debug("No significant movement needed")
            return [TrajectoryPoint(positions=end.copy(), time=0.0, time_delta=0.0)]
        
        # Berechne Bewegungsdauer basierend auf Geschwindigkeit
        duration = max_delta / (self.max_velocity * speed)
        duration = max(0.5, min(10.0, duration))  # Zwischen 0.5 und 10 Sekunden
        
        logger.debug(f"Generating {trajectory_type.value} trajectory: {num_points} points, {duration:.2f}s")
        
        # Wähle Trajektorien-Methode
        if trajectory_type == TrajectoryType.LINEAR:
            return self._generate_linear(start, end, duration, num_points)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            return self._generate_trapezoidal(start, end, duration, num_points)
        elif trajectory_type == TrajectoryType.S_CURVE:
            return self._generate_s_curve(start, end, duration, num_points)
        elif trajectory_type == TrajectoryType.SINUSOIDAL:
            return self._generate_sinusoidal(start, end, duration, num_points)
        elif trajectory_type == TrajectoryType.QUINTIC:
            return self._generate_quintic(start, end, duration, num_points)
        elif trajectory_type == TrajectoryType.MINIMUM_JERK:
            return self._generate_minimum_jerk(start, end, duration, num_points)
        else:
            logger.warning(f"Unknown trajectory type: {trajectory_type}, using linear")
            return self._generate_linear(start, end, duration, num_points)
    
    def _generate_linear(self, start: Dict[str, float], end: Dict[str, float],
                        duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert lineare Trajektorie."""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            time_point = t * duration
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + t * (end[joint] - start[joint])
                else:
                    positions[joint] = start[joint]
            
            # Berechne time_delta
            time_delta = time_point - (points[-1].time if points else 0)
            
            points.append(TrajectoryPoint(
                positions=positions,
                time=time_point,
                time_delta=time_delta
            ))
        
        return points
    
    def _generate_s_curve(self, start: Dict[str, float], end: Dict[str, float],
                         duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert S-Kurven Trajektorie für sanfte Beschleunigung."""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # S-Kurve mit Smoothstep Funktion
            s_t = 3 * t**2 - 2 * t**3
            
            time_point = t * duration
            
            positions = {}
            velocities = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s_t * delta
                    
                    # Geschwindigkeit (Ableitung der S-Kurve)
                    if duration > 0:
                        ds_dt = (6 * t - 6 * t**2) / duration
                        velocities[joint] = ds_dt * delta
                    else:
                        velocities[joint] = 0.0
                else:
                    positions[joint] = start[joint]
                    velocities[joint] = 0.0
            
            time_delta = time_point - (points[-1].time if points else 0)
            
            points.append(TrajectoryPoint(
                positions=positions,
                velocities=velocities,
                time=time_point,
                time_delta=time_delta
            ))
        
        return points
    
    def _generate_trapezoidal(self, start: Dict[str, float], end: Dict[str, float],
                             duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert trapezförmige Geschwindigkeitstrajektorie."""
        points = []
        
        # Trapez-Phasen: 1/3 Beschleunigung, 1/3 konstant, 1/3 Verzögerung
        t_acc = duration / 3
        t_const = duration / 3
        t_dec = duration / 3
        
        for i in range(num_points):
            t = i / (num_points - 1) * duration if num_points > 1 else 0
            
            # Berechne Position basierend auf Phase
            if t <= t_acc:
                # Beschleunigungsphase
                s_t = 0.5 * (t / t_acc)**2
            elif t <= t_acc + t_const:
                # Konstante Geschwindigkeit
                s_t = 0.5 + (t - t_acc) / (2 * t_const)
            else:
                # Verzögerungsphase
                t_rel = (t - t_acc - t_const) / t_dec
                s_t = 0.5 + 0.5 - 0.5 * (1 - t_rel)**2
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + s_t * (end[joint] - start[joint])
                else:
                    positions[joint] = start[joint]
            
            time_delta = t - (points[-1].time if points else 0)
            
            points.append(TrajectoryPoint(
                positions=positions,
                time=t,
                time_delta=time_delta
            ))
        
        return points
    
    def _generate_sinusoidal(self, start: Dict[str, float], end: Dict[str, float],
                            duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert sinusförmige Trajektorie."""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # Sinusförmiger Verlauf
            s_t = 0.5 * (1 - math.cos(math.pi * t))
            
            time_point = t * duration
            
            positions = {}
            velocities = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s_t * delta
                    
                    # Geschwindigkeit
                    if duration > 0:
                        ds_dt = (math.pi * math.sin(math.pi * t)) / (2 * duration)
                        velocities[joint] = ds_dt * delta
                    else:
                        velocities[joint] = 0.0
                else:
                    positions[joint] = start[joint]
                    velocities[joint] = 0.0
            
            time_delta = time_point - (points[-1].time if points else 0)
            
            points.append(TrajectoryPoint(
                positions=positions,
                velocities=velocities,
                time=time_point,
                time_delta=time_delta
            ))
        
        return points
    
    def _generate_quintic(self, start: Dict[str, float], end: Dict[str, float],
                         duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert Polynom 5. Ordnung Trajektorie."""
        return self._generate_minimum_jerk(start, end, duration, num_points)
    
    def _generate_minimum_jerk(self, start: Dict[str, float], end: Dict[str, float],
                              duration: float, num_points: int) -> List[TrajectoryPoint]:
        """Generiert Minimum-Jerk Trajektorie (optimal für schwere Lasten)."""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # Minimum-Jerk Polynom (5. Ordnung)
            s_t = 10 * t**3 - 15 * t**4 + 6 * t**5
            
            time_point = t * duration
            
            positions = {}
            velocities = {}
            accelerations = {}
            
            for joint in start:
                if joint in end:
                    delta = end[joint] - start[joint]
                    positions[joint] = start[joint] + s_t * delta
                    
                    # Ableitungen
                    if duration > 0:
                        ds_dt = (30 * t**2 - 60 * t**3 + 30 * t**4) / duration
                        d2s_dt2 = (60 * t - 180 * t**2 + 120 * t**3) / (duration**2)
                        
                        velocities[joint] = ds_dt * delta
                        accelerations[joint] = d2s_dt2 * delta
                    else:
                        velocities[joint] = 0.0
                        accelerations[joint] = 0.0
                else:
                    positions[joint] = start[joint]
                    velocities[joint] = 0.0
                    accelerations[joint] = 0.0
            
            time_delta = time_point - (points[-1].time if points else 0)
            
            points.append(TrajectoryPoint(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                time=time_point,
                time_delta=time_delta
            ))
        
        return points
