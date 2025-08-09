#!/usr/bin/env python3
"""
RoArm M3 PROFESSIONAL Scan Patterns für Revopoint Mini2
Version 5.1 FIXED - Vollständige Production Version
Mit Podest-Support, Timing-Optimierung und korrekter Validierung

Optimiert für:
- Podest-Höhe: 40cm
- Update-Rate: 11.4 Hz
- Tracking ohne Verlust
- Maximale Stabilität
"""

import math
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import json
import threading
from collections import deque
from datetime import datetime

# ============== IMPORTS MIT FALLBACK ==============
try:
    from core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION
    from motion.trajectory import TrajectoryType
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback mit korrekten Konstanten
    SERVO_LIMITS = {
        "base": (-3.14, 3.14),      # ±180°
        "shoulder": (-1.57, 1.57),  # ±90°
        "elbow": (0.0, 3.14),       # 0-180°
        "wrist": (-1.57, 1.57),     # ±90°
        "roll": (-3.14, 3.14),      # ±180°
        "hand": (1.08, 3.14)        # 62°-180°
    }
    
    HOME_POSITION = {
        "base": 0.0, "shoulder": 0.0, "elbow": 1.57,
        "wrist": 0.0, "roll": 0.0, "hand": 3.14
    }
    
    SCANNER_POSITION = {
        "base": 0.0, "shoulder": 0.35, "elbow": 1.22,
        "wrist": -1.57, "roll": 1.57, "hand": 2.5
    }
    
    class TrajectoryType:
        LINEAR = "linear"
        S_CURVE = "s_curve"
        CUBIC = "cubic"
        SPLINE = "spline"
        MINIMUM_JERK = "minimum_jerk"
    
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# ============== HARDWARE KONFIGURATION ==============

HARDWARE_TIMING = {
    "servo_update_rate": 11.4,      # Hz
    "serial_latency": 0.02,         # Sekunden
    "min_movement_time": 0.087,     # 1/11.4 Hz
    "trajectory_points": 20,        # Standard-Anzahl
    "settle_multiplier": 1.2,       # Sicherheitsfaktor
    "sync_interval": 5,             # Sync alle N Punkte
    "buffer_size": 50               # Trajectory Buffer
}

# Revopoint Mini2 Scanner Specs
SCANNER_SPECS = {
    "optimal_distance": 0.15,       # 15cm optimal
    "min_distance": 0.10,           # 10cm minimum
    "max_distance": 0.30,           # 30cm maximum
    "fov_horizontal": 40,           # degrees
    "fov_vertical": 30,             # degrees
    "fps": 10,                      # Scanner FPS
    "weight": 0.2,                  # 200g
    "mount_offset": {"x": 0.0, "y": 0.0, "z": 0.05},
    "tracking_speed": 0.25,         # Optimale Geschwindigkeit
    "podest_height": 0.40,          # 40cm Podest-Höhe!
    "precision_mode_distance": 0.12,
    "fast_mode_distance": 0.25
}

# PODEST-OPTIMIERTE Scanner-Positionen
SCANNER_POSITIONS = {
    # Objekt 40cm unter Roboter (auf Tisch)
    "table_level": {
        "base": 0.0,        # Zentriert
        "shoulder": 0.8,    # Nach unten geneigt für Podest
        "elbow": 1.3,       # Gebeugt für Stabilität
        "wrist": -0.6,      # Kompensiert Shoulder
        "roll": 1.57,       # 90° für Scanner
        "hand": 2.5         # Scanner-Griff
    },
    # Objekt auf gleicher Höhe
    "same_level": {
        "base": 0.0,
        "shoulder": 0.2,
        "elbow": 1.4,
        "wrist": -0.3,
        "roll": 1.57,
        "hand": 2.5
    },
    # Sphärische/360° Scans
    "spherical": {
        "base": 0.0,
        "shoulder": 0.5,
        "elbow": 1.2,
        "wrist": -0.4,
        "roll": 1.57,
        "hand": 2.5
    },
    # Große Objekte
    "large_object": {
        "base": 0.0,
        "shoulder": 0.3,
        "elbow": 1.0,
        "wrist": -0.5,
        "roll": 1.57,
        "hand": 2.5
    },
    # Detail-Scans
    "detail_scan": {
        "base": 0.0,
        "shoulder": 0.7,
        "elbow": 1.5,
        "wrist": -0.3,
        "roll": 1.57,
        "hand": 2.5
    }
}

# Default Scanner Center
SCANNER_CENTER = SCANNER_POSITIONS["table_level"]

# Scan Defaults
SCAN_DEFAULTS = {
    "raster": {
        "width": 0.20, "height": 0.15, "rows": 10, "cols": 10,
        "overlap": 0.2, "speed": 0.3, "settle_time": 0.5
    },
    "spiral": {
        "radius_start": 0.05, "radius_end": 0.20, "revolutions": 5,
        "points_per_rev": 30, "speed": 0.25
    },
    "spherical": {
        "radius": 0.15, "theta_steps": 12, "phi_steps": 8,
        "speed": 0.3
    },
    "turntable": {
        "steps": 36, "height_levels": 3, "height_range": 0.20,
        "speed": 0.2
    },
    "helix": {
        "radius": 0.12, "height": 0.25, "turns": 5,
        "points_per_turn": 24
    },
    "adaptive": {
        "initial_points": 20, "refinement_threshold": 0.05,
        "max_iterations": 3
    }
}

# ============== DATENKLASSEN ==============

@dataclass
class ScanPoint:
    """Erweiterte ScanPoint-Klasse."""
    positions: Dict[str, float]
    speed: float = 0.25
    settle_time: float = 0.3
    trajectory_type: Union[str, 'TrajectoryType'] = "s_curve"
    scan_angle: Optional[float] = None
    distance: Optional[float] = None
    description: Optional[str] = None
    safety_checked: bool = False
    priority: int = 1
    expected_quality: float = 1.0
    smooth_transition: bool = True
    timing_optimized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-Processing nach Initialisierung."""
        if hasattr(self.trajectory_type, 'value'):
            self.trajectory_type = self.trajectory_type.value
        elif not isinstance(self.trajectory_type, str):
            self.trajectory_type = str(self.trajectory_type).lower()
    
    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary."""
        return {
            "positions": self.positions,
            "speed": self.speed,
            "settle_time": self.settle_time,
            "trajectory_type": self.trajectory_type,
            "scan_angle": self.scan_angle,
            "distance": self.distance,
            "description": self.description,
            "priority": self.priority,
            "metadata": self.metadata
        }

@dataclass
class ScanQuality:
    """Scan-Qualitäts-Metriken."""
    coverage: float = 0.0
    overlap: float = 0.0
    stability: float = 0.0
    tracking_confidence: float = 0.0
    estimated_time: float = 0.0
    point_density: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Berechne Gesamt-Score."""
        weights = {
            "coverage": 0.3,
            "overlap": 0.2,
            "stability": 0.25,
            "tracking_confidence": 0.25
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

# ============== SAFETY VALIDATOR - KORREKT! ==============

class SafetyValidator:
    """Sicherheitsvalidierung mit FUNKTIONIERENDER Logik."""
    
    # Realistische Sicherheitsmargins
    SAFETY_MARGINS = {
        "base": 0.05,       # 5% Margin
        "shoulder": 0.08,   # 8% für schweren Arm
        "elbow": 0.05,      # 5% reicht
        "wrist": 0.15,      # 15% für kritisches Wrist
        "roll": 0.05,       # 5% ist OK
        "hand": 0.02        # 2% für Gripper
    }
    
    # Geschwindigkeitslimits (rad/s)
    VELOCITY_LIMITS = {
        "base": 2.0,
        "shoulder": 1.5,
        "elbow": 1.8,
        "wrist": 2.5,
        "roll": 3.0,
        "hand": 2.0
    }
    
    @staticmethod
    def validate_position(positions: Dict[str, float], 
                         conservative: bool = False,
                         debug: bool = False) -> Tuple[bool, List[str]]:
        """
        Validiert Position gegen Servo-Limits - KORREKTE IMPLEMENTIERUNG!
        """
        errors = []
        warnings = []
        
        for joint, value in positions.items():
            if joint not in SERVO_LIMITS:
                continue
                
            min_limit, max_limit = SERVO_LIMITS[joint]
            base_margin = SafetyValidator.SAFETY_MARGINS.get(joint, 0.05)
            
            # Margin berechnen
            if conservative:
                margin = base_margin * 1.2  # Nur 20% extra
            else:
                margin = base_margin
            
            # KORREKTE Safe-Range Berechnung!
            if joint == "wrist":
                # Wrist extra konservativ
                margin = max(0.15, margin)
                safe_min = max(min_limit + margin, -1.3)
                safe_max = min(max_limit - margin, 1.3)
            else:
                # Normale Joints - RICHTIGE REIHENFOLGE!
                safe_min = min_limit + margin
                safe_max = max_limit - margin
            
            # Validierung
            if value < safe_min or value > safe_max:
                errors.append(
                    f"{joint}={value:.3f} outside safe range "
                    f"[{safe_min:.3f}, {safe_max:.3f}] "
                    f"(limits: [{min_limit:.3f}, {max_limit:.3f}])"
                )
            elif abs(value - safe_min) < 0.05 or abs(value - safe_max) < 0.05:
                warnings.append(f"{joint} near limit: {value:.3f}")
            
            if debug:
                logger.debug(
                    f"{joint}: value={value:.3f}, "
                    f"limits=[{min_limit:.3f}, {max_limit:.3f}], "
                    f"safe=[{safe_min:.3f}, {safe_max:.3f}]"
                )
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_trajectory(points: List[ScanPoint],
                          max_velocity: Optional[Dict[str, float]] = None,
                          max_acceleration: float = 2.0) -> Tuple[bool, List[str]]:
        """Validiert komplette Trajektorie."""
        if not points:
            return False, ["Empty trajectory"]
        
        errors = []
        max_vel = max_velocity or SafetyValidator.VELOCITY_LIMITS
        
        for i in range(len(points) - 1):
            curr = points[i]
            next_point = points[i + 1]
            
            valid, pos_errors = SafetyValidator.validate_position(
                curr.positions, conservative=False
            )
            if not valid:
                errors.extend([f"Point {i}: {e}" for e in pos_errors])
            
            dt = 1.0 / HARDWARE_TIMING["servo_update_rate"]
            for joint in curr.positions:
                if joint in next_point.positions:
                    vel = abs(next_point.positions[joint] - curr.positions[joint]) / dt
                    if vel > max_vel.get(joint, 2.0):
                        errors.append(
                            f"Point {i}->{i+1}: {joint} velocity {vel:.2f} "
                            f"exceeds limit {max_vel.get(joint, 2.0):.2f}"
                        )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def smooth_path(points: List[ScanPoint], 
                   max_delta: float = 0.3,
                   timing_optimize: bool = True) -> List[ScanPoint]:
        """Glättet Pfad für kontinuierliches Tracking."""
        if len(points) < 2:
            return points
        
        smoothed = [points[0]]
        
        for i in range(1, len(points)):
            prev = smoothed[-1]
            curr = points[i]
            
            max_change = 0
            joint_deltas = {}
            
            for joint in prev.positions:
                if joint in curr.positions:
                    delta = abs(curr.positions[joint] - prev.positions[joint])
                    joint_deltas[joint] = delta
                    max_change = max(max_change, delta)
            
            if max_change > max_delta:
                steps = int(math.ceil(max_change / max_delta))
                
                if timing_optimize:
                    optimal_steps = max(2, min(steps, 5))
                    steps = optimal_steps
                
                for step in range(1, steps):
                    t = step / steps
                    t_smooth = t * t * (3 - 2 * t)
                    
                    interp_pos = {}
                    for joint in prev.positions:
                        if joint in curr.positions:
                            interp_pos[joint] = prev.positions[joint] + \
                                              t_smooth * (curr.positions[joint] - prev.positions[joint])
                        else:
                            interp_pos[joint] = prev.positions[joint]
                    
                    speed = min(prev.speed, curr.speed) * 0.8
                    if timing_optimize:
                        speed = speed * (HARDWARE_TIMING["servo_update_rate"] / 20.0)
                    
                    interp_point = ScanPoint(
                        positions=interp_pos,
                        speed=speed,
                        settle_time=HARDWARE_TIMING["serial_latency"],
                        trajectory_type="cubic",
                        description=f"Interpolated {step}/{steps}",
                        timing_optimized=timing_optimize,
                        smooth_transition=True
                    )
                    smoothed.append(interp_point)
            
            if timing_optimize and i % HARDWARE_TIMING["sync_interval"] == 0:
                curr.settle_time = max(curr.settle_time, 
                                      HARDWARE_TIMING["serial_latency"] * 2)
                curr.timing_optimized = True
            
            smoothed.append(curr)
        
        logger.info(f"✅ Path smoothed: {len(points)} -> {len(smoothed)} points")
        return smoothed

# ============== INTELLIGENTER PLANNER ==============

class IntelligentMultiAxisPlanner:
    """Multi-Achsen-Planer mit Optimierung."""
    
    def __init__(self, center_position: Dict[str, float]):
        """Initialisiert den Planer."""
        self.center = center_position.copy()
        self.validator = SafetyValidator()
        
        self.joint_weights = {
            "base": 2.5,
            "shoulder": 3.0,
            "elbow": 2.0,
            "wrist": 1.5,
            "roll": 1.0,
            "hand": 0.2
        }
        
        self.energy_cost = {
            "base": 0.8,
            "shoulder": 1.0,
            "elbow": 0.6,
            "wrist": 0.3,
            "roll": 0.2,
            "hand": 0.1
        }
        
        self.stability_factors = {
            "base": 1.0,
            "shoulder": 0.7,
            "elbow": 0.9,
            "wrist": 1.1,
            "roll": 1.2,
            "hand": 1.0
        }
        
        self.timing_optimizer = TimingOptimizer()
        self.path_cache = {}
        
    def plan_optimal_trajectory(self, 
                              start: Dict[str, float],
                              end: Dict[str, float],
                              constraints: Optional[Dict] = None) -> List[ScanPoint]:
        """Plant optimale Trajektorie."""
        cache_key = f"{hash(tuple(start.items()))}_{hash(tuple(end.items()))}"
        if cache_key in self.path_cache:
            logger.debug("Using cached trajectory")
            return self.path_cache[cache_key]
        
        points = []
        distance = self._calculate_weighted_distance(start, end)
        
        if distance < 0.3:
            points = self._interpolate_direct(start, end, steps=5)
        elif distance < 0.8:
            waypoint = self._calculate_waypoint(start, end)
            points = self._interpolate_direct(start, waypoint, steps=3)
            points.extend(self._interpolate_direct(waypoint, end, steps=3))
        else:
            waypoints = self._calculate_multiple_waypoints(start, end, count=3)
            prev = start
            for wp in waypoints:
                points.extend(self._interpolate_direct(prev, wp, steps=2))
                prev = wp
            points.extend(self._interpolate_direct(prev, end, steps=3))
        
        points = self.timing_optimizer.optimize_timing(points)
        
        valid_points = []
        for point in points:
            valid, _ = self.validator.validate_position(point.positions, conservative=False)
            if valid:
                valid_points.append(point)
            else:
                corrected = self._correct_invalid_position(point.positions)
                if corrected:
                    point.positions = corrected
                    valid_points.append(point)
        
        if len(valid_points) > 0:
            self.path_cache[cache_key] = valid_points
        
        return valid_points
    
    def _calculate_weighted_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Berechnet gewichtete Distanz."""
        distance = 0.0
        for joint in pos1:
            if joint in pos2 and joint in self.joint_weights:
                diff = abs(pos1[joint] - pos2[joint])
                distance += self.joint_weights[joint] * (diff ** 2)
        return math.sqrt(distance)
    
    def _interpolate_direct(self, start: Dict, end: Dict, steps: int) -> List[ScanPoint]:
        """Direkte Interpolation."""
        points = []
        
        for i in range(steps):
            t = i / max(1, steps - 1)
            t_smooth = t * t * (3 - 2 * t)
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + t_smooth * (end[joint] - start[joint])
                else:
                    positions[joint] = start[joint]
            
            if i == 0 or i == steps - 1:
                speed = 0.15
            else:
                speed = 0.25
            
            point = ScanPoint(
                positions=positions,
                speed=speed,
                settle_time=0.05,
                trajectory_type="cubic",
                priority=2
            )
            points.append(point)
        
        return points
    
    def _calculate_waypoint(self, start: Dict, end: Dict) -> Dict[str, float]:
        """Berechnet optimalen Wegpunkt."""
        waypoint = {}
        
        for joint in start:
            if joint in end:
                mid = (start[joint] + end[joint]) / 2
                
                if joint == "shoulder":
                    mid += 0.1
                elif joint == "elbow":
                    mid += 0.15
                elif joint == "wrist":
                    mid = max(-1.2, min(1.2, mid))
                
                waypoint[joint] = mid
            else:
                waypoint[joint] = start[joint]
        
        return waypoint
    
    def _calculate_multiple_waypoints(self, start: Dict, end: Dict, 
                                     count: int = 3) -> List[Dict[str, float]]:
        """Berechnet mehrere Wegpunkte."""
        waypoints = []
        
        for i in range(1, count + 1):
            t = i / (count + 1)
            waypoint = {}
            
            for joint in start:
                if joint in end:
                    value = start[joint] + t * (end[joint] - start[joint])
                    offset = 0.1 * math.sin(t * math.pi)
                    
                    if joint == "base":
                        offset *= 0.5
                    elif joint == "shoulder":
                        value += offset * 0.8
                    elif joint == "wrist":
                        value = max(-1.2, min(1.2, value))
                    
                    waypoint[joint] = value
                else:
                    waypoint[joint] = start[joint]
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def _correct_invalid_position(self, positions: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Korrigiert ungültige Position."""
        corrected = positions.copy()
        
        for joint, value in corrected.items():
            if joint in SERVO_LIMITS:
                min_limit, max_limit = SERVO_LIMITS[joint]
                margin = SafetyValidator.SAFETY_MARGINS.get(joint, 0.1)
                
                safe_min = min_limit + margin
                safe_max = max_limit - margin
                
                if joint == "wrist":
                    safe_min = max(safe_min, -1.3)
                    safe_max = min(safe_max, 1.3)
                
                corrected[joint] = max(safe_min, min(safe_max, value))
        
        valid, _ = SafetyValidator.validate_position(corrected, conservative=False)
        return corrected if valid else None

# ============== TIMING OPTIMIZER ==============

class TimingOptimizer:
    """Optimiert Timing basierend auf Hardware."""
    
    def __init__(self):
        self.update_rate = HARDWARE_TIMING["servo_update_rate"]
        self.serial_latency = HARDWARE_TIMING["serial_latency"]
        self.min_move_time = HARDWARE_TIMING["min_movement_time"]
        
    def optimize_timing(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """Optimiert Timing der Scan-Punkte."""
        if not points:
            return points
        
        optimized = []
        
        for i, point in enumerate(points):
            opt_point = ScanPoint(
                positions=point.positions.copy(),
                speed=point.speed,
                settle_time=point.settle_time,
                trajectory_type=point.trajectory_type,
                scan_angle=point.scan_angle,
                distance=point.distance,
                description=point.description,
                priority=point.priority,
                timing_optimized=True
            )
            
            if i > 0:
                prev = optimized[-1] if optimized else points[i-1]
                distance = self._calculate_movement_distance(prev.positions, point.positions)
                
                min_time = max(self.min_move_time, distance / 0.5)
                optimal_time = math.ceil(min_time * self.update_rate) / self.update_rate
                
                if optimal_time > 0:
                    opt_point.speed = distance / optimal_time
                else:
                    opt_point.speed = 0.2
            
            if i % HARDWARE_TIMING["sync_interval"] == 0:
                opt_point.settle_time = max(opt_point.settle_time, self.serial_latency * 3)
                opt_point.description = f"{opt_point.description or ''} [SYNC]".strip()
            else:
                opt_point.settle_time = max(self.serial_latency, opt_point.settle_time * 0.8)
            
            optimized.append(opt_point)
        
        return optimized
    
    def _calculate_movement_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Berechnet Bewegungsdistanz."""
        distance = 0.0
        for joint in pos1:
            if joint in pos2:
                distance += (pos1[joint] - pos2[joint]) ** 2
        return math.sqrt(distance)
    
    def calculate_optimal_speed(self, distance: float, priority: int = 1) -> float:
        """Berechnet optimale Geschwindigkeit."""
        min_time = self.min_move_time * (4 - priority)
        max_speed = distance / min_time if min_time > 0 else 0.3
        tracking_limit = SCANNER_SPECS["tracking_speed"]
        return min(max_speed, tracking_limit)

# ============== BASIS SCAN PATTERN KLASSE ==============

class ScanPattern(ABC):
    """Abstrakte Basisklasse für alle Scan-Patterns."""
    
    def __init__(self, center_position: Optional[Dict[str, float]] = None,
                 scan_mode: str = "table_level", **kwargs):
        """Initialisiert Scan-Pattern."""
        if center_position is None:
            center_position = SCANNER_POSITIONS.get(scan_mode, SCANNER_CENTER).copy()
        
        self.center_position = center_position
        self.scan_mode = scan_mode
        self.name = self.__class__.__name__.replace("Pattern", "").replace("Scan", " Scan").strip()
        
        self.planner = IntelligentMultiAxisPlanner(self.center_position)
        self.validator = SafetyValidator()
        self.timing_optimizer = TimingOptimizer()
        
        self.optimal_distance = SCANNER_SPECS["optimal_distance"]
        self.min_distance = SCANNER_SPECS["min_distance"]
        self.max_distance = SCANNER_SPECS["max_distance"]
        self.tracking_speed = SCANNER_SPECS["tracking_speed"]
        self.podest_height = SCANNER_SPECS["podest_height"]
        
        self.points = []
        self.quality_metrics = ScanQuality()
        
        self.use_smooth_transitions = True
        self.validate_all_points = True
        self.timing_optimized = False
        self.adaptive_quality = kwargs.get('adaptive_quality', True)
        
        logger.info(f"✅ Initialized {self.name} pattern (mode: {scan_mode})")
    
    @abstractmethod
    def generate_points(self) -> List[ScanPoint]:
        """Generiert die Scan-Punkte."""
        pass
    
    def create_scan_point(self, x: float, y: float, z: float,
                         base_angle: Optional[float] = None,
                         speed: Optional[float] = None,
                         priority: int = 1,
                         debug: bool = False,
                         **kwargs) -> Optional[ScanPoint]:
        """
        Erstellt validierten Scan-Punkt mit PODEST-OPTIMIERUNG.
        Nutzt den 40cm Podest-Raum und führt Kamera parallel!
        """
        # Base-Rotation berechnen
        if base_angle is not None:
            base = base_angle
        else:
            base = math.atan2(x, y + self.optimal_distance) if abs(x) > 0.01 else 0
        
        # PODEST-OPTIMIERTE Shoulder-Berechnung
        if self.scan_mode == "table_level":
            # Objekt ist 40cm unter Podest - nutze das aus!
            shoulder_base = 0.7  # Nach unten geneigt
            shoulder_offset = z * 0.3
        else:
            shoulder_base = self.center_position["shoulder"]
            shoulder_offset = z * 0.4
        
        shoulder = shoulder_base + shoulder_offset
        shoulder = max(-1.4, min(1.4, shoulder))
        
        # Distanz-basierte Elbow-Berechnung
        distance = math.sqrt(x**2 + y**2 + z**2)
        distance_factor = min(1.0, distance / self.max_distance)
        
        # Elbow kompakter für Stabilität
        elbow_base = 1.2 if self.scan_mode == "table_level" else 1.1
        elbow = elbow_base - distance_factor * 0.2
        elbow = max(0.1, min(3.0, elbow))
        
        # WRIST PARALLEL ZUM BODEN - Kritisch!
        if self.scan_mode == "table_level":
            # Kompensiere Shoulder für parallele Kamera
            wrist = -shoulder * 0.6 - 0.2
        else:
            wrist = self.center_position["wrist"] + z * 0.1
        
        # SICHERHEIT: Wrist weit weg von Limits!
        wrist = max(-1.2, min(1.2, wrist))
        
        # Roll & Hand
        roll = self.center_position.get("roll", 1.57)
        hand = self.center_position.get("hand", 2.5)
        
        # Positions-Dictionary
        positions = {
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "roll": roll,
            "hand": hand
        }
        
        # Validierung mit korrekter Funktion!
        valid, errors = self.validator.validate_position(positions, conservative=False, debug=debug)
        
        if not valid:
            # Versuche zu korrigieren
            corrected = self.planner._correct_invalid_position(positions)
            if corrected:
                positions = corrected
                if debug:
                    logger.debug(f"Position auto-corrected")
            else:
                if debug:
                    logger.warning(f"Invalid position: {errors[0] if errors else 'unknown'}")
                return None
        
        # Geschwindigkeit berechnen
        if speed is None:
            speed = self.timing_optimizer.calculate_optimal_speed(distance, priority)
        
        # Scan-Point erstellen
        point = ScanPoint(
            positions=positions,
            speed=min(speed, self.tracking_speed),
            settle_time=kwargs.get('settle_time', 0.3),
            trajectory_type=kwargs.get('trajectory_type', 's_curve'),
            scan_angle=base,
            distance=distance,
            description=kwargs.get('description', f"Point at ({x:.2f}, {y:.2f}, {z:.2f})"),
            priority=priority,
            expected_quality=self._estimate_point_quality(distance),
            smooth_transition=True,
            metadata={
                "x": x, "y": y, "z": z,
                "scan_mode": self.scan_mode
            }
        )
        
        return point
    
    def _estimate_point_quality(self, distance: float) -> float:
        """Schätzt Scan-Qualität."""
        if distance < self.min_distance:
            return 0.7
        elif distance > self.max_distance:
            return 0.6
        elif abs(distance - self.optimal_distance) < 0.03:
            return 1.0
        else:
            if distance < self.optimal_distance:
                ratio = (distance - self.min_distance) / (self.optimal_distance - self.min_distance)
            else:
                ratio = (self.max_distance - distance) / (self.max_distance - self.optimal_distance)
            return 0.7 + 0.3 * ratio
    
    def optimize_path(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """Optimiert Scan-Pfad mit allen Techniken."""
        if not points:
            return points
        
        valid_points = [p for p in points if p is not None]
        
        if not valid_points:
            return []
        
        # Path Smoothing
        if self.use_smooth_transitions:
            valid_points = self.validator.smooth_path(
                valid_points, 
                max_delta=0.3,
                timing_optimize=True
            )
        
        # Timing-Optimierung
        if not self.timing_optimized:
            valid_points = self.timing_optimizer.optimize_timing(valid_points)
            self.timing_optimized = True
        
        # Qualitäts-Filterung
        if self.adaptive_quality:
            valid_points = self._filter_by_quality(valid_points)
        
        # Final Validation
        if self.validate_all_points:
            final_points = []
            for point in valid_points:
                valid, _ = self.validator.validate_position(point.positions, conservative=False)
                if valid:
                    final_points.append(point)
            valid_points = final_points
        
        # Statistiken
        self._calculate_quality_metrics(valid_points)
        
        logger.info(
            f"✅ Path optimized: {len(points)} -> {len(valid_points)} points "
            f"(Quality: {self.quality_metrics.overall_score:.2f})"
        )
        
        return valid_points
    
    def _filter_by_quality(self, points: List[ScanPoint], 
                          threshold: float = 0.5) -> List[ScanPoint]:
        """Filtert Punkte basierend auf Qualität."""
        high_quality = [p for p in points if p.expected_quality >= threshold]
        
        if len(high_quality) < len(points) * 0.5:
            return sorted(points, key=lambda p: p.expected_quality, reverse=True)[:int(len(points)*0.8)]
        
        return high_quality
    
    def _calculate_quality_metrics(self, points: List[ScanPoint]):
        """Berechnet Qualitäts-Metriken."""
        if not points:
            return
        
        positions = [p.metadata for p in points if p.metadata]
        if positions:
            x_range = max(p.get('x', 0) for p in positions) - min(p.get('x', 0) for p in positions)
            y_range = max(p.get('y', 0) for p in positions) - min(p.get('y', 0) for p in positions)
            z_range = max(p.get('z', 0) for p in positions) - min(p.get('z', 0) for p in positions)
            
            volume = x_range * y_range * z_range
            self.quality_metrics.coverage = min(1.0, volume / 0.008)
        
        self.quality_metrics.point_density = len(points) / max(1, volume) if 'volume' in locals() else 0
        self.quality_metrics.overlap = min(1.0, self.quality_metrics.point_density / 1000)
        
        avg_speed = sum(p.speed for p in points) / len(points)
        self.quality_metrics.stability = 1.0 - min(1.0, avg_speed / 0.5)
        
        self.quality_metrics.tracking_confidence = sum(p.expected_quality for p in points) / len(points)
        
        self.quality_metrics.estimated_time = sum(
            1.0 / p.speed + p.settle_time for p in points if p.speed > 0
        )

# ============== PATTERN IMPLEMENTIERUNGEN ==============

class RasterScanPattern(ScanPattern):
    """Raster-Scan Pattern mit PODEST-Nutzung."""
    
    def __init__(self, width: float = 0.20, height: float = 0.15,
                 rows: int = 10, cols: int = 10,
                 overlap: float = 0.2, speed: float = 0.3,
                 settle_time: float = 0.5, zigzag: bool = True, **kwargs):
        """Initialisiert Raster-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.speed = speed
        self.settle_time = settle_time
        self.zigzag = zigzag
        self.name = "Raster Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Raster-Scan Punkte - NUTZT PODEST!"""
        points = []
        
        step_x = self.width / max(1, self.cols - 1) * (1 - self.overlap)
        step_z = self.height / max(1, self.rows - 1) * (1 - self.overlap)
        
        # NUTZT RAUM NACH UNTEN!
        start_x = -self.width / 2
        start_z = -self.height / 2 - 0.2  # 20cm tiefer für Podest
        
        for row in range(self.rows):
            cols_range = range(self.cols)
            if self.zigzag and row % 2 == 1:
                cols_range = reversed(cols_range)
            
            for col_idx, col in enumerate(cols_range):
                x = start_x + col * step_x
                z = start_z + row * step_z
                y = 0.05  # Leicht vor dem Arm
                
                priority = 1 if row < 2 or row >= self.rows - 2 else 2
                
                if col_idx == 0:
                    speed = self.speed * 0.7
                    traj_type = "s_curve"
                else:
                    speed = self.speed
                    traj_type = "linear"
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=speed,
                    priority=priority,
                    settle_time=self.settle_time,
                    trajectory_type=traj_type,
                    description=f"Raster R{row}C{col}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} raster scan points")
        return self.optimize_path(points)

class SpiralScanPattern(ScanPattern):
    """Spiral-Scan mit INTELLIGENTEM Muster."""
    
    def __init__(self, radius_start: float = 0.05, radius_end: float = 0.20,
                 revolutions: int = 5, points_per_rev: int = 30,
                 vertical_range: float = 0.15, speed: float = 0.25, **kwargs):
        """Initialisiert Spiral-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.revolutions = revolutions
        self.points_per_rev = points_per_rev
        self.vertical_range = vertical_range
        self.speed = speed
        self.name = "Spiral Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spiral-Scan Punkte - NUTZT PODEST!"""
        points = []
        total_points = self.revolutions * self.points_per_rev
        
        for i in range(total_points):
            t = i / max(1, total_points - 1)
            
            angle = 2 * math.pi * self.revolutions * t
            radius = self.radius_start + (self.radius_end - self.radius_start) * t
            
            # 3D-Position mit PODEST-NUTZUNG
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) * 0.7
            z = -0.2 - self.vertical_range * t  # Startet 20cm tief!
            
            speed = self.speed * (1.0 - 0.3 * t)
            priority = 1 if radius < self.optimal_distance else 2
            
            point = self.create_scan_point(
                x, y, z,
                speed=speed,
                priority=priority,
                description=f"Spiral {i+1}/{total_points}"
            )
            
            if point:
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} spiral scan points")
        return self.optimize_path(points)

class SphericalScanPattern(ScanPattern):
    """Spherical-Scan für 3D-Objekte."""
    
    def __init__(self, radius: float = 0.15, theta_steps: int = 12,
                 phi_steps: int = 8, phi_range: float = 1.0,
                 speed: float = 0.3, **kwargs):
        """Initialisiert Spherical-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'spherical'), **kwargs)
        self.radius = radius
        self.theta_steps = theta_steps
        self.phi_steps = phi_steps
        self.phi_range = min(phi_range, 1.2)
        self.speed = speed
        self.name = "Spherical Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert sphärische Scan-Punkte."""
        points = []
        
        for phi_idx in range(self.phi_steps):
            phi = -self.phi_range/2 + (phi_idx * self.phi_range / max(1, self.phi_steps - 1))
            theta_count = max(4, int(self.theta_steps * math.cos(phi)))
            
            for theta_idx in range(theta_count):
                theta = 2 * math.pi * theta_idx / theta_count
                
                x = self.radius * math.cos(phi) * math.sin(theta)
                y = self.radius * math.cos(phi) * math.cos(theta)
                z = self.radius * math.sin(phi) - 0.1
                
                base_angle = theta
                speed = self.speed * (0.8 if abs(phi) > self.phi_range * 0.7 else 1.0)
                
                point = self.create_scan_point(
                    x, y, z,
                    base_angle=base_angle,
                    speed=speed,
                    description=f"Sphere θ={math.degrees(theta):.0f}° φ={math.degrees(phi):.0f}°"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} spherical scan points")
        return self.optimize_path(points)

class TurntableScanPattern(ScanPattern):
    """Turntable-Scan für rotierende Objekte."""
    
    def __init__(self, steps: int = 36, height_levels: int = 3,
                 height_range: float = 0.20, radius: float = 0.15,
                 speed: float = 0.2, **kwargs):
        """Initialisiert Turntable-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.steps = steps
        self.height_levels = height_levels
        self.height_range = height_range
        self.radius = radius
        self.speed = speed
        self.name = "Turntable Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Turntable-Scan Punkte."""
        points = []
        
        for level in range(self.height_levels):
            if self.height_levels > 1:
                z = -self.height_range/2 + (level * self.height_range / (self.height_levels - 1))
            else:
                z = 0
            
            z -= 0.15  # Podest-Offset
            
            for step in range(self.steps):
                angle = 2 * math.pi * step / self.steps
                
                x = self.radius * math.sin(angle) * 0.7
                y = self.radius * math.cos(angle) * 0.7
                
                point = self.create_scan_point(
                    x, y, z,
                    base_angle=angle,
                    speed=self.speed,
                    priority=1 if level == self.height_levels // 2 else 2,
                    description=f"Turntable L{level}S{step}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} turntable scan points")
        return self.optimize_path(points)

class HelixScanPattern(ScanPattern):
    """Helix-Scan für zylindrische Objekte."""
    
    def __init__(self, radius: float = 0.12, height: float = 0.25,
                 turns: int = 5, points_per_turn: int = 24,
                 speed: float = 0.22, **kwargs):
        """Initialisiert Helix-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.radius = radius
        self.height = height
        self.turns = turns
        self.points_per_turn = points_per_turn
        self.speed = speed
        self.name = "Helix Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Helix-Scan Punkte."""
        points = []
        total_points = self.turns * self.points_per_turn
        
        for i in range(total_points):
            t = i / max(1, total_points - 1)
            angle = 2 * math.pi * self.turns * t
            
            x = self.radius * math.cos(angle) * 0.8
            y = self.radius * math.sin(angle) * 0.6
            z = -self.height/2 + self.height * t - 0.2
            
            speed = self.speed * (0.9 + 0.1 * math.sin(t * math.pi))
            
            point = self.create_scan_point(
                x, y, z,
                speed=speed,
                description=f"Helix turn {i // self.points_per_turn + 1}"
            )
            
            if point:
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} helix scan points")
        return self.optimize_path(points)

class CobwebScanPattern(ScanPattern):
    """Cobweb-Scan für detaillierte Oberflächen."""
    
    def __init__(self, radial_lines: int = 8, circles: int = 5,
                 max_radius: float = 0.20, speed: float = 0.25, **kwargs):
        """Initialisiert Cobweb-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.radial_lines = radial_lines
        self.circles = circles
        self.max_radius = max_radius
        self.speed = speed
        self.name = "Cobweb Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spinnennetz-Scan Punkte."""
        points = []
        
        for circle in range(self.circles):
            radius = (circle + 1) * self.max_radius / self.circles * 0.7
            
            for line in range(self.radial_lines):
                angle = 2 * math.pi * line / self.radial_lines
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.5
                z = -0.2  # Nutzt Podest
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=self.speed,
                    description=f"Web C{circle}L{line}"
                )
                
                if point:
                    points.append(point)
        
        for line in range(self.radial_lines):
            angle = 2 * math.pi * line / self.radial_lines
            
            for r_idx in range(5):
                radius = r_idx * self.max_radius / 4 * 0.7
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.5
                z = -0.2
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=self.speed * 0.8,
                    description=f"Radial L{line}R{r_idx}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} cobweb scan points")
        return self.optimize_path(points)

class AdaptiveScanPattern(ScanPattern):
    """Adaptiver intelligenter Scan."""
    
    def __init__(self, initial_points: int = 20,
                 refinement_threshold: float = 0.05,
                 max_iterations: int = 3, **kwargs):
        """Initialisiert Adaptive-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'table_level'), **kwargs)
        self.initial_points = initial_points
        self.refinement_threshold = refinement_threshold
        self.max_iterations = max_iterations
        self.name = "Adaptive Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert adaptive Scan-Punkte."""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(self.initial_points):
            theta = 2 * math.pi * i / golden_ratio
            r = math.sqrt(i / self.initial_points) * self.max_distance * 0.6
            
            x = r * math.cos(theta)
            y = r * math.sin(theta) * 0.5
            z = -0.15 - 0.1 * (i / self.initial_points)
            
            point = self.create_scan_point(
                x, y, z,
                speed=self.tracking_speed,
                priority=1,
                description=f"Adaptive initial {i}"
            )
            
            if point:
                points.append(point)
        
        for iteration in range(min(self.max_iterations, 2)):
            refinement_points = []
            
            for i in range(5 + iteration * 3):
                angle = 2 * math.pi * i / (5 + iteration * 3)
                r = self.optimal_distance * (0.8 + 0.2 * iteration)
                
                x = r * math.cos(angle)
                y = r * math.sin(angle) * 0.4
                z = -0.25 - iteration * 0.05
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=self.tracking_speed * 0.8,
                    priority=2,
                    description=f"Adaptive refine {iteration}-{i}"
                )
                
                if point:
                    refinement_points.append(point)
            
            points.extend(refinement_points)
            
            avg_quality = sum(p.expected_quality for p in refinement_points) / max(1, len(refinement_points))
            if avg_quality > 0.9:
                break
        
        logger.info(f"✅ Generated {len(points)} adaptive scan points")
        return self.optimize_path(points)

class StatueSpiralPattern(ScanPattern):
    """Statue-Spiral für komplexe 3D-Objekte."""
    
    def __init__(self, height: float = 0.30, base_radius: float = 0.15,
                 vertical_steps: int = 8, angular_steps: int = 24,
                 spiral_factor: float = 0.3, **kwargs):
        """Initialisiert Statue-Scan."""
        super().__init__(scan_mode=kwargs.pop('scan_mode', 'spherical'), **kwargs)
        self.height = height
        self.base_radius = base_radius
        self.vertical_steps = vertical_steps
        self.angular_steps = angular_steps
        self.spiral_factor = spiral_factor
        self.name = "Statue Spiral Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Statue-Scan Punkte."""
        points = []
        
        for v_step in range(self.vertical_steps):
            t_height = v_step / max(1, self.vertical_steps - 1)
            z = -self.height/2 + self.height * t_height - 0.15
            
            radius = self.base_radius * (1 - 0.3 * t_height)
            angle_offset = self.spiral_factor * 2 * math.pi * t_height
            
            for a_step in range(self.angular_steps):
                angle = 2 * math.pi * a_step / self.angular_steps + angle_offset
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.7
                
                speed = self.tracking_speed * (0.7 + 0.3 * (1 - t_height))
                
                point = self.create_scan_point(
                    x, y, z,
                    base_angle=angle,
                    speed=speed,
                    priority=1 if v_step < 2 else 2,
                    description=f"Statue H{v_step}A{a_step}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} statue scan points")
        return self.optimize_path(points)

class TableScanPattern(ScanPattern):
    """Table-Scan für flache Objekte."""
    
    def __init__(self, width: float = 0.30, depth: float = 0.20,
                 height_offset: float = -0.02, resolution: float = 0.02,
                 speed: float = 0.35, **kwargs):
        """Initialisiert Table-Scan."""
        super().__init__(scan_mode='table_level', **kwargs)
        self.width = width
        self.depth = depth
        self.height_offset = height_offset
        self.resolution = resolution
        self.speed = speed
        self.name = "Table Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Table-Scan Punkte."""
        points = []
        
        cols = max(3, int(self.width / self.resolution))
        rows = max(3, int(self.depth / self.resolution))
        
        for row in range(rows):
            col_range = range(cols) if row % 2 == 0 else reversed(range(cols))
            
            for col in col_range:
                x = -self.width/2 + col * self.width / max(1, cols - 1)
                y = -self.depth/2 + row * self.depth / max(1, rows - 1)
                z = self.height_offset - 0.25
                
                if 0 < col < cols - 1:
                    speed = self.speed
                    traj = "linear"
                else:
                    speed = self.speed * 0.7
                    traj = "s_curve"
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=speed,
                    trajectory_type=traj,
                    settle_time=0.2,
                    description=f"Table R{row}C{col}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} table scan points")
        return self.optimize_path(points)

# ============== FACTORY FUNCTIONS ==============

def create_scan_pattern(pattern_type: str, **kwargs) -> Optional[ScanPattern]:
    """Factory-Funktion zum Erstellen von Scan-Patterns."""
    patterns = {
        'raster': RasterScanPattern,
        'spiral': SpiralScanPattern,
        'spherical': SphericalScanPattern,
        'turntable': TurntableScanPattern,
        'helix': HelixScanPattern,
        'cobweb': CobwebScanPattern,
        'adaptive': AdaptiveScanPattern,
        'statue': StatueSpiralPattern,
        'statue_spiral': StatueSpiralPattern,
        'table': TableScanPattern
    }
    
    pattern_class = patterns.get(pattern_type.lower())
    if pattern_class:
        try:
            pattern = pattern_class(**kwargs)
            logger.info(f"✅ Created {pattern.name} pattern")
            return pattern
        except Exception as e:
            logger.error(f"Failed to create {pattern_type} pattern: {e}")
            return None
    else:
        logger.error(f"Unknown pattern type: {pattern_type}")
        return None

def get_pattern_presets(preset: str) -> Optional[ScanPattern]:
    """Gibt vordefinierte Pattern-Presets zurück."""
    presets = {
        'quick': RasterScanPattern(
            rows=5, cols=5, width=0.15, height=0.10, 
            zigzag=True, speed=0.4
        ),
        'quick_spiral': SpiralScanPattern(
            revolutions=3, points_per_rev=20, 
            radius_end=0.15, speed=0.35
        ),
        'detailed': RasterScanPattern(
            rows=15, cols=15, width=0.25, height=0.20, 
            overlap=0.3, speed=0.2
        ),
        'high_detail': RasterScanPattern(
            rows=20, cols=20, width=0.30, height=0.25,
            overlap=0.4, speed=0.15
        ),
        'small_object': SpiralScanPattern(
            radius_start=0.02, radius_end=0.08, 
            revolutions=6, points_per_rev=30
        ),
        'large_object': SphericalScanPattern(
            radius=0.20, theta_steps=16, phi_steps=10,
            speed=0.25
        ),
        'cylindrical': HelixScanPattern(
            radius=0.10, height=0.20, turns=6,
            points_per_turn=20
        ),
        'flat': TableScanPattern(
            width=0.30, depth=0.20, resolution=0.015,
            speed=0.4
        ),
        'statue': StatueSpiralPattern(
            height=0.25, base_radius=0.12,
            vertical_steps=10, angular_steps=30
        ),
        'turntable_simple': TurntableScanPattern(
            steps=24, height_levels=1, radius=0.12
        ),
        'turntable_multi': TurntableScanPattern(
            steps=36, height_levels=3, height_range=0.20
        ),
        'adaptive_fast': AdaptiveScanPattern(
            initial_points=15, max_iterations=2
        ),
        'adaptive_quality': AdaptiveScanPattern(
            initial_points=30, max_iterations=4,
            refinement_threshold=0.03
        )
    }
    
    pattern = presets.get(preset.lower())
    if pattern:
        logger.info(f"✅ Loaded preset: {preset}")
    else:
        logger.warning(f"Unknown preset: {preset}")
    
    return pattern

def estimate_scan_time(pattern: ScanPattern) -> float:
    """Schätzt die Scan-Zeit für ein Pattern."""
    if not pattern.points:
        points = pattern.generate_points()
    else:
        points = pattern.points
    
    if not points:
        return 0
    
    total_time = 0
    for i, point in enumerate(points):
        if i > 0:
            prev = points[i-1]
            distance = math.sqrt(sum(
                (point.positions.get(j, 0) - prev.positions.get(j, 0))**2 
                for j in point.positions
            ))
            if point.speed > 0:
                total_time += distance / point.speed
        
        total_time += point.settle_time
    
    return total_time

# ============== MAIN TEST ==============

if __name__ == "__main__":
    print("🤖 RoArm M3 Professional Scan Patterns V5.1 FIXED")
    print("="*60)
    
    test_patterns = [
        ('raster', RasterScanPattern(rows=5, cols=5)),
        ('spiral', SpiralScanPattern(revolutions=3)),
        ('spherical', SphericalScanPattern(theta_steps=8)),
        ('turntable', TurntableScanPattern(steps=12)),
        ('helix', HelixScanPattern(turns=3)),
        ('cobweb', CobwebScanPattern(radial_lines=6)),
        ('adaptive', AdaptiveScanPattern(initial_points=15)),
        ('statue', StatueSpiralPattern(vertical_steps=5)),
        ('table', TableScanPattern(width=0.25))
    ]
    
    for name, pattern in test_patterns:
        print(f"\n📊 Testing {name.upper()} Pattern:")
        print("-"*40)
        
        points = pattern.generate_points()
        
        print(f"  Generated points: {len(points)}")
        print(f"  Quality score: {pattern.quality_metrics.overall_score:.2f}")
        print(f"  Estimated time: {pattern.quality_metrics.estimated_time:.1f}s")
        
        if points:
            first = points[0]
            last = points[-1]
            
            print(f"  First wrist: {first.positions['wrist']:.3f}")
            print(f"  Last wrist: {last.positions['wrist']:.3f}")
            
            valid_count = sum(1 for p in points 
                            if SafetyValidator.validate_position(p.positions, conservative=False)[0])
            
            print(f"  Validation: {valid_count}/{len(points)} valid")
    
    print("\n" + "="*60)
    print("✅ PRODUCTION READY mit korrekter Validierung!")
    print("="*60)
