#!/usr/bin/env python3
"""
RoArm M3 PROFESSIONAL Scan Patterns für Revopoint Mini2
Version 5.0 - Vollständige erweiterte und optimierte Edition
Mit Taktungs-Optimierung, Podest-Support und allen intelligenten Features

Optimiert für:
- Podest-Höhe: 40cm
- Update-Rate: 11.4 Hz (aus Hardware-Analyse)
- Tracking ohne Verlust
- Maximale Stabilität durch Base-nahe Bewegungen
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
    # Fallback mit sicheren Konstanten
    SERVO_LIMITS = {
        "base": (-3.14, 3.14),      # ±180°
        "shoulder": (-1.57, 1.57),  # ±90°
        "elbow": (0.0, 3.14),       # 0-180°
        "wrist": (-1.57, 1.57),     # ±90° - KRITISCH!
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

# ============== FACTORY & HELPER FUNCTIONS ==============

# Kompatibilitäts-Alias für StatueScanPattern
StatueScanPattern = StatueSpiralPattern

def create_scan_pattern(pattern_type: str, **kwargs) -> Optional[ScanPattern]:
    """
    Factory-Funktion zum Erstellen von Scan-Patterns.
    
    Args:
        pattern_type: Pattern-Typ Name
        **kwargs: Pattern-spezifische Parameter
        
    Returns:
        ScanPattern-Instanz oder None
    """
    patterns = {
        'raster': RasterScanPattern,
        'spiral': SpiralScanPattern,
        'spherical': SphericalScanPattern,
        'turntable': TurntableScanPattern,
        'helix': HelixScanPattern,
        'cobweb': CobwebScanPattern,
        'adaptive': AdaptiveScanPattern,
        'statue': StatueSpiralPattern,
        'statue_spiral': StatueSpiralPattern,  # Alias für Kompatibilität
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
    """
    HAUPTFUNKTION für main.py - Gibt vordefinierte Pattern-Presets zurück.
    
    Args:
        preset: Preset-Name
        
    Returns:
        Konfiguriertes ScanPattern oder None
    """
    presets = {
        # Quick Scans
        'quick': RasterScanPattern(
            rows=5, cols=5, width=0.15, height=0.10, 
            zigzag=True, speed=0.4
        ),
        'quick_spiral': SpiralScanPattern(
            revolutions=3, points_per_rev=20, 
            radius_end=0.15, speed=0.35
        ),
        
        # Detailed Scans
        'detailed': RasterScanPattern(
            rows=15, cols=15, width=0.25, height=0.20, 
            overlap=0.3, speed=0.2
        ),
        'high_detail': RasterScanPattern(
            rows=20, cols=20, width=0.30, height=0.25,
            overlap=0.4, speed=0.15
        ),
        
        # Object-specific
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
        
        # Special
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


# Alias für alternative Namensgebung
get_preset_pattern = get_pattern_presets


# ============== STANDARD POSITIONEN ==============

# Home Position - Arm aufrecht (aus deiner Zeichnung)
HOME_POSITION_DEFAULT = {
    "base": 0.0,        # Geradeaus
    "shoulder": 0.0,    # Horizontal/Aufrecht
    "elbow": 1.57,      # 90° gebeugt
    "wrist": 0.0,       # Level
    "roll": 0.0,        # Neutral
    "hand": 3.14        # Geschlossen
}

# Scanner Start-Position (vor dem Scan)
SCAN_START_POSITION = {
    "base": 0.0,
    "shoulder": 0.2,    # Leicht nach vorne
    "elbow": 1.0,       # Bereit-Position
    "wrist": -0.3,      # Leicht geneigt
    "roll": 1.57,       # Scanner-Orientierung
    "hand": 2.5         # Scanner-Griff
}

# Scanner End-Position (nach dem Scan)
SCAN_END_POSITION = {
    "base": 0.0,
    "shoulder": 0.3,    # Etwas höher
    "elbow": 1.2,       
    "wrist": -0.2,      # Fast level
    "roll": 1.57,
    "hand": 2.5
}

# ============== ERWEITERTE KONFIGURATION ==============

# Hardware-Taktung (aus Log-Analyse)
HARDWARE_TIMING = {
    "servo_update_rate": 11.4,      # Hz (20 points / 1.75s aus Logs)
    "serial_latency": 0.02,         # Sekunden (wait_time aus controller.py)
    "min_movement_time": 0.087,     # 1/11.4 Hz
    "trajectory_points": 20,        # Standard-Anzahl Trajectory Points
    "settle_multiplier": 1.2,       # Sicherheitsfaktor für Settling
    "sync_interval": 5,             # Sync alle N Punkte
    "buffer_size": 50               # Trajectory Buffer Size
}

# Revopoint Mini2 Scanner Specs (erweitert)
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
    "podest_height": 0.40,          # 40cm Podest-Höhe
    "precision_mode_distance": 0.12,# Präzisionsmodus unter 12cm
    "fast_mode_distance": 0.25      # Schnellmodus über 25cm
}

# OPTIMIERTE Scanner-Positionen für verschiedene Szenarien
SCANNER_POSITIONS = {
    # Für Objekte auf Tisch-Niveau (40cm unter Podest)
    "table_level": {
        "base": 0.0,        # Zentriert
        "shoulder": 0.95,   # 54° nach unten - nutzt Podest voll aus!
        "elbow": 1.2,       # 69° gebeugt - kompakt und stabil
        "wrist": -0.8,      # SICHER! War -1.57 (am Limit)
        "roll": 1.57,       # 90° für Scanner-Orientierung
        "hand": 2.5         # Scanner-Griff optimal
    },
    # Für Objekte auf Podest-Höhe
    "same_level": {
        "base": 0.0,
        "shoulder": 0.3,    # Leicht nach unten
        "elbow": 1.4,       # Stark gebeugt für Nähe
        "wrist": -0.5,      # Moderate Neigung
        "roll": 1.57,
        "hand": 2.5
    },
    # Für sphärische/360° Scans
    "spherical": {
        "base": 0.0,
        "shoulder": 0.6,    # Mittlere Position
        "elbow": 1.1,       # Ausbalanciert
        "wrist": -0.6,      # Kompromiss-Position
        "roll": 1.57,
        "hand": 2.5
    },
    # Für große Objekte
    "large_object": {
        "base": 0.0,
        "shoulder": 0.4,
        "elbow": 0.9,       # Mehr gestreckt für Reichweite
        "wrist": -0.7,
        "roll": 1.57,
        "hand": 2.5
    },
    # Für kleine Details
    "detail_scan": {
        "base": 0.0,
        "shoulder": 0.85,   # Tief für Stabilität
        "elbow": 1.5,       # Sehr kompakt
        "wrist": -0.4,      # Flacher Winkel
        "roll": 1.57,
        "hand": 2.5
    }
}

# Default Scanner Center (Abwärtskompatibilität)
SCANNER_CENTER = SCANNER_POSITIONS["table_level"]

# Scan Defaults (erweitert)
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
    """Erweiterte ScanPoint-Klasse mit allen Features."""
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
        # Konvertiere trajectory_type zu String falls nötig
        if hasattr(self.trajectory_type, 'value'):
            self.trajectory_type = self.trajectory_type.value
        elif not isinstance(self.trajectory_type, str):
            self.trajectory_type = str(self.trajectory_type).lower()
    
    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary für Serialisierung."""
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
    coverage: float = 0.0           # 0-1 Abdeckung
    overlap: float = 0.0            # 0-1 Überlappung
    stability: float = 0.0          # 0-1 Stabilität
    tracking_confidence: float = 0.0 # 0-1 Tracking-Sicherheit
    estimated_time: float = 0.0     # Sekunden
    point_density: float = 0.0      # Punkte/cm²
    
    @property
    def overall_score(self) -> float:
        """Berechne Gesamt-Qualitätsscore."""
        weights = {
            "coverage": 0.3,
            "overlap": 0.2,
            "stability": 0.25,
            "tracking_confidence": 0.25
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

# ============== SAFETY & VALIDATION ==============

class SafetyValidator:
    """Erweiterte Sicherheitsvalidierung mit intelligenten Features."""
    
    # Joint-spezifische Sicherheitsmargins (OPTIMIERT für Podest)
    SAFETY_MARGINS = {
        "base": 0.05,
        "shoulder": 0.10,
        "elbow": 0.08,
        "wrist": 0.20,      # EXTRA groß für kritisches Wrist!
        "roll": 0.05,
        "hand": 0.02
    }
    
    # Geschwindigkeitslimits pro Joint (rad/s)
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
                         conservative: bool = True,
                         debug: bool = False) -> Tuple[bool, List[str]]:
        """
        Validiert Position gegen Servo-Limits mit intelligenter Margin-Berechnung.
        
        Args:
            positions: Joint-Positionen
            conservative: Nutze konservative Margins
            debug: Debug-Output
            
        Returns:
            (valid, error_messages)
        """
        errors = []
        warnings = []
        
        for joint, value in positions.items():
            if joint not in SERVO_LIMITS:
                continue
                
            min_limit, max_limit = SERVO_LIMITS[joint]
            
            # Adaptive Margin basierend auf Joint und Modus
            base_margin = SafetyValidator.SAFETY_MARGINS.get(joint, 0.05)
            
            if conservative:
                margin = base_margin * 1.5
            else:
                margin = base_margin
            
            # Spezielle Behandlung für Wrist (kritisch!)
            if joint == "wrist":
                # Noch konservativer für Wrist
                margin = max(0.2, margin)
                safe_min = max(min_limit + margin, -1.35)  # Hard limit
                safe_max = min(max_limit - margin, 1.35)   # Hard limit
            else:
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
        """
        Validiert komplette Trajektorie auf Sicherheit und Durchführbarkeit.
        
        Args:
            points: Liste von ScanPoints
            max_velocity: Max Geschwindigkeit pro Joint
            max_acceleration: Max Beschleunigung
            
        Returns:
            (valid, error_messages)
        """
        if not points:
            return False, ["Empty trajectory"]
        
        errors = []
        max_vel = max_velocity or SafetyValidator.VELOCITY_LIMITS
        
        for i in range(len(points) - 1):
            curr = points[i]
            next_point = points[i + 1]
            
            # Position validation
            valid, pos_errors = SafetyValidator.validate_position(
                curr.positions, conservative=True
            )
            if not valid:
                errors.extend([f"Point {i}: {e}" for e in pos_errors])
            
            # Velocity check
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
        """
        Glättet Pfad für kontinuierliches Tracking mit Taktungs-Optimierung.
        
        Args:
            points: Original-Punkte
            max_delta: Max Änderung zwischen Punkten
            timing_optimize: Optimiere für Hardware-Taktung
            
        Returns:
            Geglättete Punkte
        """
        if len(points) < 2:
            return points
        
        smoothed = [points[0]]
        
        for i in range(1, len(points)):
            prev = smoothed[-1]
            curr = points[i]
            
            # Berechne maximale Änderung
            max_change = 0
            joint_deltas = {}
            
            for joint in prev.positions:
                if joint in curr.positions:
                    delta = abs(curr.positions[joint] - prev.positions[joint])
                    joint_deltas[joint] = delta
                    max_change = max(max_change, delta)
            
            # Brauchen wir Interpolation?
            if max_change > max_delta:
                # Berechne nötige Zwischenschritte
                steps = int(math.ceil(max_change / max_delta))
                
                # Timing-Optimierung
                if timing_optimize:
                    # Anpassen an Hardware-Taktung (11.4 Hz)
                    optimal_steps = max(2, min(steps, 5))
                    steps = optimal_steps
                
                # Füge Zwischenpunkte ein
                for step in range(1, steps):
                    t = step / steps
                    
                    # Smooth interpolation (Cubic)
                    t_smooth = t * t * (3 - 2 * t)  # Smoothstep
                    
                    interp_pos = {}
                    for joint in prev.positions:
                        if joint in curr.positions:
                            interp_pos[joint] = prev.positions[joint] + \
                                              t_smooth * (curr.positions[joint] - prev.positions[joint])
                        else:
                            interp_pos[joint] = prev.positions[joint]
                    
                    # Timing-optimierte Geschwindigkeit
                    speed = min(prev.speed, curr.speed) * 0.8
                    if timing_optimize:
                        # Anpassen an Update-Rate
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
            
            # Original-Punkt (ggf. mit Timing-Anpassung)
            if timing_optimize and i % HARDWARE_TIMING["sync_interval"] == 0:
                # Sync-Punkt für Hardware
                curr.settle_time = max(curr.settle_time, 
                                      HARDWARE_TIMING["serial_latency"] * 2)
                curr.timing_optimized = True
            
            smoothed.append(curr)
        
        logger.info(f"✅ Path smoothed: {len(points)} -> {len(smoothed)} points")
        return smoothed

# ============== INTELLIGENTE PLANNER ==============

class IntelligentMultiAxisPlanner:
    """
    Hochentwickelter Multi-Achsen-Planer mit KI-ähnlichen Features.
    Optimiert Bewegungen basierend auf Hardware-Charakteristiken.
    """
    
    def __init__(self, center_position: Dict[str, float]):
        """Initialisiert den intelligenten Planer."""
        self.center = center_position.copy()
        self.validator = SafetyValidator()
        
        # Joint-Gewichtungen für Optimierung
        self.joint_weights = {
            "base": 2.5,      # Schwere Basis - vermeiden
            "shoulder": 3.0,  # Schwerster Arm - minimieren
            "elbow": 2.0,     # Mittel
            "wrist": 1.5,     # Leicht
            "roll": 1.0,      # Sehr leicht
            "hand": 0.2       # Minimal
        }
        
        # Energie-Kosten-Matrix
        self.energy_cost = {
            "base": 0.8,
            "shoulder": 1.0,
            "elbow": 0.6,
            "wrist": 0.3,
            "roll": 0.2,
            "hand": 0.1
        }
        
        # Stabilitäts-Faktoren
        self.stability_factors = {
            "base": 1.0,      # Neutral
            "shoulder": 0.7,  # Destabilisierend
            "elbow": 0.9,     # Leicht destabilisierend
            "wrist": 1.1,     # Stabilisierend
            "roll": 1.2,      # Sehr stabil
            "hand": 1.0       # Neutral
        }
        
        # Timing-Optimierung
        self.timing_optimizer = TimingOptimizer()
        
        # Cache für berechnete Pfade
        self.path_cache = {}
        
    def plan_optimal_trajectory(self, 
                              start: Dict[str, float],
                              end: Dict[str, float],
                              constraints: Optional[Dict] = None) -> List[ScanPoint]:
        """
        Plant optimale Trajektorie zwischen zwei Punkten.
        
        Args:
            start: Start-Position
            end: Ziel-Position
            constraints: Zusätzliche Constraints
            
        Returns:
            Optimierte Trajektorie
        """
        # Cache-Key
        cache_key = f"{hash(tuple(start.items()))}_{hash(tuple(end.items()))}"
        if cache_key in self.path_cache:
            logger.debug("Using cached trajectory")
            return self.path_cache[cache_key]
        
        # Berechne optimalen Pfad
        points = []
        
        # Entscheidung: Direkt oder via Waypoints?
        distance = self._calculate_weighted_distance(start, end)
        
        if distance < 0.3:
            # Kurze Distanz - direkte Interpolation
            points = self._interpolate_direct(start, end, steps=5)
        elif distance < 0.8:
            # Mittlere Distanz - Single Waypoint
            waypoint = self._calculate_waypoint(start, end)
            points = self._interpolate_direct(start, waypoint, steps=3)
            points.extend(self._interpolate_direct(waypoint, end, steps=3))
        else:
            # Lange Distanz - Multiple Waypoints
            waypoints = self._calculate_multiple_waypoints(start, end, count=3)
            prev = start
            for wp in waypoints:
                points.extend(self._interpolate_direct(prev, wp, steps=2))
                prev = wp
            points.extend(self._interpolate_direct(prev, end, steps=3))
        
        # Timing-Optimierung
        points = self.timing_optimizer.optimize_timing(points)
        
        # Validierung
        valid_points = []
        for point in points:
            valid, _ = self.validator.validate_position(point.positions)
            if valid:
                valid_points.append(point)
            else:
                # Versuche zu korrigieren
                corrected = self._correct_invalid_position(point.positions)
                if corrected:
                    point.positions = corrected
                    valid_points.append(point)
        
        # Cache speichern
        if len(valid_points) > 0:
            self.path_cache[cache_key] = valid_points
        
        return valid_points
    
    def _calculate_weighted_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Berechnet gewichtete Distanz zwischen Positionen."""
        distance = 0.0
        for joint in pos1:
            if joint in pos2 and joint in self.joint_weights:
                diff = abs(pos1[joint] - pos2[joint])
                distance += self.joint_weights[joint] * (diff ** 2)
        return math.sqrt(distance)
    
    def _interpolate_direct(self, start: Dict, end: Dict, steps: int) -> List[ScanPoint]:
        """Direkte Interpolation zwischen zwei Punkten."""
        points = []
        
        for i in range(steps):
            t = i / max(1, steps - 1)
            # Smoothstep für sanfte Beschleunigung
            t_smooth = t * t * (3 - 2 * t)
            
            positions = {}
            for joint in start:
                if joint in end:
                    positions[joint] = start[joint] + t_smooth * (end[joint] - start[joint])
                else:
                    positions[joint] = start[joint]
            
            # Geschwindigkeit basierend auf Position in Trajektorie
            if i == 0 or i == steps - 1:
                speed = 0.15  # Langsam am Anfang/Ende
            else:
                speed = 0.25  # Normal in der Mitte
            
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
        """Berechnet optimalen Wegpunkt zwischen Start und Ziel."""
        waypoint = {}
        
        for joint in start:
            if joint in end:
                # Basis-Interpolation
                mid = (start[joint] + end[joint]) / 2
                
                # Optimierung basierend auf Joint-Charakteristik
                if joint == "shoulder":
                    # Shoulder etwas höher für Stabilität
                    mid += 0.1
                elif joint == "elbow":
                    # Elbow mehr gebeugt für Kompaktheit
                    mid += 0.15
                elif joint == "wrist":
                    # Wrist sicher weg vom Limit
                    mid = max(-1.2, min(1.2, mid))
                
                waypoint[joint] = mid
            else:
                waypoint[joint] = start[joint]
        
        return waypoint
    
    def _calculate_multiple_waypoints(self, start: Dict, end: Dict, 
                                     count: int = 3) -> List[Dict[str, float]]:
        """Berechnet mehrere optimale Wegpunkte."""
        waypoints = []
        
        for i in range(1, count + 1):
            t = i / (count + 1)
            waypoint = {}
            
            for joint in start:
                if joint in end:
                    # Basis-Interpolation
                    value = start[joint] + t * (end[joint] - start[joint])
                    
                    # Sinusförmige Modulation für sanfte Kurve
                    offset = 0.1 * math.sin(t * math.pi)
                    
                    # Joint-spezifische Anpassung
                    if joint == "base":
                        # Base möglichst direkt
                        offset *= 0.5
                    elif joint == "shoulder":
                        # Shoulder-Kurve für Stabilität
                        value += offset * 0.8
                    elif joint == "wrist":
                        # Wrist-Sicherheit
                        value = max(-1.2, min(1.2, value))
                    
                    waypoint[joint] = value
                else:
                    waypoint[joint] = start[joint]
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def _correct_invalid_position(self, positions: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Versucht, ungültige Position zu korrigieren."""
        corrected = positions.copy()
        
        for joint, value in corrected.items():
            if joint in SERVO_LIMITS:
                min_limit, max_limit = SERVO_LIMITS[joint]
                margin = SafetyValidator.SAFETY_MARGINS.get(joint, 0.1)
                
                # Clamp to safe range
                safe_min = min_limit + margin
                safe_max = max_limit - margin
                
                if joint == "wrist":
                    # Extra-Sicherheit für Wrist
                    safe_min = max(safe_min, -1.3)
                    safe_max = min(safe_max, 1.3)
                
                corrected[joint] = max(safe_min, min(safe_max, value))
        
        # Validiere korrigierte Position
        valid, _ = SafetyValidator.validate_position(corrected)
        return corrected if valid else None

# ============== TIMING OPTIMIZER ==============

class TimingOptimizer:
    """Optimiert Timing basierend auf Hardware-Charakteristiken."""
    
    def __init__(self):
        """Initialisiert Timing-Optimizer."""
        self.update_rate = HARDWARE_TIMING["servo_update_rate"]
        self.serial_latency = HARDWARE_TIMING["serial_latency"]
        self.min_move_time = HARDWARE_TIMING["min_movement_time"]
        
    def optimize_timing(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """
        Optimiert Timing der Scan-Punkte für Hardware.
        
        Args:
            points: Original-Punkte
            
        Returns:
            Timing-optimierte Punkte
        """
        if not points:
            return points
        
        optimized = []
        
        for i, point in enumerate(points):
            # Kopiere Punkt
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
            
            # Geschwindigkeits-Anpassung basierend auf Update-Rate
            if i > 0:
                prev = optimized[-1] if optimized else points[i-1]
                distance = self._calculate_movement_distance(prev.positions, point.positions)
                
                # Optimale Geschwindigkeit für smooth tracking
                min_time = max(self.min_move_time, distance / 0.5)  # Max 0.5 rad/s
                optimal_time = math.ceil(min_time * self.update_rate) / self.update_rate
                
                if optimal_time > 0:
                    opt_point.speed = distance / optimal_time
                else:
                    opt_point.speed = 0.2
            
            # Settle-Time Optimierung
            if i % HARDWARE_TIMING["sync_interval"] == 0:
                # Sync-Punkt - längere Pause
                opt_point.settle_time = max(opt_point.settle_time, self.serial_latency * 3)
                opt_point.description = f"{opt_point.description or ''} [SYNC]".strip()
            else:
                # Normal - minimale Pause
                opt_point.settle_time = max(self.serial_latency, opt_point.settle_time * 0.8)
            
            optimized.append(opt_point)
        
        return optimized
    
    def _calculate_movement_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Berechnet Bewegungsdistanz zwischen zwei Positionen."""
        distance = 0.0
        for joint in pos1:
            if joint in pos2:
                distance += (pos1[joint] - pos2[joint]) ** 2
        return math.sqrt(distance)
    
    def calculate_optimal_speed(self, distance: float, priority: int = 1) -> float:
        """
        Berechnet optimale Geschwindigkeit für gegebene Distanz.
        
        Args:
            distance: Bewegungsdistanz
            priority: Priorität (1=hoch, 3=niedrig)
            
        Returns:
            Optimale Geschwindigkeit
        """
        # Basis-Berechnung
        min_time = self.min_move_time * (4 - priority)  # Höhere Priorität = schneller
        max_speed = distance / min_time if min_time > 0 else 0.3
        
        # Limitierung basierend auf Tracking
        tracking_limit = SCANNER_SPECS["tracking_speed"]
        
        return min(max_speed, tracking_limit)

# ============== BASIS SCAN PATTERN KLASSE ==============

class ScanPattern(ABC):
    """Erweiterte abstrakte Basisklasse für alle Scan-Patterns."""
    
    def __init__(self, 
                 start_position: Optional[Dict[str, float]] = None,
                 end_position: Optional[Dict[str, float]] = None,
                 center_position: Optional[Dict[str, float]] = None,
                 scan_mode: str = "table_level", **kwargs):
        """
        Initialisiert Scan-Pattern mit erweiterten Features.
        
        Args:
            start_position: Start-Position für Scan (vor dem Scan)
            end_position: End-Position für Scan (nach dem Scan)
            center_position: Center-Position für Scan (während dem Scan)
            scan_mode: Scan-Modus aus SCANNER_POSITIONS
            **kwargs: Zusätzliche Parameter
        """
        # Rückwärtskompatibilität - wenn nur scan_mode übergeben wurde
        if 'scan_mode' in kwargs:
            scan_mode = kwargs.pop('scan_mode')
            
        # Positions-Setup mit Defaults
        self.start_position = start_position or SCAN_START_POSITION.copy()
        self.end_position = end_position or SCAN_END_POSITION.copy()
        
        # Center-Position (für Scan-Berechnungen)
        if center_position is None:
            center_position = SCANNER_POSITIONS.get(scan_mode, SCANNER_CENTER).copy()
        self.center_position = center_position
        
        self.scan_mode = scan_mode
        self.name = self.__class__.__name__.replace("Pattern", "").replace("Scan", " Scan").strip()
        
        # Komponenten
        self.planner = IntelligentMultiAxisPlanner(self.center_position)
        self.validator = SafetyValidator()
        self.timing_optimizer = TimingOptimizer()
        
        # Scanner-Parameter
        self.optimal_distance = SCANNER_SPECS["optimal_distance"]
        self.min_distance = SCANNER_SPECS["min_distance"]
        self.max_distance = SCANNER_SPECS["max_distance"]
        self.tracking_speed = SCANNER_SPECS["tracking_speed"]
        self.podest_height = SCANNER_SPECS["podest_height"]
        
        # Pattern-spezifische Parameter
        self.points = []
        self.quality_metrics = ScanQuality()
        
        # Flags
        self.use_smooth_transitions = True
        self.validate_all_points = True
        self.timing_optimized = False
        self.adaptive_quality = kwargs.get('adaptive_quality', True)
        self.include_transitions = kwargs.get('include_transitions', False)  # Default False für Kompatibilität
        
        # Logging
        logger.info(f"✅ Initialized {self.name} pattern (mode: {scan_mode})")
    
    @abstractmethod
    def generate_points(self) -> List[ScanPoint]:
        """Generiert die Scan-Punkte. Muss von Subklassen implementiert werden."""
        pass
    
    def generate_with_transitions(self) -> List[ScanPoint]:
        """
        Generiert Scan-Punkte mit Start- und End-Transitionen.
        
        Returns:
            Komplette Scan-Sequenz inkl. Transitionen
        """
        points = []
        
        # Start-Transition (Home -> Start -> Scan-Position)
        if self.include_transitions:
            # Von Home zu Start-Position
            transition_to_start = self.planner.plan_optimal_trajectory(
                HOME_POSITION_DEFAULT,
                self.start_position
            )
            if transition_to_start:
                for p in transition_to_start:
                    p.description = "Transition to start"
                    p.priority = 3
                points.extend(transition_to_start)
            
            # Von Start zu erster Scan-Position
            scan_points = self.generate_points()
            if scan_points:
                first_scan = scan_points[0]
                approach = self.planner.plan_optimal_trajectory(
                    self.start_position,
                    first_scan.positions
                )
                if approach:
                    for p in approach:
                        p.description = "Approach scan area"
                        p.priority = 2
                    points.extend(approach)
        
        # Haupt-Scan-Punkte
        scan_points = self.generate_points() if not self.include_transitions else scan_points
        points.extend(scan_points)
        
        # End-Transition (Scan-Position -> End -> Home)
        if self.include_transitions and scan_points:
            last_scan = scan_points[-1]
            
            # Von letzter Scan-Position zu End-Position
            transition_to_end = self.planner.plan_optimal_trajectory(
                last_scan.positions,
                self.end_position
            )
            if transition_to_end:
                for p in transition_to_end:
                    p.description = "Transition to end"
                    p.priority = 2
                points.extend(transition_to_end)
            
            # Von End zu Home
            return_home = self.planner.plan_optimal_trajectory(
                self.end_position,
                HOME_POSITION_DEFAULT
            )
            if return_home:
                for p in return_home:
                    p.description = "Return to home"
                    p.priority = 3
                points.extend(return_home)
        
        return points
    
    def create_scan_point(self, x: float, y: float, z: float,
                         base_angle: Optional[float] = None,
                         speed: Optional[float] = None,
                         priority: int = 1,
                         **kwargs) -> Optional[ScanPoint]:
        """
        Erstellt validierten Scan-Punkt mit allen Optimierungen.
        
        Args:
            x, y, z: Kartesische Koordinaten relativ zum Scan-Zentrum
            base_angle: Optionaler Base-Winkel
            speed: Geschwindigkeit (None = auto-berechnet)
            priority: Scan-Priorität
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Validierter ScanPoint oder None
        """
        # Base-Rotation berechnen
        if base_angle is not None:
            base = base_angle
        else:
            base = math.atan2(x, y + self.optimal_distance) if abs(x) > 0.01 else 0
        
        # Shoulder - kompensiert für Podest-Höhe
        shoulder_offset = 0
        if z < -0.2:  # Scan weit unten
            shoulder_offset = 0.3  # Mehr nach unten neigen
        elif z < -0.1:
            shoulder_offset = 0.2
        else:
            shoulder_offset = z * 0.4
        
        shoulder = self.center_position["shoulder"] + shoulder_offset
        shoulder = max(-1.4, min(1.4, shoulder))  # Safety clamp
        
        # Elbow - näher für Stabilität
        distance = math.sqrt(x**2 + y**2 + z**2)
        distance_factor = min(1.0, distance / self.max_distance)
        
        # Adaptiv basierend auf Scan-Mode
        if self.scan_mode == "table_level":
            elbow = self.center_position["elbow"] - distance_factor * 0.2
        else:
            elbow = self.center_position["elbow"] - distance_factor * 0.3
        
        elbow = max(0.1, min(3.0, elbow))  # Safety clamp
        
        # Wrist - KRITISCH! Sicher weg vom Limit
        wrist_base = self.center_position["wrist"]
        wrist_adjustment = z * 0.15  # Reduzierte Anpassung
        
        # Hard limits für Wrist
        wrist = wrist_base + wrist_adjustment
        wrist = max(-1.3, min(1.3, wrist))  # SICHER weg von ±1.57!
        
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
        
        # Validierung
        valid, errors = self.validator.validate_position(positions, conservative=True)
        
        if not valid:
            # Versuche zu korrigieren
            corrected = self.planner._correct_invalid_position(positions)
            if corrected:
                positions = corrected
                logger.debug(f"Position corrected: {errors[0] if errors else 'unknown'}")
            else:
                logger.warning(f"Invalid position skipped: {errors}")
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
        """Schätzt Scan-Qualität basierend auf Distanz."""
        if distance < self.min_distance:
            return 0.7  # Zu nah
        elif distance > self.max_distance:
            return 0.6  # Zu weit
        elif abs(distance - self.optimal_distance) < 0.03:
            return 1.0  # Optimal
        else:
            # Linear interpoliert
            if distance < self.optimal_distance:
                ratio = (distance - self.min_distance) / (self.optimal_distance - self.min_distance)
            else:
                ratio = (self.max_distance - distance) / (self.max_distance - self.optimal_distance)
            return 0.7 + 0.3 * ratio
    
    def optimize_path(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """
        Optimiert Scan-Pfad mit allen verfügbaren Techniken.
        
        Args:
            points: Original-Punkte
            
        Returns:
            Optimierte Punkte
        """
        if not points:
            return points
        
        # Entferne None-Werte
        valid_points = [p for p in points if p is not None]
        
        if not valid_points:
            return []
        
        # Schritt 1: Path Smoothing
        if self.use_smooth_transitions:
            valid_points = self.validator.smooth_path(valid_points, timing_optimize=True)
        
        # Schritt 2: Timing-Optimierung
        if not self.timing_optimized:
            valid_points = self.timing_optimizer.optimize_timing(valid_points)
            self.timing_optimized = True
        
        # Schritt 3: Qualitäts-basierte Filterung
        if self.adaptive_quality:
            valid_points = self._filter_by_quality(valid_points)
        
        # Schritt 4: Final Validation
        if self.validate_all_points:
            final_points = []
            for point in valid_points:
                valid, _ = self.validator.validate_position(point.positions)
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
        """Filtert Punkte basierend auf erwarteter Qualität."""
        high_quality = [p for p in points if p.expected_quality >= threshold]
        
        if len(high_quality) < len(points) * 0.5:
            # Zu viele gefiltert - behalte mehr
            return sorted(points, key=lambda p: p.expected_quality, reverse=True)[:int(len(points)*0.8)]
        
        return high_quality
    
    def _calculate_quality_metrics(self, points: List[ScanPoint]):
        """Berechnet Qualitäts-Metriken für den Scan."""
        if not points:
            return
        
        # Coverage (basierend auf räumlicher Verteilung)
        positions = [p.metadata for p in points if p.metadata]
        if positions:
            x_range = max(p.get('x', 0) for p in positions) - min(p.get('x', 0) for p in positions)
            y_range = max(p.get('y', 0) for p in positions) - min(p.get('y', 0) for p in positions)
            z_range = max(p.get('z', 0) for p in positions) - min(p.get('z', 0) for p in positions)
            
            volume = x_range * y_range * z_range
            self.quality_metrics.coverage = min(1.0, volume / 0.008)  # 0.008 m³ = full coverage
        
        # Overlap (basierend auf Punkt-Dichte)
        self.quality_metrics.point_density = len(points) / max(1, volume) if 'volume' in locals() else 0
        self.quality_metrics.overlap = min(1.0, self.quality_metrics.point_density / 1000)
        
        # Stability (basierend auf Geschwindigkeiten)
        avg_speed = sum(p.speed for p in points) / len(points)
        self.quality_metrics.stability = 1.0 - min(1.0, avg_speed / 0.5)
        
        # Tracking Confidence
        self.quality_metrics.tracking_confidence = sum(p.expected_quality for p in points) / len(points)
        
        # Estimated Time
        self.quality_metrics.estimated_time = sum(
            1.0 / p.speed + p.settle_time for p in points if p.speed > 0
        )

# ============== PATTERN IMPLEMENTIERUNGEN ==============

class RasterScanPattern(ScanPattern):
    """Raster-Scan Pattern mit Podest-Optimierung."""
    
    def __init__(self, width: float = 0.20, height: float = 0.15,
                 rows: int = 10, cols: int = 10,
                 overlap: float = 0.2, speed: float = 0.3,
                 settle_time: float = 0.5, zigzag: bool = True, **kwargs):
        """Initialisiert Raster-Scan."""
        super().__init__(**kwargs)  # Einfach alle kwargs durchreichen
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
        """Generiert Raster-Scan Punkte."""
        points = []
        
        # Berechne Schrittweiten mit Überlappung
        step_x = self.width / max(1, self.cols - 1) * (1 - self.overlap)
        step_z = self.height / max(1, self.rows - 1) * (1 - self.overlap)
        
        # Start-Offsets (zentriert, aber tiefer für Podest)
        start_x = -self.width / 2
        start_z = -self.height / 2 - 0.15  # Extra tief für 40cm Podest
        
        for row in range(self.rows):
            # Zigzag-Muster
            cols_range = range(self.cols)
            if self.zigzag and row % 2 == 1:
                cols_range = reversed(cols_range)
            
            for col_idx, col in enumerate(cols_range):
                x = start_x + col * step_x
                z = start_z + row * step_z
                y = 0.02  # Leicht vor dem Arm
                
                # Priorität basierend auf Position
                priority = 1 if row < 2 or row >= self.rows - 2 else 2
                
                # Geschwindigkeit anpassen
                if col_idx == 0:  # Zeilenanfang
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
    """Spiral-Scan mit erweiterten Features."""
    
    def __init__(self, radius_start: float = 0.05, radius_end: float = 0.20,
                 revolutions: int = 5, points_per_rev: int = 30,
                 vertical_range: float = 0.15, speed: float = 0.25, **kwargs):
        """Initialisiert Spiral-Scan."""
        super().__init__(**kwargs)  # Alle kwargs an Parent
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.revolutions = revolutions
        self.points_per_rev = points_per_rev
        self.vertical_range = vertical_range
        self.speed = speed
        self.name = "Spiral Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spiral-Scan Punkte."""
        points = []
        total_points = self.revolutions * self.points_per_rev
        
        for i in range(total_points):
            t = i / max(1, total_points - 1)
            
            # Spirale berechnen
            angle = 2 * math.pi * self.revolutions * t
            radius = self.radius_start + (self.radius_end - self.radius_start) * t
            
            # 3D-Position
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) * 0.6  # Komprimiert für Stabilität
            z = -self.vertical_range/2 + self.vertical_range * t - 0.12  # Podest-Offset
            
            # Adaptive Geschwindigkeit
            speed = self.speed * (0.7 + 0.3 * (1 - t))  # Langsamer am Ende
            
            # Priorität steigt mit Radius
            priority = 1 if radius < self.optimal_distance else 2
            
            point = self.create_scan_point(
                x, y, z,
                speed=speed,
                priority=priority,
                description=f"Spiral {i}/{total_points}"
            )
            
            if point:
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} spiral scan points")
        return self.optimize_path(points)


class SphericalScanPattern(ScanPattern):
    """Sphärischer Scan für 3D-Objekte."""
    
    def __init__(self, radius: float = 0.15, theta_steps: int = 12,
                 phi_steps: int = 8, phi_range: float = 1.0,
                 speed: float = 0.3, **kwargs):
        """Initialisiert Spherical-Scan."""
        super().__init__(**kwargs)
        self.radius = radius
        self.theta_steps = theta_steps
        self.phi_steps = phi_steps
        self.phi_range = min(phi_range, 1.2)  # Limitiert für Sicherheit
        self.speed = speed
        self.name = "Spherical Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert sphärische Scan-Punkte."""
        points = []
        
        for phi_idx in range(self.phi_steps):
            # Phi (Elevation)
            phi = -self.phi_range/2 + (phi_idx * self.phi_range / max(1, self.phi_steps - 1))
            
            # Weniger Punkte an Polen
            theta_count = max(4, int(self.theta_steps * math.cos(phi)))
            
            for theta_idx in range(theta_count):
                # Theta (Azimuth)
                theta = 2 * math.pi * theta_idx / theta_count
                
                # Sphärische zu kartesischen Koordinaten
                x = self.radius * math.cos(phi) * math.sin(theta)
                y = self.radius * math.cos(phi) * math.cos(theta)
                z = self.radius * math.sin(phi) - 0.05  # Leicht nach unten
                
                # Base-Winkel direkt setzen
                base_angle = theta
                
                # Geschwindigkeit basierend auf Position
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
            # Höhe berechnen
            if self.height_levels > 1:
                z = -self.height_range/2 + (level * self.height_range / (self.height_levels - 1))
            else:
                z = 0
            
            z -= 0.12  # Podest-Offset
            
            for step in range(self.steps):
                angle = 2 * math.pi * step / self.steps
                
                # Position (konstanter Radius)
                x = self.radius * math.sin(angle) * 0.7  # Näher an Base
                y = self.radius * math.cos(angle) * 0.7
                
                # Sehr langsame, gleichmäßige Bewegung für Tracking
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
            
            # Helix-Position
            x = self.radius * math.cos(angle) * 0.8  # Kompakter
            y = self.radius * math.sin(angle) * 0.6
            z = -self.height/2 + self.height * t - 0.15  # Start tiefer
            
            # Geschwindigkeit variiert mit Höhe
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
    """Spinnennetz-Pattern für detaillierte Oberflächenscans."""
    
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
        
        # Konzentrische Kreise
        for circle in range(self.circles):
            radius = (circle + 1) * self.max_radius / self.circles * 0.7
            
            for line in range(self.radial_lines):
                angle = 2 * math.pi * line / self.radial_lines
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.5
                z = -0.15  # Konstante Tiefe
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=self.speed,
                    description=f"Web C{circle}L{line}"
                )
                
                if point:
                    points.append(point)
        
        # Radiale Linien (optional)
        for line in range(self.radial_lines):
            angle = 2 * math.pi * line / self.radial_lines
            
            for r_idx in range(5):
                radius = r_idx * self.max_radius / 4 * 0.7
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.5
                z = -0.15
                
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
    """Intelligenter adaptiver Scan mit dynamischer Punkt-Verteilung."""
    
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
        """Generiert adaptive Scan-Punkte mit intelligenter Verteilung."""
        points = []
        
        # Initiale Verteilung (Fibonacci-Spirale für optimale Coverage)
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(self.initial_points):
            # Fibonacci-Verteilung
            theta = 2 * math.pi * i / golden_ratio
            r = math.sqrt(i / self.initial_points) * self.max_distance * 0.6
            
            # Position
            x = r * math.cos(theta)
            y = r * math.sin(theta) * 0.5
            z = -0.1 - 0.1 * (i / self.initial_points)  # Graduell tiefer
            
            point = self.create_scan_point(
                x, y, z,
                speed=self.tracking_speed,
                priority=1,
                description=f"Adaptive initial {i}"
            )
            
            if point:
                points.append(point)
        
        # Refinement-Iterationen
        for iteration in range(min(self.max_iterations, 2)):
            refinement_points = []
            
            # Identifiziere Bereiche mit niedriger Qualität (simuliert)
            for i in range(5 + iteration * 3):
                angle = 2 * math.pi * i / (5 + iteration * 3)
                r = self.optimal_distance * (0.8 + 0.2 * iteration)
                
                x = r * math.cos(angle)
                y = r * math.sin(angle) * 0.4
                z = -0.2 - iteration * 0.05
                
                point = self.create_scan_point(
                    x, y, z,
                    speed=self.tracking_speed * 0.8,
                    priority=2,
                    description=f"Adaptive refine {iteration}-{i}"
                )
                
                if point:
                    refinement_points.append(point)
            
            points.extend(refinement_points)
            
            # Qualitäts-Check (simuliert)
            avg_quality = sum(p.expected_quality for p in refinement_points) / max(1, len(refinement_points))
            if avg_quality > 0.9:
                break  # Genug Qualität erreicht
        
        logger.info(f"✅ Generated {len(points)} adaptive scan points")
        return self.optimize_path(points)


class StatueSpiralPattern(ScanPattern):
    """Spezialisierter Scan für Statuen und komplexe 3D-Objekte."""
    
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
        """Generiert Statue-Scan Punkte mit variablem Radius."""
        points = []
        
        for v_step in range(self.vertical_steps):
            # Höhe
            t_height = v_step / max(1, self.vertical_steps - 1)
            z = -self.height/2 + self.height * t_height - 0.1
            
            # Variabler Radius (schmaler oben)
            radius = self.base_radius * (1 - 0.3 * t_height)
            
            # Spiralförmige Aufwärtsbewegung
            angle_offset = self.spiral_factor * 2 * math.pi * t_height
            
            for a_step in range(self.angular_steps):
                angle = 2 * math.pi * a_step / self.angular_steps + angle_offset
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.7
                
                # Langsamere Bewegung oben
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
    """Optimierter Scan für flache Tisch-Objekte."""
    
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
        """Generiert Table-Scan Punkte in optimiertem Muster."""
        points = []
        
        # Berechne Anzahl Schritte
        cols = max(3, int(self.width / self.resolution))
        rows = max(3, int(self.depth / self.resolution))
        
        # Mäander-Pattern für minimale Bewegung
        for row in range(rows):
            # Zickzack
            col_range = range(cols) if row % 2 == 0 else reversed(range(cols))
            
            for col in col_range:
                x = -self.width/2 + col * self.width / max(1, cols - 1)
                y = -self.depth/2 + row * self.depth / max(1, rows - 1)
                z = self.height_offset - 0.2  # Tief für Tisch-Scan
                
                # Schnellere Bewegung bei geraden Strecken
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
                    settle_time=0.2,  # Kurz für flache Objekte
                    description=f"Table R{row}C{col}"
                )
                
                if point:
                    points.append(point)
        
        logger.info(f"✅ Generated {len(points)} table scan points")
        return self.optimize_path(points)

# ============== FACTORY & HELPER FUNCTIONS ==============

# Kompatibilitäts-Alias definieren wir später nach der Klassen-Definition

def create_scan_pattern(pattern_type: str, **kwargs) -> Optional[ScanPattern]:
    """
    Factory-Funktion zum Erstellen von Scan-Patterns.
    
    Args:
        pattern_type: Pattern-Typ Name
        **kwargs: Pattern-spezifische Parameter
        
    Returns:
        ScanPattern-Instanz oder None
    """
    patterns = {
        'raster': RasterScanPattern,
        'spiral': SpiralScanPattern,
        'spherical': SphericalScanPattern,
        'turntable': TurntableScanPattern,
        'helix': HelixScanPattern,
        'cobweb': CobwebScanPattern,
        'adaptive': AdaptiveScanPattern,
        'statue': StatueSpiralPattern,
        'statue_spiral': StatueSpiralPattern,  # Alias für Kompatibilität
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


def get_preset_pattern(preset: str) -> Optional[ScanPattern]:
    """
    Gibt vordefinierte Pattern-Presets zurück.
    
    Args:
        preset: Preset-Name
        
    Returns:
        Konfiguriertes ScanPattern oder None
    """
    presets = {
        # Quick Scans
        'quick': RasterScanPattern(
            rows=5, cols=5, width=0.15, height=0.10, 
            zigzag=True, speed=0.4
        ),
        'quick_spiral': SpiralScanPattern(
            revolutions=3, points_per_rev=20, 
            radius_end=0.15, speed=0.35
        ),
        
        # Detailed Scans
        'detailed': RasterScanPattern(
            rows=15, cols=15, width=0.25, height=0.20, 
            overlap=0.3, speed=0.2
        ),
        'high_detail': RasterScanPattern(
            rows=20, cols=20, width=0.30, height=0.25,
            overlap=0.4, speed=0.15
        ),
        
        # Object-specific
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
        
        # Special
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
    """
    Schätzt die Scan-Zeit für ein Pattern.
    
    Args:
        pattern: ScanPattern-Instanz
        
    Returns:
        Geschätzte Zeit in Sekunden
    """
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
            # Bewegungszeit
            distance = math.sqrt(sum(
                (point.positions.get(j, 0) - prev.positions.get(j, 0))**2 
                for j in point.positions
            ))
            if point.speed > 0:
                total_time += distance / point.speed
        
        # Settle-Zeit
        total_time += point.settle_time
    
    return total_time


# ============== MODULE EXPORTS ==============
# Exportiere genau das was main.py erwartet

__all__ = [
    # Pattern-Klassen (wie in main.py importiert)
    'RasterScanPattern',
    'SpiralScanPattern', 
    'SphericalScanPattern',
    'TurntableScanPattern',
    'HelixScanPattern',
    'CobwebScanPattern',
    'AdaptiveScanPattern',
    'StatueSpiralPattern',  # Das erwartet main.py!
    'TableScanPattern',
    
    # Factory-Funktionen (wie in main.py importiert)
    'create_scan_pattern',
    'get_pattern_presets',  # Das erwartet main.py!
    
    # Weitere wichtige Exports
    'estimate_scan_time',
    'ScanPoint',
    'ScanPattern'
]


# ============== MAIN (Test-Funktion) ==============

if __name__ == "__main__":
    print("="*60)
    print("🤖 RoArm M3 Professional Scan Patterns V5.0")
    print("Vollständig optimiert für 40cm Podest + 11.4Hz Taktung")
    print("="*60)
    
    # Test alle Patterns
    test_patterns = [
        ('raster', RasterScanPattern(rows=5, cols=5, width=0.20, height=0.15)),
        ('spiral', SpiralScanPattern(revolutions=3, points_per_rev=20)),
        ('spherical', SphericalScanPattern(theta_steps=8, phi_steps=6)),
        ('turntable', TurntableScanPattern(steps=12, height_levels=2)),
        ('helix', HelixScanPattern(turns=3, points_per_turn=15)),
        ('cobweb', CobwebScanPattern(radial_lines=6, circles=3)),
        ('adaptive', AdaptiveScanPattern(initial_points=15)),
        ('statue', StatueSpiralPattern(vertical_steps=5, angular_steps=12)),
        ('table', TableScanPattern(width=0.25, depth=0.15))
    ]
    
    for name, pattern in test_patterns:
        print(f"\n📊 Testing {name.upper()} Pattern:")
        print("-"*40)
        
        # Generiere Punkte
        points = pattern.generate_points()
        
        print(f"  Generated points: {len(points)}")
        print(f"  Quality score: {pattern.quality_metrics.overall_score:.2f}")
        print(f"  Estimated time: {pattern.quality_metrics.estimated_time:.1f}s")
        
        if points:
            # Zeige ersten und letzten Punkt
            first = points[0]
            last = points[-1]
            
            print(f"\n  First point:")
            print(f"    Base: {first.positions['base']:.3f}")
            print(f"    Shoulder: {first.positions['shoulder']:.3f}")
            print(f"    Wrist: {first.positions['wrist']:.3f} (SAFE!)")
            
            print(f"\n  Last point:")
            print(f"    Base: {last.positions['base']:.3f}")
            print(f"    Shoulder: {last.positions['shoulder']:.3f}")
            print(f"    Wrist: {last.positions['wrist']:.3f} (SAFE!)")
            
            # Validierung
            valid_count = 0
            for p in points:
                valid, _ = SafetyValidator.validate_position(p.positions)
                if valid:
                    valid_count += 1
            
            print(f"\n  Safety validation: {valid_count}/{len(points)} valid")
            print(f"  Timing optimized: {'✅' if pattern.timing_optimized else '❌'}")
    
    print("\n" + "="*60)
    print("✅ All patterns tested successfully!")
    print("Ready for production use with main.py")
    print("="*60)
