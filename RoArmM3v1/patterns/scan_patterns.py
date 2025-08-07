#!/usr/bin/env python3
"""
RoArm M3 Scan Patterns für Revopoint Mini2
Optimierte Bewegungsmuster für 3D-Scanning.
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.constants import SCAN_DEFAULTS, SCANNER_SPECS
from motion.trajectory import TrajectoryType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScanPoint:
    """Ein Punkt im Scan-Pattern."""
    positions: Dict[str, float]  # Joint-Positionen
    speed: float = 0.3           # Bewegungsgeschwindigkeit
    settle_time: float = 0.5     # Wartezeit für Scanner-Stabilität
    trajectory_type: TrajectoryType = TrajectoryType.S_CURVE
    scan_angle: Optional[float] = None  # Winkel relativ zum Objekt
    distance: Optional[float] = None    # Abstand zum Objekt


class ScanPattern(ABC):
    """Abstrakte Basisklasse für Scan-Patterns."""
    
    def __init__(self, center_position: Optional[Dict[str, float]] = None):
        """
        Initialisiert das Scan-Pattern.
        
        Args:
            center_position: Zentrumsposition für den Scan
        """
        self.center_position = center_position or {
            "base": 0.0,
            "shoulder": 0.35,
            "elbow": 1.22,
            "wrist": -1.57,
            "roll": 1.57,
            "hand": 2.5
        }
        self.name = self.__class__.__name__
        self.points = []
    
    @abstractmethod
    def generate_points(self) -> List[ScanPoint]:
        """Generiert die Scan-Punkte."""
        pass
    
    def optimize_path(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """
        Optimiert die Reihenfolge der Scan-Punkte für minimale Bewegung.
        
        Args:
            points: Unoptimierte Punkte
            
        Returns:
            Optimierte Punktliste
        """
        if len(points) <= 2:
            return points
        
        # Greedy nearest-neighbor für einfache Optimierung
        optimized = [points[0]]
        remaining = points[1:]
        
        while remaining:
            current = optimized[-1]
            nearest_idx = self._find_nearest(current, remaining)
            optimized.append(remaining.pop(nearest_idx))
        
        return optimized
    
    def _find_nearest(self, current: ScanPoint, points: List[ScanPoint]) -> int:
        """Findet den nächsten Punkt."""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, point in enumerate(points):
            dist = self._calculate_distance(current.positions, point.positions)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Berechnet die Distanz zwischen zwei Positionen."""
        dist = 0.0
        for joint in pos1:
            if joint in pos2:
                dist += (pos1[joint] - pos2[joint]) ** 2
        return math.sqrt(dist)


class RasterScanPattern(ScanPattern):
    """
    Raster-Scan (Zeilen-basiert).
    Optimal für flache oder rechteckige Objekte.
    """
    
    def __init__(self, width: float = 0.20, height: float = 0.15,
                 rows: int = 10, cols: int = 10,
                 overlap: float = 0.2,
                 speed: float = 0.3,
                 settle_time: float = 0.5,
                 zigzag: bool = True,
                 **kwargs):
        """
        Initialisiert Raster-Scan.
        
        Args:
            width: Scan-Breite in Metern
            height: Scan-Höhe in Metern
            rows: Anzahl Zeilen
            cols: Anzahl Spalten
            overlap: Überlappung zwischen Scans (0.0-1.0)
            speed: Bewegungsgeschwindigkeit
            settle_time: Wartezeit pro Position
            zigzag: Zickzack-Muster statt immer links anfangen
        """
        super().__init__(**kwargs)
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
        step_x = self.width / (self.cols - 1) * (1 - self.overlap)
        step_y = self.height / (self.rows - 1) * (1 - self.overlap)
        
        # Start-Offsets (zentriert um center_position)
        start_x = -self.width / 2
        start_y = -self.height / 2
        
        for row in range(self.rows):
            # Zigzag: Jede zweite Zeile rückwärts
            cols_range = range(self.cols)
            if self.zigzag and row % 2 == 1:
                cols_range = reversed(cols_range)
            
            for col in cols_range:
                # Berechne Position relativ zum Zentrum
                x_offset = start_x + col * step_x
                y_offset = start_y + row * step_y
                
                # Konvertiere zu Joint-Positionen
                positions = self._calculate_joint_positions(x_offset, y_offset, 0)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.LINEAR if col > 0 else TrajectoryType.S_CURVE
                )
                points.append(point)
        
        logger.info(f"Generated {len(points)} raster scan points")
        return points
    
    def _calculate_joint_positions(self, x: float, y: float, z: float) -> Dict[str, float]:
        """
        Konvertiert kartesische Offsets zu Joint-Positionen.
        Vereinfachte Implementierung - kann durch inverse Kinematik ersetzt werden.
        """
        positions = self.center_position.copy()
        
        # Einfache Mapping (sollte durch richtige IK ersetzt werden)
        positions["base"] += math.atan2(x, SCANNER_SPECS["optimal_distance"])
        positions["shoulder"] += y / 0.3  # Skalierung
        positions["elbow"] += z / 0.3
        
        return positions


class SpiralScanPattern(ScanPattern):
    """
    Spiral-Scan.
    Optimal für runde oder zylindrische Objekte.
    """
    
    def __init__(self, radius_start: float = 0.05, radius_end: float = 0.15,
                 revolutions: int = 5, points_per_rev: int = 36,
                 height_range: float = 0.1,
                 speed: float = 0.25,
                 settle_time: float = 0.3,
                 continuous: bool = True,
                 **kwargs):
        """
        Initialisiert Spiral-Scan.
        
        Args:
            radius_start: Start-Radius in Metern
            radius_end: End-Radius in Metern
            revolutions: Anzahl Umdrehungen
            points_per_rev: Punkte pro Umdrehung
            height_range: Höhenvariation während Spirale
            speed: Bewegungsgeschwindigkeit
            settle_time: Wartezeit (0 für kontinuierlich)
            continuous: Kontinuierliche Bewegung ohne Stops
        """
        super().__init__(**kwargs)
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.revolutions = revolutions
        self.points_per_rev = points_per_rev
        self.height_range = height_range
        self.speed = speed
        self.settle_time = 0.0 if continuous else settle_time
        self.continuous = continuous
        self.name = "Spiral Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spiral-Scan Punkte."""
        points = []
        total_points = self.revolutions * self.points_per_rev
        
        for i in range(total_points):
            # Fortschritt (0.0 bis 1.0)
            t = i / (total_points - 1)
            
            # Spirale
            angle = 2 * math.pi * self.revolutions * t
            radius = self.radius_start + (self.radius_end - self.radius_start) * t
            
            # Position
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = -self.height_range / 2 + self.height_range * t
            
            # Konvertiere zu Joint-Positionen
            positions = self._calculate_spiral_positions(x, y, z, angle)
            
            point = ScanPoint(
                positions=positions,
                speed=self.speed,
                settle_time=self.settle_time,
                trajectory_type=TrajectoryType.LINEAR if self.continuous else TrajectoryType.S_CURVE,
                scan_angle=angle,
                distance=radius
            )
            points.append(point)
        
        logger.info(f"Generated {len(points)} spiral scan points")
        return points
    
    def _calculate_spiral_positions(self, x: float, y: float, z: float, angle: float) -> Dict[str, float]:
        """Berechnet Joint-Positionen für Spiralpunkt."""
        positions = self.center_position.copy()
        
        # Basis rotiert um das Objekt
        positions["base"] = angle
        
        # Schulter und Ellbogen für Radius
        distance = math.sqrt(x**2 + y**2)
        positions["shoulder"] += (distance - SCANNER_SPECS["optimal_distance"]) * 2
        
        # Höhenanpassung
        positions["elbow"] += z * 2
        
        # Handgelenk bleibt level
        positions["wrist"] = -1.57 - positions["shoulder"]
        
        return positions


class SphericalScanPattern(ScanPattern):
    """
    Sphärischer Scan.
    Optimal für komplexe 3D-Objekte von allen Seiten.
    """
    
    def __init__(self, radius: float = 0.15,
                 theta_steps: int = 12,  # Horizontal
                 phi_steps: int = 8,      # Vertikal
                 phi_range: Tuple[float, float] = (-60, 60),
                 speed: float = 0.3,
                 settle_time: float = 0.7,
                 **kwargs):
        """
        Initialisiert sphärischen Scan.
        
        Args:
            radius: Scan-Radius in Metern
            theta_steps: Horizontale Schritte (360°)
            phi_steps: Vertikale Schritte
            phi_range: Vertikaler Bereich in Grad
            speed: Bewegungsgeschwindigkeit
            settle_time: Wartezeit pro Position
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.theta_steps = theta_steps
        self.phi_steps = phi_steps
        self.phi_range = (math.radians(phi_range[0]), math.radians(phi_range[1]))
        self.speed = speed
        self.settle_time = settle_time
        self.name = "Spherical Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert sphärische Scan-Punkte."""
        points = []
        
        # Winkelschritte
        theta_step = 2 * math.pi / self.theta_steps
        phi_step = (self.phi_range[1] - self.phi_range[0]) / (self.phi_steps - 1)
        
        for phi_idx in range(self.phi_steps):
            phi = self.phi_range[0] + phi_idx * phi_step
            
            for theta_idx in range(self.theta_steps):
                theta = theta_idx * theta_step
                
                # Sphärische zu kartesische Koordinaten
                x = self.radius * math.cos(phi) * math.cos(theta)
                y = self.radius * math.cos(phi) * math.sin(theta)
                z = self.radius * math.sin(phi)
                
                # Konvertiere zu Joint-Positionen
                positions = self._calculate_spherical_positions(x, y, z, theta, phi)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.S_CURVE,
                    scan_angle=theta,
                    distance=self.radius
                )
                points.append(point)
        
        # Optimiere Pfad für minimale Bewegung
        points = self.optimize_path(points)
        
        logger.info(f"Generated {len(points)} spherical scan points")
        return points
    
    def _calculate_spherical_positions(self, x: float, y: float, z: float, 
                                      theta: float, phi: float) -> Dict[str, float]:
        """Berechnet Joint-Positionen für sphärischen Punkt."""
        positions = self.center_position.copy()
        
        # Basis für horizontale Rotation
        positions["base"] = theta
        
        # Schulter für vertikale Position
        positions["shoulder"] = self.center_position["shoulder"] + phi
        
        # Ellbogen für Distanz
        positions["elbow"] = self.center_position["elbow"] + (self.radius - SCANNER_SPECS["optimal_distance"]) * 3
        
        # Handgelenk kompensiert für Scanner-Ausrichtung
        positions["wrist"] = -1.57 - positions["shoulder"]
        
        return positions


class TurntableScanPattern(ScanPattern):
    """
    Drehtisch-Scan.
    Roboter dreht nur die Basis, Objekt wird von allen Seiten gescannt.
    """
    
    def __init__(self, steps: int = 36,
                 radius: float = 0.15,
                 height_levels: int = 1,
                 height_range: float = 0.1,
                 speed: float = 0.5,
                 settle_time: float = 1.0,
                 **kwargs):
        """
        Initialisiert Drehtisch-Scan.
        
        Args:
            steps: Anzahl Rotationsschritte
            radius: Abstand zum Objekt
            height_levels: Anzahl Höhenebenen
            height_range: Höhenbereich für mehrere Ebenen
            speed: Rotationsgeschwindigkeit
            settle_time: Wartezeit pro Position
        """
        super().__init__(**kwargs)
        self.steps = steps
        self.radius = radius
        self.height_levels = height_levels
        self.height_range = height_range
        self.speed = speed
        self.settle_time = settle_time
        self.name = "Turntable Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Drehtisch-Scan Punkte."""
        points = []
        
        angle_step = 2 * math.pi / self.steps
        
        if self.height_levels > 1:
            height_step = self.height_range / (self.height_levels - 1)
        else:
            height_step = 0
        
        for level in range(self.height_levels):
            # Höhe für diese Ebene
            if self.height_levels > 1:
                z_offset = -self.height_range / 2 + level * height_step
            else:
                z_offset = 0
            
            for step in range(self.steps):
                angle = step * angle_step
                
                # Nur Basis dreht sich
                positions = self.center_position.copy()
                positions["base"] = angle
                
                # Höhenanpassung falls mehrere Ebenen
                if z_offset != 0:
                    positions["shoulder"] += z_offset
                    positions["wrist"] = -1.57 - positions["shoulder"]
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.S_CURVE if step == 0 else TrajectoryType.LINEAR,
                    scan_angle=angle,
                    distance=self.radius
                )
                points.append(point)
        
        logger.info(f"Generated {len(points)} turntable scan points")
        return points


class AdaptiveScanPattern(ScanPattern):
    """
    Adaptiver Scan.
    Passt sich dynamisch an die Objektgeometrie an.
    """
    
    def __init__(self, initial_points: int = 20,
                 refinement_threshold: float = 0.05,
                 max_iterations: int = 3,
                 **kwargs):
        """
        Initialisiert adaptiven Scan.
        
        Args:
            initial_points: Initiale Anzahl Scan-Punkte
            refinement_threshold: Schwellwert für Verfeinerung
            max_iterations: Maximale Verfeinerungsiterationen
        """
        super().__init__(**kwargs)
        self.initial_points = initial_points
        self.refinement_threshold = refinement_threshold
        self.max_iterations = max_iterations
        self.name = "Adaptive Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """
        Generiert adaptive Scan-Punkte.
        Beginnt mit groben Scan und verfeinert basierend auf Geometrie.
        """
        # Starte mit sphärischem Grobscan
        coarse_scan = SphericalScanPattern(
            radius=SCANNER_SPECS["optimal_distance"],
            theta_steps=6,
            phi_steps=4,
            center_position=self.center_position
        )
        
        points = coarse_scan.generate_points()
        
        # Hier würde normalerweise eine Analyse der Scan-Daten erfolgen
        # und basierend darauf weitere Punkte hinzugefügt werden.
        # Da wir keine echten Scan-Daten haben, fügen wir beispielhaft
        # Verfeinerungspunkte hinzu.
        
        logger.info(f"Generated {len(points)} adaptive scan points")
        return points


class CobwebScanPattern(ScanPattern):
    """
    Spinnennetz-Scan.
    Radiale Linien kombiniert mit konzentrischen Kreisen.
    """
    
    def __init__(self, radial_lines: int = 8,
                 circles: int = 5,
                 max_radius: float = 0.15,
                 speed: float = 0.3,
                 **kwargs):
        """
        Initialisiert Cobweb-Scan.
        
        Args:
            radial_lines: Anzahl radialer Linien
            circles: Anzahl konzentrischer Kreise
            max_radius: Maximaler Radius
            speed: Bewegungsgeschwindigkeit
        """
        super().__init__(**kwargs)
        self.radial_lines = radial_lines
        self.circles = circles
        self.max_radius = max_radius
        self.speed = speed
        self.name = "Cobweb Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spinnennetz-Scan Punkte."""
        points = []
        
        # Zentrum
        points.append(ScanPoint(
            positions=self.center_position.copy(),
            speed=self.speed,
            settle_time=1.0
        ))
        
        # Konzentrische Kreise
        for circle in range(1, self.circles + 1):
            radius = self.max_radius * circle / self.circles
            
            for line in range(self.radial_lines):
                angle = 2 * math.pi * line / self.radial_lines
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                
                positions = self.center_position.copy()
                positions["base"] += angle
                positions["shoulder"] += (radius - SCANNER_SPECS["optimal_distance"]) * 2
                
                points.append(ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=0.5,
                    scan_angle=angle,
                    distance=radius
                ))
        
        logger.info(f"Generated {len(points)} cobweb scan points")
        return points
