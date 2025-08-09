#!/usr/bin/env python3
"""
RoArm M3 - Comprehensive Scan Patterns
Professional 3D Scanning Patterns for RoArm M3
Version 3.1.0 - Complete Implementation
"""

import math
import time
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import motion trajectory if available
try:
    from motion.trajectory import TrajectoryGenerator, TrajectoryType, TrajectoryPoint
except ImportError:
    # Fallback definitions
    class TrajectoryType(Enum):
        LINEAR = "linear"
        S_CURVE = "s_curve"
        SPLINE = "spline"
    
    TrajectoryGenerator = None
    TrajectoryPoint = None

logger = logging.getLogger(__name__)

# ================================
# BASE CLASSES
# ================================

@dataclass
class ScanPoint:
    """Einzelner Scan-Punkt mit allen Parametern."""
    positions: Tuple[float, float, float, float, float, float]  # base, shoulder, elbow, wrist, hand, head
    speed: float = 0.3
    settle_time: float = 0.5
    trajectory_type: TrajectoryType = TrajectoryType.LINEAR
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ScanPattern:
    """
    Basisklasse für alle Scan-Patterns.
    Definiert das Interface für Pattern-Generierung.
    """
    
    def __init__(self, 
                 center_position: Tuple[float, float, float] = (0.0, 0.15, 0.0),
                 optimal_distance: float = 0.15,
                 speed: float = 0.3,
                 settle_time: float = 0.5,
                 name: str = "Generic Scan"):
        """
        Initialisiert Basis-Pattern.
        
        Args:
            center_position: Zentrum des Scan-Bereichs (x, y, z)
            optimal_distance: Optimaler Abstand zum Objekt
            speed: Standard-Bewegungsgeschwindigkeit
            settle_time: Standard-Wartezeit pro Position
            name: Name des Patterns
        """
        self.center_position = center_position
        self.optimal_distance = optimal_distance
        self.speed = speed
        self.settle_time = settle_time
        self.name = name
        self.points: List[ScanPoint] = []
        
    def generate_points(self) -> List[ScanPoint]:
        """
        Generiert die Scan-Punkte für dieses Pattern.
        Muss von Subklassen implementiert werden.
        """
        raise NotImplementedError("Subclasses must implement generate_points()")
    
    def get_points(self) -> List[ScanPoint]:
        """Gibt die generierten Punkte zurück, generiert sie falls nötig."""
        if not self.points:
            self.points = self.generate_points()
        return self.points
    
    def _calculate_joint_positions(self, x_offset: float, y_offset: float, z_offset: float) -> Tuple[float, float, float, float, float, float]:
        """
        Berechnet Joint-Positionen für einen Offset vom Zentrum.
        Vereinfachte Inverse Kinematik für Scan-Positionen.
        """
        # Basis-Position + Offsets
        target_x = self.center_position[0] + x_offset
        target_y = self.center_position[1] + y_offset
        target_z = self.center_position[2] + z_offset
        
        # Vereinfachte inverse Kinematik
        # Base rotation
        base = math.atan2(target_y, target_x)
        
        # Distance to target
        distance = math.sqrt(target_x**2 + target_y**2)
        
        # Shoulder and elbow (vereinfacht)
        # Für Scanner-Position zeigen wir meist nach unten
        shoulder = math.radians(45)  # 45° nach unten
        elbow = math.radians(-90)    # 90° Beugung
        
        # Wrist zeigt Scanner nach unten
        wrist = math.radians(-45)
        
        # Hand bleibt meist neutral
        hand = 0.0
        
        # Head kann für bessere Sicht adjustiert werden
        head = base  # Folgt der Base-Rotation
        
        return (base, shoulder, elbow, wrist, hand, head)
    
    def get_estimated_duration(self) -> float:
        """Schätzt die Scan-Dauer in Sekunden."""
        points = self.get_points()
        total_time = 0.0
        
        for point in points:
            # Bewegungszeit (vereinfacht) + Settle-Zeit
            total_time += (1.0 / point.speed) + point.settle_time
        
        return total_time
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """Gibt Pattern-Informationen zurück."""
        points = self.get_points()
        duration = self.get_estimated_duration()
        
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'points_count': len(points),
            'estimated_duration': duration,
            'estimated_duration_minutes': duration / 60.0,
            'center_position': self.center_position,
            'optimal_distance': self.optimal_distance
        }


# ================================
# BASIC SCAN PATTERNS
# ================================

class RasterScanPattern(ScanPattern):
    """
    Raster-/Grid-Scan Pattern.
    Optimal für flache oder rechteckige Objekte.
    """
    
    def __init__(self, 
                 width: float = 0.20, 
                 height: float = 0.15,
                 rows: int = 10, 
                 cols: int = 10,
                 overlap: float = 0.2,
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
            zigzag: Zickzack-Muster
        """
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.zigzag = zigzag
        self.name = "Raster Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Raster-Scan Punkte."""
        points = []
        
        # Schrittweiten mit Überlappung
        step_x = self.width / (self.cols - 1) * (1 - self.overlap) if self.cols > 1 else 0
        step_y = self.height / (self.rows - 1) * (1 - self.overlap) if self.rows > 1 else 0
        
        # Start-Offsets (zentriert)
        start_x = -self.width / 2
        start_y = -self.height / 2
        
        for row in range(self.rows):
            # Zigzag: Jede zweite Zeile rückwärts
            cols_range = range(self.cols)
            if self.zigzag and row % 2 == 1:
                cols_range = reversed(cols_range)
            
            for col in cols_range:
                x_offset = start_x + col * step_x
                y_offset = start_y + row * step_y
                
                positions = self._calculate_joint_positions(x_offset, y_offset, 0)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.LINEAR,
                    metadata={'row': row, 'col': col}
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} raster scan points")
        return points


class SpiralScanPattern(ScanPattern):
    """
    Spiral-Scan Pattern.
    Optimal für runde oder zylindrische Objekte.
    """
    
    def __init__(self, 
                 radius_start: float = 0.05,
                 radius_end: float = 0.15,
                 revolutions: float = 4.0,
                 points_per_rev: int = 32,
                 **kwargs):
        """
        Initialisiert Spiral-Scan.
        
        Args:
            radius_start: Start-Radius
            radius_end: End-Radius
            revolutions: Anzahl Umdrehungen
            points_per_rev: Punkte pro Umdrehung
        """
        super().__init__(**kwargs)
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.revolutions = revolutions
        self.points_per_rev = points_per_rev
        self.name = "Spiral Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Spiral-Scan Punkte."""
        points = []
        
        total_points = int(self.revolutions * self.points_per_rev)
        
        for i in range(total_points):
            # Parameter für Spirale
            t = i / total_points
            angle = 2 * math.pi * self.revolutions * t
            radius = self.radius_start + (self.radius_end - self.radius_start) * t
            
            # Kartesische Koordinaten
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)
            
            positions = self._calculate_joint_positions(x_offset, y_offset, 0)
            
            point = ScanPoint(
                positions=positions,
                speed=self.speed,
                settle_time=self.settle_time,
                trajectory_type=TrajectoryType.S_CURVE,
                metadata={'angle': angle, 'radius': radius, 'revolution': i / self.points_per_rev}
            )
            points.append(point)
        
        logger.info(f"✅ Generated {len(points)} spiral scan points")
        return points


class SphericalScanPattern(ScanPattern):
    """
    Sphärischer Scan.
    Optimal für 3D-Objekte mit komplexer Geometrie.
    """
    
    def __init__(self, 
                 radius: float = 0.15,
                 theta_steps: int = 12,  # Azimuth
                 phi_steps: int = 8,     # Elevation
                 **kwargs):
        """
        Initialisiert sphärischen Scan.
        
        Args:
            radius: Scan-Radius
            theta_steps: Azimuth-Schritte (horizontal)
            phi_steps: Elevations-Schritte (vertikal)
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.theta_steps = theta_steps
        self.phi_steps = phi_steps
        self.name = "Spherical Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert sphärische Scan-Punkte."""
        points = []
        
        for phi_i in range(self.phi_steps):
            # Elevation: von -60° bis +60°
            phi = math.radians(-60 + 120 * phi_i / (self.phi_steps - 1)) if self.phi_steps > 1 else 0
            
            for theta_i in range(self.theta_steps):
                # Azimuth: volle 360°
                theta = 2 * math.pi * theta_i / self.theta_steps
                
                # Sphärische zu kartesische Koordinaten
                x_offset = self.radius * math.cos(phi) * math.cos(theta)
                y_offset = self.radius * math.cos(phi) * math.sin(theta)
                z_offset = self.radius * math.sin(phi)
                
                positions = self._calculate_joint_positions(x_offset, y_offset, z_offset)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.S_CURVE,
                    metadata={'theta': theta, 'phi': phi, 'theta_deg': math.degrees(theta), 'phi_deg': math.degrees(phi)}
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} spherical scan points")
        return points


class TurntableScanPattern(ScanPattern):
    """
    Turntable-Scan.
    Objekt wird gedreht, Scanner bleibt in festen Positionen.
    """
    
    def __init__(self, 
                 rotation_steps: int = 24,
                 elevation_angles: List[float] = None,
                 **kwargs):
        """
        Initialisiert Turntable-Scan.
        
        Args:
            rotation_steps: Anzahl Rotationsschritte
            elevation_angles: Liste der Elevationswinkel (in Grad)
        """
        super().__init__(**kwargs)
        self.rotation_steps = rotation_steps
        self.elevation_angles = elevation_angles or [0, 30, -30]  # Standard-Winkel
        self.name = "Turntable Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Turntable-Scan Punkte."""
        points = []
        
        for elevation in self.elevation_angles:
            elevation_rad = math.radians(elevation)
            
            for step in range(self.rotation_steps):
                # Turntable-Rotation
                rotation_angle = 2 * math.pi * step / self.rotation_steps
                
                # Scanner-Position berechnen
                x_offset = self.optimal_distance * math.cos(elevation_rad) * math.cos(rotation_angle)
                y_offset = self.optimal_distance * math.cos(elevation_rad) * math.sin(rotation_angle)
                z_offset = self.optimal_distance * math.sin(elevation_rad)
                
                positions = self._calculate_joint_positions(x_offset, y_offset, z_offset)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.LINEAR,
                    metadata={'rotation_deg': math.degrees(rotation_angle), 'elevation_deg': elevation}
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} turntable scan points")
        return points


# ================================
# ADVANCED SCAN PATTERNS
# ================================

class HelixScanPattern(ScanPattern):
    """
    Helix-/Schrauben-Scan.
    Optimal für zylindrische Objekte.
    """
    
    def __init__(self, 
                 radius: float = 0.12,
                 height: float = 0.20,
                 turns: float = 3.0,
                 points_per_turn: int = 24,
                 **kwargs):
        """
        Initialisiert Helix-Scan.
        
        Args:
            radius: Helix-Radius
            height: Helix-Höhe
            turns: Anzahl Windungen
            points_per_turn: Punkte pro Windung
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.height = height
        self.turns = turns
        self.points_per_turn = points_per_turn
        self.name = "Helix Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Helix-Scan Punkte."""
        points = []
        
        total_points = int(self.turns * self.points_per_turn)
        
        for i in range(total_points):
            t = i / total_points
            
            # Helix-Parameter
            angle = 2 * math.pi * self.turns * t
            height = -self.height/2 + self.height * t  # Von unten nach oben
            
            # Kartesische Koordinaten
            x_offset = self.radius * math.cos(angle)
            y_offset = self.radius * math.sin(angle)
            z_offset = height
            
            positions = self._calculate_joint_positions(x_offset, y_offset, z_offset)
            
            point = ScanPoint(
                positions=positions,
                speed=self.speed,
                settle_time=self.settle_time,
                trajectory_type=TrajectoryType.S_CURVE,
                metadata={'angle': angle, 'height': height, 'turn': i / self.points_per_turn}
            )
            points.append(point)
        
        logger.info(f"✅ Generated {len(points)} helix scan points")
        return points


class AdaptiveScanPattern(ScanPattern):
    """
    Adaptiver Scan.
    Passt sich dynamisch an die Objektgeometrie an.
    """
    
    def __init__(self, 
                 initial_points: int = 25,
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
        """Generiert adaptive Scan-Punkte."""
        # Starte mit sphärischem Grobscan
        coarse_scan = SphericalScanPattern(
            radius=self.optimal_distance,
            theta_steps=6,
            phi_steps=4,
            center_position=self.center_position
        )
        
        points = coarse_scan.generate_points()
        
        # Füge Verfeinerungspunkte hinzu (vereinfacht)
        refinement_points = []
        for i in range(self.initial_points // 4):
            angle = 2 * math.pi * i / (self.initial_points // 4)
            radius = self.optimal_distance * 0.8
            
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)
            
            positions = self._calculate_joint_positions(x_offset, y_offset, 0)
            
            point = ScanPoint(
                positions=positions,
                speed=0.2,  # Langsamer für Details
                settle_time=0.8,
                trajectory_type=TrajectoryType.S_CURVE,
                metadata={'type': 'refinement', 'iteration': 1}
            )
            refinement_points.append(point)
        
        points.extend(refinement_points)
        
        logger.info(f"✅ Generated {len(points)} adaptive scan points")
        return points


class CobwebScanPattern(ScanPattern):
    """
    Spinnennetz-/Cobweb-Scan.
    Radiale Linien kombiniert mit konzentrischen Kreisen.
    """
    
    def __init__(self, 
                 radial_lines: int = 8,
                 circles: int = 5,
                 max_radius: float = 0.15,
                 **kwargs):
        """
        Initialisiert Cobweb-Scan.
        
        Args:
            radial_lines: Anzahl radialer Linien
            circles: Anzahl konzentrischer Kreise
            max_radius: Maximaler Radius
        """
        super().__init__(**kwargs)
        self.radial_lines = radial_lines
        self.circles = circles
        self.max_radius = max_radius
        self.name = "Cobweb Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert Cobweb-Scan Punkte."""
        points = []
        
        # Konzentrische Kreise
        for circle in range(self.circles):
            radius = self.max_radius * (circle + 1) / self.circles
            
            for line in range(self.radial_lines):
                angle = 2 * math.pi * line / self.radial_lines
                
                x_offset = radius * math.cos(angle)
                y_offset = radius * math.sin(angle)
                
                positions = self._calculate_joint_positions(x_offset, y_offset, 0)
                
                point = ScanPoint(
                    positions=positions,
                    speed=self.speed,
                    settle_time=self.settle_time,
                    trajectory_type=TrajectoryType.LINEAR,
                    metadata={'circle': circle, 'line': line, 'radius': radius, 'angle': angle}
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} cobweb scan points")
        return points


# ================================
# SPECIALIZED PATTERNS
# ================================

class TableScanPattern(ScanPattern):
    """Table/Surface Scan - für flache Oberflächen."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Table Scan"
    
    def generate_points(self) -> List[ScanPoint]:
        # Verwende RasterScan als Basis
        raster = RasterScanPattern(
            width=0.25, height=0.20, rows=8, cols=10,
            center_position=self.center_position,
            speed=self.speed, settle_time=self.settle_time
        )
        return raster.generate_points()


class StatueSpiralPattern(ScanPattern):
    """Statue/Complex Object Spiral Pattern."""
    
    def __init__(self, 
                 radius_start: float = 0.08,
                 radius_end: float = 0.18,
                 height_range: float = 0.25,
                 revolutions: float = 5.0,
                 points_per_rev: int = 36,
                 **kwargs):
        super().__init__(**kwargs)
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.height_range = height_range
        self.revolutions = revolutions
        self.points_per_rev = points_per_rev
        self.name = "Statue Spiral"
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        total_points = int(self.revolutions * self.points_per_rev)
        
        for i in range(total_points):
            t = i / total_points
            
            # Spirale mit Höhenvariation
            angle = 2 * math.pi * self.revolutions * t
            radius = self.radius_start + (self.radius_end - self.radius_start) * t
            height = -self.height_range/2 + self.height_range * math.sin(math.pi * t)
            
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)
            z_offset = height
            
            positions = self._calculate_joint_positions(x_offset, y_offset, z_offset)
            
            point = ScanPoint(
                positions=positions,
                speed=self.speed,
                settle_time=self.settle_time,
                trajectory_type=TrajectoryType.S_CURVE,
                metadata={'angle': angle, 'radius': radius, 'height': height}
            )
            points.append(point)
        
        logger.info(f"✅ Generated {len(points)} statue spiral points")
        return points


# ================================
# PATTERN FACTORY & UTILITIES
# ================================

def create_scan_pattern(pattern_type: str, **params) -> ScanPattern:
    """
    Factory-Funktion für Scan-Patterns.
    
    Args:
        pattern_type: Typ des Patterns
        **params: Parameter für das Pattern
    
    Returns:
        ScanPattern: Erstelltes Pattern
    """
    patterns = {
        'raster': RasterScanPattern,
        'spiral': SpiralScanPattern,
        'spherical': SphericalScanPattern,
        'turntable': TurntableScanPattern,
        'helix': HelixScanPattern,
        'adaptive': AdaptiveScanPattern,
        'cobweb': CobwebScanPattern,
        'table': TableScanPattern,
        'statue': StatueSpiralPattern
    }
    
    pattern_class = patterns.get(pattern_type.lower())
    if not pattern_class:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return pattern_class(**params)


def get_pattern_presets() -> Dict[str, Dict[str, Any]]:
    """Gibt vordefinierte Pattern-Presets zurück."""
    return {
        'quick_preview': {
            'type': 'raster',
            'params': {'width': 0.15, 'height': 0.12, 'rows': 6, 'cols': 6, 'speed': 0.6}
        },
        'high_quality': {
            'type': 'spherical',
            'params': {'radius': 0.15, 'theta_steps': 16, 'phi_steps': 12, 'speed': 0.25}
        },
        'small_object': {
            'type': 'spiral',
            'params': {'radius_start': 0.05, 'radius_end': 0.10, 'revolutions': 3, 'speed': 0.4}
        },
        'large_object': {
            'type': 'spherical',
            'params': {'radius': 0.20, 'theta_steps': 20, 'phi_steps': 14, 'speed': 0.3}
        }
    }


# ================================
# MAIN (for testing)
# ================================

if __name__ == "__main__":
    # Test pattern generation
    print("🧪 Testing Scan Patterns...")
    
    patterns = [
        RasterScanPattern(rows=5, cols=5),
        SpiralScanPattern(revolutions=2),
        SphericalScanPattern(theta_steps=8, phi_steps=6),
        HelixScanPattern(turns=2),
        CobwebScanPattern(radial_lines=6, circles=3)
    ]
    
    for pattern in patterns:
        info = pattern.get_pattern_info()
        print(f"\n{info['name']}:")
        print(f"  Points: {info['points_count']}")
        print(f"  Duration: {info['estimated_duration_minutes']:.1f} min")
