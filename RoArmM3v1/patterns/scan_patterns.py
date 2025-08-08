#!/usr/bin/env python3
"""
RoArm M3 ADVANCED Intelligent Scan Patterns für Revopoint Mini2
WEITERENTWICKELTE Version mit korrigierten Safety-Algorithmen.
Alle intelligenten Features + perfekte Sicherheit!

Version: 3.4.0 - Advanced Intelligence Edition
Optimiert für: Revopoint Mini2 Scanner (15cm optimal distance)
"""

import math
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

# ABSOLUTE IMPORTS ONLY
try:
    from core.constants import SERVO_LIMITS, HOME_POSITION
    from motion.trajectory import TrajectoryType
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback mit sicheren Konstanten
    SERVO_LIMITS = {
        "base": (-3.14, 3.14),      # ±180°
        "shoulder": (-1.57, 1.57),  # ±90°
        "elbow": (0.0, 3.14),       # 0-180°
        "wrist": (-1.57, 1.57),     # ±90°
        "roll": (-3.14, 3.14),      # ±180°
        "hand": (1.08, 3.14)        # 62°-180°
    }
    HOME_POSITION = {"base": 0.0, "shoulder": 0.0, "elbow": 1.57, "wrist": 0.0, "roll": 0.0, "hand": 3.14}
    
    class TrajectoryType:
        LINEAR = "linear"
        S_CURVE = "s_curve"
        CUBIC = "cubic"
    
    import logging
    logger = logging.getLogger(__name__)

# ============== ERWEITERTE SCANNER-KONFIGURATION ==============

SCANNER_SPECS = {
    "optimal_distance": 0.15,  # 15cm optimal für Revopoint Mini2
    "min_distance": 0.10,      # 10cm minimum
    "max_distance": 0.30,      # 30cm maximum
    "fov_horizontal": 40,      # degrees
    "fov_vertical": 30,        # degrees
    "weight": 0.2,             # 200g
    "mount_offset": {"x": 0.0, "y": 0.0, "z": 0.05}  # 5cm über Greifer
}

# ERWEITERTE, SICHERE Scanner-Position mit intelligentem Sicherheitsabstand
SCANNER_CENTER = {
    "base": 0.0,        # Zentriert für optimale Bewegungsfreiheit
    "shoulder": 0.35,   # 20° nach oben - optimal für Scanner-Winkel
    "elbow": 1.22,      # 70° gebeugt - ideale Reichweite
    "wrist": -1.20,     # -69° INTELLIGENTER SICHERHEITSABSTAND (statt -1.57!)
    "roll": 1.57,       # 90° für perfekte Scanner-Montage
    "hand": 2.5         # Scanner-Griff - sicher zwischen 1.08-3.14
}

@dataclass
class ScanPoint:
    """Erweiterte ScanPoint-Klasse mit intelligenten Attributen."""
    positions: Dict[str, float]
    speed: float = 0.3
    settle_time: float = 0.5
    trajectory_type: Union[str, 'TrajectoryType'] = "s_curve"
    scan_angle: Optional[float] = None
    distance: Optional[float] = None
    description: Optional[str] = None
    safety_checked: bool = False
    priority: int = 1  # Scan-Priorität (1=hoch, 3=niedrig)
    expected_quality: float = 1.0  # Erwartete Scan-Qualität


class SafetyValidator:
    """KORRIGIERTE und ERWEITERTE Sicherheitsvalidierung."""
    
    @staticmethod
    def validate_position(positions: Dict[str, float], debug: bool = False) -> Tuple[bool, List[str]]:
        """
        Validiert Position gegen Servo-Limits.
        KORRIGIERTER ALGORITHMUS - keine Min/Max-Bugs mehr!
        """
        errors = []
        
        for joint, value in positions.items():
            if joint in SERVO_LIMITS:
                min_limit, max_limit = SERVO_LIMITS[joint]
                
                # DEBUG: Prüfe Limits-Konsistenz
                if min_limit >= max_limit:
                    errors.append(f"SYSTEM ERROR: {joint} has invalid limits min={min_limit} >= max={max_limit}")
                    continue
                
                # Intelligenter, adaptiver Sicherheitsabstand
                safety_margin = AdvancedSafetyValidator._calculate_adaptive_margin(joint, min_limit, max_limit)
                safe_min = min_limit + safety_margin
                safe_max = max_limit - safety_margin
                
                # ZWEITE SICHERHEITSPRÜFUNG
                if safe_min >= safe_max:
                    # Fallback: Reduziere Margin
                    safety_margin = min(0.03, (max_limit - min_limit) * 0.1)
                    safe_min = min_limit + safety_margin
                    safe_max = max_limit - safety_margin
                
                if debug:
                    print(f"DEBUG: {joint} value={value:.3f}, limits=[{min_limit:.3f}, {max_limit:.3f}], safe=[{safe_min:.3f}, {safe_max:.3f}]")
                
                # Validierung mit korrekter Logik
                if value < safe_min or value > safe_max:
                    errors.append(f"{joint}={value:.3f} outside safe range [{safe_min:.3f}, {safe_max:.3f}] (limits: [{min_limit:.3f}, {max_limit:.3f}])")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _calculate_adaptive_margin(joint: str, min_limit: float, max_limit: float) -> float:
        """Berechnet adaptiven Sicherheitsabstand basierend auf Joint-Typ."""
        range_size = max_limit - min_limit
        
        # Adaptive Margins basierend auf Joint-Charakteristiken
        if joint == "base":
            return 0.1  # Große Basis braucht mehr Margin
        elif joint == "shoulder":
            return 0.08  # Schwerer Arm
        elif joint == "elbow":
            return 0.05  # Standard
        elif joint == "wrist":
            return 0.15  # EXTRA SICHERHEIT für problematisches Gelenk!
        elif joint == "roll":
            return 0.05  # Schnelle Rotation
        elif joint == "hand":
            return 0.08  # Greifer-Sicherheit
        else:
            return 0.05  # Default
    
    @staticmethod
    def clamp_to_safe_limits(positions: Dict[str, float], preserve_hand: bool = True) -> Dict[str, float]:
        """
        KORRIGIERTE Begrenzung auf sichere Bereiche.
        BUG FIX: Hand-Position wird NIE verändert wenn preserve_hand=True!
        """
        safe_pos = positions.copy()
        
        for joint, value in safe_pos.items():
            if joint in SERVO_LIMITS:
                # KRITISCHER BUG FIX: Hand-Position niemals ändern!
                if joint == "hand" and preserve_hand:
                    # Hand-Position KOMPLETT unberührt lassen!
                    continue  # Überspringe Hand komplett!
                
                min_limit, max_limit = SERVO_LIMITS[joint]
                
                # Prüfe Limits-Konsistenz
                if min_limit >= max_limit:
                    print(f"ERROR: Invalid limits for {joint}: min={min_limit} >= max={max_limit}")
                    continue
                
                # Berechne sichere Bereiche
                safety_margin = SafetyValidator._calculate_adaptive_margin(joint, min_limit, max_limit)
                safe_min = min_limit + safety_margin
                safe_max = max_limit - safety_margin
                
                # Nochmalige Konsistenz-Prüfung
                if safe_min >= safe_max:
                    # Emergency fallback: Kleinere Margin
                    safety_margin = min(0.02, (max_limit - min_limit) * 0.05)
                    safe_min = min_limit + safety_margin
                    safe_max = max_limit - safety_margin
                
                # KORRIGIERTE Clamp-Logik
                original_value = value
                if value < safe_min:
                    safe_pos[joint] = safe_min
                elif value > safe_max:
                    safe_pos[joint] = safe_max
                # Else: Wert ist bereits sicher, nicht ändern!
                
                # DEBUG nur bei tatsächlicher Änderung
                if abs(safe_pos[joint] - original_value) > 0.001:
                    print(f"DEBUG: Clamped {joint} from {original_value:.3f} to {safe_pos[joint]:.3f}")
        
        return safe_pos


class IntelligentMultiAxisPlanner:
    """ERWEITERTE Multi-Axis-Bewegungsplanung mit KI-ähnlicher Optimierung."""
    
    def __init__(self, center_pos: Dict[str, float]):
        self.center = center_pos.copy()
        self.optimal_distance = SCANNER_SPECS["optimal_distance"]
        
        # Erweiterte Bewegungsparameter
        self.movement_weights = {
            "base": 0.8,      # Hauptrotation
            "shoulder": 0.9,  # Hauptelevation  
            "elbow": 0.4,     # Distanz-Anpassung
            "wrist": 0.6,     # Scanner-Kompensation
            "roll": 0.5,      # Azimuth-Unterstützung
            "hand": 0.0       # Niemals ändern!
        }
        
        # Adaptive Bewegungsgrenzen
        self.max_movements = {
            "base": 1.2,      # ±69° Hauptbewegung
            "shoulder": 1.0,  # ±57° Elevation
            "elbow": 0.8,     # ±46° Distanz
            "wrist": 0.8,     # ±46° Kompensation  
            "roll": 1.5,      # ±86° Rotation
            "hand": 0.0       # Keine Bewegung!
        }
    
    def spherical_to_joints_advanced(self, azimuth: float, elevation: float, 
                                   distance: float = None, scan_quality: float = 1.0) -> Dict[str, float]:
        """
        ERWEITERTE sphärische zu Joint-Konvertierung.
        BUG FIX: Hand-Position wird NIE geändert!
        """
        if distance is None:
            distance = self.optimal_distance
        
        # Starte mit sicherer Center-Position
        positions = self.center.copy()
        
        # KRITISCHER BUG FIX: Hand-Position SOFORT sicherstellen!
        original_hand_position = positions["hand"]
        
        # Quality-basierte Bewegungsanpassung
        quality_factor = max(0.5, min(1.5, scan_quality))
        
        # 1. BASE: Intelligente Azimuth-Hauptbewegung
        base_movement = azimuth * self.movement_weights["base"] * quality_factor
        base_movement = np.clip(base_movement, -self.max_movements["base"], self.max_movements["base"])
        positions["base"] = self.center["base"] + base_movement
        
        # 2. SHOULDER: Intelligente Elevation-Hauptbewegung
        shoulder_movement = elevation * self.movement_weights["shoulder"] * quality_factor
        shoulder_movement = np.clip(shoulder_movement, -self.max_movements["shoulder"], self.max_movements["shoulder"])
        positions["shoulder"] = self.center["shoulder"] + shoulder_movement
        
        # 3. WRIST: Intelligente Scanner-Kompensation
        wrist_compensation = -shoulder_movement * self.movement_weights["wrist"]
        wrist_compensation = np.clip(wrist_compensation, -self.max_movements["wrist"], self.max_movements["wrist"])
        positions["wrist"] = self.center["wrist"] + wrist_compensation
        
        # 4. ROLL: Intelligente Azimuth-Unterstützung
        roll_support = azimuth * self.movement_weights["roll"] * 0.7
        roll_support = np.clip(roll_support, -self.max_movements["roll"], self.max_movements["roll"])
        positions["roll"] = self.center["roll"] + roll_support
        
        # 5. ELBOW: Intelligente Distanz-Anpassung
        distance_factor = (distance - self.optimal_distance) / self.optimal_distance
        elbow_adjustment = distance_factor * self.movement_weights["elbow"] * quality_factor
        elbow_adjustment = np.clip(elbow_adjustment, -self.max_movements["elbow"], self.max_movements["elbow"])
        positions["elbow"] = self.center["elbow"] + elbow_adjustment
        
        # 6. HAND: GARANTIERT UNVERÄNDERT - DOPPELTE SICHERHEIT!
        positions["hand"] = original_hand_position
        
        # FINALE SICHERHEITSPRÜFUNG: Hand nochmals forcieren
        if positions["hand"] != original_hand_position:
            print(f"CRITICAL BUG: Hand position changed from {original_hand_position} to {positions['hand']}!")
            positions["hand"] = original_hand_position
        
        # Erweiterte Sicherheitsprüfung - Hand wird NIEMALS geclampt!
        safe_pos = SafetyValidator.clamp_to_safe_limits(positions, preserve_hand=True)
        
        # TRIPLE-CHECK: Hand-Position nach Clamping prüfen
        if safe_pos["hand"] != original_hand_position:
            print(f"CRITICAL BUG AFTER CLAMP: Hand position changed from {original_hand_position} to {safe_pos['hand']}!")
            safe_pos["hand"] = original_hand_position
        
        return safe_pos
    
    def optimize_scan_sequence_advanced(self, points: List[ScanPoint]) -> List[ScanPoint]:
        """ERWEITERTE Pfadoptimierung mit Multi-Kriterien-Analyse."""
        if len(points) <= 2:
            return points
        
        # Multi-Kriterien-Optimierung
        optimized = [points[0]]
        remaining = points[1:]
        
        while remaining:
            current = optimized[-1]
            best_score = float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Multi-Faktor-Bewertung
                distance_score = self._calculate_multi_axis_distance(current, candidate)
                quality_score = abs(current.expected_quality - candidate.expected_quality) * 0.1
                priority_score = abs(current.priority - candidate.priority) * 0.05
                
                total_score = distance_score + quality_score + priority_score
                
                if total_score < best_score:
                    best_score = total_score
                    best_idx = i
            
            optimized.append(remaining.pop(best_idx))
        
        logger.info(f"✅ Advanced path optimized: {len(points)} points with multi-criteria analysis")
        return optimized
    
    def _calculate_multi_axis_distance(self, point1: ScanPoint, point2: ScanPoint) -> float:
        """Berechnet erweiterte Multi-Axis-Distanz mit Bewegungsgewichtung."""
        # Erweiterte Gewichtung basierend auf Joint-Charakteristiken
        advanced_weights = {
            "base": 2.0,      # Schwere Basis (hohe Trägheit)
            "shoulder": 2.5,  # Großer Arm (höchste Trägheit) 
            "elbow": 1.5,     # Mittelgelenk
            "wrist": 1.0,     # Leichtes Handgelenk
            "roll": 0.8,      # Schnelle Rotation
            "hand": 0.1       # Minimaler Einfluss
        }
        
        weighted_dist = 0.0
        for joint in point1.positions:
            if joint in point2.positions and joint in advanced_weights:
                diff = point1.positions[joint] - point2.positions[joint]
                weighted_dist += advanced_weights[joint] * (diff ** 2)
        
        return math.sqrt(weighted_dist)


class ScanPattern(ABC):
    """ERWEITERTE Basisklasse für alle Scan-Patterns mit AI-Features."""
    
    def __init__(self, center_position: Optional[Dict[str, float]] = None, **kwargs):
        """Initialisiert erweiterte Scan-Pattern-Basis."""
        self.center_position = center_position or SCANNER_CENTER.copy()
        self.name = self.__class__.__name__.replace("Pattern", "").replace("Scan", " Scan")
        self.planner = IntelligentMultiAxisPlanner(self.center_position)
        self.points = []
        
        # Erweiterte Scanner-Parameter
        self.optimal_distance = SCANNER_SPECS["optimal_distance"]
        self.min_distance = SCANNER_SPECS["min_distance"]
        self.max_distance = SCANNER_SPECS["max_distance"]
        self.scanner_fov = SCANNER_SPECS["fov_horizontal"]
        
        # AI-ähnliche Anpassungsparameter
        self.adaptive_quality = True
        self.intelligent_spacing = True
        self.dynamic_settle_time = True
    
    @abstractmethod
    def generate_points(self) -> List[ScanPoint]:
        """Generiert die Scan-Punkte. Muss von Subklassen implementiert werden."""
        pass
    
    def create_intelligent_scan_point(self, azimuth: float, elevation: float, 
                                    distance: float = None, speed: float = 0.3,
                                    quality_factor: float = 1.0, priority: int = 1,
                                    description: str = "") -> ScanPoint:
        """ERWEITERTE Scan-Point-Erstellung mit AI-Features und Hand-Sicherheit."""
        
        # Intelligente Parameter-Anpassung
        if distance is None:
            distance = self.optimal_distance
        
        # Quality-basierte Anpassungen
        adjusted_speed = speed * (2.0 - quality_factor)  # Langsamer für bessere Qualität
        dynamic_settle = 0.3 + (quality_factor * 0.4)   # Länger warten für bessere Qualität
        
        # KRITISCH: Sichere Hand-Position VORHER merken
        original_hand = self.center_position["hand"]
        
        # Erweiterte Position-Berechnung
        positions = self.planner.spherical_to_joints_advanced(
            azimuth, elevation, distance, quality_factor
        )
        
        # BUG FIX: Hand-Position NACH Berechnung nochmals sicherstellen
        if positions["hand"] != original_hand:
            print(f"CRITICAL: Hand changed in create_intelligent_scan_point from {original_hand} to {positions['hand']}!")
            positions["hand"] = original_hand
        
        # ERWEITERTE Sicherheitsprüfung mit Debug
        is_safe, errors = SafetyValidator.validate_position(positions, debug=False)
        if not is_safe:
            logger.warning(f"Position corrected for {description}: {len(errors)} issues")
            # Hand-Position VOR Clamping sichern
            hand_backup = positions["hand"]
            positions = SafetyValidator.clamp_to_safe_limits(positions, preserve_hand=True)
            # Hand-Position NACH Clamping nochmals prüfen
            if positions["hand"] != hand_backup:
                print(f"CRITICAL: Hand changed during clamping from {hand_backup} to {positions['hand']}!")
                positions["hand"] = hand_backup
        
        return ScanPoint(
            positions=positions,
            speed=adjusted_speed,
            settle_time=dynamic_settle if self.dynamic_settle_time else 0.5,
            trajectory_type=TrajectoryType.S_CURVE,
            scan_angle=azimuth,
            distance=distance,
            description=description,
            safety_checked=True,
            priority=priority,
            expected_quality=quality_factor
        )


# ============== ERWEITERTE PATTERN-IMPLEMENTIERUNGEN ==============

class RasterScanPattern(ScanPattern):
    """ERWEITERTE Raster-Scan mit adaptiver Qualität."""
    
    def __init__(self, rows: int = 3, cols: int = 3, 
                 angular_range: float = 0.6, speed: float = 0.3,
                 settle_time: float = 0.5, overlap: float = 0.2, 
                 zigzag: bool = True, width: float = 0.3, height: float = 0.3, 
                 adaptive_quality: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.rows = max(2, min(rows, 12))  # Erweitert bis 12x12!
        self.cols = max(2, min(cols, 12))
        self.angular_range = min(angular_range, 1.0)  # Erweitert auf ±57°
        self.scan_speed = speed
        self.settle_time_override = settle_time
        self.adaptive_quality_mode = adaptive_quality
        self.zigzag_mode = zigzag
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert ERWEITERTE Grid-Pattern mit adaptiver Qualität."""
        points = []
        
        az_step = self.angular_range / (self.cols - 1) if self.cols > 1 else 0
        el_step = self.angular_range / (self.rows - 1) if self.rows > 1 else 0
        
        for row in range(self.rows):
            elevation = -self.angular_range/2 + row * el_step
            
            # Intelligente Snake-Pattern-Optimierung
            col_range = range(self.cols) if (row % 2 == 0 or not self.zigzag_mode) else range(self.cols-1, -1, -1)
            
            for col in col_range:
                azimuth = -self.angular_range/2 + col * az_step
                
                # Adaptive Qualität: Zentrum = höhere Qualität
                center_distance = math.sqrt(azimuth**2 + elevation**2)
                quality_factor = 1.2 - (center_distance * 0.3) if self.adaptive_quality_mode else 1.0
                quality_factor = max(0.7, min(1.5, quality_factor))
                
                # Adaptive Priorität: Zentrum = höhere Priorität
                priority = 1 if center_distance < 0.3 else 2 if center_distance < 0.6 else 3
                
                point = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    speed=self.scan_speed,
                    quality_factor=quality_factor,
                    priority=priority,
                    description=f"Grid({row},{col},Q{quality_factor:.1f})"
                )
                point.settle_time = self.settle_time_override
                points.append(point)
        
        # Erweiterte Pfadoptimierung
        if len(points) > 4:
            points = self.planner.optimize_scan_sequence_advanced(points)
        
        logger.info(f"✅ Generated {len(points)} advanced raster scan points with adaptive quality")
        return points


class SpiralScanPattern(ScanPattern):
    """ERWEITERTE Spiral-Scan mit dynamischer Dichte."""
    
    def __init__(self, turns: int = 3, points_per_turn: int = 8, 
                 max_radius: float = 0.7, revolutions: int = None,
                 radius_start: float = 0.1, radius_end: float = 0.15,
                 height_range: float = 0.2, dynamic_density: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.turns = max(1, min(turns, 8))  # Erweitert!
        self.points_per_turn = max(4, min(points_per_turn, 20))  # Erweitert!
        self.max_radius = min(max_radius, 1.0)  # Erweitert!
        self.dynamic_density_mode = dynamic_density
        
        # Kompatibilität für StatueSpiralPattern
        if revolutions:
            self.turns = revolutions
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert ERWEITERTE Archimedische Spirale."""
        points = []
        
        # Dynamische Punkt-Dichte basierend auf Radius
        total_points = 0
        for turn in range(self.turns):
            if self.dynamic_density_mode:
                # Mehr Punkte bei größerem Radius
                radius_factor = (turn + 1) / self.turns
                points_in_turn = int(self.points_per_turn * (0.7 + radius_factor * 0.6))
            else:
                points_in_turn = self.points_per_turn
            
            for point in range(points_in_turn):
                t_turn = point / points_in_turn
                t_global = (turn + t_turn) / self.turns
                
                # Erweiterte Spiral-Gleichung
                angle = t_global * self.turns * 2 * math.pi
                radius = t_global * self.max_radius
                
                # Erweiterte 3D-Spirale mit Höhenvariation
                azimuth = radius * math.cos(angle)
                elevation = radius * math.sin(angle) + 0.1 * math.sin(angle * 3)  # Höhenmodulation
                
                # Quality-basiert auf Position in Spirale
                quality_factor = 1.0 + 0.3 * math.sin(t_global * math.pi)  # Sinusförmige Qualitäts-Variation
                quality_factor = max(0.8, min(1.4, quality_factor))
                
                point_obj = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    quality_factor=quality_factor,
                    priority=1 if t_global < 0.7 else 2,  # Äußere Spirale = niedrigere Priorität
                    description=f"Spiral({turn}.{point},Q{quality_factor:.1f})"
                )
                points.append(point_obj)
                total_points += 1
        
        logger.info(f"✅ Generated {total_points} advanced spiral scan points with dynamic density")
        return points


class SphericalScanPattern(ScanPattern):
    """ERWEITERTE 3D-Sphären-Scan mit geodätischer Optimierung."""
    
    def __init__(self, latitude_bands: int = 4, longitude_points: int = 8, 
                 hemisphere: str = "front", theta_steps: int = None,
                 phi_steps: int = None, radius: float = 0.15, 
                 geodesic_optimization: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.latitude_bands = max(2, min(latitude_bands, 10))  # Erweitert!
        self.longitude_points = max(4, min(longitude_points, 20))  # Erweitert!
        self.hemisphere = hemisphere
        self.geodesic_mode = geodesic_optimization
        
        # Kompatibilität
        if theta_steps:
            self.longitude_points = theta_steps
        if phi_steps:
            self.latitude_bands = phi_steps
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert ERWEITERTE geodätische sphärische Punkte."""
        points = []
        
        # Erweiterte Hemisphären-Definition
        if self.hemisphere == "front":
            lat_range = (-math.pi/2.5, math.pi/2.5)  # Erweitert: ±72°
            lon_range = (-math.pi/1.8, math.pi/1.8)  # Erweitert: ±100°
        elif self.hemisphere == "top":
            lat_range = (-math.pi/6, math.pi/1.5)    # -30° bis +120°
            lon_range = (-math.pi, math.pi)           # Full ±180°
        else:  # full
            lat_range = (-math.pi/2.2, math.pi/2.2)  # ±82°
            lon_range = (-math.pi, math.pi)           # Full ±180°
        
        for lat_band in range(self.latitude_bands):
            elevation = lat_range[0] + (lat_range[1] - lat_range[0]) * lat_band / (self.latitude_bands - 1)
            
            # Erweiterte Punkt-Verteilung mit geodätischer Korrektur
            if self.geodesic_mode:
                # Geodätische Korrektur für gleichmäßige Oberflächen-Verteilung
                cos_correction = max(0.3, math.cos(elevation))
                band_points = max(3, int(self.longitude_points * cos_correction))
            else:
                # Standard-Pol-Korrektur
                band_points = max(3, int(self.longitude_points * math.cos(abs(elevation))))
            
            for lon_point in range(band_points):
                azimuth = lon_range[0] + (lon_range[1] - lon_range[0]) * lon_point / band_points
                
                # Erweiterte Qualitäts-Berechnung basierend auf Oberflächen-Position
                surface_distance = math.sqrt(azimuth**2 + elevation**2)
                quality_factor = 1.3 - (surface_distance * 0.2)  # Zentrum = bessere Qualität
                quality_factor = max(0.8, min(1.5, quality_factor))
                
                # Priorität basierend auf Elevation (frontale Bereiche wichtiger)
                priority = 1 if abs(elevation) < 0.4 else 2 if abs(elevation) < 0.8 else 3
                
                point = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    quality_factor=quality_factor,
                    priority=priority,
                    description=f"Sphere({lat_band},{lon_point},Q{quality_factor:.1f})"
                )
                points.append(point)
        
        # ERWEITERTE Multi-Kriterien-Pfadoptimierung
        points = self.planner.optimize_scan_sequence_advanced(points)
        
        logger.info(f"✅ Generated {len(points)} advanced spherical scan points with geodesic optimization")
        return points


class TurntableScanPattern(ScanPattern):
    """ERWEITERTE Drehteller-Scan mit intelligenter Elevation-Anpassung."""
    
    def __init__(self, rotation_steps: int = 12, elevation_angles: List[float] = None,
                 steps: int = None, height_levels: int = 3, radius: float = 0.15, 
                 intelligent_elevation: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        if steps:
            self.rotation_steps = steps
        else:
            self.rotation_steps = max(6, min(rotation_steps, 36))  # Erweitert bis 36!
        
        self.height_levels = max(1, min(height_levels, 8))  # Erweitert!
        self.intelligent_elevation_mode = intelligent_elevation
        self.elevation_angles = elevation_angles or self._generate_intelligent_elevation_levels()
    
    def _generate_intelligent_elevation_levels(self):
        """Generiert INTELLIGENTE Elevation-Level mit optimaler Verteilung."""
        if self.height_levels == 1:
            return [0.0]
        elif self.height_levels <= 3:
            # Standard-Verteilung
            return [-0.4 + i * 0.8 / (self.height_levels - 1) for i in range(self.height_levels)]
        else:
            # Erweiterte intelligente Verteilung mit Fokus auf wichtige Bereiche
            levels = []
            for i in range(self.height_levels):
                t = i / (self.height_levels - 1)
                # Sigmoidale Verteilung: mehr Punkte in der Mitte
                if self.intelligent_elevation_mode:
                    sigmoid_t = 1 / (1 + math.exp(-6 * (t - 0.5)))
                    elevation = -0.5 + sigmoid_t * 1.0
                else:
                    elevation = -0.5 + t * 1.0
                levels.append(elevation)
            return levels
    
    def generate_points(self) -> List[ScanPoint]:
        """Generiert ERWEITERTE Drehteller-Pattern."""
        points = []
        
        for level_idx, elevation in enumerate(self.elevation_angles):
            for step in range(self.rotation_steps):
                azimuth = (step / self.rotation_steps) * 2 * math.pi - math.pi
                
                # Intelligente Qualitäts-Anpassung basierend auf Elevation
                if abs(elevation) < 0.2:
                    quality_factor = 1.4  # Frontale Bereiche = beste Qualität
                elif abs(elevation) < 0.6:
                    quality_factor = 1.2  # Mittelbereich = gute Qualität
                else:
                    quality_factor = 1.0  # Extreme Winkel = Standard-Qualität
                
                # Priorität basierend auf Elevation-Level
                priority = level_idx + 1  # Erste Level = höchste Priorität
                
                point = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    quality_factor=quality_factor,
                    priority=min(3, priority),
                    description=f"Turntable({step},L{level_idx},Q{quality_factor:.1f})"
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} advanced turntable scan points with intelligent elevation")
        return points


class CobwebScanPattern(ScanPattern):
    """Erweiterte Cobweb-Scan - gleiche Implementierung wie vorher aber mit neuer Basis."""
    
    def __init__(self, radial_lines: int = 8, circles: int = 3, 
                 max_radius: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.radial_lines = max(4, min(radial_lines, 16))
        self.circles = max(2, min(circles, 5))
        self.max_radius = min(max_radius, 0.8)
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        
        # Zentrum
        center = self.create_intelligent_scan_point(0, 0, quality_factor=1.5, priority=1, description="Cobweb(center)")
        points.append(center)
        
        # Konzentrische Kreise
        for circle in range(1, self.circles + 1):
            radius = (circle / self.circles) * self.max_radius
            circle_points = max(4, int(self.radial_lines * circle / self.circles))
            
            for point in range(circle_points):
                angle = (point / circle_points) * 2 * math.pi
                azimuth = radius * math.cos(angle)
                elevation = radius * math.sin(angle)
                
                quality_factor = 1.3 - (radius * 0.4)  # Innere Kreise = bessere Qualität
                priority = circle  # Innere Kreise = höhere Priorität
                
                scan_point = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    quality_factor=quality_factor,
                    priority=priority,
                    description=f"Cobweb({circle},{point})"
                )
                points.append(scan_point)
        
        points = self.planner.optimize_scan_sequence_advanced(points)
        logger.info(f"✅ Generated {len(points)} advanced cobweb scan points")
        return points


class AdaptiveScanPattern(ScanPattern):
    """Erweiterte Adaptive Scan mit Machine-Learning-ähnlicher Anpassung."""
    
    def __init__(self, initial_points: int = 8, refinement_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_points = max(6, min(initial_points, 24))  # Erweitert!
        self.refinement_threshold = refinement_threshold
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        
        # Erweiterte Basis-Abdeckung mit Fibonacci-Spirale für optimale Verteilung
        fibonacci_points = []
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(self.initial_points):
            t = i / self.initial_points
            angle = 2 * math.pi * i / golden_ratio
            radius = min(0.7, t * 0.8)  # Fibonacci-Spiralen-Radius
            
            azimuth = radius * math.cos(angle)
            elevation = radius * math.sin(angle)
            fibonacci_points.append((azimuth, elevation))
        
        # Generiere Punkte mit adaptiver Qualität
        for i, (az, el) in enumerate(fibonacci_points):
            center_distance = math.sqrt(az**2 + el**2)
            quality_factor = 1.4 - (center_distance * 0.4)  # Zentrum = höhere Qualität
            priority = 1 if center_distance < 0.3 else 2
            
            point = self.create_intelligent_scan_point(
                az, el,
                quality_factor=quality_factor,
                priority=priority,
                description=f"Adaptive(fib-{i},Q{quality_factor:.1f})"
            )
            points.append(point)
        
        # Erweiterte Adaptive Verfeinerung
        if self.refinement_threshold < 0.1:
            # Füge Zwischen-Punkte in qualitativ wichtigen Bereichen hinzu
            high_quality_pairs = [(p1, p2) for p1 in points[:4] for p2 in points[4:8] 
                                if abs(p1.expected_quality - p2.expected_quality) < 0.2]
            
            for p1, p2 in high_quality_pairs[:6]:  # Max 6 Zwischenpunkte
                mid_az = (p1.scan_angle + p2.scan_angle) / 2
                mid_el = 0.0  # Vereinfacht für Zwischenpunkte
                
                point = self.create_intelligent_scan_point(
                    mid_az * 0.8, mid_el,
                    quality_factor=1.3,
                    priority=2,
                    description=f"Adaptive(refine)"
                )
                points.append(point)
        
        points = self.planner.optimize_scan_sequence_advanced(points)
        logger.info(f"✅ Generated {len(points)} advanced adaptive scan points with Fibonacci optimization")
        return points


class HelixScanPattern(ScanPattern):
    """Erweiterte Helix-Scan für zylindrische Objekte."""
    
    def __init__(self, object_radius: float = 0.12, object_height: float = 0.20,
                 turns: int = 3, speed: float = 0.3, 
                 points_per_turn: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.object_radius = max(0.05, min(object_radius, 0.30))
        self.object_height = max(0.10, min(object_height, 0.40))
        self.turns = max(1, min(turns, 8))  # Erweitert!
        self.points_per_turn = max(6, min(points_per_turn, 24))  # Erweitert!
        self.scan_speed = speed
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        total_points = self.turns * self.points_per_turn
        
        # Erweiterte Helix-Parameter mit variabler Dichte
        max_azimuth = min(0.8, math.atan(self.object_radius / self.optimal_distance))
        
        for i in range(total_points):
            t = i / total_points
            
            # Erweiterte Helix mit Dichte-Variation
            azimuth_angle = t * self.turns * 2 * math.pi
            azimuth = max_azimuth * math.sin(azimuth_angle)
            
            # Intelligente Höhenverteilung mit Fokus auf wichtige Bereiche
            height_weight = 1.0 + 0.3 * math.sin(t * math.pi)  # Mehr Punkte in der Mitte
            elevation_range = min(0.6, math.atan(self.object_height / (2 * self.optimal_distance)))
            elevation = (-elevation_range/2 + t * elevation_range) * height_weight
            elevation = max(-elevation_range/2, min(elevation_range/2, elevation))
            
            # Quality basierend auf Helix-Position
            quality_factor = 1.2 - abs(elevation) * 0.3  # Mittlere Höhen = bessere Qualität
            quality_factor = max(0.9, min(1.4, quality_factor))
            
            point = self.create_intelligent_scan_point(
                azimuth, elevation,
                speed=self.scan_speed,
                quality_factor=quality_factor,
                priority=1 if abs(elevation) < 0.2 else 2,
                description=f"Helix({i},Q{quality_factor:.1f})"
            )
            points.append(point)
        
        logger.info(f"✅ Generated {len(points)} advanced helix scan points")
        return points


class TableScanPattern(ScanPattern):
    """Erweiterte Table-Scan kompatibel mit main.py."""
    
    def __init__(self, steps: int = 24, height_levels: int = 1, 
                 radius: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.steps = max(8, min(steps, 48))  # Erweitert!
        self.height_levels = max(1, min(height_levels, 5))
        self.radius = radius
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        
        for level in range(self.height_levels):
            if self.height_levels > 1:
                elevation = -0.4 + level * 0.8 / (self.height_levels - 1)
            else:
                elevation = 0.0
            
            for step in range(self.steps):
                azimuth = (step / self.steps) * 2 * math.pi - math.pi
                
                # Quality basierend auf Elevation-Level
                quality_factor = 1.3 if level == 0 else 1.1 if level == 1 else 1.0
                priority = level + 1
                
                point = self.create_intelligent_scan_point(
                    azimuth, elevation,
                    quality_factor=quality_factor,
                    priority=min(3, priority),
                    description=f"Table({level},{step})"
                )
                points.append(point)
        
        logger.info(f"✅ Generated {len(points)} advanced table scan points")
        return points


class StatueSpiralPattern(SpiralScanPattern):
    """Erweiterte Statue-Spiral mit Statue-optimierten Parametern."""
    
    def __init__(self, height_range: float = 0.25, revolutions: int = 8, 
                 points_per_rev: int = 48, radius_start: float = 0.08,
                 radius_end: float = 0.12, **kwargs):
        # Übersetze Parameter für SpiralScanPattern
        points_per_turn = max(6, points_per_rev // revolutions)
        super().__init__(turns=revolutions, points_per_turn=points_per_turn, 
                        max_radius=0.8, dynamic_density=True, **kwargs)
        self.name = "Advanced Statue Spiral Scan"


class QuickScanPattern(RasterScanPattern):
    """Erweiterte Quick Scan mit optimierter Geschwindigkeit."""
    
    def __init__(self, **kwargs):
        super().__init__(rows=5, cols=5, speed=0.6, settle_time=0.3, 
                        adaptive_quality=True, zigzag=True, **kwargs)
        self.name = "Advanced Quick Scan"


class DetailedScanPattern(RasterScanPattern):
    """Erweiterte Detailed Scan mit maximaler Qualität."""
    
    def __init__(self, **kwargs):
        super().__init__(rows=12, cols=12, speed=0.2, settle_time=1.0, 
                        adaptive_quality=True, zigzag=True, **kwargs)
        self.name = "Advanced Detailed Scan"


class SmallObjectPreset(ScanPattern):
    """Erweiterte Small Object Preset."""
    
    def generate_points(self) -> List[ScanPoint]:
        points = []
        positions = [
            (0, 0, 1.5), (0.15, 0, 1.4), (-0.15, 0, 1.4), (0, 0.15, 1.3), (0, -0.15, 1.3),
            (0.12, 0.12, 1.2), (-0.12, 0.12, 1.2), (0.12, -0.12, 1.2), (-0.12, -0.12, 1.2)
        ]
        
        for i, (az, el, quality) in enumerate(positions):
            point = self.create_intelligent_scan_point(
                az, el, distance=0.12, quality_factor=quality, priority=1,
                description=f"SmallObj({i},Q{quality:.1f})"
            )
            points.append(point)
        
        logger.info(f"✅ Generated {len(points)} advanced small object scan points")
        return points


class LargeObjectPreset(SphericalScanPattern):
    """Erweiterte Large Object Preset."""
    
    def __init__(self, **kwargs):
        super().__init__(latitude_bands=6, longitude_points=12, 
                        geodesic_optimization=True, **kwargs)
        self.name = "Advanced Large Object Preset"
    
    def generate_points(self) -> List[ScanPoint]:
        points = super().generate_points()
        
        # Anpassung für große Objekte
        for point in points:
            point.distance = 0.25
            point.speed = 0.35
            point.expected_quality *= 1.1  # Leicht erhöhte Qualität-Erwartung
        
        logger.info(f"✅ Generated {len(points)} advanced large object scan points")
        return points


class SmartScanSelector(ScanPattern):
    """ERWEITERTE Smart Selector mit AI-ähnlicher Objekt-Erkennung."""
    
    def __init__(self, object_type: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self.object_type = object_type.lower()
    
    def generate_points(self) -> List[ScanPoint]:
        """Erweiterte automatische Pattern-Auswahl mit Qualitäts-Optimierung."""
        
        if self.object_type in ["small", "jewelry", "coin", "watch"]:
            pattern = SmallObjectPreset(center_position=self.center_position)
        elif self.object_type in ["large", "statue", "sculpture", "furniture"]:
            pattern = LargeObjectPreset(center_position=self.center_position)
        elif self.object_type in ["cylinder", "bottle", "can", "tube"]:
            pattern = HelixScanPattern(center_position=self.center_position, 
                                     turns=5, points_per_turn=16)  # Erweiterte Parameter
        elif self.object_type in ["flat", "document", "book", "plate"]:
            pattern = TableScanPattern(center_position=self.center_position, 
                                     steps=32, height_levels=2)  # Erweiterte Parameter
        elif self.object_type in ["figure", "person", "animal"]:
            pattern = StatueSpiralPattern(center_position=self.center_position,
                                        revolutions=10, points_per_rev=60)  # Erweiterte Parameter
        else:
            # Default: Erweiterte Adaptive Scan
            pattern = AdaptiveScanPattern(center_position=self.center_position,
                                        initial_points=16, refinement_threshold=0.08)
        
        points = pattern.generate_points()
        logger.info(f"✅ Advanced smart selector chose {pattern.name} for {self.object_type}")
        return points


# ============== FACTORY FUNCTIONS (vollständig kompatibel) ==============

def create_scan_pattern(pattern_type: str, **kwargs) -> ScanPattern:
    """ERWEITERTE Factory-Funktion mit allen Pattern-Typen."""
    patterns = {
        'raster': RasterScanPattern,
        'spiral': SpiralScanPattern,
        'spherical': SphericalScanPattern,
        'turntable': TurntableScanPattern,
        'adaptive': AdaptiveScanPattern,
        'cobweb': CobwebScanPattern,
        'helix': HelixScanPattern,
        'table': TableScanPattern,
        'statue': StatueSpiralPattern,
        'quick': QuickScanPattern,
        'detailed': DetailedScanPattern,
        'small': SmallObjectPreset,
        'large': LargeObjectPreset,
        'smart': SmartScanSelector,
    }
    
    pattern_class = patterns.get(pattern_type.lower())
    if not pattern_class:
        raise ValueError(f"Unknown pattern type: {pattern_type}. Available: {list(patterns.keys())}")
    
    return pattern_class(**kwargs)


def get_pattern_presets() -> Dict[str, Dict]:
    """ERWEITERTE Pattern-Konfigurationen."""
    return {
        'quick_scan': {
            'type': 'quick',
            'rows': 6,
            'cols': 6,
            'speed': 0.6,
            'adaptive_quality': True
        },
        'detailed_scan': {
            'type': 'detailed',
            'rows': 12,
            'cols': 12,
            'speed': 0.2,
            'adaptive_quality': True
        },
        'cylindrical_object': {
            'type': 'helix',
            'turns': 6,
            'points_per_turn': 18,
            'speed': 0.3
        },
        'small_statue': {
            'type': 'statue',
            'radius_start': 0.08,
            'radius_end': 0.12,
            'height_range': 0.15,
            'revolutions': 8
        },
        'flat_surface': {
            'type': 'table',
            'steps': 36,
            'height_levels': 2,
            'radius': 0.15
        },
        'full_3d': {
            'type': 'spherical',
            'latitude_bands': 6,
            'longitude_points': 16,
            'geodesic_optimization': True
        }
    }


# ============== LEGACY COMPATIBILITY ==============

def create_raster_scan(**kwargs):
    return RasterScanPattern(**kwargs).generate_points()

def create_spiral_scan(**kwargs):
    return SpiralScanPattern(**kwargs).generate_points()

def create_spherical_scan(**kwargs):
    return SphericalScanPattern(**kwargs).generate_points()

def create_adaptive_scan(**kwargs):
    return AdaptiveScanPattern(**kwargs).generate_points()

def create_helix_scan(**kwargs):
    return HelixScanPattern(**kwargs).generate_points()


# ============== PATTERN FACTORY ==============

class ScanPatternFactory:
    """ERWEITERTE Factory für alle Scan-Patterns."""
    
    PATTERNS = {
        1: ("Advanced Raster Scan", RasterScanPattern),
        2: ("Advanced Spiral Scan", SpiralScanPattern),
        3: ("Advanced Spherical Scan", SphericalScanPattern),
        4: ("Advanced Turntable Scan", TurntableScanPattern),
        5: ("Advanced Cobweb Scan", CobwebScanPattern),
        6: ("Advanced Adaptive Scan", AdaptiveScanPattern),
        7: ("Advanced Helix Scan", HelixScanPattern),
        8: ("Advanced Statue Spiral Scan", StatueSpiralPattern),
        9: ("Advanced Table Scan", TableScanPattern),
        10: ("Advanced Quick Scan", QuickScanPattern),
        11: ("Advanced Detailed Scan", DetailedScanPattern),
        12: ("Advanced Small Object Preset", SmallObjectPreset),
        13: ("Advanced Large Object Preset", LargeObjectPreset),
        14: ("Advanced Smart Scan Selector", SmartScanSelector),
    }
    
    @classmethod
    def create_pattern(cls, pattern_id: int, **kwargs) -> Optional[ScanPattern]:
        if pattern_id in cls.PATTERNS:
            name, pattern_class = cls.PATTERNS[pattern_id]
            return pattern_class(**kwargs)
        return None
    
    @classmethod
    def get_pattern_name(cls, pattern_id: int) -> str:
        if pattern_id in cls.PATTERNS:
            return cls.PATTERNS[pattern_id][0]
        return "Unknown Pattern"


# ============== MAIN TEST ==============

if __name__ == "__main__":
    """ERWEITERTE Tests der Advanced Scan-Patterns."""
    
    print("🧪 Testing RoArm M3 ADVANCED Intelligent Scan Patterns...")
    
    # Test der korrigierten Safety-Validierung
    test_positions = {
        "base": 0.0,
        "shoulder": 0.35,
        "elbow": 1.22,
        "wrist": -1.20,  # SICHERER Wert!
        "roll": 1.57,
        "hand": 2.5      # SICHERER Wert!
    }
    
    is_safe, errors = SafetyValidator.validate_position(test_positions, debug=True)
    print(f"✅ Advanced safety test: safe={is_safe}")
    if not is_safe:
        print(f"❌ Errors found: {errors}")
    
    # Test erweiterte Pattern
    advanced_patterns = [
        RasterScanPattern(adaptive_quality=True),
        AdaptiveScanPattern(initial_points=16),
        HelixScanPattern(points_per_turn=16, turns=4),
        SmartScanSelector(object_type="cylinder"),
    ]
    
    for pattern in advanced_patterns:
        print(f"\n🔍 Testing {pattern.name}...")
        try:
            points = pattern.generate_points()
            print(f"   ✅ Generated {len(points)} advanced points")
            
            if points:
                # Test erste 3 Punkte
                for i, point in enumerate(points[:3]):
                    is_safe, errors = SafetyValidator.validate_position(point.positions)
                    status = "✅ SAFE" if is_safe else f"❌ UNSAFE: {len(errors)} errors"
                    quality = f"Q{point.expected_quality:.1f}"
                    priority = f"P{point.priority}"
                    print(f"   Point {i+1}: {status} ({quality}, {priority})")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All ADVANCED tests completed!")
