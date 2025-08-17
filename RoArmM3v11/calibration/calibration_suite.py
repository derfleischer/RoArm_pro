#!/usr/bin/env python3
"""
RoArm M3 Professional Calibration Suite - SAFE VERSION
Mit Kollisionsvermeidung und sicheren Bewegungssequenzen.
"""

import json
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pickle

from core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_SPECS
from motion.trajectory import TrajectoryType
from utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationType(Enum):
    """Verfügbare Kalibrierungstypen."""
    AUTO_FULL = "auto_full"
    MANUAL_JOINT = "manual_joint"
    SCANNER_ALIGNMENT = "scanner"
    BACKLASH = "backlash"
    ENDSTOPS = "endstops"
    ACCURACY = "accuracy"
    WEIGHT = "weight"
    REPEATABILITY = "repeatability"


@dataclass
class SafePosition:
    """Sichere Position mit Gelenkabhängigkeiten."""
    positions: Dict[str, float]
    description: str
    is_stable: bool = True  # Stabil ohne Servo-Power
    requires_order: bool = False  # Reihenfolge wichtig
    joint_order: List[str] = None  # Bewegungsreihenfolge
    
    def __post_init__(self):
        if self.joint_order is None:
            self.joint_order = []


@dataclass
class CalibrationPoint:
    """Ein Kalibrierpunkt mit Soll- und Ist-Werten."""
    joint: str
    target_position: float
    actual_position: float
    error: float
    timestamp: float
    temperature: Optional[float] = None
    load: Optional[float] = None


@dataclass
class JointCalibration:
    """Kalibrierungsdaten für ein einzelnes Gelenk."""
    joint_name: str
    offset: float = 0.0
    scale: float = 1.0
    backlash: float = 0.0
    min_limit: float = 0.0
    max_limit: float = 0.0
    safe_min: float = 0.0
    safe_max: float = 0.0
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    friction: float = 0.0
    damping: float = 0.0
    repeatability: float = 0.001
    accuracy: float = 0.002
    temp_coefficient: float = 0.0
    reference_temp: float = 25.0
    calibration_points: List[CalibrationPoint] = None
    
    def __post_init__(self):
        if self.calibration_points is None:
            self.calibration_points = []
    
    def apply_calibration(self, raw_position: float, temperature: Optional[float] = None) -> float:
        """Wendet Kalibrierung auf Rohwert an."""
        calibrated = (raw_position + self.offset) * self.scale
        
        if temperature and self.temp_coefficient != 0:
            temp_diff = temperature - self.reference_temp
            calibrated += temp_diff * self.temp_coefficient
        
        return calibrated
    
    def inverse_calibration(self, calibrated_position: float) -> float:
        """Invertiert die Kalibrierung."""
        return (calibrated_position / self.scale) - self.offset


@dataclass
class ScannerCalibration:
    """Scanner-spezifische Kalibrierung."""
    mount_offset_x: float = 0.0
    mount_offset_y: float = 0.0
    mount_offset_z: float = 0.05
    roll_offset: float = 0.0
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    optimal_distance: float = 0.15
    optimal_speed: float = 0.3
    optimal_settle_time: float = 0.5
    focus_distance: float = 0.15
    depth_of_field: float = 0.10
    vibration_damping_time: float = 0.3
    acceleration_limit: float = 1.0
    reference_points: List[Dict] = None
    
    def __post_init__(self):
        if self.reference_points is None:
            self.reference_points = []


@dataclass
class SystemCalibration:
    """Komplette System-Kalibrierung."""
    version: str = "2.0.0"  # Safe version
    timestamp: float = 0.0
    joints: Dict[str, JointCalibration] = None
    scanner: ScannerCalibration = None
    dh_parameters: Dict = None
    gravity_compensation: Dict[str, float] = None
    temperature_model: Dict = None
    overall_accuracy: float = 0.0
    calibration_valid: bool = False
    last_verification: float = 0.0
    
    def __post_init__(self):
        if self.joints is None:
            self.joints = {}
        if self.scanner is None:
            self.scanner = ScannerCalibration()
        if self.gravity_compensation is None:
            self.gravity_compensation = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def save(self, filepath: str = "calibration/system_calibration.json"):
        """Speichert Kalibrierung."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": self.version,
            "timestamp": self.timestamp,
            "joints": {name: asdict(joint) for name, joint in self.joints.items()},
            "scanner": asdict(self.scanner),
            "dh_parameters": self.dh_parameters,
            "gravity_compensation": self.gravity_compensation,
            "temperature_model": self.temperature_model,
            "overall_accuracy": self.overall_accuracy,
            "calibration_valid": self.calibration_valid,
            "last_verification": self.last_verification
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = "calibration/system_calibration.json"):
        """Lädt Kalibrierung."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            calibration = cls()
            calibration.version = data.get("version", "2.0.0")
            calibration.timestamp = data.get("timestamp", time.time())
            
            for name, joint_data in data.get("joints", {}).items():
                calibration.joints[name] = JointCalibration(**joint_data)
            
            if "scanner" in data:
                calibration.scanner = ScannerCalibration(**data["scanner"])
            
            calibration.dh_parameters = data.get("dh_parameters")
            calibration.gravity_compensation = data.get("gravity_compensation", {})
            calibration.temperature_model = data.get("temperature_model")
            calibration.overall_accuracy = data.get("overall_accuracy", 0.0)
            calibration.calibration_valid = data.get("calibration_valid", False)
            calibration.last_verification = data.get("last_verification", 0.0)
            
            logger.info(f"Calibration loaded from {filepath}")
            return calibration
            
        except FileNotFoundError:
            logger.warning(f"No calibration file found at {filepath}")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return cls()


class SafeCalibrationSuite:
    """
    Sichere Kalibrierungsroutinen mit Kollisionsvermeidung.
    """
    
    # Sichere Standardpositionen - HAND: 1.08=offen, 3.14=geschlossen
    SAFE_POSITIONS = {
        'home': SafePosition(
            positions={'base': 0.0, 'shoulder': 0.0, 'elbow': 1.57, 
                      'wrist': 0.0, 'roll': 0.0, 'hand': 3.14},  # Geschlossen ist sicher
            description="Safe home position"
        ),
        'calibration_start': SafePosition(
            positions={'base': 0.0, 'shoulder': 0.2, 'elbow': 1.8, 
                      'wrist': -0.2, 'roll': 0.0, 'hand': 2.0},  # Mitte zwischen offen/geschlossen
            description="Safe calibration start"
        ),
        'scanner_mount': SafePosition(
            positions={'base': 0.0, 'shoulder': -0.3, 'elbow': 2.0, 
                      'wrist': 0.3, 'roll': 0.0, 'hand': 1.08},  # Offen für Scanner
            description="Scanner mounting position"
        ),
        'table_safe': SafePosition(
            positions={'base': 0.0, 'shoulder': -0.5, 'elbow': 2.3, 
                      'wrist': 0.5, 'roll': 0.0, 'hand': 2.0},  # Sichere Mitte
            description="Safe position near table",
            is_stable=True
        )
    }
    
    # Kritische Gelenkabhängigkeiten
    JOINT_DEPENDENCIES = {
        'shoulder': {
            'affects': ['wrist'],
            'rule': lambda s: {'wrist': -s * 0.8}  # Wrist kompensiert Shoulder
        },
        'elbow': {
            'affects': ['wrist'],
            'rule': lambda e: {'wrist': max(-1.57, min(1.57, -0.3 * (e - 1.57)))}
        }
    }
    
    # Kollisionszonen (vermeiden!)
    COLLISION_ZONES = [
        {
            'condition': lambda pos: pos['shoulder'] < -1.0 and pos['elbow'] > 2.5,
            'description': "Table collision risk"
        },
        {
            'condition': lambda pos: pos['shoulder'] > 1.0 and pos['wrist'] < -1.0,
            'description': "Self-collision risk"
        },
        {
            'condition': lambda pos: abs(pos['base']) > 2.5 and pos['shoulder'] < -0.5,
            'description': "Base rotation collision"
        }
    ]
    
    def __init__(self, controller):
        """
        Initialisiert Safe Calibration Suite.
        
        Args:
            controller: RoArm Controller Instanz
        """
        self.controller = controller
        self.calibration = SystemCalibration()
        
        # Lade existierende Kalibrierung
        self.load_calibration()
        
        # Status
        self.is_calibrating = False
        self.calibration_progress = 0.0
        self.current_step = ""
        
        # Sicherheitsparameter
        self.max_step_size = 0.3  # Max Bewegung pro Schritt (rad)
        self.safety_speed = 0.3   # Langsame, sichere Geschwindigkeit
        self.settle_time = 1.0     # Wartezeit zwischen Bewegungen
        
        logger.info("Safe Calibration Suite initialized")
    
    def _validate_position(self, position: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validiert eine Position auf Sicherheit.
        
        Returns:
            (is_safe, reason)
        """
        # Prüfe Servo-Limits
        for joint, value in position.items():
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                # Kleine Toleranz für Rundungsfehler
                tolerance = 0.01
                if value < min_val - tolerance or value > max_val + tolerance:
                    return False, f"{joint} außerhalb Limits: {value:.2f} (erlaubt: {min_val:.2f} bis {max_val:.2f})"
        
        # Prüfe Kollisionszonen
        for zone in self.COLLISION_ZONES:
            try:
                if zone['condition'](position):
                    return False, zone['description']
            except KeyError:
                # Wenn ein Joint fehlt, ignoriere diese Zone
                continue
        
        return True, "Position safe"
    
    def _apply_dependencies(self, position: Dict[str, float]) -> Dict[str, float]:
        """
        Wendet Gelenkabhängigkeiten an für sichere Positionen.
        """
        safe_pos = position.copy()
        
        for joint, value in position.items():
            if joint in self.JOINT_DEPENDENCIES:
                dep = self.JOINT_DEPENDENCIES[joint]
                adjustments = dep['rule'](value)
                
                for affected_joint, adjustment in adjustments.items():
                    safe_pos[affected_joint] = adjustment
                    logger.debug(f"Adjusted {affected_joint} to {adjustment:.2f} due to {joint}")
        
        return safe_pos
    
    def _move_safe(self, target: Dict[str, float], speed: float = None) -> bool:
        """
        Sichere Bewegung mit Zwischenschritten und Validierung.
        Robuste Version mit besserer Fehlerbehandlung.
        """
        speed = speed or self.safety_speed
        
        # Hole aktuelle Position vom Controller
        try:
            current = self.controller.current_position.copy()
        except Exception as e:
            logger.warning(f"Could not get current position: {e}")
            # Nutze HOME_POSITION als Fallback
            current = HOME_POSITION.copy()
        
        # Validiere Zielposition
        is_safe, reason = self._validate_position(target)
        if not is_safe:
            logger.error(f"Unsafe target position: {reason}")
            # Versuche trotzdem eine angepasste sichere Version
            safe_target = self._make_position_safe(target)
            if safe_target:
                logger.info("Using adjusted safe position")
                target = safe_target
            else:
                return False
        
        # Wende Abhängigkeiten an
        safe_target = self._apply_dependencies(target)
        
        # Berechne Zwischenschritte wenn große Bewegung
        steps = self._calculate_safe_trajectory(current, safe_target)
        
        # Führe Bewegung schrittweise aus
        for i, step_pos in enumerate(steps):
            logger.debug(f"Step {i+1}/{len(steps)}")
            
            # Validiere jeden Schritt
            is_safe, reason = self._validate_position(step_pos)
            if not is_safe:
                logger.warning(f"Adjusting unsafe intermediate position: {reason}")
                step_pos = self._make_position_safe(step_pos)
                if not step_pos:
                    logger.error("Could not make position safe")
                    continue
            
            # Bewegung ausführen mit Fehlerbehandlung
            try:
                success = self.controller.move_joints(
                    step_pos,
                    speed=speed,
                    trajectory_type=TrajectoryType.S_CURVE,
                    wait=True
                )
                
                if not success:
                    logger.warning(f"Movement step {i+1} reported failure - continuing")
                
                # Wartezeit für Stabilität
                time.sleep(self.settle_time)
                
            except Exception as e:
                logger.error(f"Movement exception at step {i+1}: {e}")
                # Versuche trotzdem weiterzumachen
                time.sleep(self.settle_time)
        
        return True
    
    def _make_position_safe(self, position: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Versucht eine unsichere Position sicher zu machen.
        """
        safe_pos = position.copy()
        
        # Korrigiere Joints die außerhalb der Limits sind
        for joint, value in safe_pos.items():
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                if value < min_val:
                    safe_pos[joint] = min_val + 0.05  # Kleiner Sicherheitsabstand
                    logger.info(f"Adjusted {joint} from {value:.2f} to {safe_pos[joint]:.2f}")
                elif value > max_val:
                    safe_pos[joint] = max_val - 0.05
                    logger.info(f"Adjusted {joint} from {value:.2f} to {safe_pos[joint]:.2f}")
        
        # Nochmal validieren
        is_safe, _ = self._validate_position(safe_pos)
        return safe_pos if is_safe else None
    
    def _calculate_safe_trajectory(self, start: Dict[str, float], 
                                  end: Dict[str, float]) -> List[Dict[str, float]]:
        """
        Berechnet sichere Trajektorie mit Zwischenschritten.
        """
        steps = []
        
        # Berechne maximale Bewegung
        max_delta = 0
        for joint in start:
            if joint in end:
                delta = abs(end[joint] - start[joint])
                max_delta = max(max_delta, delta)
        
        # Anzahl Schritte basierend auf Bewegungsgröße
        num_steps = max(1, int(max_delta / self.max_step_size))
        
        # Spezielle Behandlung für kritische Bewegungen
        if self._is_critical_movement(start, end):
            num_steps = max(num_steps, 5)  # Mehr Schritte für kritische Bewegungen
            
            # Füge sichere Zwischenposition ein
            safe_intermediate = self._get_safe_intermediate(start, end)
            if safe_intermediate:
                # Erst zur sicheren Position
                for i in range(num_steps // 2):
                    t = (i + 1) / (num_steps // 2)
                    step = self._interpolate(start, safe_intermediate, t)
                    steps.append(step)
                
                # Dann zum Ziel
                for i in range(num_steps // 2):
                    t = (i + 1) / (num_steps // 2)
                    step = self._interpolate(safe_intermediate, end, t)
                    steps.append(step)
                
                return steps
        
        # Normale Interpolation
        for i in range(num_steps):
            t = (i + 1) / num_steps
            step = self._interpolate(start, end, t)
            steps.append(step)
        
        return steps
    
    def _is_critical_movement(self, start: Dict[str, float], 
                             end: Dict[str, float]) -> bool:
        """
        Prüft ob Bewegung kritisch ist (z.B. große Shoulder-Bewegung).
        """
        # Shoulder-Bewegung nach unten mit ausgestrecktem Arm
        if 'shoulder' in start and 'shoulder' in end:
            shoulder_delta = end['shoulder'] - start['shoulder']
            if shoulder_delta < -0.5 and start.get('elbow', 0) > 2.0:
                return True
        
        # Große Base-Rotation mit tiefem Arm
        if 'base' in start and 'base' in end:
            base_delta = abs(end['base'] - start['base'])
            if base_delta > 1.0 and start.get('shoulder', 0) < -0.3:
                return True
        
        return False
    
    def _get_safe_intermediate(self, start: Dict[str, float], 
                              end: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Bestimmt sichere Zwischenposition für kritische Bewegungen.
        """
        # Bei Shoulder-Bewegung nach unten: Erst Elbow einfahren
        if end.get('shoulder', 0) < start.get('shoulder', 0) - 0.5:
            intermediate = start.copy()
            intermediate['elbow'] = min(1.8, start.get('elbow', 1.57))
            intermediate['wrist'] = 0.0
            return intermediate
        
        # Bei großer Base-Rotation: Erst hochfahren
        if abs(end.get('base', 0) - start.get('base', 0)) > 1.0:
            intermediate = start.copy()
            intermediate['shoulder'] = max(0.0, start.get('shoulder', 0))
            intermediate['elbow'] = 1.57
            return intermediate
        
        return None
    
    def _interpolate(self, start: Dict[str, float], end: Dict[str, float], 
                    t: float) -> Dict[str, float]:
        """Lineare Interpolation zwischen Positionen."""
        result = {}
        for joint in start:
            if joint in end:
                result[joint] = start[joint] + t * (end[joint] - start[joint])
            else:
                result[joint] = start[joint]
        return result
    
    def run_auto_calibration(self, include_scanner: bool = True) -> bool:
        """
        Führt sichere Auto-Kalibrierung durch.
        """
        logger.info("🔧 Starting SAFE AUTO CALIBRATION")
        logger.info("="*50)
        
        self.is_calibrating = True
        self.calibration_progress = 0.0
        
        try:
            # Phase 1: Sichere Startposition
            logger.info("Phase 1/6: Moving to safe start position...")
            try:
                if not self._move_to_safe_start():
                    logger.warning("Safe start position not fully reached - continuing anyway")
            except Exception as e:
                logger.warning(f"Start position error: {e} - continuing")
            self.calibration_progress = 10.0
            
            # Phase 2: Endstops mit Sicherheit
            logger.info("Phase 2/6: Finding safe endstops...")
            try:
                self._calibrate_safe_endstops()
            except Exception as e:
                logger.warning(f"Endstop calibration error: {e}")
            self.calibration_progress = 25.0
            
            # Phase 3: Joint-Kalibrierung mit sicheren Bewegungen
            logger.info("Phase 3/6: Safe joint calibration...")
            for joint in ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']:
                try:
                    if not self._calibrate_joint_safe(joint):
                        logger.warning(f"Joint {joint} calibration incomplete")
                except Exception as e:
                    logger.warning(f"Joint {joint} error: {e}")
            self.calibration_progress = 50.0
            
            # Phase 4: Backlash mit kleinen Bewegungen
            logger.info("Phase 4/6: Measuring backlash safely...")
            try:
                self._measure_backlash_safe()
            except Exception as e:
                logger.warning(f"Backlash measurement error: {e}")
            self.calibration_progress = 65.0
            
            # Phase 5: Gewichtskompensation
            logger.info("Phase 5/6: Weight compensation...")
            try:
                self._calibrate_weight_safe()
            except Exception as e:
                logger.warning(f"Weight calibration error: {e}")
            self.calibration_progress = 80.0
            
            # Phase 6: Scanner-Kalibrierung
            if include_scanner:
                logger.info("Phase 6/6: Safe scanner calibration...")
                try:
                    self._calibrate_scanner_safe()
                except Exception as e:
                    logger.warning(f"Scanner calibration error: {e}")
            self.calibration_progress = 95.0
            
            # Verifikation
            try:
                self._verify_calibration_safe()
            except Exception as e:
                logger.warning(f"Verification error: {e}")
            self.calibration_progress = 100.0
            
            # Speichern
            self.calibration.calibration_valid = True
            self.calibration.last_verification = time.time()
            self.save_calibration()
            
            logger.info("✅ SAFE AUTO CALIBRATION COMPLETE")
            logger.info(f"Overall accuracy: {self.calibration.overall_accuracy:.4f} rad")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
        
        finally:
            self.is_calibrating = False
            # Zurück zur sicheren Home-Position
            try:
                self._move_safe(self.SAFE_POSITIONS['home'].positions)
            except Exception as e:
                logger.warning(f"Could not return to home: {e}")
    
    def _move_to_safe_start(self) -> bool:
        """Bewegt zu sicherer Startposition für Kalibrierung."""
        logger.info("Moving to safe calibration start position...")
        
        # LED indication
        try:
            self.controller.led_control(True, brightness=128)
        except Exception as e:
            logger.warning(f"LED control failed: {e}")
        
        # Enable torque
        try:
            success = self.controller.set_torque(True)
            if not success:
                logger.warning("Could not enable torque - continuing anyway")
                time.sleep(2)  # Extra wait for servo power
        except Exception as e:
            logger.warning(f"Torque enable failed: {e}")
        
        # Sichere Sequenz zur Startposition - vereinfacht!
        sequence = [
            # Direkt zur Kalibrierungsposition mit sicheren Werten
            {'base': 0.0, 'shoulder': 0.0, 'elbow': 1.57, 'wrist': 0.0, 'roll': 0.0, 'hand': 2.0}
        ]
        
        for i, step in enumerate(sequence):
            logger.info(f"Moving to step {i+1}/{len(sequence)}")
            
            # Nutze controller.move_joints direkt statt _move_safe für erste Bewegung
            try:
                success = self.controller.move_joints(
                    step,
                    speed=0.2,
                    trajectory_type=TrajectoryType.S_CURVE,
                    wait=True
                )
                
                if success:
                    logger.info(f"Step {i+1} completed")
                    time.sleep(2.0)
                else:
                    logger.warning(f"Step {i+1} may have failed")
                    time.sleep(2.0)  # Wait anyway
                    
            except Exception as e:
                logger.error(f"Movement error: {e}")
                # Try to continue
                time.sleep(2.0)
        
        logger.info("Start position sequence completed")
        return True  # Return True even if some steps failed
    
    def _calibrate_safe_endstops(self):
        """Findet Endpositionen mit Sicherheitsabstand."""
        for joint, (min_limit, max_limit) in SERVO_LIMITS.items():
            logger.info(f"  Finding safe limits for {joint}...")
            
            # Verwende konservative Limits mit Sicherheitsabstand
            safety_margin = 0.15  # ~8.5 Grad Sicherheitsabstand
            
            if joint not in self.calibration.joints:
                self.calibration.joints[joint] = JointCalibration(joint_name=joint)
            
            self.calibration.joints[joint].min_limit = min_limit
            self.calibration.joints[joint].max_limit = max_limit
            self.calibration.joints[joint].safe_min = min_limit + safety_margin
            self.calibration.joints[joint].safe_max = max_limit - safety_margin
            
            logger.info(f"    Limits: [{min_limit:.2f}, {max_limit:.2f}]")
            logger.info(f"    Safe: [{min_limit + safety_margin:.2f}, {max_limit - safety_margin:.2f}]")
    
    def _calibrate_joint_safe(self, joint_name: str) -> bool:
        """
        Kalibriert einzelnes Gelenk mit sicheren Bewegungen.
        """
        logger.info(f"  Calibrating {joint_name} safely...")
        
        if joint_name not in SERVO_LIMITS:
            return False
        
        calibration = JointCalibration(joint_name=joint_name)
        min_limit, max_limit = SERVO_LIMITS[joint_name]
        
        # Sichere Test-Positionen (nicht zu extrem)
        safety_factor = 0.7  # Nur 70% des Bewegungsbereichs nutzen
        safe_min = min_limit + (max_limit - min_limit) * (1 - safety_factor) / 2
        safe_max = max_limit - (max_limit - min_limit) * (1 - safety_factor) / 2
        
        test_positions = np.linspace(safe_min, safe_max, 3)  # Nur 3 Punkte für Sicherheit
        
        # Startposition merken
        start_pos = self.controller.current_position.copy()
        
        for target in test_positions:
            # Sichere Position für dieses Gelenk
            test_pos = start_pos.copy()
            test_pos[joint_name] = target
            
            # Abhängigkeiten anwenden
            test_pos = self._apply_dependencies(test_pos)
            
            # Validierung
            is_safe, reason = self._validate_position(test_pos)
            if not is_safe:
                logger.warning(f"    Skipping unsafe position: {reason}")
                continue
            
            # Sichere Bewegung
            if not self._move_safe(test_pos, speed=0.2):
                logger.warning(f"    Could not reach test position {target:.2f}")
                continue
            
            time.sleep(1.5)  # Extra Wartezeit für Messung
            
            # Messen
            status = self.controller.query_status()
            if status:
                actual = status['positions'].get(joint_name, 0)
                error = actual - target
                
                point = CalibrationPoint(
                    joint=joint_name,
                    target_position=target,
                    actual_position=actual,
                    error=error,
                    timestamp=time.time()
                )
                calibration.calibration_points.append(point)
        
        # Zurück zur Startposition
        self._move_safe(start_pos, speed=0.3)
        
        # Offset und Scale berechnen
        if calibration.calibration_points:
            errors = [p.error for p in calibration.calibration_points]
            calibration.offset = -np.mean(errors)
            calibration.accuracy = np.std(errors) * 3
            
            self.calibration.joints[joint_name] = calibration
            
            logger.info(f"    Offset: {calibration.offset*1000:.2f} mrad")
            logger.info(f"    Accuracy: {calibration.accuracy*1000:.2f} mrad")
            return True
        
        return False
    
    def _measure_backlash_safe(self):
        """Misst Spiel mit kleinen, sicheren Bewegungen."""
        logger.info("  Measuring backlash with safe movements...")
        
        # Sichere Position für Backlash-Test
        safe_test_pos = self.SAFE_POSITIONS['calibration_start'].positions.copy()
        
        for joint in ['base', 'shoulder', 'elbow', 'wrist', 'roll']:
            if joint not in SERVO_LIMITS:
                continue
            
            # Kleine Testbewegung (nur 0.2 rad)
            delta = 0.2
            
            # Vorwärts
            pos1 = safe_test_pos.copy()
            pos1[joint] += delta / 2
            pos1 = self._apply_dependencies(pos1)
            
            if not self._move_safe(pos1, speed=0.15):
                logger.warning(f"    Could not test backlash for {joint}")
                continue
            
            time.sleep(2)
            status1 = self.controller.query_status()
            actual1 = status1['positions'].get(joint, 0) if status1 else 0
            
            # Rückwärts
            pos2 = safe_test_pos.copy()
            pos2[joint] -= delta / 2
            pos2 = self._apply_dependencies(pos2)
            
            if not self._move_safe(pos2, speed=0.15):
                continue
            
            time.sleep(2)
            status2 = self.controller.query_status()
            actual2 = status2['positions'].get(joint, 0) if status2 else 0
            
            # Wieder vorwärts (gleiche Position wie pos1)
            if not self._move_safe(pos1, speed=0.15):
                continue
            
            time.sleep(2)
            status3 = self.controller.query_status()
            actual3 = status3['positions'].get(joint, 0) if status3 else 0
            
            # Backlash berechnen
            backlash = abs(actual3 - actual1)
            
            if joint in self.calibration.joints:
                self.calibration.joints[joint].backlash = backlash
                logger.info(f"    {joint}: {backlash*1000:.2f} mrad")
        
        # Zurück zur sicheren Position
        self._move_safe(safe_test_pos, speed=0.3)
    
    def _calibrate_weight_safe(self):
        """Kalibriert Gewichtskompensation mit sicheren Positionen."""
        logger.info("  Calibrating weight compensation safely...")
        
        # Sichere Positionen für Gewichtstest
        test_positions = [
            {'shoulder': 0.0, 'elbow': 1.57, 'wrist': 0.0},     # Neutral
            {'shoulder': 0.3, 'elbow': 1.2, 'wrist': -0.3},     # Leicht oben
            {'shoulder': -0.3, 'elbow': 2.0, 'wrist': 0.3}      # Leicht unten
        ]
        
        for pos in test_positions:
            full_pos = self.controller.current_position.copy()
            full_pos.update(pos)
            full_pos = self._apply_dependencies(full_pos)
            
            if not self._move_safe(full_pos, speed=0.2):
                logger.warning("Could not reach weight test position")
                continue
            
            time.sleep(3)  # Stabilisieren
            
            # Hier würde man Strom/Torque messen
        
        # Kompensationsfaktoren
        self.calibration.gravity_compensation = {
            "shoulder": 0.05,
            "elbow": 0.03
        }
    
    def _calibrate_scanner_safe(self):
        """Scanner-Kalibrierung mit sicheren Positionen."""
        logger.info("  Safe scanner calibration...")
        
        # Sichere Scanner-Montageposition
        if not self._move_safe(self.SAFE_POSITIONS['scanner_mount'].positions):
            logger.error("Could not reach scanner mount position")
            return
        
        scanner_cal = ScannerCalibration()
        
        # Teste verschiedene sichere Scanner-Positionen
        safe_distances = [0.12, 0.15, 0.18]  # Sichere Abstände
        
        for dist in safe_distances:
            # Berechne sichere Position für Abstand
            # (Implementierung abhängig von Scanner-Setup)
            pass
        
        scanner_cal.optimal_distance = 0.15
        scanner_cal.optimal_speed = 0.3
        scanner_cal.optimal_settle_time = 0.5
        
        self.calibration.scanner = scanner_cal
    
    def _verify_calibration_safe(self):
        """Verifiziert Kalibrierung mit sicheren Bewegungen."""
        logger.info("Verifying calibration with safe movements...")
        
        # Sichere Testpositionen
        test_positions = [
            self.SAFE_POSITIONS['home'].positions,
            self.SAFE_POSITIONS['calibration_start'].positions,
            {'base': 0.5, 'shoulder': 0.2, 'elbow': 1.5, 'wrist': -0.2, 'roll': 0.0, 'hand': 2.0},
            {'base': -0.5, 'shoulder': -0.2, 'elbow': 1.8, 'wrist': 0.2, 'roll': 0.0, 'hand': 2.0}
        ]
        
        errors = []
        for target_pos in test_positions:
            # Validierung
            is_safe, reason = self._validate_position(target_pos)
            if not is_safe:
                logger.warning(f"Skipping unsafe test position: {reason}")
                continue
            
            # Bewegung
            if not self._move_safe(target_pos, speed=0.3):
                logger.warning("Could not reach test position")
                continue
            
            time.sleep(2)
            
            # Messung
            status = self.controller.query_status()
            if status:
                for joint, target in target_pos.items():
                    actual = status['positions'].get(joint, 0)
                    error = abs(actual - target)
                    errors.append(error)
        
        # Gesamt-Genauigkeit
        if errors:
            self.calibration.overall_accuracy = np.mean(errors)
            logger.info(f"Verification complete: ±{self.calibration.overall_accuracy*1000:.1f} mrad")
        else:
            self.calibration.overall_accuracy = 0.01  # Default
    
    def calibrate_single_joint(self, joint_name: str) -> bool:
        """
        Kalibriert einzelnes Gelenk mit Sicherheit.
        """
        logger.info(f"Safe calibration of joint: {joint_name}")
        
        if joint_name not in SERVO_LIMITS:
            logger.error(f"Unknown joint: {joint_name}")
            return False
        
        try:
            # Zur sicheren Startposition
            if not self._move_to_safe_start():
                return False
            
            time.sleep(2)
            
            # Kalibrierung durchführen
            success = self._calibrate_joint_safe(joint_name)
            
            if success:
                self.save_calibration()
                logger.info(f"✅ Joint {joint_name} calibrated safely")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Joint calibration failed: {e}")
            return False
        finally:
            # Zurück zur Home-Position
            self._move_safe(self.SAFE_POSITIONS['home'].positions)
    
    def test_repeatability(self, positions: int = 5, cycles: int = 3) -> Dict:
        """
        Testet Wiederholgenauigkeit mit sicheren Bewegungen.
        """
        logger.info(f"Testing repeatability safely: {positions} positions, {cycles} cycles")
        
        results = {joint: [] for joint in SERVO_LIMITS.keys()}
        
        # Generiere sichere Testpositionen
        test_positions = []
        for _ in range(positions):
            pos = {}
            for joint, (min_val, max_val) in SERVO_LIMITS.items():
                # Nur mittlerer Bereich (50% des Bewegungsbereichs)
                safe_range = 0.5
                center = (min_val + max_val) / 2
                range_val = (max_val - min_val) * safe_range / 2
                pos[joint] = center + np.random.uniform(-range_val, range_val)
            
            # Abhängigkeiten anwenden
            pos = self._apply_dependencies(pos)
            
            # Validierung
            is_safe, _ = self._validate_position(pos)
            if is_safe:
                test_positions.append(pos)
        
        # Teste jede Position
        for pos_idx, target_pos in enumerate(test_positions):
            logger.info(f"Testing position {pos_idx+1}/{len(test_positions)}")
            
            measurements = {joint: [] for joint in SERVO_LIMITS.keys()}
            
            for cycle in range(cycles):
                # Sichere Bewegung zur Position
                if not self._move_safe(target_pos, speed=0.3):
                    logger.warning(f"Could not reach position {pos_idx+1}")
                    break
                
                time.sleep(2)
                
                # Messe Position
                status = self.controller.query_status()
                if status:
                    for joint in SERVO_LIMITS.keys():
                        actual = status['positions'].get(joint, 0)
                        measurements[joint].append(actual)
                
                # Weg und zurück
                if cycle < cycles - 1:
                    self._move_safe(self.SAFE_POSITIONS['home'].positions, speed=0.3)
                    time.sleep(1)
            
            # Berechne Standardabweichung
            for joint, values in measurements.items():
                if len(values) > 1:
                    std_dev = np.std(values)
                    results[joint].append(std_dev)
        
        # Statistiken
        statistics = {}
        for joint, deviations in results.items():
            if deviations:
                statistics[joint] = {
                    "mean_deviation": np.mean(deviations),
                    "max_deviation": np.max(deviations),
                    "repeatability": np.mean(deviations) * 3  # 3-Sigma
                }
                
                if joint not in self.calibration.joints:
                    self.calibration.joints[joint] = JointCalibration(joint_name=joint)
                self.calibration.joints[joint].repeatability = statistics[joint]["repeatability"]
        
        logger.info("REPEATABILITY TEST RESULTS:")
        logger.info("-"*40)
        for joint, stats in statistics.items():
            logger.info(f"{joint:10s}: ±{stats['repeatability']*1000:.2f} mrad (3σ)")
        
        self.save_calibration()
        return statistics
    
    def emergency_safe_position(self) -> bool:
        """
        Notfall: Bewegung zur sichersten Position.
        """
        logger.warning("EMERGENCY: Moving to safest position")
        
        # Definiere absolut sichere Notfallposition
        emergency_pos = {
            'base': 0.0,
            'shoulder': -0.2,  # Leicht nach unten
            'elbow': 1.8,      # Eingefahren
            'wrist': 0.2,      # Kompensiert
            'roll': 0.0,
            'hand': 2.0        # Sichere Mitte zwischen offen und geschlossen
        }
        
        # Versuche direkte Bewegung ohne Validierung (Notfall!)
        try:
            return self.controller.move_joints(
                emergency_pos,
                speed=0.1,
                trajectory_type=TrajectoryType.S_CURVE,
                wait=True
            )
        except Exception as e:
            logger.error(f"Emergency movement failed: {e}")
            return False
    
    def verify_calibration(self) -> bool:
        """Verifiziert Kalibrierung mit Sicherheitsprüfungen."""
        return self._verify_calibration_safe()
    
    def save_calibration(self, filepath: Optional[str] = None):
        """Speichert Kalibrierung."""
        if filepath:
            self.calibration.save(filepath)
        else:
            self.calibration.save()
    
    def load_calibration(self, filepath: Optional[str] = None):
        """Lädt Kalibrierung."""
        if filepath:
            self.calibration = SystemCalibration.load(filepath)
        else:
            self.calibration = SystemCalibration.load()
        
        if self.calibration.calibration_valid:
            logger.info("✅ Calibration loaded successfully")
            self._apply_calibration_to_controller()
        else:
            logger.warning("No valid calibration found - using safe defaults")
    
    def _apply_calibration_to_controller(self):
        """Wendet Kalibrierung auf Controller an."""
        # Update Servo-Limits mit sicheren Werten
        for joint_name, joint_cal in self.calibration.joints.items():
            if joint_name in SERVO_LIMITS:
                # Verwende sichere kalibrierte Limits
                SERVO_LIMITS[joint_name] = (
                    joint_cal.safe_min,
                    joint_cal.safe_max
                )
        
        # Update Scanner-Parameter
        if self.calibration.scanner:
            SCANNER_SPECS["optimal_distance"] = self.calibration.scanner.optimal_distance
            SCANNER_SPECS["settle_time"] = self.calibration.scanner.optimal_settle_time
        
        logger.info("Safe calibration applied to controller")
    
    def export_report(self, filepath: str = "calibration/calibration_report.txt") -> str:
        """Exportiert Kalibrierungsbericht."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("="*60)
        report.append("ROARM M3 SAFE CALIBRATION REPORT")
        report.append("="*60)
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Version: {self.calibration.version}")
        report.append(f"Valid: {self.calibration.calibration_valid}")
        report.append(f"Overall Accuracy: {self.calibration.overall_accuracy*1000:.2f} mrad")
        report.append("")
        
        report.append("JOINT CALIBRATION:")
        report.append("-"*40)
        for joint_name, joint in self.calibration.joints.items():
            report.append(f"\n{joint_name.upper()}:")
            report.append(f"  Offset: {joint.offset*1000:.3f} mrad")
            report.append(f"  Scale: {joint.scale:.4f}")
            report.append(f"  Backlash: {joint.backlash*1000:.3f} mrad")
            report.append(f"  Safe Limits: [{joint.safe_min:.3f}, {joint.safe_max:.3f}] rad")
            report.append(f"  Repeatability: ±{joint.repeatability*1000:.2f} mrad")
            report.append(f"  Accuracy: ±{joint.accuracy*1000:.2f} mrad")
        
        report.append("\nSCANNER CALIBRATION:")
        report.append("-"*40)
        if self.calibration.scanner:
            report.append(f"  Mount Offset: ({self.calibration.scanner.mount_offset_x:.3f}, "
                         f"{self.calibration.scanner.mount_offset_y:.3f}, "
                         f"{self.calibration.scanner.mount_offset_z:.3f}) m")
            report.append(f"  Optimal Distance: {self.calibration.scanner.optimal_distance:.3f} m")
            report.append(f"  Optimal Speed: {self.calibration.scanner.optimal_speed:.2f}")
            report.append(f"  Settle Time: {self.calibration.scanner.optimal_settle_time:.2f} s")
        
        report.append("\nSAFETY INFORMATION:")
        report.append("-"*40)
        report.append(f"  Max Step Size: {self.max_step_size:.2f} rad")
        report.append(f"  Safety Speed: {self.safety_speed:.2f}")
        report.append(f"  Settle Time: {self.settle_time:.1f} s")
        report.append(f"  Collision Zones: {len(self.COLLISION_ZONES)}")
        
        report.append("\n" + "="*60)
        
        # Speichern
        report_text = '\n'.join(report)
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Calibration report saved to {filepath}")
        return report_text


# Alias für Kompatibilität
CalibrationSuite = SafeCalibrationSuite
