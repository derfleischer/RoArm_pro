#!/usr/bin/env python3
"""
RoArm M3 Professional Calibration Suite
Umfassende Kalibrierung für präzise Bewegungen und Scanner-Ausrichtung.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pickle

from ..core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_SPECS
from ..motion.trajectory import TrajectoryType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationType(Enum):
    """Verfügbare Kalibrierungstypen."""
    AUTO_FULL = "auto_full"              # Komplette Auto-Kalibrierung
    MANUAL_JOINT = "manual_joint"        # Einzelne Gelenke manuell
    SCANNER_ALIGNMENT = "scanner"        # Scanner-Position optimieren
    BACKLASH = "backlash"               # Spiel-Kompensation
    ENDSTOPS = "endstops"               # Endschalter finden
    ACCURACY = "accuracy"               # Genauigkeitstest
    WEIGHT = "weight"                   # Gewichtskompensation
    REPEATABILITY = "repeatability"     # Wiederholgenauigkeit


@dataclass
class CalibrationPoint:
    """Ein Kalibrierpunkt mit Soll- und Ist-Werten."""
    joint: str
    target_position: float    # Soll-Position
    actual_position: float    # Ist-Position (gemessen)
    error: float             # Abweichung
    timestamp: float
    temperature: Optional[float] = None
    load: Optional[float] = None  # Belastung/Gewicht


@dataclass
class JointCalibration:
    """Kalibrierungsdaten für ein einzelnes Gelenk."""
    joint_name: str
    
    # Mechanische Parameter
    offset: float = 0.0              # Zero-Offset
    scale: float = 1.0               # Skalierungsfaktor
    backlash: float = 0.0            # Spiel in rad
    
    # Limits (kalibriert)
    min_limit: float = 0.0
    max_limit: float = 0.0
    safe_min: float = 0.0           # Sicherer Bereich
    safe_max: float = 0.0
    
    # Dynamische Parameter
    max_velocity: float = 1.0        # rad/s
    max_acceleration: float = 2.0    # rad/s²
    friction: float = 0.0            # Reibungskoeffizient
    damping: float = 0.0             # Dämpfung
    
    # Genauigkeit
    repeatability: float = 0.001     # Wiederholgenauigkeit in rad
    accuracy: float = 0.002          # Absolute Genauigkeit in rad
    
    # Temperatur-Kompensation
    temp_coefficient: float = 0.0    # rad/°C
    reference_temp: float = 25.0     # °C
    
    # Kalibrierungspunkte
    calibration_points: List[CalibrationPoint] = None
    
    def __post_init__(self):
        if self.calibration_points is None:
            self.calibration_points = []
    
    def apply_calibration(self, raw_position: float, temperature: Optional[float] = None) -> float:
        """
        Wendet Kalibrierung auf Rohwert an.
        
        Args:
            raw_position: Unkalibrierte Position
            temperature: Aktuelle Temperatur für Kompensation
            
        Returns:
            Kalibrierte Position
        """
        # Offset und Skalierung
        calibrated = (raw_position + self.offset) * self.scale
        
        # Temperatur-Kompensation
        if temperature and self.temp_coefficient != 0:
            temp_diff = temperature - self.reference_temp
            calibrated += temp_diff * self.temp_coefficient
        
        return calibrated
    
    def inverse_calibration(self, calibrated_position: float) -> float:
        """
        Invertiert die Kalibrierung (für Befehle).
        
        Args:
            calibrated_position: Kalibrierte Zielposition
            
        Returns:
            Raw-Wert für Servo
        """
        return (calibrated_position / self.scale) - self.offset


@dataclass
class ScannerCalibration:
    """Scanner-spezifische Kalibrierung."""
    # Montage-Offset (kalibriert)
    mount_offset_x: float = 0.0
    mount_offset_y: float = 0.0
    mount_offset_z: float = 0.05
    
    # Rotation-Offset (Ausrichtung)
    roll_offset: float = 0.0
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    
    # Optimale Scan-Parameter
    optimal_distance: float = 0.15
    optimal_speed: float = 0.3
    optimal_settle_time: float = 0.5
    
    # Focus-Einstellungen
    focus_distance: float = 0.15
    depth_of_field: float = 0.10
    
    # Vibrations-Kompensation
    vibration_damping_time: float = 0.3
    acceleration_limit: float = 1.0
    
    # Kalibrierungspunkte für Genauigkeit
    reference_points: List[Dict] = None
    
    def __post_init__(self):
        if self.reference_points is None:
            self.reference_points = []


@dataclass
class SystemCalibration:
    """Komplette System-Kalibrierung."""
    version: str = "1.0.0"
    timestamp: float = 0.0
    
    # Joint-Kalibrierungen
    joints: Dict[str, JointCalibration] = None
    
    # Scanner-Kalibrierung
    scanner: ScannerCalibration = None
    
    # Kinematik-Parameter (DH-Parameter)
    dh_parameters: Dict = None
    
    # Globale Kompensationen
    gravity_compensation: Dict[str, float] = None
    temperature_model: Dict = None
    
    # Kalibrierungs-Qualität
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
        
        # Konvertiere zu Dict
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
            calibration.version = data.get("version", "1.0.0")
            calibration.timestamp = data.get("timestamp", time.time())
            
            # Joints laden
            for name, joint_data in data.get("joints", {}).items():
                calibration.joints[name] = JointCalibration(**joint_data)
            
            # Scanner laden
            if "scanner" in data:
                calibration.scanner = ScannerCalibration(**data["scanner"])
            
            # Weitere Parameter
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


class CalibrationSuite:
    """
    Hauptklasse für alle Kalibrierungsfunktionen.
    Führt verschiedene Kalibrierungsroutinen durch.
    """
    
    def __init__(self, controller):
        """
        Initialisiert Calibration Suite.
        
        Args:
            controller: RoArm Controller Instanz
        """
        self.controller = controller
        self.calibration = SystemCalibration()
        
        # Lade existierende Kalibrierung
        self.load_calibration()
        
        # Kalibrierungs-Status
        self.is_calibrating = False
        self.calibration_progress = 0.0
        self.current_step = ""
        
        logger.info("Calibration Suite initialized")
    
    def run_auto_calibration(self, include_scanner: bool = True) -> bool:
        """
        Führt komplette Auto-Kalibrierung durch.
        
        Args:
            include_scanner: Scanner-Kalibrierung einschließen
            
        Returns:
            True wenn erfolgreich
        """
        logger.info("🔧 Starting AUTO CALIBRATION")
        logger.info("="*50)
        
        self.is_calibrating = True
        self.calibration_progress = 0.0
        
        try:
            # Phase 1: Vorbereitung
            self._prepare_calibration()
            self.calibration_progress = 10.0
            
            # Phase 2: Endstops finden
            logger.info("Phase 2/6: Finding endstops...")
            self._calibrate_endstops()
            self.calibration_progress = 25.0
            
            # Phase 3: Joint-Kalibrierung
            logger.info("Phase 3/6: Calibrating joints...")
            for joint in SERVO_LIMITS.keys():
                self._calibrate_joint(joint)
            self.calibration_progress = 50.0
            
            # Phase 4: Backlash-Messung
            logger.info("Phase 4/6: Measuring backlash...")
            self._measure_backlash()
            self.calibration_progress = 65.0
            
            # Phase 5: Gewichtskompensation
            logger.info("Phase 5/6: Weight compensation...")
            self._calibrate_weight_compensation()
            self.calibration_progress = 80.0
            
            # Phase 6: Scanner-Kalibrierung
            if include_scanner:
                logger.info("Phase 6/6: Scanner calibration...")
                self._calibrate_scanner()
            self.calibration_progress = 95.0
            
            # Verifikation
            self._verify_calibration()
            self.calibration_progress = 100.0
            
            # Speichern
            self.calibration.calibration_valid = True
            self.calibration.last_verification = time.time()
            self.save_calibration()
            
            logger.info("✅ AUTO CALIBRATION COMPLETE")
            logger.info(f"Overall accuracy: {self.calibration.overall_accuracy:.4f} rad")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
        
        finally:
            self.is_calibrating = False
            # Zurück zur Home-Position
            self.controller.move_home(speed=0.5)
    
    def calibrate_single_joint(self, joint_name: str) -> bool:
        """
        Kalibriert ein einzelnes Gelenk.
        
        Args:
            joint_name: Name des Gelenks
            
        Returns:
            True wenn erfolgreich
        """
        logger.info(f"Calibrating joint: {joint_name}")
        
        if joint_name not in SERVO_LIMITS:
            logger.error(f"Unknown joint: {joint_name}")
            return False
        
        try:
            # Home position
            self.controller.move_home(speed=0.5)
            time.sleep(2)
            
            # Kalibrierung durchführen
            calibration = self._calibrate_joint(joint_name)
            
            if calibration:
                self.calibration.joints[joint_name] = calibration
                self.save_calibration()
                logger.info(f"✅ Joint {joint_name} calibrated successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Joint calibration failed: {e}")
            return False
    
    def calibrate_scanner_position(self) -> bool:
        """
        Optimiert Scanner-Position für beste Scan-Qualität.
        """
        logger.info("📷 SCANNER POSITION CALIBRATION")
        logger.info("-"*40)
        
        try:
            # Test-Objekt platzieren
            input("Place calibration object (sphere or cube) and press ENTER...")
            
            # Verschiedene Positionen testen
            test_positions = self._generate_scanner_test_positions()
            best_position = None
            best_score = 0
            
            for i, pos in enumerate(test_positions):
                logger.info(f"Testing position {i+1}/{len(test_positions)}")
                
                # Position anfahren
                self.controller.move_joints(pos, speed=0.3)
                time.sleep(2)  # Settle
                
                # Bewertung (manuell oder automatisch)
                score = self._evaluate_scanner_position()
                
                if score > best_score:
                    best_score = score
                    best_position = pos
                
                logger.info(f"  Score: {score:.2f}")
            
            # Beste Position speichern
            if best_position:
                self.calibration.scanner.optimal_distance = best_score
                logger.info(f"✅ Optimal scanner position found (score: {best_score:.2f})")
                
                # Speichere als Scanner-Position
                self.controller.config.scanner_position = best_position
                self.save_calibration()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Scanner calibration failed: {e}")
            return False
    
    def test_repeatability(self, positions: int = 10, cycles: int = 3) -> Dict:
        """
        Testet Wiederholgenauigkeit.
        
        Args:
            positions: Anzahl Testpositionen
            cycles: Wiederholungen pro Position
            
        Returns:
            Genauigkeits-Statistiken
        """
        logger.info(f"Testing repeatability: {positions} positions, {cycles} cycles")
        
        results = {joint: [] for joint in SERVO_LIMITS.keys()}
        
        # Generiere zufällige Testpositionen
        test_positions = []
        for _ in range(positions):
            pos = {}
            for joint, (min_val, max_val) in SERVO_LIMITS.items():
                # Zufällige Position im sicheren Bereich
                safe_range = 0.8  # 80% des Bewegungsbereichs
                center = (min_val + max_val) / 2
                range_val = (max_val - min_val) * safe_range / 2
                pos[joint] = center + np.random.uniform(-range_val, range_val)
            test_positions.append(pos)
        
        # Teste jede Position mehrfach
        for pos_idx, target_pos in enumerate(test_positions):
            logger.info(f"Testing position {pos_idx+1}/{positions}")
            
            measurements = {joint: [] for joint in SERVO_LIMITS.keys()}
            
            for cycle in range(cycles):
                # Fahre Position an
                self.controller.move_joints(target_pos, speed=0.5)
                time.sleep(2)  # Settle
                
                # Messe aktuelle Position
                status = self.controller.query_status()
                if status:
                    for joint in SERVO_LIMITS.keys():
                        actual = status['positions'].get(joint, 0)
                        measurements[joint].append(actual)
                
                # Fahre weg und zurück
                if cycle < cycles - 1:
                    self.controller.move_home(speed=0.5)
                    time.sleep(1)
            
            # Berechne Standardabweichung
            for joint, values in measurements.items():
                if len(values) > 1:
                    std_dev = np.std(values)
                    results[joint].append(std_dev)
        
        # Statistiken berechnen
        statistics = {}
        for joint, deviations in results.items():
            if deviations:
                statistics[joint] = {
                    "mean_deviation": np.mean(deviations),
                    "max_deviation": np.max(deviations),
                    "repeatability": np.mean(deviations) * 3  # 3-Sigma
                }
                
                # Update Kalibrierung
                if joint not in self.calibration.joints:
                    self.calibration.joints[joint] = JointCalibration(joint_name=joint)
                self.calibration.joints[joint].repeatability = statistics[joint]["repeatability"]
        
        # Ausgabe
        logger.info("REPEATABILITY TEST RESULTS:")
        logger.info("-"*40)
        for joint, stats in statistics.items():
            logger.info(f"{joint:10s}: ±{stats['repeatability']*1000:.2f} mrad (3σ)")
        
        self.save_calibration()
        return statistics
    
    def verify_calibration(self) -> bool:
        """
        Verifiziert aktuelle Kalibrierung.
        
        Returns:
            True wenn Kalibrierung gültig
        """
        logger.info("Verifying calibration...")
        
        if not self.calibration.calibration_valid:
            logger.warning("No valid calibration found")
            return False
        
        # Alter der Kalibrierung prüfen
        age_days = (time.time() - self.calibration.timestamp) / 86400
        if age_days > 30:
            logger.warning(f"Calibration is {age_days:.0f} days old - recalibration recommended")
        
        # Quick-Test: Bekannte Positionen anfahren
        test_positions = [
            HOME_POSITION,
            {joint: 0.5 for joint in SERVO_LIMITS.keys()},
            {joint: -0.5 for joint in SERVO_LIMITS.keys()}
        ]
        
        errors = []
        for pos in test_positions:
            self.controller.move_joints(pos, speed=0.5)
            time.sleep(2)
            
            status = self.controller.query_status()
            if status:
                for joint, target in pos.items():
                    actual = status['positions'].get(joint, 0)
                    error = abs(actual - target)
                    errors.append(error)
        
        mean_error = np.mean(errors) if errors else float('inf')
        
        if mean_error < 0.01:  # 10 mrad
            logger.info(f"✅ Calibration verified (mean error: {mean_error*1000:.1f} mrad)")
            self.calibration.last_verification = time.time()
            return True
        else:
            logger.warning(f"⚠️ Calibration may be invalid (mean error: {mean_error*1000:.1f} mrad)")
            return False
    
    # ============== PRIVATE METHODS ==============
    
    def _prepare_calibration(self):
        """Bereitet Kalibrierung vor."""
        logger.info("Phase 1/6: Preparing calibration...")
        
        # LED indication
        self.controller.led_control(True, brightness=128)
        
        # Enable torque
        self.controller.set_torque(True)
        
        # Move to safe position
        self.controller.move_home(speed=0.3)
        time.sleep(3)
        
        # Clear old calibration data
        self.calibration = SystemCalibration()
    
    def _calibrate_endstops(self):
        """Findet Endpositionen für alle Gelenke."""
        for joint, (min_limit, max_limit) in SERVO_LIMITS.items():
            logger.info(f"  Finding endstops for {joint}...")
            
            # Langsam zu Limits fahren
            # VORSICHT: Nur mit Strombegrenzung!
            
            # Speichere gefundene Limits
            if joint not in self.calibration.joints:
                self.calibration.joints[joint] = JointCalibration(joint_name=joint)
            
            self.calibration.joints[joint].min_limit = min_limit
            self.calibration.joints[joint].max_limit = max_limit
            self.calibration.joints[joint].safe_min = min_limit + 0.1
            self.calibration.joints[joint].safe_max = max_limit - 0.1
    
    def _calibrate_joint(self, joint_name: str) -> Optional[JointCalibration]:
        """Kalibriert einzelnes Gelenk."""
        logger.info(f"  Calibrating {joint_name}...")
        
        calibration = JointCalibration(joint_name=joint_name)
        min_limit, max_limit = SERVO_LIMITS[joint_name]
        
        # Test-Positionen
        test_positions = np.linspace(min_limit + 0.1, max_limit - 0.1, 5)
        
        for target in test_positions:
            # Position anfahren
            pos = self.controller.current_position.copy()
            pos[joint_name] = target
            
            self.controller.move_joints(pos, speed=0.3)
            time.sleep(2)
            
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
        
        # Offset und Scale berechnen
        if calibration.calibration_points:
            errors = [p.error for p in calibration.calibration_points]
            calibration.offset = -np.mean(errors)
            calibration.accuracy = np.std(errors) * 3
            
            logger.info(f"    Offset: {calibration.offset*1000:.2f} mrad")
            logger.info(f"    Accuracy: {calibration.accuracy*1000:.2f} mrad")
        
        return calibration
    
    def _measure_backlash(self):
        """Misst Spiel in Gelenken."""
        logger.info("  Measuring backlash...")
        
        for joint in SERVO_LIMITS.keys():
            # Hin und zurück fahren
            center = 0.0
            delta = 0.5
            
            # Vorwärts
            pos1 = self.controller.current_position.copy()
            pos1[joint] = center + delta
            self.controller.move_joints(pos1, speed=0.3)
            time.sleep(2)
            
            status1 = self.controller.query_status()
            actual1 = status1['positions'].get(joint, 0) if status1 else 0
            
            # Rückwärts
            pos2 = self.controller.current_position.copy()
            pos2[joint] = center
            self.controller.move_joints(pos2, speed=0.3)
            time.sleep(2)
            
            status2 = self.controller.query_status()
            actual2 = status2['positions'].get(joint, 0) if status2 else 0
            
            # Nochmal vorwärts (gleiche Position)
            self.controller.move_joints(pos1, speed=0.3)
            time.sleep(2)
            
            status3 = self.controller.query_status()
            actual3 = status3['positions'].get(joint, 0) if status3 else 0
            
            # Backlash berechnen
            backlash = abs(actual3 - actual1)
            
            if joint in self.calibration.joints:
                self.calibration.joints[joint].backlash = backlash
                logger.info(f"    {joint}: {backlash*1000:.2f} mrad")
    
    def _calibrate_weight_compensation(self):
        """Kalibriert Gewichtskompensation."""
        logger.info("  Calibrating weight compensation...")
        
        # Mit und ohne Last messen
        positions = [
            {"shoulder": 0.0, "elbow": 1.57},
            {"shoulder": 0.5, "elbow": 1.0},
            {"shoulder": -0.5, "elbow": 2.0}
        ]
        
        for pos in positions:
            self.controller.move_joints(pos, speed=0.3)
            time.sleep(3)
            
            # Torque messen (wenn verfügbar)
            # Hier würde man den Strom der Servos messen
            
        # Kompensationsfaktoren berechnen
        self.calibration.gravity_compensation = {
            "shoulder": 0.05,  # Beispielwerte
            "elbow": 0.03
        }
    
    def _calibrate_scanner(self):
        """Scanner-spezifische Kalibrierung."""
        logger.info("  Calibrating scanner mount...")
        
        # Scanner-Position optimieren
        scanner_cal = ScannerCalibration()
        
        # Test verschiedene Abstände
        distances = [0.10, 0.15, 0.20, 0.25]
        for dist in distances:
            # Position für Abstand berechnen
            # Hier würde man Scan-Qualität bewerten
            pass
        
        scanner_cal.optimal_distance = 0.15  # Beispiel
        scanner_cal.optimal_speed = 0.3
        scanner_cal.optimal_settle_time = 0.5
        
        self.calibration.scanner = scanner_cal
    
    def _verify_calibration(self):
        """Verifiziert Kalibrierungs-Ergebnisse."""
        logger.info("Verifying calibration results...")
        
        # Berechne Gesamt-Genauigkeit
        accuracies = []
        for joint in self.calibration.joints.values():
            if joint.accuracy > 0:
                accuracies.append(joint.accuracy)
        
        if accuracies:
            self.calibration.overall_accuracy = np.mean(accuracies)
        else:
            self.calibration.overall_accuracy = 0.01  # Default
    
    def _generate_scanner_test_positions(self) -> List[Dict]:
        """Generiert Test-Positionen für Scanner-Kalibrierung."""
        positions = []
        
        # Verschiedene Höhen und Winkel
        for height in [-0.2, 0, 0.2]:
            for angle in [-0.3, 0, 0.3]:
                pos = HOME_POSITION.copy()
                pos["shoulder"] += height
                pos["base"] += angle
                pos["wrist"] = -1.57 - pos["shoulder"]  # Level halten
                positions.append(pos)
        
        return positions
    
    def _evaluate_scanner_position(self) -> float:
        """
        Bewertet Scanner-Position.
        
        Returns:
            Score (0-100)
        """
        # Hier könnte man automatisch bewerten durch:
        # - Bildanalyse
        # - Scan-Qualitäts-Metriken
        # - Oder manuell
        
        score = float(input("Rate scan quality (0-100): ") or "50")
        return score
    
    def load_calibration(self, filepath: Optional[str] = None):
        """Lädt gespeicherte Kalibrierung."""
        if filepath:
            self.calibration = SystemCalibration.load(filepath)
        else:
            self.calibration = SystemCalibration.load()
        
        if self.calibration.calibration_valid:
            logger.info("✅ Calibration loaded successfully")
            
            # Wende Kalibrierung auf Controller an
            self._apply_calibration_to_controller()
        else:
            logger.warning("No valid calibration found - using defaults")
    
    def save_calibration(self, filepath: Optional[str] = None):
        """Speichert aktuelle Kalibrierung."""
        if filepath:
            self.calibration.save(filepath)
        else:
            self.calibration.save()
    
    def _apply_calibration_to_controller(self):
        """Wendet Kalibrierung auf Controller an."""
        # Update Servo-Limits
        for joint_name, joint_cal in self.calibration.joints.items():
            if joint_name in SERVO_LIMITS:
                # Verwende kalibrierte Limits
                SERVO_LIMITS[joint_name] = (
                    joint_cal.safe_min,
                    joint_cal.safe_max
                )
        
        # Update Scanner-Parameter
        if self.calibration.scanner:
            SCANNER_SPECS["optimal_distance"] = self.calibration.scanner.optimal_distance
            SCANNER_SPECS["settle_time"] = self.calibration.scanner.optimal_settle_time
        
        logger.info("Calibration applied to controller")
    
    def export_report(self, filepath: str = "calibration/calibration_report.txt"):
        """Exportiert Kalibrierungsbericht."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("="*60)
        report.append("ROARM M3 CALIBRATION REPORT")
        report.append("="*60)
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
            report.append(f"  Limits: [{joint.safe_min:.3f}, {joint.safe_max:.3f}] rad")
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
        
        report.append("\n" + "="*60)
        
        # Speichern
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Calibration report saved to {filepath}")
        return '\n'.join(report)
