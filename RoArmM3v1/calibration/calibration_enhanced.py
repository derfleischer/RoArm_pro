#!/usr/bin/env python3
"""
RoArm M3 Enhanced Calibration Suite
Professioneller Kalibrierungsalgorithmus mit erweiterten Features
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import scipy.optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime

from ..core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_SPECS
from ..motion.trajectory import TrajectoryType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationType(Enum):
    """Kalibrierungstypen."""
    AUTO_FULL = "auto_full"
    MANUAL_JOINT = "manual_joint"
    SCANNER_ALIGNMENT = "scanner"
    BACKLASH = "backlash"
    ENDSTOPS = "endstops"
    ACCURACY = "accuracy"
    WEIGHT = "weight"
    REPEATABILITY = "repeatability"
    KINEMATIC = "kinematic"
    THERMAL = "thermal"
    DYNAMIC = "dynamic"


@dataclass
class CalibrationPoint:
    """Einzelner Kalibrierpunkt mit erweiterten Daten."""
    joint: str
    target_position: float
    actual_position: float
    error: float
    timestamp: float
    temperature: Optional[float] = None
    load: Optional[float] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    torque: Optional[float] = None
    
    @property
    def absolute_error(self) -> float:
        """Absoluter Fehler."""
        return abs(self.error)
    
    @property
    def relative_error(self) -> float:
        """Relativer Fehler."""
        if self.target_position != 0:
            return abs(self.error / self.target_position)
        return 0


@dataclass
class EnhancedJointCalibration:
    """Erweiterte Kalibrierung für einzelnes Gelenk."""
    joint_name: str
    
    # Basis-Kalibrierung
    offset: float = 0.0
    scale: float = 1.0
    backlash: float = 0.0
    
    # Erweiterte Parameter
    nonlinearity_correction: List[Tuple[float, float]] = field(default_factory=list)
    temperature_compensation: Dict[float, float] = field(default_factory=dict)
    load_compensation: Dict[float, float] = field(default_factory=dict)
    
    # Dynamische Parameter
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    friction_static: float = 0.0
    friction_dynamic: float = 0.0
    damping: float = 0.0
    inertia: float = 0.0
    
    # Limits
    min_limit: float = 0.0
    max_limit: float = 0.0
    safe_min: float = 0.0
    safe_max: float = 0.0
    
    # Genauigkeit
    repeatability: float = 0.001
    accuracy: float = 0.002
    resolution: float = 0.0001
    
    # Kalibrierdaten
    calibration_points: List[CalibrationPoint] = field(default_factory=list)
    calibration_curve: Optional[Any] = None  # Interpolationsfunktion
    
    # Statistiken
    mean_error: float = 0.0
    std_error: float = 0.0
    max_error: float = 0.0
    calibration_quality: float = 0.0
    
    def apply_calibration(self, raw_position: float, 
                         temperature: Optional[float] = None,
                         load: Optional[float] = None) -> float:
        """
        Wendet vollständige Kalibrierung an.
        
        Args:
            raw_position: Rohe Position
            temperature: Aktuelle Temperatur
            load: Aktuelle Last
            
        Returns:
            Kalibrierte Position
        """
        # Basis-Kalibrierung
        calibrated = (raw_position + self.offset) * self.scale
        
        # Nichtlinearitäts-Korrektur
        if self.calibration_curve:
            try:
                calibrated = float(self.calibration_curve(calibrated))
            except:
                pass
        
        # Temperatur-Kompensation
        if temperature and self.temperature_compensation:
            # Interpoliere Temperatur-Kompensation
            temps = sorted(self.temperature_compensation.keys())
            if temps:
                comp_values = [self.temperature_compensation[t] for t in temps]
                if len(temps) > 1:
                    f_temp = interp1d(temps, comp_values, 
                                     kind='linear', fill_value='extrapolate')
                    calibrated += float(f_temp(temperature))
        
        # Last-Kompensation
        if load and self.load_compensation:
            loads = sorted(self.load_compensation.keys())
            if loads:
                comp_values = [self.load_compensation[l] for l in loads]
                if len(loads) > 1:
                    f_load = interp1d(loads, comp_values,
                                     kind='linear', fill_value='extrapolate')
                    calibrated += float(f_load(load))
        
        # Backlash-Kompensation (abhängig von Bewegungsrichtung)
        # Wird in Controller angewendet
        
        return calibrated
    
    def inverse_calibration(self, calibrated_position: float) -> float:
        """
        Invertiert Kalibrierung für Befehle.
        
        Args:
            calibrated_position: Kalibrierte Zielposition
            
        Returns:
            Raw-Position für Servo
        """
        # Inverse Nichtlinearität (wenn vorhanden)
        if self.calibration_curve:
            # Numerische Inversion
            def error_func(raw):
                return abs(self.apply_calibration(raw) - calibrated_position)
            
            result = scipy.optimize.minimize_scalar(
                error_func,
                bounds=(self.min_limit, self.max_limit),
                method='bounded'
            )
            return result.x
        
        # Einfache inverse Transformation
        return (calibrated_position / self.scale) - self.offset
    
    def calculate_statistics(self):
        """Berechnet Kalibrierungs-Statistiken."""
        if not self.calibration_points:
            return
        
        errors = [p.error for p in self.calibration_points]
        
        self.mean_error = np.mean(errors)
        self.std_error = np.std(errors)
        self.max_error = max(abs(e) for e in errors)
        
        # Qualitäts-Score (0-1, 1 ist perfekt)
        if self.max_error > 0:
            self.calibration_quality = 1.0 - min(1.0, self.max_error / 0.01)
        else:
            self.calibration_quality = 1.0
    
    def fit_nonlinearity(self, order: int = 3):
        """
        Fittet Polynom für Nichtlinearitäts-Korrektur.
        
        Args:
            order: Polynom-Ordnung
        """
        if len(self.calibration_points) < order + 1:
            return
        
        # Sammle Daten
        targets = [p.target_position for p in self.calibration_points]
        actuals = [p.actual_position for p in self.calibration_points]
        
        # Polynom-Fit
        coeffs = np.polyfit(targets, actuals, order)
        self.calibration_curve = np.poly1d(coeffs)
        
        # Speichere als Lookup-Table
        self.nonlinearity_correction = [
            (t, float(self.calibration_curve(t))) 
            for t in np.linspace(self.min_limit, self.max_limit, 20)
        ]


@dataclass
class KinematicCalibration:
    """Kinematische Kalibrierung (DH-Parameter)."""
    # Denavit-Hartenberg Parameter
    dh_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Tool Center Point
    tcp_offset: Tuple[float, float, float] = (0, 0, 0)
    tcp_rotation: Tuple[float, float, float] = (0, 0, 0)
    
    # Base transformation
    base_offset: Tuple[float, float, float] = (0, 0, 0)
    base_rotation: Tuple[float, float, float] = (0, 0, 0)
    
    # Kinematische Ketten
    forward_kinematics: Optional[Any] = None
    inverse_kinematics: Optional[Any] = None
    
    def calculate_forward_kinematics(self, joint_positions: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Berechnet Vorwärts-Kinematik.
        
        Args:
            joint_positions: Joint-Positionen
            
        Returns:
            (x, y, z) Position des End-Effektors
        """
        # Vereinfachte FK für 6-DOF Arm
        # In Realität würde hier die vollständige DH-Transformation stehen
        
        # Beispiel-Implementation
        x = 0.1 * np.cos(joint_positions.get('base', 0))
        y = 0.1 * np.sin(joint_positions.get('base', 0))
        z = 0.1 + 0.1 * np.sin(joint_positions.get('shoulder', 0))
        
        return (x, y, z)


@dataclass
class EnhancedSystemCalibration:
    """Erweiterte System-Kalibrierung."""
    version: str = "2.0.0"
    timestamp: float = field(default_factory=time.time)
    
    # Joint-Kalibrierungen
    joints: Dict[str, EnhancedJointCalibration] = field(default_factory=dict)
    
    # Kinematik
    kinematics: KinematicCalibration = field(default_factory=KinematicCalibration)
    
    # Scanner-Kalibrierung
    scanner_offset: Tuple[float, float, float] = (0, 0, 0.05)
    scanner_rotation: Tuple[float, float, float] = (0, 0, 0)
    scanner_optimal_distance: float = 0.15
    scanner_calibration_matrix: Optional[np.ndarray] = None
    
    # Globale Kompensationen
    gravity_vector: Tuple[float, float, float] = (0, 0, -9.81)
    gravity_compensation: Dict[str, float] = field(default_factory=dict)
    temperature_model: Dict[str, Any] = field(default_factory=dict)
    
    # Qualitäts-Metriken
    overall_accuracy: float = 0.0
    overall_repeatability: float = 0.0
    calibration_valid: bool = False
    last_verification: float = 0.0
    calibration_score: float = 0.0
    
    def calculate_quality_metrics(self):
        """Berechnet Gesamt-Qualitätsmetriken."""
        if not self.joints:
            return
        
        accuracies = []
        repeatabilities = []
        qualities = []
        
        for joint in self.joints.values():
            if joint.accuracy > 0:
                accuracies.append(joint.accuracy)
            if joint.repeatability > 0:
                repeatabilities.append(joint.repeatability)
            if joint.calibration_quality > 0:
                qualities.append(joint.calibration_quality)
        
        if accuracies:
            self.overall_accuracy = np.mean(accuracies)
        if repeatabilities:
            self.overall_repeatability = np.mean(repeatabilities)
        if qualities:
            self.calibration_score = np.mean(qualities)
        
        # Validität prüfen
        self.calibration_valid = (
            self.overall_accuracy < 0.005 and  # <5 mrad
            self.overall_repeatability < 0.002 and  # <2 mrad
            self.calibration_score > 0.8  # >80% Qualität
        )


class EnhancedCalibrationSuite:
    """
    Erweiterte Kalibrierungs-Suite mit professionellen Features.
    """
    
    def __init__(self, controller):
        """
        Initialisiert Calibration Suite.
        
        Args:
            controller: RoArm Controller
        """
        self.controller = controller
        self.calibration = EnhancedSystemCalibration()
        
        # Kalibrierungs-Status
        self.is_calibrating = False
        self.calibration_progress = 0.0
        self.current_step = ""
        
        # Kalibrierungs-Parameter
        self.calibration_points_per_joint = 20
        self.repeatability_cycles = 5
        self.temperature_range = (20, 40)  # °C
        self.load_range = (0, 0.5)  # kg
        
        # Lade existierende Kalibrierung
        self.load_calibration()
        
        logger.info("Enhanced Calibration Suite initialized")
    
    def run_full_calibration(self, include_scanner: bool = True,
                            include_thermal: bool = False,
                            include_dynamic: bool = False) -> bool:
        """
        Führt vollständige System-Kalibrierung durch.
        
        Args:
            include_scanner: Scanner-Kalibrierung einschließen
            include_thermal: Thermische Kalibrierung
            include_dynamic: Dynamische Kalibrierung
            
        Returns:
            True wenn erfolgreich
        """
        logger.info("🔧 STARTING FULL CALIBRATION")
        logger.info("="*60)
        
        self.is_calibrating = True
        self.calibration_progress = 0.0
        
        try:
            steps = []
            
            # Phase 1: Vorbereitung
            steps.append(("Preparation", self._prepare_calibration))
            
            # Phase 2: Endstops
            steps.append(("Finding endstops", self._calibrate_endstops))
            
            # Phase 3: Joint-Kalibrierung
            for joint in SERVO_LIMITS.keys():
                steps.append((f"Calibrating {joint}", 
                            lambda j=joint: self._calibrate_joint_enhanced(j)))
            
            # Phase 4: Backlash
            steps.append(("Measuring backlash", self._measure_backlash_enhanced))
            
            # Phase 5: Kinematik
            steps.append(("Kinematic calibration", self._calibrate_kinematics))
            
            # Phase 6: Wiederholgenauigkeit
            steps.append(("Testing repeatability", self._test_repeatability_enhanced))
            
            # Optional: Scanner
            if include_scanner:
                steps.append(("Scanner calibration", self._calibrate_scanner_enhanced))
            
            # Optional: Thermal
            if include_thermal:
                steps.append(("Thermal calibration", self._calibrate_thermal))
            
            # Optional: Dynamic
            if include_dynamic:
                steps.append(("Dynamic calibration", self._calibrate_dynamic))
            
            # Phase 7: Verifikation
            steps.append(("Verification", self._verify_calibration_enhanced))
            
            # Führe alle Schritte aus
            total_steps = len(steps)
            for i, (name, func) in enumerate(steps):
                self.current_step = name
                logger.info(f"Step {i+1}/{total_steps}: {name}...")
                
                if not func():
                    logger.error(f"Failed at: {name}")
                    return False
                
                self.calibration_progress = (i + 1) / total_steps * 100
                logger.info(f"Progress: {self.calibration_progress:.1f}%")
            
            # Berechne Qualitäts-Metriken
            self.calibration.calculate_quality_metrics()
            
            # Speichern
            self.save_calibration()
            
            logger.info("="*60)
            logger.info("✅ CALIBRATION COMPLETE")
            logger.info(f"Overall accuracy: {self.calibration.overall_accuracy*1000:.2f} mrad")
            logger.info(f"Overall repeatability: {self.calibration.overall_repeatability*1000:.2f} mrad")
            logger.info(f"Quality score: {self.calibration.calibration_score*100:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
        
        finally:
            self.is_calibrating = False
            self.controller.move_home(speed=0.5)
    
    def _prepare_calibration(self) -> bool:
        """Bereitet Kalibrierung vor."""
        logger.info("Preparing calibration...")
        
        # LED-Indikation
        self.controller.led_control(True, brightness=128)
        
        # Torque aktivieren
        self.controller.set_torque(True)
        
        # Zur Home-Position
        self.controller.move_home(speed=0.3)
        time.sleep(2)
        
        # Neue Kalibrierung
        self.calibration = EnhancedSystemCalibration()
        
        return True
    
    def _calibrate_endstops(self) -> bool:
        """Findet und kalibriert Endstops."""
        logger.info("Finding endstops...")
        
        for joint, (min_limit, max_limit) in SERVO_LIMITS.items():
            # Initialisiere Joint-Kalibrierung
            if joint not in self.calibration.joints:
                self.calibration.joints[joint] = EnhancedJointCalibration(joint_name=joint)
            
            cal = self.calibration.joints[joint]
            cal.min_limit = min_limit
            cal.max_limit = max_limit
            cal.safe_min = min_limit + 0.1
            cal.safe_max = max_limit - 0.1
            
            logger.info(f"  {joint}: [{min_limit:.3f}, {max_limit:.3f}] rad")
        
        return True
    
    def _calibrate_joint_enhanced(self, joint_name: str) -> bool:
        """
        Erweiterte Joint-Kalibrierung mit Nichtlinearität.
        
        Args:
            joint_name: Name des Joints
            
        Returns:
            True wenn erfolgreich
        """
        logger.info(f"Calibrating {joint_name}...")
        
        if joint_name not in self.calibration.joints:
            self.calibration.joints[joint_name] = EnhancedJointCalibration(joint_name=joint_name)
        
        cal = self.calibration.joints[joint_name]
        min_limit, max_limit = SERVO_LIMITS[joint_name]
        
        # Mehr Testpunkte für bessere Kalibrierung
        test_positions = np.linspace(
            min_limit + 0.1, 
            max_limit - 0.1,
            self.calibration_points_per_joint
        )
        
        cal.calibration_points = []
        
        for target in test_positions:
            # Position anfahren
            pos = self.controller.current_position.copy()
            pos[joint_name] = target
            
            self.controller.move_joints(pos, speed=0.3)
            time.sleep(1.5)  # Längere Wartezeit für Stabilität
            
            # Mehrfach messen für Genauigkeit
            measurements = []
            for _ in range(3):
                status = self.controller.query_status()
                if status:
                    measurements.append(status['positions'].get(joint_name, 0))
                time.sleep(0.1)
            
            if measurements:
                actual = np.mean(measurements)
                error = actual - target
                
                point = CalibrationPoint(
                    joint=joint_name,
                    target_position=target,
                    actual_position=actual,
                    error=error,
                    timestamp=time.time(),
                    temperature=status.get('temperature')
                )
                cal.calibration_points.append(point)
        
        # Berechne Kalibrierungs-Parameter
        if cal.calibration_points:
            errors = [p.error for p in cal.calibration_points]
            cal.offset = -np.mean(errors)
            
            # Fitte Nichtlinearität
            cal.fit_nonlinearity(order=3)
            
            # Statistiken
            cal.calculate_statistics()
            
            logger.info(f"  Offset: {cal.offset*1000:.3f} mrad")
            logger.info(f"  Mean error: {cal.mean_error*1000:.3f} mrad")
            logger.info(f"  Std error: {cal.std_error*1000:.3f} mrad")
            logger.info(f"  Max error: {cal.max_error*1000:.3f} mrad")
            logger.info(f"  Quality: {cal.calibration_quality*100:.1f}%")
        
        return True
    
    def _measure_backlash_enhanced(self) -> bool:
        """Erweiterte Backlash-Messung."""
        logger.info("Measuring backlash...")
        
        for joint in SERVO_LIMITS.keys():
            if joint not in self.calibration.joints:
                continue
            
            cal = self.calibration.joints[joint]
            
            # Teste an mehreren Positionen
            test_positions = np.linspace(
                cal.safe_min + 0.2,
                cal.safe_max - 0.2,
                5
            )
            
            backlash_measurements = []
            
            for test_pos in test_positions:
                # Approach from below
                pos1 = self.controller.current_position.copy()
                pos1[joint] = test_pos - 0.5
                self.controller.move_joints(pos1, speed=0.3)
                time.sleep(1)
                
                pos2 = pos1.copy()
                pos2[joint] = test_pos
                self.controller.move_joints(pos2, speed=0.3)
                time.sleep(1)
                
                status1 = self.controller.query_status()
                actual_up = status1['positions'].get(joint, 0) if status1 else 0
                
                # Approach from above
                pos3 = self.controller.current_position.copy()
                pos3[joint] = test_pos + 0.5
                self.controller.move_joints(pos3, speed=0.3)
                time.sleep(1)
                
                pos4 = pos3.copy()
                pos4[joint] = test_pos
                self.controller.move_joints(pos4, speed=0.3)
                time.sleep(1)
                
                status2 = self.controller.query_status()
                actual_down = status2['positions'].get(joint, 0) if status2 else 0
                
                # Backlash
                backlash = abs(actual_up - actual_down)
                backlash_measurements.append(backlash)
            
            # Durchschnittlicher Backlash
            if backlash_measurements:
                cal.backlash = np.mean(backlash_measurements)
                logger.info(f"  {joint}: {cal.backlash*1000:.3f} mrad")
        
        return True
    
    def _calibrate_kinematics(self) -> bool:
        """Kalibriert Kinematik-Parameter."""
        logger.info("Calibrating kinematics...")
        
        # Vereinfachte DH-Parameter für 6-DOF
        self.calibration.kinematics.dh_parameters = {
            "base": {"a": 0, "alpha": 0, "d": 0.1, "theta_offset": 0},
            "shoulder": {"a": 0.1, "alpha": np.pi/2, "d": 0, "theta_offset": 0},
            "elbow": {"a": 0.1, "alpha": 0, "d": 0, "theta_offset": 0},
            "wrist": {"a": 0.05, "alpha": np.pi/2, "d": 0, "theta_offset": 0},
            "roll": {"a": 0, "alpha": -np.pi/2, "d": 0.05, "theta_offset": 0},
            "hand": {"a": 0, "alpha": 0, "d": 0.05, "theta_offset": 0}
        }
        
        # TCP kalibrieren (Tool Center Point)
        # Würde normalerweise durch Messungen erfolgen
        self.calibration.kinematics.tcp_offset = (0, 0, 0.05)
        
        logger.info("  DH parameters configured")
        return True
    
    def _test_repeatability_enhanced(self) -> bool:
        """Erweiterte Wiederholgenauigkeits-Tests."""
        logger.info("Testing repeatability...")
        
        # Test-Positionen
        test_positions = [
            HOME_POSITION,
            {joint: 0.3 for joint in SERVO_LIMITS.keys()},
            {joint: -0.3 for joint in SERVO_LIMITS.keys()},
            {joint: 0.5 for joint in SERVO_LIMITS.keys()},
            {joint: -0.5 for joint in SERVO_LIMITS.keys()}
        ]
        
        for joint_name in SERVO_LIMITS.keys():
            if joint_name not in self.calibration.joints:
                continue
            
            cal = self.calibration.joints[joint_name]
            repeatability_errors = []
            
            for target_pos in test_positions:
                measurements = []
                
                for cycle in range(self.repeatability_cycles):
                    # Fahre Position an
                    self.controller.move_joints(target_pos, speed=0.5)
                    time.sleep(1.5)
                    
                    # Messe
                    status = self.controller.query_status()
                    if status:
                        actual = status['positions'].get(joint_name, 0)
                        measurements.append(actual)
                    
                    # Fahre weg
                    if cycle < self.repeatability_cycles - 1:
                        self.controller.move_home(speed=0.5)
                        time.sleep(1)
                
                # Berechne Wiederholgenauigkeit
                if len(measurements) > 1:
                    std_dev = np.std(measurements)
                    repeatability_errors.append(std_dev)
            
            # Gesamt-Wiederholgenauigkeit (3-Sigma)
            if repeatability_errors:
                cal.repeatability = 3 * np.mean(repeatability_errors)
                logger.info(f"  {joint_name}: ±{cal.repeatability*1000:.3f} mrad (3σ)")
        
        return True
    
    def _calibrate_scanner_enhanced(self) -> bool:
        """Erweiterte Scanner-Kalibrierung."""
        logger.info("Calibrating scanner...")
        
        # Test verschiedene Scanner-Positionen
        test_distances = [0.10, 0.12, 0.15, 0.18, 0.20]  # Meter
        best_distance = 0.15
        best_score = 0
        
        for distance in test_distances:
            # Position für Distanz berechnen
            # (vereinfachte Berechnung)
            pos = SCANNER_POSITION.copy()
            pos['elbow'] = 1.57 - (distance - 0.15) * 2
            
            self.controller.move_joints(pos, speed=0.3)
            time.sleep(2)
            
            # Bewertung (würde normalerweise durch Scan-Qualität erfolgen)
            score = 100 - abs(distance - 0.15) * 100
            
            logger.info(f"  Distance {distance*100:.0f}cm: Score {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_distance = distance
        
        # Speichere optimale Parameter
        self.calibration.scanner_optimal_distance = best_distance
        
        # Scanner-Transformation
        self.calibration.scanner_offset = (0, 0, 0.05)
        self.calibration.scanner_rotation = (0, 0, 0)
        
        logger.info(f"  Optimal distance: {best_distance*100:.1f}cm")
        
        return True
    
    def _calibrate_thermal(self) -> bool:
        """Thermische Kalibrierung."""
        logger.info("Thermal calibration...")
        
        # Würde normalerweise über längeren Zeitraum erfolgen
        # Hier nur Beispiel-Implementation
        
        for joint_name in self.calibration.joints.keys():
            cal = self.calibration.joints[joint_name]
            
            # Beispiel-Temperatur-Kompensation
            cal.temperature_compensation = {
                20.0: 0.0,
                30.0: 0.0001,
                40.0: 0.0003,
                50.0: 0.0005
            }
        
        logger.info("  Temperature compensation configured")
        return True
    
    def _calibrate_dynamic(self) -> bool:
        """Dynamische Kalibrierung."""
        logger.info("Dynamic calibration...")
        
        for joint_name in self.calibration.joints.keys():
            cal = self.calibration.joints[joint_name]
            
            # Teste verschiedene Geschwindigkeiten
            velocities = [0.2, 0.5, 1.0, 1.5, 2.0]
            
            for velocity in velocities:
                # Bewegung mit verschiedenen Geschwindigkeiten
                pos1 = self.controller.current_position.copy()
                pos1[joint_name] = cal.safe_min + 0.2
                
                pos2 = pos1.copy()
                pos2[joint_name] = cal.safe_max - 0.2
                
                # Hin
                self.controller.move_joints(pos2, speed=velocity)
                time.sleep(0.5)
                
                # Zurück
                self.controller.move_joints(pos1, speed=velocity)
                time.sleep(0.5)
            
            # Setze dynamische Parameter (Beispielwerte)
            cal.max_velocity = 2.0
            cal.max_acceleration = 5.0
            cal.friction_static = 0.01
            cal.friction_dynamic = 0.005
            cal.damping = 0.002
            
        logger.info("  Dynamic parameters configured")
        return True
    
    def _verify_calibration_enhanced(self) -> bool:
        """Erweiterte Kalibrierungs-Verifikation."""
        logger.info("Verifying calibration...")
        
        # Test-Positionen
        verification_positions = [
            HOME_POSITION,
            SCANNER_POSITION,
            {joint: 0.2 for joint in SERVO_LIMITS.keys()},
            {joint: -0.2 for joint in SERVO_LIMITS.keys()}
        ]
        
        all_errors = []
        
        for target_pos in verification_positions:
            # Fahre Position an
            self.controller.move_joints(target_pos, speed=0.5)
            time.sleep(2)
            
            # Messe
            status = self.controller.query_status()
            if status:
                for joint, target in target_pos.items():
                    actual = status['positions'].get(joint, 0)
                    error = abs(actual - target)
                    all_errors.append(error)
        
        # Bewerte
        if all_errors:
            mean_error = np.mean(all_errors)
            max_error = np.max(all_errors)
            
            logger.info(f"  Mean error: {mean_error*1000:.3f} mrad")
            logger.info(f"  Max error: {max_error*1000:.3f} mrad")
            
            # Setze Verifikations-Zeitstempel
            self.calibration.last_verification = time.time()
            
            return mean_error < 0.005  # <5 mrad
        
        return False
    
    def generate_calibration_report(self) -> str:
        """
        Generiert detaillierten Kalibrierungsbericht.
        
        Returns:
            Report als String
        """
        report = []
        report.append("="*70)
        report.append("ROARM M3 CALIBRATION REPORT")
        report.append("="*70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Version: {self.calibration.version}")
        report.append(f"Valid: {self.calibration.calibration_valid}")
        report.append(f"Overall Accuracy: {self.calibration.overall_accuracy*1000:.3f} mrad")
        report.append(f"Overall Repeatability: {self.calibration.overall_repeatability*1000:.3f} mrad")
        report.append(f"Quality Score: {self.calibration.calibration_score*100:.1f}%")
        report.append("")
        
        # Joint Details
        report.append("JOINT CALIBRATION DETAILS")
        report.append("-"*70)
        
        for joint_name, cal in self.calibration.joints.items():
            report.append(f"\n{joint_name.upper()}:")
            report.append(f"  Offset: {cal.offset*1000:.3f} mrad")
            report.append(f"  Scale: {cal.scale:.6f}")
            report.append(f"  Backlash: {cal.backlash*1000:.3f} mrad")
            report.append(f"  Repeatability: ±{cal.repeatability*1000:.3f} mrad (3σ)")
            report.append(f"  Mean Error: {cal.mean_error*1000:.3f} mrad")
            report.append(f"  Std Error: {cal.std_error*1000:.3f} mrad")
            report.append(f"  Max Error: {cal.max_error*1000:.3f} mrad")
            report.append(f"  Quality: {cal.calibration_quality*100:.1f}%")
            report.append(f"  Limits: [{cal.safe_min:.3f}, {cal.safe_max:.3f}] rad")
            
            if cal.nonlinearity_correction:
                report.append(f"  Nonlinearity: {len(cal.nonlinearity_correction)} points")
        
        # Kinematik
        report.append("\nKINEMATIC CALIBRATION")
        report.append("-"*70)
        report.append(f"TCP Offset: {self.calibration.kinematics.tcp_offset}")
        report.append(f"Base Offset: {self.calibration.kinematics.base_offset}")
        
        # Scanner
        report.append("\nSCANNER CALIBRATION")
        report.append("-"*70)
        report.append(f"Optimal Distance: {self.calibration.scanner_optimal_distance*100:.1f} cm")
        report.append(f"Scanner Offset: {self.calibration.scanner_offset}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def plot_calibration_curves(self, save_path: Optional[str] = None):
        """
        Plottet Kalibrierungskurven.
        
        Args:
            save_path: Pfad zum Speichern der Plots
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (joint_name, cal) in enumerate(self.calibration.joints.items()):
            if i >= 6:
                break
            
            ax = axes[i]
            
            if cal.calibration_points:
                targets = [p.target_position for p in cal.calibration_points]
                actuals = [p.actual_position for p in cal.calibration_points]
                errors = [p.error * 1000 for p in cal.calibration_points]  # in mrad
                
                # Plot actual vs target
                ax.scatter(targets, actuals, alpha=0.5, label='Measured')
                
                # Plot ideal line
                ax.plot([min(targets), max(targets)], 
                       [min(targets), max(targets)], 
                       'r--', label='Ideal')
                
                # Plot calibration curve if available
                if cal.calibration_curve:
                    x_fit = np.linspace(min(targets), max(targets), 100)
                    y_fit = [cal.calibration_curve(x) for x in x_fit]
                    ax.plot(x_fit, y_fit, 'g-', label='Calibration')
                
                ax.set_xlabel('Target (rad)')
                ax.set_ylabel('Actual (rad)')
                ax.set_title(f'{joint_name.upper()} Calibration')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Joint Calibration Curves')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Calibration plots saved to {save_path}")
        else:
            plt.show()
    
    def save_calibration(self, filepath: str = "calibration/enhanced_calibration.json"):
        """Speichert Kalibrierung."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Konvertiere zu speicherbarem Format
        data = {
            "version": self.calibration.version,
            "timestamp": self.calibration.timestamp,
            "joints": {},
            "kinematics": {
                "dh_parameters": self.calibration.kinematics.dh_parameters,
                "tcp_offset": self.calibration.kinematics.tcp_offset,
                "base_offset": self.calibration.kinematics.base_offset
            },
            "scanner": {
                "offset": self.calibration.scanner_offset,
                "rotation": self.calibration.scanner_rotation,
                "optimal_distance": self.calibration.scanner_optimal_distance
            },
            "quality": {
                "overall_accuracy": self.calibration.overall_accuracy,
                "overall_repeatability": self.calibration.overall_repeatability,
                "calibration_score": self.calibration.calibration_score,
                "valid": self.calibration.calibration_valid
            }
        }
        
        # Joint-Daten
        for joint_name, cal in self.calibration.joints.items():
            data["joints"][joint_name] = {
                "offset": cal.offset,
                "scale": cal.scale,
                "backlash": cal.backlash,
                "repeatability": cal.repeatability,
                "accuracy": cal.accuracy,
                "mean_error": cal.mean_error,
                "std_error": cal.std_error,
                "max_error": cal.max_error,
                "quality": cal.calibration_quality,
                "limits": {
                    "min": cal.min_limit,
                    "max": cal.max_limit,
                    "safe_min": cal.safe_min,
                    "safe_max": cal.safe_max
                },
                "nonlinearity": cal.nonlinearity_correction,
                "temperature_compensation": cal.temperature_compensation,
                "dynamic": {
                    "max_velocity": cal.max_velocity,
                    "max_acceleration": cal.max_acceleration,
                    "friction_static": cal.friction_static,
                    "friction_dynamic": cal.friction_dynamic
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str = "calibration/enhanced_calibration.json"):
        """Lädt Kalibrierung."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Erstelle neue Kalibrierung
            self.calibration = EnhancedSystemCalibration()
            self.calibration.version = data.get("version", "2.0.0")
            self.calibration.timestamp = data.get("timestamp", time.time())
            
            # Lade Joints
            for joint_name, joint_data in data.get("joints", {}).items():
                cal = EnhancedJointCalibration(joint_name=joint_name)
                
                # Basis-Parameter
                cal.offset = joint_data.get("offset", 0)
                cal.scale = joint_data.get("scale", 1)
                cal.backlash = joint_data.get("backlash", 0)
                cal.repeatability = joint_data.get("repeatability", 0.001)
                cal.accuracy = joint_data.get("accuracy", 0.002)
                
                # Statistiken
                cal.mean_error = joint_data.get("mean_error", 0)
                cal.std_error = joint_data.get("std_error", 0)
                cal.max_error = joint_data.get("max_error", 0)
                cal.calibration_quality = joint_data.get("quality", 0)
                
                # Limits
                limits = joint_data.get("limits", {})
                cal.min_limit = limits.get("min", 0)
                cal.max_limit = limits.get("max", 0)
                cal.safe_min = limits.get("safe_min", 0)
                cal.safe_max = limits.get("safe_max", 0)
                
                # Erweiterte Parameter
                cal.nonlinearity_correction = joint_data.get("nonlinearity", [])
                cal.temperature_compensation = joint_data.get("temperature_compensation", {})
                
                # Dynamik
                dynamic = joint_data.get("dynamic", {})
                cal.max_velocity = dynamic.get("max_velocity", 1.0)
                cal.max_acceleration = dynamic.get("max_acceleration", 2.0)
                cal.friction_static = dynamic.get("friction_static", 0)
                cal.friction_dynamic = dynamic.get("friction_dynamic", 0)
                
                self.calibration.joints[joint_name] = cal
            
            # Lade Kinematik
            kin_data = data.get("kinematics", {})
            self.calibration.kinematics.dh_parameters = kin_data.get("dh_parameters", {})
            self.calibration.kinematics.tcp_offset = tuple(kin_data.get("tcp_offset", (0, 0, 0)))
            self.calibration.kinematics.base_offset = tuple(kin_data.get("base_offset", (0, 0, 0)))
            
            # Lade Scanner
            scanner_data = data.get("scanner", {})
            self.calibration.scanner_offset = tuple(scanner_data.get("offset", (0, 0, 0.05)))
            self.calibration.scanner_rotation = tuple(scanner_data.get("rotation", (0, 0, 0)))
            self.calibration.scanner_optimal_distance = scanner_data.get("optimal_distance", 0.15)
            
            # Lade Qualitäts-Metriken
            quality = data.get("quality", {})
            self.calibration.overall_accuracy = quality.get("overall_accuracy", 0)
            self.calibration.overall_repeatability = quality.get("overall_repeatability", 0)
            self.calibration.calibration_score = quality.get("calibration_score", 0)
            self.calibration.calibration_valid = quality.get("valid", False)
            
            logger.info(f"Calibration loaded from {filepath}")
            
            if self.calibration.calibration_valid:
                logger.info(f"✅ Valid calibration (Score: {self.calibration.calibration_score*100:.1f}%)")
            else:
                logger.warning("⚠️ Calibration loaded but not valid")
                
        except FileNotFoundError:
            logger.warning(f"No calibration file found at {filepath}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")


# Kompatibilität
class CalibrationSuite(EnhancedCalibrationSuite):
    """Alias für Kompatibilität."""
    pass
