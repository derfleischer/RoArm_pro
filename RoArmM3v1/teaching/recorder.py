#!/usr/bin/env python3
"""
RoArm M3 Teaching Mode - Erweiterte Version
Aufzeichnung von Positionen mit Geschwindigkeit, Beschleunigung und Bewegungsprofilen.
"""

import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import numpy as np

from ..core.constants import TEACHING_DEFAULTS, TRAJECTORY_PROFILES
from ..motion.trajectory import TrajectoryType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RecordingMode(Enum):
    """Aufzeichnungsmodi."""
    MANUAL = "manual"          # Manuell per Tastendruck
    CONTINUOUS = "continuous"  # Kontinuierlich mit Sampling
    TRIGGERED = "triggered"     # Bei Positionsänderung
    TIMED = "timed"            # Zeitbasiert


@dataclass
class TeachingWaypoint:
    """
    Ein Wegpunkt in der Teaching-Sequenz.
    Erweitert um Bewegungsparameter.
    """
    index: int                          # Waypoint-Nummer
    timestamp: float                     # Zeitstempel
    positions: Dict[str, float]         # Joint-Positionen (rad)
    speed: float = 1.0                  # Geschwindigkeitsfaktor (0.1-2.0)
    acceleration: float = 2.0           # Beschleunigung (rad/s²)
    jerk: float = 5.0                   # Ruck (rad/s³)
    trajectory_type: str = "s_curve"    # Bewegungsprofil
    settle_time: float = 0.0            # Wartezeit nach Erreichen (s)
    torque: float = 0.8                 # Servo-Torque (0.0-1.0)
    gripper_force: float = 0.5          # Greifkraft (0.0-1.0)
    comment: str = ""                   # Optionaler Kommentar
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für JSON."""
        data = asdict(self)
        # Trajectory type als String
        if isinstance(data['trajectory_type'], Enum):
            data['trajectory_type'] = data['trajectory_type'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Erstellt aus Dictionary."""
        return cls(**data)


@dataclass
class TeachingSequence:
    """
    Komplette Teaching-Sequenz mit Metadaten.
    """
    name: str
    description: str = ""
    created_at: float = 0.0
    waypoints: List[TeachingWaypoint] = None
    loop: bool = False                  # Sequenz wiederholen
    speed_override: Optional[float] = None  # Globaler Speed-Override
    total_duration: float = 0.0
    
    def __post_init__(self):
        if self.waypoints is None:
            self.waypoints = []
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def add_waypoint(self, waypoint: TeachingWaypoint):
        """Fügt einen Waypoint hinzu."""
        self.waypoints.append(waypoint)
        self._update_duration()
    
    def _update_duration(self):
        """Berechnet die Gesamtdauer."""
        if len(self.waypoints) > 1:
            self.total_duration = (
                self.waypoints[-1].timestamp - self.waypoints[0].timestamp
            )
    
    def optimize(self):
        """Optimiert die Sequenz (entfernt redundante Punkte)."""
        if len(self.waypoints) < 3:
            return
        
        optimized = [self.waypoints[0]]  # Erster Punkt bleibt
        
        for i in range(1, len(self.waypoints) - 1):
            prev = self.waypoints[i - 1]
            curr = self.waypoints[i]
            next = self.waypoints[i + 1]
            
            # Prüfe ob Punkt auf gerader Linie liegt
            if not self._is_redundant(prev, curr, next):
                optimized.append(curr)
        
        optimized.append(self.waypoints[-1])  # Letzter Punkt bleibt
        
        old_count = len(self.waypoints)
        self.waypoints = optimized
        logger.info(f"Optimized sequence: {old_count} -> {len(self.waypoints)} waypoints")
    
    def _is_redundant(self, prev: TeachingWaypoint, 
                      curr: TeachingWaypoint, 
                      next: TeachingWaypoint,
                      threshold: float = 0.01) -> bool:
        """Prüft ob ein Punkt redundant ist."""
        # Prüfe nur wenn gleiche Parameter
        if (curr.speed != prev.speed or 
            curr.acceleration != prev.acceleration or
            curr.trajectory_type != prev.trajectory_type):
            return False
        
        # Prüfe Linearität der Positionen
        for joint in curr.positions:
            if joint not in prev.positions or joint not in next.positions:
                continue
            
            # Interpolierte Position
            t = (curr.timestamp - prev.timestamp) / (next.timestamp - prev.timestamp)
            interpolated = prev.positions[joint] + t * (next.positions[joint] - prev.positions[joint])
            
            # Vergleiche mit tatsächlicher Position
            if abs(curr.positions[joint] - interpolated) > threshold:
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für JSON."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "loop": self.loop,
            "speed_override": self.speed_override,
            "total_duration": self.total_duration
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Erstellt aus Dictionary."""
        sequence = cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            loop=data.get("loop", False),
            speed_override=data.get("speed_override"),
            total_duration=data.get("total_duration", 0.0)
        )
        
        # Waypoints hinzufügen
        for wp_data in data.get("waypoints", []):
            waypoint = TeachingWaypoint.from_dict(wp_data)
            sequence.waypoints.append(waypoint)
        
        return sequence


class TeachingRecorder:
    """
    Teaching Mode Recorder mit erweiterten Features.
    Zeichnet Bewegungen mit allen Parametern auf.
    """
    
    def __init__(self, controller, config: Optional[Dict] = None):
        """
        Initialisiert den Teaching Recorder.
        
        Args:
            controller: RoArm Controller Instanz
            config: Optionale Konfiguration
        """
        self.controller = controller
        self.config = config or TEACHING_DEFAULTS.copy()
        
        # Recording state
        self.is_recording = False
        self.recording_mode = RecordingMode.MANUAL
        self.current_sequence = None
        self.recording_thread = None
        
        # Current parameters (können während Recording geändert werden)
        self.current_speed = 1.0
        self.current_acceleration = 2.0
        self.current_jerk = 5.0
        self.current_trajectory = TrajectoryType.S_CURVE
        self.current_settle_time = 0.0
        self.current_torque = 0.8
        
        # Recording statistics
        self.waypoint_count = 0
        self.start_time = 0.0
        
        # Thread lock
        self._lock = threading.Lock()
        
        logger.info("Teaching Recorder initialized")
    
    def start_recording(self, name: str, 
                       mode: RecordingMode = RecordingMode.MANUAL,
                       description: str = "") -> bool:
        """
        Startet eine neue Aufzeichnung.
        
        Args:
            name: Name der Sequenz
            mode: Aufzeichnungsmodus
            description: Optionale Beschreibung
            
        Returns:
            True wenn erfolgreich gestartet
        """
        try:
            with self._lock:
                if self.is_recording:
                    logger.warning("Already recording")
                    return False
                
                # Neue Sequenz erstellen
                self.current_sequence = TeachingSequence(
                    name=name,
                    description=description
                )
                
                self.recording_mode = mode
                self.waypoint_count = 0
                self.start_time = time.time()
                self.is_recording = True
                
                # LED zur Indikation
                self.controller.led_control(True, brightness=255)
                
                # Starte Recording Thread für kontinuierliche Modi
                if mode in [RecordingMode.CONTINUOUS, RecordingMode.TRIGGERED]:
                    self._start_recording_thread()
                
                logger.info(f"Started recording '{name}' in {mode.value} mode")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self, optimize: bool = True) -> Optional[TeachingSequence]:
        """
        Stoppt die Aufzeichnung.
        
        Args:
            optimize: Sequenz optimieren (redundante Punkte entfernen)
            
        Returns:
            Die aufgezeichnete Sequenz oder None
        """
        try:
            with self._lock:
                if not self.is_recording:
                    logger.warning("Not recording")
                    return None
                
                self.is_recording = False
                
                # Stop recording thread
                if self.recording_thread:
                    self.recording_thread.join(timeout=2)
                    self.recording_thread = None
                
                # LED aus
                self.controller.led_control(False)
                
                # Optimierung
                if optimize and self.current_sequence:
                    self.current_sequence.optimize()
                
                sequence = self.current_sequence
                self.current_sequence = None
                
                duration = time.time() - self.start_time
                logger.info(f"Recording stopped: {self.waypoint_count} waypoints in {duration:.1f}s")
                
                return sequence
                
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return None
    
    def record_waypoint(self, comment: str = "") -> bool:
        """
        Zeichnet einen einzelnen Waypoint auf (MANUAL mode).
        Verwendet die aktuell eingestellten Parameter.
        
        Args:
            comment: Optionaler Kommentar
            
        Returns:
            True wenn erfolgreich
        """
        if not self.is_recording:
            logger.warning("Not recording")
            return False
        
        if self.recording_mode != RecordingMode.MANUAL:
            logger.warning(f"Not in manual mode (current: {self.recording_mode.value})")
            return False
        
        try:
            # Aktuelle Position abfragen
            status = self.controller.query_status()
            if not status:
                logger.error("Failed to query position")
                return False
            
            # Waypoint erstellen
            waypoint = TeachingWaypoint(
                index=self.waypoint_count,
                timestamp=time.time() - self.start_time,
                positions=status['positions'].copy(),
                speed=self.current_speed,
                acceleration=self.current_acceleration,
                jerk=self.current_jerk,
                trajectory_type=self.current_trajectory.value if isinstance(self.current_trajectory, Enum) else self.current_trajectory,
                settle_time=self.current_settle_time,
                torque=self.current_torque,
                gripper_force=self._get_gripper_force(status['positions'].get('hand', 2.0)),
                comment=comment
            )
            
            # Zur Sequenz hinzufügen
            self.current_sequence.add_waypoint(waypoint)
            self.waypoint_count += 1
            
            # Feedback
            logger.info(f"Waypoint {self.waypoint_count} recorded (speed={self.current_speed:.1f}, acc={self.current_acceleration:.1f})")
            
            # LED blink
            self.controller.led_control(False)
            time.sleep(0.1)
            self.controller.led_control(True, brightness=255)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record waypoint: {e}")
            return False
    
    def set_parameters(self, speed: Optional[float] = None,
                       acceleration: Optional[float] = None,
                       jerk: Optional[float] = None,
                       trajectory_type: Optional[TrajectoryType] = None,
                       settle_time: Optional[float] = None,
                       torque: Optional[float] = None):
        """
        Setzt die Parameter für zukünftige Waypoints.
        
        Args:
            speed: Geschwindigkeitsfaktor (0.1-2.0)
            acceleration: Beschleunigung (rad/s²)
            jerk: Ruck (rad/s³)
            trajectory_type: Bewegungsprofil
            settle_time: Wartezeit nach Position
            torque: Servo-Torque (0.0-1.0)
        """
        if speed is not None:
            self.current_speed = max(0.1, min(2.0, speed))
            logger.info(f"Speed set to {self.current_speed:.1f}")
        
        if acceleration is not None:
            self.current_acceleration = max(0.5, min(5.0, acceleration))
            logger.info(f"Acceleration set to {self.current_acceleration:.1f}")
        
        if jerk is not None:
            self.current_jerk = max(1.0, min(10.0, jerk))
            logger.info(f"Jerk set to {self.current_jerk:.1f}")
        
        if trajectory_type is not None:
            self.current_trajectory = trajectory_type
            logger.info(f"Trajectory type set to {trajectory_type.value if isinstance(trajectory_type, Enum) else trajectory_type}")
        
        if settle_time is not None:
            self.current_settle_time = max(0.0, settle_time)
            logger.info(f"Settle time set to {self.current_settle_time:.1f}s")
        
        if torque is not None:
            self.current_torque = max(0.0, min(1.0, torque))
            logger.info(f"Torque set to {self.current_torque:.1f}")
    
    def save_sequence(self, filepath: Optional[str] = None) -> bool:
        """
        Speichert die aktuelle oder zuletzt aufgezeichnete Sequenz.
        
        Args:
            filepath: Dateipfad oder None für automatischen Namen
            
        Returns:
            True wenn erfolgreich gespeichert
        """
        sequence = self.current_sequence
        if not sequence and hasattr(self, 'last_sequence'):
            sequence = self.last_sequence
        
        if not sequence:
            logger.error("No sequence to save")
            return False
        
        try:
            # Dateipfad generieren
            if not filepath:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = f"sequences/{sequence.name}_{timestamp}.json"
            
            # Verzeichnis erstellen
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Speichern
            with open(filepath, 'w') as f:
                json.dump(sequence.to_dict(), f, indent=2)
            
            logger.info(f"Sequence saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sequence: {e}")
            return False
    
    def load_sequence(self, filepath: str) -> Optional[TeachingSequence]:
        """
        Lädt eine gespeicherte Sequenz.
        
        Args:
            filepath: Dateipfad
            
        Returns:
            Geladene Sequenz oder None
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sequence = TeachingSequence.from_dict(data)
            logger.info(f"Loaded sequence '{sequence.name}' with {len(sequence.waypoints)} waypoints")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Failed to load sequence: {e}")
            return None
    
    # ============== PRIVATE METHODS ==============
    
    def _start_recording_thread(self):
        """Startet den Recording Thread für kontinuierliche Aufzeichnung."""
        self.recording_thread = threading.Thread(
            target=self._continuous_recording,
            daemon=True
        )
        self.recording_thread.start()
    
    def _continuous_recording(self):
        """Thread-Funktion für kontinuierliche Aufzeichnung."""
        sample_interval = 1.0 / self.config['sample_rate']
        last_positions = {}
        
        while self.is_recording:
            try:
                # Position abfragen
                status = self.controller.query_status()
                if not status:
                    time.sleep(sample_interval)
                    continue
                
                positions = status['positions']
                
                # Modus-spezifische Aufzeichnung
                if self.recording_mode == RecordingMode.CONTINUOUS:
                    # Immer aufzeichnen
                    self._record_continuous_waypoint(positions)
                    
                elif self.recording_mode == RecordingMode.TRIGGERED:
                    # Nur bei Änderung aufzeichnen
                    if self._has_position_changed(positions, last_positions):
                        self._record_continuous_waypoint(positions)
                        last_positions = positions.copy()
                
                time.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Recording thread error: {e}")
                time.sleep(0.1)
    
    def _record_continuous_waypoint(self, positions: Dict[str, float]):
        """Zeichnet einen Waypoint im kontinuierlichen Modus auf."""
        waypoint = TeachingWaypoint(
            index=self.waypoint_count,
            timestamp=time.time() - self.start_time,
            positions=positions.copy(),
            speed=self.current_speed,
            acceleration=self.current_acceleration,
            jerk=self.current_jerk,
            trajectory_type=self.current_trajectory.value if isinstance(self.current_trajectory, Enum) else self.current_trajectory,
            settle_time=0.0,  # Keine Pause bei kontinuierlicher Aufzeichnung
            torque=self.current_torque,
            gripper_force=self._get_gripper_force(positions.get('hand', 2.0))
        )
        
        self.current_sequence.add_waypoint(waypoint)
        self.waypoint_count += 1
    
    def _has_position_changed(self, current: Dict[str, float], 
                             previous: Dict[str, float]) -> bool:
        """Prüft ob sich die Position geändert hat."""
        if not previous:
            return True
        
        threshold = self.config['position_threshold']
        
        for joint, pos in current.items():
            if joint not in previous:
                return True
            if abs(pos - previous[joint]) > threshold:
                return True
        
        return False
    
    def _get_gripper_force(self, hand_position: float) -> float:
        """Berechnet die Greifkraft aus der Hand-Position."""
        # Map hand position to force (1.08=open=0.0, 3.14=closed=1.0)
        min_pos = 1.08
        max_pos = 3.14
        force = (hand_position - min_pos) / (max_pos - min_pos)
        return max(0.0, min(1.0, force))
