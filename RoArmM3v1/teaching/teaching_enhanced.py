#!/usr/bin/env python3
"""
RoArm M3 Enhanced Teaching Mode
Vollständige Aufzeichnung, Speicherung und Wiedergabe von Bewegungssequenzen
"""

import json
import time
import threading
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import pickle
import csv

from ..core.constants import TEACHING_DEFAULTS, TRAJECTORY_PROFILES, SERVO_LIMITS
from ..motion.trajectory import TrajectoryType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RecordingMode(Enum):
    """Aufzeichnungsmodi."""
    MANUAL = "manual"          # Manuell per Tastendruck
    CONTINUOUS = "continuous"  # Kontinuierlich mit Sampling
    TRIGGERED = "triggered"    # Bei Positionsänderung
    TIMED = "timed"           # Zeitbasiert mit Intervallen
    SMART = "smart"           # Intelligent mit Optimierung


class PlaybackMode(Enum):
    """Wiedergabe-Modi."""
    NORMAL = "normal"          # Normale Geschwindigkeit
    SLOW = "slow"             # Langsame Wiedergabe
    FAST = "fast"             # Schnelle Wiedergabe
    STEP = "step"             # Schritt für Schritt
    LOOP = "loop"             # Endlos-Schleife
    REVERSE = "reverse"       # Rückwärts


@dataclass
class EnhancedWaypoint:
    """
    Erweiterter Wegpunkt mit allen Parametern.
    """
    index: int
    timestamp: float
    positions: Dict[str, float]
    velocities: Dict[str, float] = None
    accelerations: Dict[str, float] = None
    speed: float = 1.0
    acceleration: float = 2.0
    jerk: float = 5.0
    trajectory_type: str = "s_curve"
    settle_time: float = 0.0
    torque: float = 0.8
    gripper_force: float = 0.5
    led_state: bool = False
    led_brightness: int = 0
    comment: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.velocities is None:
            self.velocities = {}
        if self.accelerations is None:
            self.accelerations = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Erstellt aus Dictionary."""
        return cls(**data)
    
    def interpolate_to(self, other: 'EnhancedWaypoint', t: float) -> 'EnhancedWaypoint':
        """
        Interpoliert zu anderem Waypoint.
        
        Args:
            other: Ziel-Waypoint
            t: Interpolationsfaktor (0.0 bis 1.0)
            
        Returns:
            Interpolierter Waypoint
        """
        # Linear interpolation for positions
        interp_positions = {}
        for joint in self.positions:
            if joint in other.positions:
                interp_positions[joint] = (
                    self.positions[joint] * (1 - t) + 
                    other.positions[joint] * t
                )
        
        # Interpolate other parameters
        interp_speed = self.speed * (1 - t) + other.speed * t
        interp_timestamp = self.timestamp * (1 - t) + other.timestamp * t
        
        return EnhancedWaypoint(
            index=-1,  # Interpolated point
            timestamp=interp_timestamp,
            positions=interp_positions,
            speed=interp_speed,
            acceleration=self.acceleration,
            trajectory_type=self.trajectory_type
        )


@dataclass
class TeachingSequence:
    """
    Vollständige Teaching-Sequenz mit Metadaten.
    """
    name: str
    description: str = ""
    created_at: float = 0.0
    modified_at: float = 0.0
    waypoints: List[EnhancedWaypoint] = None
    metadata: Dict[str, Any] = None
    
    # Playback settings
    loop: bool = False
    speed_override: Optional[float] = None
    reverse_compatible: bool = True
    
    # Statistics
    total_duration: float = 0.0
    total_distance: float = 0.0
    joint_ranges: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.waypoints is None:
            self.waypoints = []
        if self.metadata is None:
            self.metadata = {}
        if self.joint_ranges is None:
            self.joint_ranges = {}
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.modified_at = time.time()
    
    def add_waypoint(self, waypoint: EnhancedWaypoint):
        """Fügt einen Waypoint hinzu."""
        waypoint.index = len(self.waypoints)
        self.waypoints.append(waypoint)
        self._update_statistics()
    
    def remove_waypoint(self, index: int):
        """Entfernt einen Waypoint."""
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)
            self._reindex_waypoints()
            self._update_statistics()
    
    def insert_waypoint(self, index: int, waypoint: EnhancedWaypoint):
        """Fügt einen Waypoint an bestimmter Stelle ein."""
        if 0 <= index <= len(self.waypoints):
            self.waypoints.insert(index, waypoint)
            self._reindex_waypoints()
            self._update_statistics()
    
    def _reindex_waypoints(self):
        """Neu-Indizierung der Waypoints."""
        for i, wp in enumerate(self.waypoints):
            wp.index = i
    
    def _update_statistics(self):
        """Aktualisiert Sequenz-Statistiken."""
        if len(self.waypoints) < 2:
            return
        
        # Duration
        self.total_duration = (
            self.waypoints[-1].timestamp - self.waypoints[0].timestamp
        )
        
        # Distance and ranges
        self.total_distance = 0.0
        self.joint_ranges = {}
        
        for i in range(1, len(self.waypoints)):
            prev = self.waypoints[i-1]
            curr = self.waypoints[i]
            
            # Distance
            for joint in curr.positions:
                if joint in prev.positions:
                    self.total_distance += abs(
                        curr.positions[joint] - prev.positions[joint]
                    )
                    
                    # Update ranges
                    if joint not in self.joint_ranges:
                        self.joint_ranges[joint] = (
                            curr.positions[joint],
                            curr.positions[joint]
                        )
                    else:
                        min_val, max_val = self.joint_ranges[joint]
                        self.joint_ranges[joint] = (
                            min(min_val, curr.positions[joint]),
                            max(max_val, curr.positions[joint])
                        )
    
    def optimize(self, tolerance: float = 0.01):
        """
        Optimiert die Sequenz (entfernt redundante Punkte).
        
        Args:
            tolerance: Toleranz für Redundanz-Prüfung
        """
        if len(self.waypoints) < 3:
            return
        
        optimized = [self.waypoints[0]]
        
        for i in range(1, len(self.waypoints) - 1):
            if not self._is_redundant(i, tolerance):
                optimized.append(self.waypoints[i])
        
        optimized.append(self.waypoints[-1])
        
        old_count = len(self.waypoints)
        self.waypoints = optimized
        self._reindex_waypoints()
        
        logger.info(f"Optimized: {old_count} -> {len(self.waypoints)} waypoints")
    
    def _is_redundant(self, index: int, tolerance: float) -> bool:
        """Prüft ob ein Waypoint redundant ist."""
        if index <= 0 or index >= len(self.waypoints) - 1:
            return False
        
        prev = self.waypoints[index - 1]
        curr = self.waypoints[index]
        next = self.waypoints[index + 1]
        
        # Check if on straight line
        for joint in curr.positions:
            if joint not in prev.positions or joint not in next.positions:
                continue
            
            # Linear interpolation
            t = (curr.timestamp - prev.timestamp) / (
                next.timestamp - prev.timestamp
            )
            expected = prev.positions[joint] + t * (
                next.positions[joint] - prev.positions[joint]
            )
            
            if abs(curr.positions[joint] - expected) > tolerance:
                return False
        
        return True
    
    def smooth(self, window_size: int = 3):
        """
        Glättet die Sequenz mit Moving Average.
        
        Args:
            window_size: Fenster-Größe für Glättung
        """
        if len(self.waypoints) < window_size:
            return
        
        smoothed = []
        
        for i in range(len(self.waypoints)):
            # Window indices
            start = max(0, i - window_size // 2)
            end = min(len(self.waypoints), i + window_size // 2 + 1)
            window = self.waypoints[start:end]
            
            # Average positions
            avg_positions = {}
            for joint in self.waypoints[i].positions:
                values = [wp.positions.get(joint, 0) for wp in window]
                avg_positions[joint] = sum(values) / len(values)
            
            # Create smoothed waypoint
            smoothed_wp = EnhancedWaypoint(
                index=i,
                timestamp=self.waypoints[i].timestamp,
                positions=avg_positions,
                speed=self.waypoints[i].speed,
                acceleration=self.waypoints[i].acceleration,
                trajectory_type=self.waypoints[i].trajectory_type
            )
            smoothed.append(smoothed_wp)
        
        self.waypoints = smoothed
        logger.info(f"Sequence smoothed with window size {window_size}")
    
    def reverse(self):
        """Kehrt die Sequenz um."""
        self.waypoints.reverse()
        
        # Adjust timestamps
        if self.waypoints:
            total_time = self.waypoints[0].timestamp
            for wp in self.waypoints:
                wp.timestamp = total_time - wp.timestamp
        
        self._reindex_waypoints()
        logger.info("Sequence reversed")
    
    def scale_time(self, factor: float):
        """
        Skaliert die Zeit der Sequenz.
        
        Args:
            factor: Zeitfaktor (>1 = langsamer, <1 = schneller)
        """
        for wp in self.waypoints:
            wp.timestamp *= factor
        
        self._update_statistics()
        logger.info(f"Time scaled by factor {factor}")
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "metadata": self.metadata,
            "loop": self.loop,
            "speed_override": self.speed_override,
            "total_duration": self.total_duration,
            "total_distance": self.total_distance,
            "joint_ranges": self.joint_ranges
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Erstellt aus Dictionary."""
        sequence = cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time()),
            metadata=data.get("metadata", {}),
            loop=data.get("loop", False),
            speed_override=data.get("speed_override")
        )
        
        # Load waypoints
        for wp_data in data.get("waypoints", []):
            waypoint = EnhancedWaypoint.from_dict(wp_data)
            sequence.waypoints.append(waypoint)
        
        sequence._update_statistics()
        return sequence


class EnhancedTeachingRecorder:
    """
    Erweiterter Teaching Mode Recorder mit allen Features.
    """
    
    def __init__(self, controller, config: Optional[Dict] = None):
        """
        Initialisiert den Teaching Recorder.
        
        Args:
            controller: RoArm Controller
            config: Optionale Konfiguration
        """
        self.controller = controller
        self.config = config or TEACHING_DEFAULTS.copy()
        
        # Recording state
        self.is_recording = False
        self.recording_mode = RecordingMode.MANUAL
        self.current_sequence = None
        self.recording_thread = None
        
        # Playback state
        self.is_playing = False
        self.playback_mode = PlaybackMode.NORMAL
        self.playback_thread = None
        self.playback_paused = False
        self.playback_position = 0
        
        # Parameters
        self.current_speed = 1.0
        self.current_acceleration = 2.0
        self.current_jerk = 5.0
        self.current_trajectory = TrajectoryType.S_CURVE
        self.current_settle_time = 0.0
        
        # Statistics
        self.waypoint_count = 0
        self.start_time = 0.0
        
        # Sequence library
        self.sequences = {}  # Name -> Sequence
        self.sequence_directory = Path("sequences")
        self.sequence_directory.mkdir(exist_ok=True)
        
        # Load saved sequences
        self._load_sequence_library()
        
        logger.info("Enhanced Teaching Recorder initialized")
    
    def start_recording(self, name: str, 
                       mode: RecordingMode = RecordingMode.MANUAL,
                       description: str = "") -> bool:
        """
        Startet eine neue Aufzeichnung.
        
        Args:
            name: Name der Sequenz
            mode: Aufzeichnungsmodus
            description: Beschreibung
            
        Returns:
            True wenn erfolgreich
        """
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        if self.is_playing:
            logger.warning("Cannot record while playing")
            return False
        
        # Create new sequence
        self.current_sequence = TeachingSequence(
            name=name,
            description=description
        )
        
        self.recording_mode = mode
        self.waypoint_count = 0
        self.start_time = time.time()
        self.is_recording = True
        
        # LED indication
        self.controller.led_control(True, brightness=255)
        
        # Start recording thread for continuous modes
        if mode in [RecordingMode.CONTINUOUS, RecordingMode.TRIGGERED, 
                   RecordingMode.TIMED, RecordingMode.SMART]:
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True
            )
            self.recording_thread.start()
        
        logger.info(f"Started recording '{name}' in {mode.value} mode")
        return True
    
    def stop_recording(self, optimize: bool = True, 
                      save: bool = True) -> Optional[TeachingSequence]:
        """
        Stoppt die Aufzeichnung.
        
        Args:
            optimize: Sequenz optimieren
            save: Automatisch speichern
            
        Returns:
            Aufgezeichnete Sequenz oder None
        """
        if not self.is_recording:
            logger.warning("Not recording")
            return None
        
        self.is_recording = False
        
        # Stop thread
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
            self.recording_thread = None
        
        # LED off
        self.controller.led_control(False)
        
        # Optimize if requested
        if optimize and self.current_sequence:
            self.current_sequence.optimize()
        
        # Save if requested
        if save and self.current_sequence:
            self.save_sequence(self.current_sequence)
        
        sequence = self.current_sequence
        self.current_sequence = None
        
        if sequence:
            duration = time.time() - self.start_time
            logger.info(
                f"Recording stopped: {len(sequence.waypoints)} waypoints "
                f"in {duration:.1f}s"
            )
        
        return sequence
    
    def record_waypoint(self, comment: str = "", 
                       tags: List[str] = None) -> bool:
        """
        Zeichnet einen Waypoint auf (MANUAL mode).
        
        Args:
            comment: Kommentar
            tags: Tags für den Waypoint
            
        Returns:
            True wenn erfolgreich
        """
        if not self.is_recording:
            logger.warning("Not recording")
            return False
        
        if self.recording_mode != RecordingMode.MANUAL:
            logger.warning("Not in manual mode")
            return False
        
        # Get current state
        status = self.controller.query_status()
        if not status:
            return False
        
        # Create waypoint
        waypoint = EnhancedWaypoint(
            index=self.waypoint_count,
            timestamp=time.time() - self.start_time,
            positions=status['positions'].copy(),
            velocities=status.get('velocities', {}),
            speed=self.current_speed,
            acceleration=self.current_acceleration,
            jerk=self.current_jerk,
            trajectory_type=self.current_trajectory.value if hasattr(self.current_trajectory, 'value') else str(self.current_trajectory),
            settle_time=self.current_settle_time,
            comment=comment,
            tags=tags or []
        )
        
        # Add to sequence
        self.current_sequence.add_waypoint(waypoint)
        self.waypoint_count += 1
        
        # Feedback
        logger.info(f"Waypoint {self.waypoint_count} recorded")
        
        # LED blink
        self.controller.led_control(False)
        time.sleep(0.1)
        self.controller.led_control(True, brightness=255)
        
        return True
    
    def start_playback(self, sequence: TeachingSequence,
                      mode: PlaybackMode = PlaybackMode.NORMAL,
                      start_index: int = 0) -> bool:
        """
        Startet die Wiedergabe einer Sequenz.
        
        Args:
            sequence: Abzuspielende Sequenz
            mode: Wiedergabe-Modus
            start_index: Start-Waypoint
            
        Returns:
            True wenn erfolgreich
        """
        if self.is_playing:
            logger.warning("Already playing")
            return False
        
        if self.is_recording:
            logger.warning("Cannot play while recording")
            return False
        
        if not sequence.waypoints:
            logger.warning("Sequence is empty")
            return False
        
        self.is_playing = True
        self.playback_mode = mode
        self.playback_position = start_index
        self.playback_paused = False
        
        # Start playback thread
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(sequence,),
            daemon=True
        )
        self.playback_thread.start()
        
        logger.info(f"Started playback of '{sequence.name}' in {mode.value} mode")
        return True
    
    def stop_playback(self):
        """Stoppt die Wiedergabe."""
        if not self.is_playing:
            logger.warning("Not playing")
            return
        
        self.is_playing = False
        
        # Stop thread
        if self.playback_thread:
            self.playback_thread.join(timeout=2)
            self.playback_thread = None
        
        logger.info("Playback stopped")
    
    def pause_playback(self):
        """Pausiert die Wiedergabe."""
        if self.is_playing:
            self.playback_paused = True
            logger.info("Playback paused")
    
    def resume_playback(self):
        """Setzt die Wiedergabe fort."""
        if self.is_playing and self.playback_paused:
            self.playback_paused = False
            logger.info("Playback resumed")
    
    def save_sequence(self, sequence: TeachingSequence, 
                     filepath: Optional[str] = None) -> bool:
        """
        Speichert eine Sequenz.
        
        Args:
            sequence: Zu speichernde Sequenz
            filepath: Dateipfad oder None für Auto
            
        Returns:
            True wenn erfolgreich
        """
        try:
            # Generate filepath if not provided
            if not filepath:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = self.sequence_directory / f"{sequence.name}_{timestamp}.json"
            else:
                filepath = Path(filepath)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(sequence.to_dict(), f, indent=2)
            
            # Add to library
            self.sequences[sequence.name] = sequence
            
            logger.info(f"Sequence saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sequence: {e}")
            return False
    
    def load_sequence(self, filepath: str) -> Optional[TeachingSequence]:
        """
        Lädt eine Sequenz.
        
        Args:
            filepath: Dateipfad
            
        Returns:
            Geladene Sequenz oder None
        """
        try:
            filepath = Path(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sequence = TeachingSequence.from_dict(data)
            
            # Add to library
            self.sequences[sequence.name] = sequence
            
            logger.info(f"Loaded sequence '{sequence.name}'")
            return sequence
            
        except Exception as e:
            logger.error(f"Failed to load sequence: {e}")
            return None
    
    def export_sequence(self, sequence: TeachingSequence,
                       filepath: str,
                       format: str = "csv") -> bool:
        """
        Exportiert eine Sequenz in verschiedenen Formaten.
        
        Args:
            sequence: Sequenz
            filepath: Ziel-Datei
            format: Format (csv, txt, pickle)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            filepath = Path(filepath)
            
            if format == "csv":
                self._export_csv(sequence, filepath)
            elif format == "txt":
                self._export_txt(sequence, filepath)
            elif format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(sequence, f)
            else:
                logger.error(f"Unknown format: {format}")
                return False
            
            logger.info(f"Exported to {filepath} as {format}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def list_sequences(self) -> List[str]:
        """
        Listet alle verfügbaren Sequenzen.
        
        Returns:
            Liste der Sequenz-Namen
        """
        return list(self.sequences.keys())
    
    def get_sequence(self, name: str) -> Optional[TeachingSequence]:
        """
        Holt eine Sequenz aus der Library.
        
        Args:
            name: Sequenz-Name
            
        Returns:
            Sequenz oder None
        """
        return self.sequences.get(name)
    
    def delete_sequence(self, name: str) -> bool:
        """
        Löscht eine Sequenz.
        
        Args:
            name: Sequenz-Name
            
        Returns:
            True wenn erfolgreich
        """
        if name in self.sequences:
            del self.sequences[name]
            logger.info(f"Deleted sequence '{name}'")
            return True
        return False
    
    def merge_sequences(self, sequences: List[TeachingSequence],
                       name: str) -> TeachingSequence:
        """
        Verbindet mehrere Sequenzen.
        
        Args:
            sequences: Liste von Sequenzen
            name: Name der neuen Sequenz
            
        Returns:
            Neue kombinierte Sequenz
        """
        merged = TeachingSequence(name=name)
        
        time_offset = 0.0
        for seq in sequences:
            for wp in seq.waypoints:
                # Create copy with adjusted timestamp
                new_wp = EnhancedWaypoint(
                    index=len(merged.waypoints),
                    timestamp=wp.timestamp + time_offset,
                    positions=wp.positions.copy(),
                    speed=wp.speed,
                    acceleration=wp.acceleration,
                    trajectory_type=wp.trajectory_type
                )
                merged.add_waypoint(new_wp)
            
            # Update time offset for next sequence
            if seq.waypoints:
                time_offset += seq.waypoints[-1].timestamp + 1.0
        
        logger.info(f"Merged {len(sequences)} sequences into '{name}'")
        return merged
    
    # Private methods
    
    def _recording_loop(self):
        """Thread-Loop für kontinuierliche Aufzeichnung."""
        sample_interval = 1.0 / self.config['sample_rate']
        last_positions = {}
        
        while self.is_recording:
            try:
                # Get current state
                status = self.controller.query_status()
                if not status:
                    time.sleep(sample_interval)
                    continue
                
                positions = status['positions']
                
                # Mode-specific recording
                if self.recording_mode == RecordingMode.CONTINUOUS:
                    self._record_continuous(positions)
                    
                elif self.recording_mode == RecordingMode.TRIGGERED:
                    if self._has_changed(positions, last_positions):
                        self._record_continuous(positions)
                        last_positions = positions.copy()
                        
                elif self.recording_mode == RecordingMode.TIMED:
                    if self.waypoint_count == 0 or (
                        time.time() - self.start_time - 
                        self.current_sequence.waypoints[-1].timestamp
                    ) > 1.0:  # Every second
                        self._record_continuous(positions)
                        
                elif self.recording_mode == RecordingMode.SMART:
                    # Smart recording with curve detection
                    if self._should_record_smart(positions, last_positions):
                        self._record_continuous(positions)
                        last_positions = positions.copy()
                
                time.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Recording error: {e}")
                time.sleep(0.1)
    
    def _playback_loop(self, sequence: TeachingSequence):
        """Thread-Loop für Wiedergabe."""
        waypoints = sequence.waypoints
        
        # Playback modes
        if self.playback_mode == PlaybackMode.REVERSE:
            waypoints = list(reversed(waypoints))
        
        # Speed factor
        speed_factor = 1.0
        if self.playback_mode == PlaybackMode.SLOW:
            speed_factor = 0.5
        elif self.playback_mode == PlaybackMode.FAST:
            speed_factor = 2.0
        
        if sequence.speed_override:
            speed_factor = sequence.speed_override
        
        # Main playback loop
        while self.is_playing:
            try:
                # Loop handling
                if self.playback_position >= len(waypoints):
                    if self.playback_mode == PlaybackMode.LOOP or sequence.loop:
                        self.playback_position = 0
                    else:
                        self.is_playing = False
                        break
                
                # Pause handling
                if self.playback_paused:
                    time.sleep(0.1)
                    continue
                
                # Get current waypoint
                wp = waypoints[self.playback_position]
                
                # Move to waypoint
                self.controller.move_joints(
                    wp.positions,
                    speed=wp.speed * speed_factor,
                    trajectory_type=TrajectoryType.S_CURVE,
                    wait=True
                )
                
                # Settle time
                if wp.settle_time > 0:
                    time.sleep(wp.settle_time)
                
                # LED state
                if wp.led_state:
                    self.controller.led_control(True, wp.led_brightness)
                
                # Step mode
                if self.playback_mode == PlaybackMode.STEP:
                    self.playback_paused = True
                    logger.info(f"Step {self.playback_position + 1}/{len(waypoints)}")
                
                # Next waypoint
                self.playback_position += 1
                
            except Exception as e:
                logger.error(f"Playback error: {e}")
                self.is_playing = False
    
    def _record_continuous(self, positions: Dict[str, float]):
        """Records a waypoint in continuous mode."""
        waypoint = EnhancedWaypoint(
            index=self.waypoint_count,
            timestamp=time.time() - self.start_time,
            positions=positions.copy(),
            speed=self.current_speed,
            acceleration=self.current_acceleration,
            trajectory_type=str(self.current_trajectory)
        )
        
        self.current_sequence.add_waypoint(waypoint)
        self.waypoint_count += 1
    
    def _has_changed(self, current: Dict, previous: Dict,
                    threshold: float = None) -> bool:
        """Checks if position has changed significantly."""
        if not previous:
            return True
        
        threshold = threshold or self.config['position_threshold']
        
        for joint in current:
            if joint in previous:
                if abs(current[joint] - previous[joint]) > threshold:
                    return True
        return False
    
    def _should_record_smart(self, current: Dict, previous: Dict) -> bool:
        """Smart decision whether to record waypoint."""
        if not previous:
            return True
        
        # Check for direction change (curve detection)
        if len(self.current_sequence.waypoints) >= 2:
            prev_wp = self.current_sequence.waypoints[-1]
            prev_prev_wp = self.current_sequence.waypoints[-2]
            
            for joint in current:
                if joint in previous and joint in prev_prev_wp.positions:
                    # Calculate velocities
                    v1 = previous[joint] - prev_prev_wp.positions[joint]
                    v2 = current[joint] - previous[joint]
                    
                    # Check for direction change
                    if v1 * v2 < 0:  # Sign change
                        return True
                    
                    # Check for significant acceleration
                    if abs(v2 - v1) > 0.05:
                        return True
        
        # Regular threshold check
        return self._has_changed(current, previous)
    
    def _load_sequence_library(self):
        """Loads all sequences from directory."""
        for filepath in self.sequence_directory.glob("*.json"):
            try:
                sequence = self.load_sequence(filepath)
                if sequence:
                    self.sequences[sequence.name] = sequence
            except:
                pass
        
        logger.info(f"Loaded {len(self.sequences)} sequences")
    
    def _export_csv(self, sequence: TeachingSequence, filepath: Path):
        """Exports sequence as CSV."""
        with open(filepath, 'w', newline='') as f:
            # Get all joint names
            joints = set()
            for wp in sequence.waypoints:
                joints.update(wp.positions.keys())
            joints = sorted(joints)
            
            # Write header
            writer = csv.writer(f)
            header = ['index', 'timestamp'] + joints + ['speed', 'comment']
            writer.writerow(header)
            
            # Write waypoints
            for wp in sequence.waypoints:
                row = [wp.index, wp.timestamp]
                row += [wp.positions.get(j, 0) for j in joints]
                row += [wp.speed, wp.comment]
                writer.writerow(row)
    
    def _export_txt(self, sequence: TeachingSequence, filepath: Path):
        """Exports sequence as readable text."""
        with open(filepath, 'w') as f:
            f.write(f"Teaching Sequence: {sequence.name}\n")
            f.write(f"Description: {sequence.description}\n")
            f.write(f"Created: {time.ctime(sequence.created_at)}\n")
            f.write(f"Duration: {sequence.total_duration:.1f}s\n")
            f.write(f"Distance: {sequence.total_distance:.2f} rad\n")
            f.write(f"Waypoints: {len(sequence.waypoints)}\n")
            f.write("\n" + "="*60 + "\n\n")
            
            for wp in sequence.waypoints:
                f.write(f"Waypoint {wp.index}:\n")
                f.write(f"  Time: {wp.timestamp:.2f}s\n")
                f.write(f"  Positions:\n")
                for joint, pos in wp.positions.items():
                    f.write(f"    {joint:10s}: {pos:+.3f} rad\n")
                f.write(f"  Speed: {wp.speed:.1f}\n")
                if wp.comment:
                    f.write(f"  Comment: {wp.comment}\n")
                f.write("\n")


# Kompatibilität mit altem TeachingRecorder
class TeachingRecorder(EnhancedTeachingRecorder):
    """Alias für Kompatibilität."""
    pass
