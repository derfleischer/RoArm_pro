#!/usr/bin/env python3
"""
RoArm M3 Enhanced Manual Control
Intuitive Echtzeit-Steuerung ohne Enter-Taste
Mit visueller Rückmeldung und erweiterten Features
"""

import time
import threading
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.constants import SERVO_LIMITS, HOME_POSITION
from ..motion.trajectory import TrajectoryType
from ..utils.terminal_enhanced import EnhancedTerminalController, KeyMode
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ControlMode(Enum):
    """Steuerungsmodi."""
    JOINT = "joint"          # Einzelne Joints
    CARTESIAN = "cartesian"  # Kartesische Steuerung (XYZ)
    TOOL = "tool"           # Tool-zentriert
    PRESET = "preset"       # Vordefinierte Positionen


@dataclass
class ControlState:
    """Aktueller Steuerungszustand."""
    mode: ControlMode = ControlMode.JOINT
    speed: float = 1.0
    step_size: float = 0.05
    continuous: bool = True
    smooth: bool = True
    recording: bool = False
    paused: bool = False
    
    # Visual feedback
    show_positions: bool = True
    show_velocities: bool = False
    show_limits: bool = True
    
    # Safety
    soft_limits: bool = True
    collision_check: bool = True
    emergency_active: bool = False


class EnhancedManualControl:
    """
    Erweiterte manuelle Steuerung mit Echtzeit-Feedback.
    """
    
    def __init__(self, controller, teaching_recorder=None):
        """
        Initialisiert Manual Control.
        
        Args:
            controller: RoArm Controller
            teaching_recorder: Optional Teaching Recorder für Aufzeichnung
        """
        self.controller = controller
        self.teaching_recorder = teaching_recorder
        self.terminal = EnhancedTerminalController()
        
        # Control state
        self.state = ControlState()
        self.active = False
        
        # Movement tracking
        self.current_movements = {}  # Joint -> direction
        self.movement_thread = None
        self.last_update = time.time()
        
        # Position tracking
        self.target_positions = controller.current_position.copy()
        self.actual_positions = controller.current_position.copy()
        
        # Presets
        self.presets = {
            '1': ('Home', HOME_POSITION),
            '2': ('Scanner', {
                "base": 0.0, "shoulder": 0.35, "elbow": 1.22,
                "wrist": -1.57, "roll": 1.57, "hand": 2.5
            }),
            '3': ('Park', {
                "base": 0.0, "shoulder": -1.57, "elbow": 3.14,
                "wrist": 0.0, "roll": 0.0, "hand": 3.14
            }),
            '4': ('Up', {
                "base": 0.0, "shoulder": 1.0, "elbow": 0.5,
                "wrist": 0.0, "roll": 0.0, "hand": 2.0
            }),
            '5': ('Forward', {
                "base": 0.0, "shoulder": 0.0, "elbow": 0.0,
                "wrist": 0.0, "roll": 0.0, "hand": 2.0
            })
        }
        
        # Visual elements
        self.status_lines = []
        self.last_screen_update = 0
        self.screen_update_rate = 0.1  # 10 Hz
        
        logger.info("Enhanced Manual Control initialized")
    
    def start(self):
        """Startet die manuelle Steuerung."""
        if self.active:
            logger.warning("Manual control already active")
            return
        
        self.active = True
        self.terminal.start()
        
        # Setup key mappings
        self._setup_controls()
        
        # Start movement thread
        self.movement_thread = threading.Thread(
            target=self._movement_loop,
            daemon=True
        )
        self.movement_thread.start()
        
        # Clear screen and show interface
        self.terminal.clear_screen()
        self.terminal.hide_cursor()
        self._draw_interface()
        
        logger.info("Manual control started")
        
        # Main loop
        try:
            while self.active:
                self._update_display()
                time.sleep(0.05)
                
                # Check for exit
                if self.terminal.is_key_pressed('x') or self.terminal.is_key_pressed('X'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stoppt die manuelle Steuerung."""
        if not self.active:
            return
        
        self.active = False
        
        # Stop movement thread
        if self.movement_thread:
            self.movement_thread.join(timeout=1)
        
        # Restore terminal
        self.terminal.show_cursor()
        self.terminal.stop()
        
        # Clear screen
        print("\033[2J\033[H", end='')
        
        logger.info("Manual control stopped")
    
    def _setup_controls(self):
        """Konfiguriert alle Steuerungs-Mappings."""
        
        # Joint control keys (continuous mode)
        joint_mappings = {
            # Base
            'q': ('base', -1),
            'Q': ('base', -1),
            'a': ('base', 1),
            'A': ('base', 1),
            # Shoulder
            'w': ('shoulder', 1),
            'W': ('shoulder', 1),
            's': ('shoulder', -1),
            'S': ('shoulder', -1),
            # Elbow
            'e': ('elbow', 1),
            'E': ('elbow', 1),
            'd': ('elbow', -1),
            'D': ('elbow', -1),
            # Wrist
            'r': ('wrist', 1),
            'R': ('wrist', 1),
            'f': ('wrist', -1),
            'F': ('wrist', -1),
            # Roll
            't': ('roll', 1),
            'T': ('roll', 1),
            'g': ('roll', -1),
            'G': ('roll', -1),
            # Gripper
            'y': ('hand', -1),
            'Y': ('hand', -1),
            'h': ('hand', 1),
            'H': ('hand', 1)
        }
        
        # Register movement keys
        for key, (joint, direction) in joint_mappings.items():
            self.terminal.register_key(
                key,
                lambda j=joint, d=direction: self._start_movement(j, d),
                KeyMode.SINGLE
            )
            
            # Also register for continuous
            self.terminal.register_key(
                key,
                lambda j=joint, d=direction: self._continue_movement(j, d),
                KeyMode.CONTINUOUS
            )
        
        # Control keys
        self.terminal.register_key('+', self._increase_speed, KeyMode.SINGLE)
        self.terminal.register_key('=', self._increase_speed, KeyMode.SINGLE)
        self.terminal.register_key('-', self._decrease_speed, KeyMode.SINGLE)
        self.terminal.register_key('_', self._decrease_speed, KeyMode.SINGLE)
        
        # Step size
        self.terminal.register_key('[', self._decrease_step, KeyMode.SINGLE)
        self.terminal.register_key(']', self._increase_step, KeyMode.SINGLE)
        
        # Modes
        self.terminal.register_key('m', self._cycle_mode, KeyMode.SINGLE)
        self.terminal.register_key('M', self._cycle_mode, KeyMode.SINGLE)
        self.terminal.register_key('c', self._toggle_continuous, KeyMode.SINGLE)
        self.terminal.register_key('C', self._toggle_continuous, KeyMode.SINGLE)
        
        # Presets (1-5)
        for key, (name, _) in self.presets.items():
            self.terminal.register_key(
                key,
                lambda k=key: self._goto_preset(k),
                KeyMode.SINGLE
            )
        
        # Special functions
        self.terminal.register_key('SPACE', self._emergency_stop, KeyMode.SINGLE)
        self.terminal.register_key(' ', self._emergency_stop, KeyMode.SINGLE)
        self.terminal.register_key('p', self._toggle_pause, KeyMode.SINGLE)
        self.terminal.register_key('P', self._toggle_pause, KeyMode.SINGLE)
        self.terminal.register_key('0', self._go_home, KeyMode.SINGLE)
        
        # Recording
        if self.teaching_recorder:
            self.terminal.register_key('ENTER', self._record_waypoint, KeyMode.SINGLE)
            self.terminal.register_key('\r', self._record_waypoint, KeyMode.SINGLE)
            self.terminal.register_key('TAB', self._toggle_recording, KeyMode.SINGLE)
            self.terminal.register_key('\t', self._toggle_recording, KeyMode.SINGLE)
        
        # Display toggles
        self.terminal.register_key('v', self._toggle_velocities, KeyMode.SINGLE)
        self.terminal.register_key('V', self._toggle_velocities, KeyMode.SINGLE)
        self.terminal.register_key('l', self._toggle_limits, KeyMode.SINGLE)
        self.terminal.register_key('L', self._toggle_limits, KeyMode.SINGLE)
    
    def _movement_loop(self):
        """Thread-Loop für kontinuierliche Bewegungen."""
        last_time = time.time()
        
        while self.active:
            try:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                if self.state.paused:
                    time.sleep(0.01)
                    continue
                
                # Process active movements
                if self.current_movements and self.state.continuous:
                    # Calculate new positions
                    new_positions = {}
                    
                    for joint, direction in self.current_movements.items():
                        if joint in self.target_positions:
                            # Calculate delta
                            delta = self.state.step_size * direction * dt * 10
                            
                            # Apply to target
                            new_pos = self.target_positions[joint] + delta
                            
                            # Check limits
                            if self.state.soft_limits and joint in SERVO_LIMITS:
                                min_val, max_val = SERVO_LIMITS[joint]
                                new_pos = max(min_val, min(max_val, new_pos))
                            
                            new_positions[joint] = new_pos
                            self.target_positions[joint] = new_pos
                    
                    # Send movement command
                    if new_positions:
                        self.controller.move_joints(
                            new_positions,
                            speed=self.state.speed,
                            trajectory_type=TrajectoryType.LINEAR if self.state.smooth else TrajectoryType.DIRECT,
                            wait=False
                        )
                
                # Update actual positions
                status = self.controller.query_status()
                if status:
                    self.actual_positions = status['positions'].copy()
                
                time.sleep(0.01)  # 100 Hz update rate
                
            except Exception as e:
                logger.error(f"Movement loop error: {e}")
                time.sleep(0.1)
    
    def _start_movement(self, joint: str, direction: int):
        """Startet eine Bewegung."""
        if self.state.paused or self.state.emergency_active:
            return
        
        self.current_movements[joint] = direction
        
        # Immediate feedback
        if not self.state.continuous:
            # Single step mode
            delta = self.state.step_size * direction
            new_pos = self.target_positions[joint] + delta
            
            # Check limits
            if self.state.soft_limits and joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                new_pos = max(min_val, min(max_val, new_pos))
            
            self.target_positions[joint] = new_pos
            
            # Send command
            self.controller.move_joints(
                {joint: new_pos},
                speed=self.state.speed,
                wait=False
            )
            
            # Remove from active movements
            self.current_movements.pop(joint, None)
    
    def _continue_movement(self, joint: str, direction: int):
        """Setzt eine Bewegung fort (continuous mode)."""
        if self.state.continuous:
            self.current_movements[joint] = direction
    
    def _stop_movement(self, joint: str):
        """Stoppt eine Bewegung."""
        self.current_movements.pop(joint, None)
    
    def _stop_all_movements(self):
        """Stoppt alle Bewegungen."""
        self.current_movements.clear()
    
    def _increase_speed(self):
        """Erhöht die Geschwindigkeit."""
        self.state.speed = min(2.0, self.state.speed + 0.1)
        self._update_status(f"Speed: {self.state.speed:.1f}")
    
    def _decrease_speed(self):
        """Verringert die Geschwindigkeit."""
        self.state.speed = max(0.1, self.state.speed - 0.1)
        self._update_status(f"Speed: {self.state.speed:.1f}")
    
    def _increase_step(self):
        """Erhöht die Schrittweite."""
        self.state.step_size = min(0.2, self.state.step_size + 0.01)
        self._update_status(f"Step size: {self.state.step_size:.3f} rad")
    
    def _decrease_step(self):
        """Verringert die Schrittweite."""
        self.state.step_size = max(0.001, self.state.step_size - 0.01)
        self._update_status(f"Step size: {self.state.step_size:.3f} rad")
    
    def _cycle_mode(self):
        """Wechselt den Steuerungsmodus."""
        modes = list(ControlMode)
        current_idx = modes.index(self.state.mode)
        self.state.mode = modes[(current_idx + 1) % len(modes)]
        self._update_status(f"Mode: {self.state.mode.value}")
    
    def _toggle_continuous(self):
        """Wechselt zwischen Continuous und Single Step."""
        self.state.continuous = not self.state.continuous
        self._stop_all_movements()
        mode = "CONTINUOUS" if self.state.continuous else "SINGLE STEP"
        self._update_status(f"Movement: {mode}")
    
    def _toggle_pause(self):
        """Pausiert/Fortsetzt Bewegungen."""
        self.state.paused = not self.state.paused
        if self.state.paused:
            self._stop_all_movements()
            self._update_status("PAUSED")
        else:
            self._update_status("RESUMED")
    
    def _emergency_stop(self):
        """Führt Emergency Stop aus."""
        self.state.emergency_active = True
        self._stop_all_movements()
        self.controller.emergency_stop()
        self._update_status("🚨 EMERGENCY STOP!")
        
        # Auto-reset after 2 seconds
        threading.Timer(2.0, self._reset_emergency).start()
    
    def _reset_emergency(self):
        """Reset nach Emergency Stop."""
        self.state.emergency_active = False
        self.controller.reset_emergency()
        self._update_status("Emergency reset")
    
    def _go_home(self):
        """Fährt zur Home-Position."""
        self._stop_all_movements()
        self.controller.move_home(speed=self.state.speed)
        self.target_positions = HOME_POSITION.copy()
        self._update_status("Moving to HOME")
    
    def _goto_preset(self, preset_key: str):
        """Fährt zu einer vordefinierten Position."""
        if preset_key in self.presets:
            name, position = self.presets[preset_key]
            self._stop_all_movements()
            self.controller.move_joints(position, speed=self.state.speed)
            self.target_positions = position.copy()
            self._update_status(f"Moving to {name}")
    
    def _record_waypoint(self):
        """Zeichnet einen Waypoint auf (Teaching Mode)."""
        if self.teaching_recorder and self.state.recording:
            if self.teaching_recorder.record_waypoint(f"Manual control point"):
                self._update_status(f"Waypoint {self.teaching_recorder.waypoint_count} recorded")
    
    def _toggle_recording(self):
        """Startet/Stoppt Recording."""
        if not self.teaching_recorder:
            return
        
        if self.state.recording:
            # Stop recording
            seq = self.teaching_recorder.stop_recording()
            if seq:
                self._update_status(f"Recording stopped: {len(seq.waypoints)} points")
            self.state.recording = False
        else:
            # Start recording
            name = f"manual_{time.strftime('%H%M%S')}"
            if self.teaching_recorder.start_recording(name):
                self._update_status(f"Recording started: {name}")
                self.state.recording = True
    
    def _toggle_velocities(self):
        """Zeigt/Versteckt Geschwindigkeiten."""
        self.state.show_velocities = not self.state.show_velocities
    
    def _toggle_limits(self):
        """Zeigt/Versteckt Limits."""
        self.state.show_limits = not self.state.show_limits
    
    def _update_status(self, message: str):
        """Aktualisiert Status-Nachricht."""
        self.status_lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Keep only last 5 messages
        self.status_lines = self.status_lines[-5:]
    
    def _draw_interface(self):
        """Zeichnet das komplette Interface."""
        cols, rows = self.terminal.get_terminal_size()
        
        # Title
        self.terminal.print_at(1, 1, "="*min(cols-2, 80))
        self.terminal.print_at(1, 2, "🎮 ENHANCED MANUAL CONTROL".center(min(cols-2, 80)))
        self.terminal.print_at(1, 3, "="*min(cols-2, 80))
        
        # Mode info
        y = 5
        mode_text = f"Mode: {self.state.mode.value.upper()}"
        move_text = "CONTINUOUS" if self.state.continuous else "SINGLE STEP"
        self.terminal.print_at(1, y, f"{mode_text} | Movement: {move_text} | Speed: {self.state.speed:.1f}")
        
        # Controls help
        y = 7
        self.terminal.set_color('cyan')
        self.terminal.print_at(1, y, "CONTROLS:")
        self.terminal.set_color('reset')
        
        controls = [
            "Q/A: Base  W/S: Shoulder  E/D: Elbow",
            "R/F: Wrist  T/G: Roll  Y/H: Gripper",
            "",
            "+/-: Speed  [/]: Step size  M: Mode  C: Continuous",
            "1-5: Presets  0: Home  SPACE: Emergency  P: Pause",
            "TAB: Record  ENTER: Waypoint  V: Velocities  X: Exit"
        ]
        
        for i, line in enumerate(controls):
            self.terminal.print_at(3, y+i+1, line)
        
        # Position display area
        self.pos_display_y = y + len(controls) + 3
    
    def _update_display(self):
        """Aktualisiert die Anzeige."""
        current_time = time.time()
        
        # Throttle screen updates
        if current_time - self.last_screen_update < self.screen_update_rate:
            return
        
        self.last_screen_update = current_time
        
        # Update positions
        y = self.pos_display_y
        
        # Header
        self.terminal.set_color('green')
        self.terminal.print_at(1, y, "JOINT POSITIONS:")
        self.terminal.set_color('reset')
        
        # Joint positions
        for i, (joint, actual) in enumerate(self.actual_positions.items()):
            y_pos = y + i + 2
            
            # Joint name
            self.terminal.print_at(3, y_pos, f"{joint:10s}")
            
            # Actual position
            self.terminal.print_at(15, y_pos, f"{actual:+7.3f}")
            
            # Target position
            target = self.target_positions.get(joint, actual)
            if abs(target - actual) > 0.001:
                self.terminal.set_color('yellow')
            self.terminal.print_at(25, y_pos, f"→ {target:+7.3f}")
            self.terminal.set_color('reset')
            
            # Progress bar
            if joint in SERVO_LIMITS and self.state.show_limits:
                min_val, max_val = SERVO_LIMITS[joint]
                range_val = max_val - min_val
                if range_val > 0:
                    progress = (actual - min_val) / range_val
                    bar = self.terminal.create_progress_bar(progress, 1.0, 20)
                    self.terminal.print_at(40, y_pos, bar)
                    
                    # Limits
                    self.terminal.print_at(62, y_pos, f"[{min_val:+.2f}, {max_val:+.2f}]")
            
            # Movement indicator
            if joint in self.current_movements:
                direction = self.current_movements[joint]
                arrow = "↑" if direction > 0 else "↓"
                self.terminal.set_color('cyan')
                self.terminal.print_at(80, y_pos, arrow)
                self.terminal.set_color('reset')
        
        # Status messages
        status_y = y + len(self.actual_positions) + 4
        self.terminal.set_color('yellow')
        self.terminal.print_at(1, status_y, "STATUS:")
        self.terminal.set_color('reset')
        
        for i, msg in enumerate(self.status_lines[-3:]):
            self.terminal.print_at(3, status_y + i + 1, msg + " " * 50)
        
        # Recording indicator
        if self.state.recording:
            self.terminal.set_color('red')
            self.terminal.print_at(1, rows-2, "● RECORDING")
            self.terminal.set_color('reset')
        
        # Pause indicator
        if self.state.paused:
            self.terminal.set_color('yellow')
            self.terminal.print_at(20, rows-2, "║║ PAUSED")
            self.terminal.set_color('reset')
        
        # Emergency indicator
        if self.state.emergency_active:
            self.terminal.set_color('red')
            self.terminal.print_at(35, rows-2, "🚨 EMERGENCY")
            self.terminal.set_color('reset')


def create_manual_control(controller, teaching_recorder=None):
    """
    Factory-Funktion für Manual Control.
    
    Args:
        controller: RoArm Controller
        teaching_recorder: Optional Teaching Recorder
        
    Returns:
        EnhancedManualControl Instanz
    """
    return EnhancedManualControl(controller, teaching_recorder)
