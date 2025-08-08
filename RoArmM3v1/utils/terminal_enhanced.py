#!/usr/bin/env python3
"""
Enhanced Terminal Controller für RoArm M3
Ermöglicht Echtzeit-Tastatureingabe ohne Enter-Taste
Optimiert für macOS mit erweiterten Features
"""

import sys
import tty
import termios
import select
import threading
import time
from typing import Optional, Dict, Callable, Tuple
from enum import Enum
import os

class KeyMode(Enum):
    """Tastatur-Modi."""
    SINGLE = "single"          # Einzelne Tastenanschläge
    CONTINUOUS = "continuous"  # Gedrückt halten
    COMBO = "combo"           # Tastenkombinationen


class EnhancedTerminalController:
    """
    Erweiterte Terminal-Steuerung mit Echtzeit-Input.
    Unterstützt gedrückt gehaltene Tasten und Kombinationen.
    """
    
    def __init__(self):
        """Initialisiert Terminal Controller."""
        self.old_settings = None
        self.running = False
        self.key_thread = None
        self.current_keys = set()  # Aktuell gedrückte Tasten
        self.key_callbacks = {}    # Callbacks für Tasten
        self.continuous_callbacks = {}  # Callbacks für gehaltene Tasten
        self.combo_callbacks = {}  # Callbacks für Kombinationen
        
        # Key repeat settings
        self.repeat_delay = 0.5   # Verzögerung vor Wiederholung (s)
        self.repeat_rate = 0.05   # Wiederholungsrate (s)
        
        # Key states
        self.key_states = {}      # Zeitstempel wann Taste gedrückt wurde
        self.key_repeating = {}   # Ob Taste wiederholt wird
        
        # Special keys mapping
        self.special_keys = {
            '\x1b[A': 'UP',
            '\x1b[B': 'DOWN',
            '\x1b[C': 'RIGHT',
            '\x1b[D': 'LEFT',
            '\x1b[H': 'HOME',
            '\x1b[F': 'END',
            '\x1b[5~': 'PGUP',
            '\x1b[6~': 'PGDN',
            '\x7f': 'BACKSPACE',
            '\r': 'ENTER',
            '\t': 'TAB',
            '\x1b': 'ESC',
            ' ': 'SPACE'
        }
        
    def start(self):
        """Startet den Terminal Controller."""
        if not self.running:
            self.running = True
            self._setup_terminal()
            self.key_thread = threading.Thread(target=self._key_listener, daemon=True)
            self.key_thread.start()
    
    def stop(self):
        """Stoppt den Terminal Controller."""
        self.running = False
        if self.key_thread:
            self.key_thread.join(timeout=1)
        self._restore_terminal()
    
    def _setup_terminal(self):
        """Konfiguriert Terminal für Raw Input."""
        if sys.platform == 'darwin':  # macOS
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            # Non-blocking mode
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
    
    def _restore_terminal(self):
        """Stellt Terminal-Einstellungen wieder her."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def _key_listener(self):
        """Thread-Funktion für Tastatur-Listener."""
        while self.running:
            try:
                # Check if input available
                if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                    key = self._read_key()
                    if key:
                        self._process_key(key)
                
                # Process continuous keys
                self._process_continuous_keys()
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
            except Exception as e:
                # Ignore errors in thread
                pass
    
    def _read_key(self) -> Optional[str]:
        """Liest eine Taste vom Terminal."""
        try:
            key = sys.stdin.read(1)
            
            # Check for escape sequences (arrow keys, etc.)
            if key == '\x1b':
                # Read additional bytes for escape sequence
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key += sys.stdin.read(2)
                    # Might be longer sequence
                    if key[-1] == '~':
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            key += sys.stdin.read(1)
            
            return key
            
        except:
            return None
    
    def _process_key(self, key: str):
        """Verarbeitet eine gedrückte Taste."""
        # Map special keys
        if key in self.special_keys:
            key_name = self.special_keys[key]
        else:
            key_name = key
        
        # Track key state
        if key_name not in self.key_states:
            # Key pressed
            self.key_states[key_name] = time.time()
            self.current_keys.add(key_name)
            
            # Check for combinations
            self._check_combos()
            
            # Execute single press callback
            if key_name in self.key_callbacks:
                self.key_callbacks[key_name]()
        
        # Reset repeat state
        self.key_repeating[key_name] = False
    
    def _process_continuous_keys(self):
        """Verarbeitet gedrückt gehaltene Tasten."""
        current_time = time.time()
        keys_to_remove = []
        
        for key_name, press_time in self.key_states.items():
            elapsed = current_time - press_time
            
            # Check if key should start repeating
            if elapsed > self.repeat_delay and not self.key_repeating.get(key_name, False):
                self.key_repeating[key_name] = True
                self.key_states[key_name] = current_time  # Reset for repeat rate
            
            # Execute continuous callback if repeating
            if self.key_repeating.get(key_name, False):
                if current_time - self.key_states[key_name] > self.repeat_rate:
                    if key_name in self.continuous_callbacks:
                        self.continuous_callbacks[key_name]()
                    self.key_states[key_name] = current_time
            
            # Remove old keys (key release detection)
            if elapsed > 0.5 and not self.key_repeating.get(key_name, False):
                keys_to_remove.append(key_name)
        
        # Clean up released keys
        for key in keys_to_remove:
            self.key_states.pop(key, None)
            self.key_repeating.pop(key, None)
            self.current_keys.discard(key)
    
    def _check_combos(self):
        """Prüft auf Tastenkombinationen."""
        for combo, callback in self.combo_callbacks.items():
            combo_keys = set(combo.split('+'))
            if combo_keys.issubset(self.current_keys):
                callback()
    
    def register_key(self, key: str, callback: Callable, mode: KeyMode = KeyMode.SINGLE):
        """
        Registriert einen Callback für eine Taste.
        
        Args:
            key: Taste oder Kombination (z.B. 'a' oder 'ctrl+c')
            callback: Funktion die aufgerufen wird
            mode: Tastatur-Modus
        """
        if mode == KeyMode.SINGLE:
            self.key_callbacks[key] = callback
        elif mode == KeyMode.CONTINUOUS:
            self.continuous_callbacks[key] = callback
        elif mode == KeyMode.COMBO:
            self.combo_callbacks[key] = callback
    
    def unregister_key(self, key: str, mode: KeyMode = KeyMode.SINGLE):
        """Entfernt einen Key-Callback."""
        if mode == KeyMode.SINGLE:
            self.key_callbacks.pop(key, None)
        elif mode == KeyMode.CONTINUOUS:
            self.continuous_callbacks.pop(key, None)
        elif mode == KeyMode.COMBO:
            self.combo_callbacks.pop(key, None)
    
    def get_key(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wartet auf eine Taste (Kompatibilität).
        
        Args:
            timeout: Timeout in Sekunden
            
        Returns:
            Gedrückte Taste oder None
        """
        start_time = time.time()
        
        while self.running:
            if self.current_keys:
                key = next(iter(self.current_keys))
                self.current_keys.clear()
                return key
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)
        
        return None
    
    def is_key_pressed(self, key: str) -> bool:
        """Prüft ob eine Taste aktuell gedrückt ist."""
        return key in self.current_keys
    
    def clear_keys(self):
        """Löscht alle aktuellen Tasten."""
        self.current_keys.clear()
        self.key_states.clear()
        self.key_repeating.clear()
    
    def set_repeat_rate(self, delay: float, rate: float):
        """
        Setzt die Wiederholungsrate für gehaltene Tasten.
        
        Args:
            delay: Verzögerung vor erster Wiederholung (s)
            rate: Wiederholungsrate (s)
        """
        self.repeat_delay = delay
        self.repeat_rate = rate
    
    def print_at(self, x: int, y: int, text: str):
        """
        Druckt Text an bestimmter Position (ANSI).
        
        Args:
            x: X-Position (Spalte)
            y: Y-Position (Zeile)
            text: Zu druckender Text
        """
        print(f"\033[{y};{x}H{text}", end='', flush=True)
    
    def clear_screen(self):
        """Löscht den Bildschirm."""
        print("\033[2J\033[H", end='', flush=True)
    
    def hide_cursor(self):
        """Versteckt den Cursor."""
        print("\033[?25l", end='', flush=True)
    
    def show_cursor(self):
        """Zeigt den Cursor."""
        print("\033[?25h", end='', flush=True)
    
    def set_color(self, color: str):
        """
        Setzt die Textfarbe (ANSI).
        
        Args:
            color: Farbname oder ANSI-Code
        """
        colors = {
            'black': '30',
            'red': '31',
            'green': '32',
            'yellow': '33',
            'blue': '34',
            'magenta': '35',
            'cyan': '36',
            'white': '37',
            'reset': '0'
        }
        
        if color in colors:
            print(f"\033[{colors[color]}m", end='', flush=True)
    
    def get_terminal_size(self) -> Tuple[int, int]:
        """
        Gibt die Terminal-Größe zurück.
        
        Returns:
            (Breite, Höhe) in Zeichen
        """
        try:
            import shutil
            cols, rows = shutil.get_terminal_size()
            return cols, rows
        except:
            return 80, 24  # Default
    
    def create_progress_bar(self, current: float, total: float, width: int = 50) -> str:
        """
        Erstellt einen Fortschrittsbalken.
        
        Args:
            current: Aktueller Wert
            total: Maximaler Wert
            width: Breite des Balkens in Zeichen
            
        Returns:
            Fortschrittsbalken als String
        """
        if total == 0:
            percent = 0
        else:
            percent = current / total
        
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        
        return f"[{bar}] {percent*100:.1f}%"
    
    def __enter__(self):
        """Context Manager Entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit."""
        self.stop()


# Kompatibilität mit altem TerminalController
class TerminalController(EnhancedTerminalController):
    """Alias für Kompatibilität."""
    pass


# Beispiel-Verwendung für Manual Control
class ManualControlHelper:
    """
    Helper-Klasse für intuitive manuelle Steuerung.
    """
    
    def __init__(self, controller):
        """
        Initialisiert Manual Control Helper.
        
        Args:
            controller: RoArm Controller Instanz
        """
        self.controller = controller
        self.terminal = EnhancedTerminalController()
        self.speed = 1.0
        self.step_size = 0.05
        self.continuous_mode = True
        self.movement_active = {}
        
    def start_manual_control(self):
        """Startet die manuelle Steuerung."""
        self.terminal.start()
        self._setup_controls()
        
        print("\n🎮 ENHANCED MANUAL CONTROL")
        print("="*50)
        print("Movement Keys (hold for continuous):")
        print("  Q/A: Base left/right")
        print("  W/S: Shoulder up/down")
        print("  E/D: Elbow up/down")
        print("  R/F: Wrist up/down")
        print("  T/G: Roll left/right")
        print("  Y/H: Gripper open/close")
        print("\nControl Keys:")
        print("  +/-: Speed up/down")
        print("  SPACE: Emergency stop")
        print("  C: Toggle continuous mode")
        print("  X: Exit")
        print("="*50)
        print(f"\nMode: {'CONTINUOUS' if self.continuous_mode else 'SINGLE'}")
        print(f"Speed: {self.speed:.1f}")
        
        # Main loop
        while True:
            time.sleep(0.1)
            
            # Check for exit
            if self.terminal.is_key_pressed('x') or self.terminal.is_key_pressed('X'):
                break
        
        self.terminal.stop()
    
    def _setup_controls(self):
        """Konfiguriert die Steuerungs-Callbacks."""
        # Movement controls
        movements = {
            'q': ('base', -1),
            'Q': ('base', -1),
            'a': ('base', 1),
            'A': ('base', 1),
            'w': ('shoulder', 1),
            'W': ('shoulder', 1),
            's': ('shoulder', -1),
            'S': ('shoulder', -1),
            'e': ('elbow', 1),
            'E': ('elbow', 1),
            'd': ('elbow', -1),
            'D': ('elbow', -1),
            'r': ('wrist', 1),
            'R': ('wrist', 1),
            'f': ('wrist', -1),
            'F': ('wrist', -1),
            't': ('roll', 1),
            'T': ('roll', 1),
            'g': ('roll', -1),
            'G': ('roll', -1),
            'y': ('hand', -1),
            'Y': ('hand', -1),
            'h': ('hand', 1),
            'H': ('hand', 1)
        }
        
        # Register movement callbacks
        for key, (joint, direction) in movements.items():
            if self.continuous_mode:
                self.terminal.register_key(
                    key,
                    lambda j=joint, d=direction: self._move_joint(j, d),
                    KeyMode.CONTINUOUS
                )
            else:
                self.terminal.register_key(
                    key,
                    lambda j=joint, d=direction: self._move_joint(j, d),
                    KeyMode.SINGLE
                )
        
        # Control keys
        self.terminal.register_key('+', self._increase_speed, KeyMode.SINGLE)
        self.terminal.register_key('-', self._decrease_speed, KeyMode.SINGLE)
        self.terminal.register_key('SPACE', self._emergency_stop, KeyMode.SINGLE)
        self.terminal.register_key('c', self._toggle_mode, KeyMode.SINGLE)
        self.terminal.register_key('C', self._toggle_mode, KeyMode.SINGLE)
    
    def _move_joint(self, joint: str, direction: int):
        """Bewegt ein Gelenk."""
        delta = self.step_size * direction
        current = self.controller.current_position[joint]
        new_pos = {joint: current + delta}
        
        self.controller.move_joints(
            new_pos,
            speed=self.speed,
            wait=False
        )
        
        # Visual feedback
        print(f"\r{joint:10s}: {current + delta:+.3f} rad", end='', flush=True)
    
    def _increase_speed(self):
        """Erhöht die Geschwindigkeit."""
        self.speed = min(2.0, self.speed + 0.1)
        print(f"\nSpeed: {self.speed:.1f}")
    
    def _decrease_speed(self):
        """Verringert die Geschwindigkeit."""
        self.speed = max(0.1, self.speed - 0.1)
        print(f"\nSpeed: {self.speed:.1f}")
    
    def _emergency_stop(self):
        """Führt Emergency Stop aus."""
        self.controller.emergency_stop()
        print("\n🚨 EMERGENCY STOP!")
    
    def _toggle_mode(self):
        """Wechselt zwischen Single und Continuous Mode."""
        self.continuous_mode = not self.continuous_mode
        print(f"\nMode: {'CONTINUOUS' if self.continuous_mode else 'SINGLE'}")
        self._setup_controls()  # Re-register with new mode
