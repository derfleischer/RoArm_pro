#!/usr/bin/env python3
"""
RoArm M3 Professional Controller
Hauptcontroller für Waveshare RoArm M3 - macOS optimiert
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
import queue
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager

# Interne Imports aus dem gleichen Package
from core.constants import (
    SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION,
    COMMANDS, DEFAULT_SPEED
)
from core.serial_comm import SerialManager

# Imports aus anderen Packages
from motion.trajectory import TrajectoryGenerator, TrajectoryType
from utils.logger import get_logger
from utils.safety import SafetyMonitor

logger = get_logger(__name__)


@dataclass
class RoArmConfig:
    """Konfiguration für RoArm Controller."""
    port: str = "/dev/tty.usbserial-110"
    baudrate: int = 115200
    timeout: float = 2.0
    default_speed: float = 1.0
    scanner_weight: float = 0.2  # kg
    enable_weight_compensation: bool = True
    auto_connect: bool = True
    debug: bool = False


class RoArmController:
    """
    Hauptcontroller für RoArm M3.
    Thread-safe mit Queue-basierter Kommandoverarbeitung.
    """
    
    def __init__(self, config: Optional[RoArmConfig] = None):
        """
        Initialisiert den RoArm Controller.
        
        Args:
            config: Konfigurationsobjekt oder None für Defaults
        """
        self.config = config or RoArmConfig()
        
        # Serial Manager
        self.serial = SerialManager(
            port=self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout
        )
        
        # Trajectory Generator
        self.trajectory = TrajectoryGenerator()
        
        # Safety Monitor
        self.safety = SafetyMonitor(SERVO_LIMITS)
        
        # Safety System (wird nach Verbindung initialisiert)
        self.safety_system = None
        
        # Command Queue
        self.command_queue = queue.Queue()
        self.queue_thread = None
        self.running = False
        
        # Current State
        self.current_position = HOME_POSITION.copy()
        self.current_speed = self.config.default_speed
        self.torque_enabled = True
        self.emergency_stop_flag = False
        
        # Scanner compensation
        self.scanner_mounted = False
        self.scanner_weight = self.config.scanner_weight
        
        # Thread Lock
        self._lock = threading.Lock()
        
        # Auto-connect
        if self.config.auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """
        Verbindet mit dem RoArm.
        
        Returns:
            True wenn erfolgreich verbunden
        """
        try:
            # Auto-detect port für macOS
            if "auto" in self.config.port.lower():
                self.config.port = self._auto_detect_port()
                logger.info(f"Auto-detected port: {self.config.port}")
            
            # Verbinden
            if self.serial.connect():
                logger.info(f"✅ Connected to RoArm on {self.config.port}")
                
                # Start command queue processor
                self._start_queue_processor()
                
                # Initial setup
                self._initialize_robot()
                
                return True
            else:
                logger.error("Failed to connect to RoArm")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Trennt die Verbindung zum RoArm."""
        try:
            # Stop queue processor
            self._stop_queue_processor()
            
            # Safe shutdown
            self.move_home(speed=0.5)
            time.sleep(1)
            
            # Disconnect serial
            self.serial.disconnect()
            logger.info("✅ Disconnected from RoArm")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def _auto_detect_port(self) -> str:
        """
        Auto-detect USB serial port für macOS.
        
        Returns:
            Gefundener Port oder Default
        """
        try:
            ports = list(serial.tools.list_ports.comports())
            
            # Suche nach USB Serial
            for port in ports:
                if 'usbserial' in port.device.lower():
                    return port.device
                elif 'cu.' in port.device:
                    return port.device
            
            # Fallback
            logger.warning("No USB serial found, using default")
            return "/dev/tty.usbserial-110"
            
        except Exception as e:
            logger.error(f"Port detection error: {e}")
            return "/dev/tty.usbserial-110"
    
    def _initialize_robot(self):
        """Initialisiert den Roboter nach Verbindung."""
        try:
            # LED blink zur Bestätigung
            self.led_control(True, brightness=128)
            time.sleep(0.5)
            self.led_control(False)
            
            # Query current position
            self.query_status()
            
            # Enable torque
            self.set_torque(True)
            
            logger.info("Robot initialized")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
    
    def _start_queue_processor(self):
        """Startet den Command Queue Processor Thread."""
        if not self.queue_thread:
            self.running = True
            self.queue_thread = threading.Thread(
                target=self._process_command_queue,
                daemon=True
            )
            self.queue_thread.start()
            logger.debug("Command queue processor started")
    
    def _stop_queue_processor(self):
        """Stoppt den Command Queue Processor Thread."""
        self.running = False
        if self.queue_thread:
            self.queue_thread.join(timeout=2)
            self.queue_thread = None
            logger.debug("Command queue processor stopped")
    
    def _process_command_queue(self):
        """Verarbeitet Commands aus der Queue (Thread)."""
        while self.running:
            try:
                # Get command with timeout
                command = self.command_queue.get(timeout=0.1)
                
                # Send command
                self.serial.send_command(command)
                
                # Small delay between commands
                time.sleep(0.01)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    # ============== BASIC MOVEMENTS ==============
    
    def move_joints(self, positions: Dict[str, float], 
                   speed: Optional[float] = None,
                   trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                   wait: bool = True) -> bool:
        """
        Bewegt Joints zu Zielpositionen.
        
        Args:
            positions: Dict mit Joint-Namen und Ziel-Positionen (rad)
            speed: Geschwindigkeit (0.1-2.0) oder None für default
            trajectory_type: Bewegungsprofil
            wait: Warte auf Bewegungsende
            
        Returns:
            True wenn erfolgreich
        """
        try:
            # Safety check
            if self.emergency_stop_flag:
                logger.warning("Movement blocked - emergency stop active")
                return False
            
            # Validate positions
            if not self.safety.validate_positions(positions):
                logger.error("Invalid positions - outside limits")
                return False
            
            # Speed
            speed = speed or self.current_speed
            
            # Weight compensation wenn Scanner montiert
            if self.scanner_mounted and self.config.enable_weight_compensation:
                positions = self._apply_weight_compensation(positions)
            
            # Generate trajectory
            trajectory_points = self.trajectory.generate(
                start=self.current_position,
                end=positions,
                speed=speed,
                trajectory_type=trajectory_type
            )
            
            # Execute trajectory
            for point in trajectory_points:
                command = self._create_joint_command(point.positions)
                self.command_queue.put(command)
                
                if wait:
                    time.sleep(point.time_delta)
            
            # Update current position
            self.current_position.update(positions)
            
            return True
            
        except Exception as e:
            logger.error(f"Movement error: {e}")
            return False
    
    def move_home(self, speed: float = 1.0) -> bool:
        """
        Bewegt den Roboter zur Home-Position.
        
        Args:
            speed: Bewegungsgeschwindigkeit
            
        Returns:
            True wenn erfolgreich
        """
        logger.info("Moving to home position...")
        return self.move_joints(
            HOME_POSITION,
            speed=speed,
            trajectory_type=TrajectoryType.S_CURVE
        )
    
    def move_to_scanner_position(self, speed: float = 0.5) -> bool:
        """
        Bewegt zur optimalen Scanner-Position.
        
        Args:
            speed: Bewegungsgeschwindigkeit
            
        Returns:
            True wenn erfolgreich
        """
        logger.info("Moving to scanner position...")
        self.scanner_mounted = True
        return self.move_joints(
            SCANNER_POSITION,
            speed=speed,
            trajectory_type=TrajectoryType.S_CURVE
        )
    
    # ============== GRIPPER CONTROL ==============
    
    def gripper_control(self, position: float) -> bool:
        """
        Steuert den Greifer.
        
        Args:
            position: 0.0 (offen) bis 1.0 (geschlossen)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            # Map to servo range
            min_pos = SERVO_LIMITS["hand"][0]  # 1.08 rad (open)
            max_pos = SERVO_LIMITS["hand"][1]  # 3.14 rad (closed)
            
            servo_pos = min_pos + (max_pos - min_pos) * position
            
            return self.move_joints({"hand": servo_pos}, speed=1.0)
            
        except Exception as e:
            logger.error(f"Gripper control error: {e}")
            return False
    
    # ============== SERVO CONTROL ==============
    
    def set_torque(self, enabled: bool) -> bool:
        """
        Aktiviert/Deaktiviert Servo-Torque.
        
        Args:
            enabled: True zum Aktivieren
            
        Returns:
            True wenn erfolgreich
        """
        try:
            command = {
                "T": COMMANDS["TORQUE_CONTROL"],
                "enabled": 1 if enabled else 0
            }
            
            self.serial.send_command(command)
            self.torque_enabled = enabled
            
            logger.info(f"Torque {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            logger.error(f"Torque control error: {e}")
            return False
    
    def led_control(self, on: bool, brightness: int = 255) -> bool:
        """
        Steuert die LED.
        
        Args:
            on: True für an
            brightness: 0-255
            
        Returns:
            True wenn erfolgreich
        """
        try:
            command = {
                "T": COMMANDS["LED_CONTROL"],
                "led": 1 if on else 0,
                "brightness": max(0, min(255, brightness))
            }
            
            self.serial.send_command(command)
            return True
            
        except Exception as e:
            logger.error(f"LED control error: {e}")
            return False
    
    # ============== STATUS & SAFETY ==============
    
    def emergency_stop(self):
        """Führt einen Notstopp aus."""
        logger.warning("🚨 EMERGENCY STOP")
        
        with self._lock:
            self.emergency_stop_flag = True
            
            # Clear command queue
            while not self.command_queue.empty():
                self.command_queue.get()
            
            # Send emergency stop command
            command = {"T": COMMANDS["EMERGENCY_STOP"]}
            self.serial.send_command(command)
            
            # Disable torque
            self.set_torque(False)
    
    def reset_emergency(self):
        """Setzt den Notstopp zurück."""
        with self._lock:
            self.emergency_stop_flag = False
            logger.info("Emergency stop reset")
    
    def query_status(self) -> Optional[Dict]:
        """
        Fragt den aktuellen Status ab.
        
        Returns:
            Status-Dictionary oder None
        """
        try:
            command = {"T": COMMANDS["STATUS_QUERY"]}
            response = self.serial.send_command(command, wait_response=True)
            
            if response:
                # Parse response
                return self._parse_status(response)
            
            return None
            
        except Exception as e:
            logger.error(f"Status query error: {e}")
            return None
    
    # ============== HELPER METHODS ==============
    
    def _create_joint_command(self, positions: Dict[str, float]) -> Dict:
        """Erstellt ein Joint-Control Command."""
        # Convert positions to servo values
        servo_values = {}
        for joint, pos in positions.items():
            if joint in SERVO_LIMITS:
                # Map to servo range (implementation specific)
                servo_values[joint] = pos
        
        return {
            "T": COMMANDS["JOINT_CONTROL"],
            "positions": servo_values
        }
    
    def _apply_weight_compensation(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Kompensiert das Gewicht des Scanners.
        
        Args:
            positions: Original-Positionen
            
        Returns:
            Kompensierte Positionen
        """
        compensated = positions.copy()
        
        # Simple compensation - adjust shoulder and elbow
        if "shoulder" in compensated:
            # Add slight upward compensation
            compensated["shoulder"] += 0.05 * self.scanner_weight
        
        if "elbow" in compensated:
            # Adjust elbow for balance
            compensated["elbow"] -= 0.03 * self.scanner_weight
        
        return compensated
    
    def _parse_status(self, response: str) -> Optional[Dict]:
        """Parst die Status-Antwort."""
        try:
            # Parse JSON response
            data = json.loads(response)
            
            status = {
                "positions": data.get("positions", {}),
                "velocities": data.get("velocities", {}),
                "torque_enabled": data.get("torque", False),
                "temperature": data.get("temperature", 0),
                "voltage": data.get("voltage", 0)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status parse error: {e}")
            return None
    
    # ============== CONTEXT MANAGER ==============
    
    def __enter__(self):
        """Context manager entry."""
        if not self.serial.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    # ============== PATTERN EXECUTION ==============
    
    def execute_pattern(self, pattern) -> bool:
        """
        Führt ein Bewegungsmuster aus.
        
        Args:
            pattern: Pattern-Objekt mit generate_points() Methode
            
        Returns:
            True wenn erfolgreich
        """
        try:
            # Generate pattern points
            points = pattern.generate_points()
            
            logger.info(f"Executing pattern: {pattern.name} ({len(points)} points)")
            
            # Execute each point
            for i, point in enumerate(points):
                if self.emergency_stop_flag:
                    logger.warning("Pattern aborted - emergency stop")
                    return False
                
                # Move to point
                success = self.move_joints(
                    point.positions,
                    speed=point.speed,
                    trajectory_type=point.trajectory_type
                )
                
                if not success:
                    logger.error(f"Failed at point {i+1}/{len(points)}")
                    return False
                
                # Settle time (wichtig für Scanner)
                if hasattr(point, 'settle_time'):
                    time.sleep(point.settle_time)
                
                # Progress
                if i % 10 == 0:
                    progress = (i + 1) / len(points) * 100
                    logger.info(f"Pattern progress: {progress:.1f}%")
            
            logger.info("✅ Pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pattern execution error: {e}")
            return False
