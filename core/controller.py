#!/usr/bin/env python3
"""
RoArm M3 Professional Controller
Based on WORKING communication from RoArm3v7.py
Compatible with existing architecture
"""

import time
import threading
import queue
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Use absolute imports to avoid issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.serial_comm import SerialManager
from core.constants import (
    SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION,
    COMMANDS, DEFAULT_SPEED
)
from motion.trajectory import TrajectoryGenerator, TrajectoryType
from utils.logger import get_logger
from utils.safety import SafetyMonitor

logger = get_logger(__name__)


@dataclass
class RoArmConfig:
    """Configuration for RoArm Controller."""
    port: str = "/dev/tty.usbserial-110"
    baudrate: int = 115200
    timeout: float = 2.0
    default_speed: float = 1.0
    scanner_weight: float = 0.2
    enable_weight_compensation: bool = True
    auto_connect: bool = True
    debug: bool = False


class RoArmController:
    """
    RoArm M3 Controller with WORKING communication from RoArm3v7.py
    """
    
    # Command IDs from RoArm3v7.py that WORK
    COMMANDS = {
        "EMERGENCY_STOP": {"T": 0},
        "JOINT_CONTROL": {"T": 102},
        "STATUS_QUERY": {"T": 1},
        "POSITION_QUERY": {"T": 2},
        "LED_CONTROL": {"T": 51}  # Just T: 51, brightness is added separately
    }
    
    def __init__(self, config: Optional[RoArmConfig] = None):
        """Initialize the RoArm Controller."""
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
        """Connect to the RoArm."""
        try:
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
        """Disconnect from the RoArm."""
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
    
    def _initialize_robot(self):
        """Initialize the robot after connection."""
        try:
            # LED blink for confirmation
            self.led_on(128)
            time.sleep(0.5)
            self.led_off()
            
            # Query current position
            self.query_status()
            
            # Enable torque
            self.set_torque(True)
            
            logger.info("Robot initialized")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
    
    def _start_queue_processor(self):
        """Start the command queue processor thread."""
        if not self.queue_thread:
            self.running = True
            self.queue_thread = threading.Thread(
                target=self._process_command_queue,
                daemon=True
            )
            self.queue_thread.start()
            logger.debug("Command queue processor started")
    
    def _stop_queue_processor(self):
        """Stop the command queue processor thread."""
        self.running = False
        if self.queue_thread:
            self.queue_thread.join(timeout=2)
            self.queue_thread = None
            logger.debug("Command queue processor stopped")
    
    def _process_command_queue(self):
        """Process commands from the queue."""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                self.serial.send_command(command)
                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    # ============== LED CONTROL (EXACT from RoArm3v7.py) ==============
    
    def led_on(self, brightness: int = 255) -> bool:
        """Turn on LED with specified brightness - EXACT from RoArm3v7.py"""
        brightness = max(0, min(255, brightness))
        command = {**self.COMMANDS["LED_CONTROL"], "brightness": brightness}
        
        response = self.serial.send_command(command)
        return response is not None
    
    def led_off(self) -> bool:
        """Turn off LED - EXACT from RoArm3v7.py"""
        return self.led_on(0)
    
    def led_control(self, on: bool, brightness: int = 255) -> bool:
        """LED control wrapper for compatibility"""
        if on:
            return self.led_on(brightness)
        else:
            return self.led_off()
    
    def led_blink_sequence(self, count: int = 3, brightness: int = 200, interval: float = 0.5):
        """Blink LED sequence - from RoArm3v7.py"""
        for i in range(count):
            self.led_on(brightness)
            time.sleep(interval)
            self.led_off()
            if i < count - 1:
                time.sleep(interval)
    
    # ============== MOVEMENT COMMANDS ==============
    
    def move_joints(self, positions: Dict[str, float], 
                   speed: Optional[float] = None,
                   trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                   wait: bool = True) -> bool:
        """Move joints to target positions with trajectory planning."""
        try:
            # Safety check
            if self.emergency_stop_flag:
                logger.warning("Movement blocked - emergency stop active")
                return False
            
            # Validate positions
            if not self.safety.validate_positions(positions):
                logger.error("Invalid positions - outside limits")
                return False
            
            # Build complete target position (wichtig!)
            # Kalibrierung übergibt nur einzelne Joints
            target_positions = self.current_position.copy()
            target_positions.update(positions)
            
            # Speed
            speed = speed or self.current_speed
            
            # Weight compensation if scanner mounted
            if self.scanner_mounted and self.config.enable_weight_compensation:
                target_positions = self._apply_weight_compensation(target_positions)
            
            # Generate trajectory
            trajectory_points = self.trajectory.generate(
                start=self.current_position,
                end=target_positions,
                speed=speed,
                trajectory_type=trajectory_type,
                num_points=20  # Angemessene Anzahl Punkte
            )
            
            if not trajectory_points:
                logger.error("No trajectory points generated")
                return False
            
            logger.debug(f"Generated {len(trajectory_points)} trajectory points")
            
            # Execute trajectory
            for i, point in enumerate(trajectory_points):
                # Create joint command
                command = {
                    "T": 102,  # Joint control command
                    **point.positions  # Alle Joint-Positionen
                }
                
                # Send command directly (nicht über Queue für Kalibrierung)
                response = self.serial.send_command(command, wait_time=0.02)
                
                if not response:
                    logger.warning(f"No response at trajectory point {i}")
                
                # Wait for timing
                if wait and point.time_delta > 0:
                    time.sleep(point.time_delta)
            
            # Update current position
            self.current_position.update(target_positions)
            
            logger.debug(f"Movement complete, new position: {self.current_position}")
            
            return True
            
        except Exception as e:
            logger.error(f"Movement error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def move_home(self, speed: float = 1.0) -> bool:
        """Move the robot to home position."""
        logger.info("Moving to home position...")
        return self.move_joints(HOME_POSITION, speed=speed)
    
    def move_to_scanner_position(self, speed: float = 0.5) -> bool:
        """Move to optimal scanner position."""
        logger.info("Moving to scanner position...")
        self.scanner_mounted = True
        return self.move_joints(SCANNER_POSITION, speed=speed)
    
    # ============== CONTROL COMMANDS ==============
    
    def gripper_control(self, position: float) -> bool:
        """Control the gripper (0.0=open, 1.0=closed)."""
        try:
            min_pos = SERVO_LIMITS["hand"][0]
            max_pos = SERVO_LIMITS["hand"][1]
            servo_pos = min_pos + (max_pos - min_pos) * position
            
            return self.move_joints({"hand": servo_pos}, speed=1.0)
            
        except Exception as e:
            logger.error(f"Gripper control error: {e}")
            return False
    
    def set_torque(self, enabled: bool) -> bool:
        """Enable/disable servo torque."""
        try:
            command = {
                "T": 210,
                "enabled": 1 if enabled else 0
            }
            
            response = self.serial.send_command(command)
            if response:
                self.torque_enabled = enabled
                logger.info(f"Torque {'enabled' if enabled else 'disabled'}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Torque control error: {e}")
            return False
    
    def emergency_stop(self):
        """Execute emergency stop."""
        logger.warning("🚨 EMERGENCY STOP")
        
        with self._lock:
            self.emergency_stop_flag = True
            
            # Clear command queue
            while not self.command_queue.empty():
                self.command_queue.get()
            
            # Send emergency stop command
            command = {"T": 0}
            self.serial.send_command(command)
            
            # Disable torque
            self.set_torque(False)
    
    def reset_emergency(self):
        """Reset emergency stop."""
        with self._lock:
            self.emergency_stop_flag = False
            logger.info("Emergency stop reset")
    
    def query_status(self) -> Optional[Dict]:
        """Query current status."""
        try:
            command = {"T": 1}
            response = self.serial.send_command(command, wait_time=0.1)
            
            if response:
                return self._parse_status(response)
            
            return None
            
        except Exception as e:
            logger.error(f"Status query error: {e}")
            return None
    
    def _parse_status(self, response) -> Dict:
        """Parse status response."""
        status = {
            "positions": self.current_position.copy(),
            "torque_enabled": self.torque_enabled,
            "temperature": response.get("temperature", 0),
            "voltage": response.get("voltage", 0)
        }
        
        if "positions" in response:
            status["positions"] = response["positions"]
        
        return status
    
    def _create_joint_command(self, positions: Dict[str, float]) -> Dict:
        """Create joint control command."""
        return {
            "T": 102,
            **positions
        }
    
    def _apply_weight_compensation(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Apply weight compensation for scanner."""
        compensated = positions.copy()
        
        if "shoulder" in compensated:
            compensated["shoulder"] += 0.05 * self.scanner_weight
        
        if "elbow" in compensated:
            compensated["elbow"] -= 0.03 * self.scanner_weight
        
        return compensated
    
    def execute_pattern(self, pattern) -> bool:
        """Execute a scan pattern."""
        try:
            points = pattern.generate_points()
            
            logger.info(f"Executing pattern: {pattern.name} ({len(points)} points)")
            
            for i, point in enumerate(points):
                if self.emergency_stop_flag:
                    logger.warning("Pattern aborted - emergency stop")
                    return False
                
                success = self.move_joints(
                    point.positions,
                    speed=point.speed,
                    trajectory_type=point.trajectory_type
                )
                
                if not success:
                    logger.error(f"Failed at point {i+1}/{len(points)}")
                    return False
                
                if hasattr(point, 'settle_time'):
                    time.sleep(point.settle_time)
                
                if i % 10 == 0:
                    progress = (i + 1) / len(points) * 100
                    logger.info(f"Pattern progress: {progress:.1f}%")
            
            logger.info("✅ Pattern completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pattern execution error: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        if not self.serial.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
