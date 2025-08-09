#!/usr/bin/env python3
"""
RoArm M3 Controller - Fixed for actual JSON protocol
Based on the actual response format from your RoArm
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
import queue
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

from .constants import SERVO_LIMITS, SPEED_LIMITS
from ..motion.trajectory import TrajectoryGenerator, TrajectoryType
from ..utils.logger import get_logger

logger = get_logger(__name__)


# CORRECTED COMMAND IDs based on your RoArm's responses
ROARM_COMMANDS = {
    "STATUS": 1051,      # Returns position data (from your log)
    "MOVE": 104,         # Most likely movement command
    "LED": 51,           # Try this first for LED
    "TORQUE": 210,       # Standard torque command
    "EMERGENCY": 0,      # Emergency stop
    
    # Alternatives to try if above don't work
    "MOVE_ALT1": 102,    
    "MOVE_ALT2": 105,
    "LED_ALT1": 107,
    "LED_ALT2": 1052,
    "TORQUE_ALT": 208,
}

# Position mappings from your response
POSITION_KEYS = {
    "base": "b",
    "shoulder": "s", 
    "elbow": "e",
    "wrist": "t",  # 't' in response
    "roll": "r",
    "hand": "g"    # 'g' for gripper
}

# Home position based on your response data
ACTUAL_HOME_POSITION = {
    "base": 0.0015,      # From "b":0.001533981
    "shoulder": -0.0015,  # From "s":-0.001533981
    "elbow": 2.968,      # From "e":2.968252825
    "wrist": -0.0276,    # From "t":-0.027611654
    "roll": 0.0031,      # From "r":0.003067962
    "hand": 3.151        # From "g":3.150796538
}


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
    # Use actual command IDs
    use_actual_commands: bool = True
    command_delay: float = 0.1  # Delay between commands


class RoArmController:
    """
    RoArm M3 Controller using actual JSON protocol.
    """
    
    def __init__(self, config: Optional[RoArmConfig] = None):
        """Initialize controller."""
        self.config = config or RoArmConfig()
        
        # Serial connection
        self.serial = None
        self.connected = False
        
        # Trajectory Generator
        self.trajectory = TrajectoryGenerator()
        
        # Command Queue
        self.command_queue = queue.Queue(maxsize=100)
        self.queue_thread = None
        self.running = False
        
        # Current State (from actual response)
        self.current_position = ACTUAL_HOME_POSITION.copy()
        self.current_speed = self.config.default_speed
        self.torque_enabled = False
        self.emergency_stop_flag = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Response buffer
        self._response_buffer = ""
        
        logger.info("RoArm Controller initialized (Fixed JSON protocol)")
        
        # Auto-connect
        if self.config.auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to RoArm."""
        try:
            # Auto-detect port if needed
            if "auto" in self.config.port.lower():
                self.config.port = self._auto_detect_port()
                logger.info(f"Auto-detected port: {self.config.port}")
            
            # Open serial connection
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=0.5,
                write_timeout=2.0
            )
            
            # Wait for Arduino reset
            time.sleep(2)
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            self.connected = True
            logger.info(f"✅ Connected to RoArm on {self.config.port}")
            
            # Test connection with status query
            if self._test_connection():
                # Start queue processor
                self._start_queue_processor()
                
                # Initialize robot
                self._initialize_robot()
                
                return True
            else:
                logger.error("Connection test failed")
                self.connected = False
                self.serial.close()
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from RoArm."""
        try:
            logger.info("Disconnecting...")
            
            # Stop queue processor
            self._stop_queue_processor()
            
            # Close serial
            if self.serial:
                self.serial.close()
            
            self.connected = False
            logger.info("✅ Disconnected")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def _test_connection(self) -> bool:
        """Test connection with status query."""
        try:
            # Send status query using actual command ID
            response = self._send_command({"T": ROARM_COMMANDS["STATUS"]}, wait_response=True)
            
            if response and "T" in response:
                logger.info(f"Connection test successful, response T:{response.get('T')}")
                
                # Update current position from response
                self._update_position_from_response(response)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False
    
    def _initialize_robot(self):
        """Initialize robot after connection."""
        try:
            # Query current status
            self.query_status()
            
            # Try to blink LED (might not work with wrong command ID)
            self.led_control(True, 255)
            time.sleep(0.5)
            self.led_control(False)
            
            # Enable torque
            self.set_torque(True)
            
            logger.info("Robot initialized")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
    
    def _send_command(self, command: Dict, wait_response: bool = False) -> Optional[Dict]:
        """
        Send JSON command and optionally wait for response.
        
        Args:
            command: Command dictionary
            wait_response: Wait for response
            
        Returns:
            Response dict if wait_response=True
        """
        if not self.connected:
            logger.error("Not connected")
            return None
        
        try:
            # Send command as JSON
            cmd_str = json.dumps(command) + '\n'
            
            if self.config.debug:
                logger.debug(f"Sending: {cmd_str.strip()}")
            
            self.serial.write(cmd_str.encode('utf-8'))
            
            if wait_response:
                # Wait for response
                time.sleep(self.config.command_delay)
                
                # Read response
                response = self._read_response()
                
                if self.config.debug and response:
                    logger.debug(f"Response: {response}")
                
                return response
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return None
    
    def _read_response(self, timeout: float = 0.5) -> Optional[Dict]:
        """Read and parse JSON response."""
        try:
            start_time = time.time()
            response_data = ""
            
            while time.time() - start_time < timeout:
                if self.serial.in_waiting:
                    chunk = self.serial.read(self.serial.in_waiting)
                    response_data += chunk.decode('utf-8', errors='ignore')
                    
                    # Try to parse JSON from response
                    if '{' in response_data and '}' in response_data:
                        # Extract JSON object
                        start_idx = response_data.find('{')
                        end_idx = response_data.rfind('}') + 1
                        
                        if end_idx > start_idx:
                            json_str = response_data[start_idx:end_idx]
                            
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                # Keep trying
                                pass
                
                time.sleep(0.01)
            
            return None
            
        except Exception as e:
            logger.error(f"Read response error: {e}")
            return None
    
    def _update_position_from_response(self, response: Dict):
        """Update current position from response."""
        if not response:
            return
        
        with self._lock:
            # Update using actual response keys
            for joint, key in POSITION_KEYS.items():
                if key in response:
                    self.current_position[joint] = response[key]
    
    def move_joints(self, positions: Dict[str, float], 
                   speed: Optional[float] = None,
                   trajectory_type: TrajectoryType = TrajectoryType.S_CURVE,
                   wait: bool = True) -> bool:
        """
        Move joints to target positions.
        
        Args:
            positions: Target positions in radians
            speed: Speed factor (0.1-2.0)
            trajectory_type: Movement profile
            wait: Wait for completion
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                if self.emergency_stop_flag:
                    logger.warning("Movement blocked - emergency stop active")
                    return False
            
            # Validate positions
            for joint, pos in positions.items():
                if joint in SERVO_LIMITS:
                    min_val, max_val = SERVO_LIMITS[joint]
                    if pos < min_val or pos > max_val:
                        logger.error(f"{joint} position {pos:.3f} out of limits [{min_val:.3f}, {max_val:.3f}]")
                        return False
            
            # Set speed
            speed = speed or self.current_speed
            speed = max(SPEED_LIMITS["min"], min(SPEED_LIMITS["max"], speed))
            
            # Try different movement command formats
            move_commands = [
                # Format 1: Using position keys from response
                self._create_move_command_format1(positions),
                # Format 2: Direct joint names
                self._create_move_command_format2(positions),
                # Format 3: Cartesian if available
                self._create_move_command_format3(positions),
            ]
            
            success = False
            for cmd in move_commands:
                if cmd:
                    response = self._send_command(cmd, wait_response=True)
                    
                    if response:
                        # Check if movement was accepted
                        if response.get("T") == ROARM_COMMANDS["STATUS"]:
                            # Got status response, movement might have worked
                            success = True
                            break
                        elif "error" not in str(response).lower():
                            success = True
                            break
            
            if success:
                # Update current position
                with self._lock:
                    self.current_position.update(positions)
                
                if wait:
                    # Calculate movement time based on distance and speed
                    max_delta = max(abs(positions.get(j, 0) - self.current_position.get(j, 0)) 
                                   for j in positions.keys())
                    move_time = max_delta / speed
                    time.sleep(min(move_time, 3.0))
                
                logger.debug(f"Movement completed to {positions}")
                return True
            else:
                logger.error("Movement command failed")
                return False
            
        except Exception as e:
            logger.error(f"Movement error: {e}")
            return False
    
    def _create_move_command_format1(self, positions: Dict[str, float]) -> Dict:
        """Create movement command using actual response keys."""
        cmd = {"T": ROARM_COMMANDS["MOVE"]}
        
        for joint, pos in positions.items():
            if joint in POSITION_KEYS:
                key = POSITION_KEYS[joint]
                cmd[key] = pos
        
        return cmd
    
    def _create_move_command_format2(self, positions: Dict[str, float]) -> Dict:
        """Create movement command using joint names."""
        cmd = {"T": ROARM_COMMANDS["MOVE"]}
        cmd.update(positions)
        return cmd
    
    def _create_move_command_format3(self, positions: Dict[str, float]) -> Optional[Dict]:
        """Create cartesian movement command if needed."""
        # Only if we have cartesian data in response
        # For now, return None
        return None
    
    def move_home(self, speed: float = 1.0) -> bool:
        """Move to home position."""
        logger.info("Moving to home position...")
        
        # Use actual home position from response
        return self.move_joints(ACTUAL_HOME_POSITION, speed=speed)
    
    def gripper_control(self, position: float) -> bool:
        """
        Control gripper.
        
        Args:
            position: 0.0 (open) to 1.0 (closed)
            
        Returns:
            True if successful
        """
        try:
            # Map to servo range
            min_pos = SERVO_LIMITS["hand"][0]  # 1.08 rad
            max_pos = SERVO_LIMITS["hand"][1]  # 3.14 rad
            
            servo_pos = min_pos + (max_pos - min_pos) * position
            
            return self.move_joints({"hand": servo_pos}, speed=1.0)
            
        except Exception as e:
            logger.error(f"Gripper control error: {e}")
            return False
    
    def set_torque(self, enabled: bool) -> bool:
        """Enable/disable torque."""
        try:
            # Try different torque command formats
            commands = [
                {"T": ROARM_COMMANDS["TORQUE"], "enabled": 1 if enabled else 0},
                {"T": ROARM_COMMANDS["TORQUE"], "torque": 1 if enabled else 0},
                {"T": ROARM_COMMANDS["TORQUE_ALT"], "enabled": 1 if enabled else 0},
            ]
            
            for cmd in commands:
                response = self._send_command(cmd, wait_response=True)
                
                if response and "error" not in str(response).lower():
                    self.torque_enabled = enabled
                    logger.info(f"Torque {'enabled' if enabled else 'disabled'}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Torque control error: {e}")
            return False
    
    def led_control(self, on: bool, brightness: int = 255) -> bool:
        """Control LED."""
        try:
            # Try different LED command formats
            commands = [
                {"T": ROARM_COMMANDS["LED"], "led": 1 if on else 0, "brightness": brightness},
                {"T": ROARM_COMMANDS["LED"], "led": 1 if on else 0},
                {"T": ROARM_COMMANDS["LED_ALT1"], "led": 1 if on else 0},
                {"T": ROARM_COMMANDS["LED_ALT2"], "led": 1 if on else 0},
            ]
            
            for cmd in commands:
                response = self._send_command(cmd)
                # LED might not return response
                return True
            
        except Exception as e:
            logger.error(f"LED control error: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop."""
        logger.warning("🚨 EMERGENCY STOP")
        
        with self._lock:
            self.emergency_stop_flag = True
        
        # Clear queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                pass
        
        # Send stop command
        self._send_command({"T": ROARM_COMMANDS["EMERGENCY"]})
        
        # Disable torque
        self.set_torque(False)
    
    def query_status(self) -> Optional[Dict]:
        """Query robot status."""
        try:
            response = self._send_command({"T": ROARM_COMMANDS["STATUS"]}, wait_response=True)
            
            if response:
                # Update internal state
                self._update_position_from_response(response)
                
                # Convert to expected format
                status = {
                    "positions": {},
                    "cartesian": {}
                }
                
                # Extract positions
                for joint, key in POSITION_KEYS.items():
                    if key in response:
                        status["positions"][joint] = response[key]
                
                # Extract cartesian if available
                if "x" in response:
                    status["cartesian"]["x"] = response["x"]
                    status["cartesian"]["y"] = response["y"]
                    status["cartesian"]["z"] = response["z"]
                
                return status
            
            return None
            
        except Exception as e:
            logger.error(f"Status query error: {e}")
            return None
    
    def execute_pattern(self, pattern) -> bool:
        """Execute a scan pattern."""
        try:
            points = pattern.generate_points()
            
            logger.info(f"Executing {pattern.name} with {len(points)} points")
            
            for i, point in enumerate(points):
                if self.emergency_stop_flag:
                    logger.warning("Pattern aborted - emergency stop")
                    return False
                
                # Move to point
                if not self.move_joints(
                    point.positions,
                    speed=point.speed,
                    trajectory_type=point.trajectory_type
                ):
                    logger.error(f"Failed at point {i+1}/{len(points)}")
                    return False
                
                # Settle time
                if hasattr(point, 'settle_time'):
                    time.sleep(point.settle_time)
                
                # Progress
                if i % 10 == 0:
                    logger.info(f"Progress: {(i+1)/len(points)*100:.1f}%")
            
            logger.info("✅ Pattern completed")
            return True
            
        except Exception as e:
            logger.error(f"Pattern execution error: {e}")
            return False
    
    def _start_queue_processor(self):
        """Start command queue processor."""
        if not self.queue_thread:
            self.running = True
            self.queue_thread = threading.Thread(
                target=self._process_queue,
                daemon=True
            )
            self.queue_thread.start()
    
    def _stop_queue_processor(self):
        """Stop command queue processor."""
        self.running = False
        if self.queue_thread:
            self.queue_thread.join(timeout=2)
            self.queue_thread = None
    
    def _process_queue(self):
        """Process command queue."""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                self._send_command(command)
                time.sleep(self.config.command_delay)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    def _auto_detect_port(self) -> str:
        """Auto-detect USB serial port."""
        try:
            ports = list(serial.tools.list_ports.comports())
            
            for port in ports:
                if 'usbserial' in port.device.lower():
                    return port.device
                elif 'cu.' in port.device:
                    return port.device
            
            logger.warning("No USB serial found, using default")
            return "/dev/tty.usbserial-110"
            
        except Exception as e:
            logger.error(f"Port detection error: {e}")
            return "/dev/tty.usbserial-110"
    
    def reset_emergency(self):
        """Reset emergency stop."""
        with self._lock:
            self.emergency_stop_flag = False
        logger.info("Emergency stop reset")
    
    def move_to_scanner_position(self, speed: float = 0.5) -> bool:
        """Move to scanner position."""
        # Define scanner position based on your needs
        scanner_pos = {
            "base": 0.0,
            "shoulder": 0.35,
            "elbow": 1.22,
            "wrist": -1.57,
            "roll": 1.57,
            "hand": 2.5
        }
        
        logger.info("Moving to scanner position...")
        return self.move_joints(scanner_pos, speed=speed)
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
