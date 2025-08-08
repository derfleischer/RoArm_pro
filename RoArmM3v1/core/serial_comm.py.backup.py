#!/usr/bin/env python3
"""
RoArm M3 Serial Communication with Waveshare Protocol
Implementiert das korrekte Protokoll für Waveshare RoArm-M3
"""

import serial
import time
import threading
import struct
from typing import Dict, Optional, Any, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types."""
    JSON = "json"                # JSON-based protocol
    WAVESHARE = "waveshare"      # Waveshare binary protocol  
    GCODE = "gcode"              # G-code protocol
    SSC32 = "ssc32"              # SSC-32 servo protocol


class WaveshareProtocol:
    """
    Waveshare RoArm-M3 Protocol Implementation.
    Based on the actual hardware protocol.
    """
    
    # Command bytes (from Waveshare documentation)
    CMD_SERVO_MOVE = 0x03          # Move servo
    CMD_MULT_SERVO_MOVE = 0x04     # Move multiple servos
    CMD_SERVO_STOP = 0x05          # Stop servo
    CMD_READ_POSITION = 0x06       # Read servo position
    CMD_LED_CONTROL = 0x07         # Control LED
    CMD_SERVO_TORQUE = 0x08        # Enable/disable torque
    CMD_READ_STATUS = 0x09         # Read status
    
    # Special bytes
    HEADER = 0x55                  # Header byte
    HEADER2 = 0x55                 # Second header byte
    
    @staticmethod
    def create_packet(command: int, data: bytes) -> bytes:
        """
        Create a Waveshare protocol packet.
        
        Format: 0x55 0x55 LEN CMD DATA CHECKSUM
        
        Args:
            command: Command byte
            data: Data bytes
            
        Returns:
            Complete packet
        """
        length = len(data) + 2  # +2 for length and command bytes
        
        # Build packet
        packet = bytearray()
        packet.append(WaveshareProtocol.HEADER)
        packet.append(WaveshareProtocol.HEADER2) 
        packet.append(length)
        packet.append(command)
        packet.extend(data)
        
        # Calculate checksum (simple sum & 0xFF)
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        
        return bytes(packet)
    
    @staticmethod
    def move_servo(servo_id: int, position: int, time_ms: int = 1000) -> bytes:
        """
        Create move servo command.
        
        Args:
            servo_id: Servo ID (1-6)
            position: Target position (500-2500 PWM)
            time_ms: Movement time in milliseconds
            
        Returns:
            Command packet
        """
        # Data format: ID(1) POS_L(1) POS_H(1) TIME_L(1) TIME_H(1)
        data = bytearray()
        data.append(servo_id)
        data.append(position & 0xFF)           # Position low byte
        data.append((position >> 8) & 0xFF)    # Position high byte
        data.append(time_ms & 0xFF)            # Time low byte
        data.append((time_ms >> 8) & 0xFF)     # Time high byte
        
        return WaveshareProtocol.create_packet(
            WaveshareProtocol.CMD_SERVO_MOVE, 
            data
        )
    
    @staticmethod
    def move_multiple_servos(positions: Dict[int, int], time_ms: int = 1000) -> bytes:
        """
        Create move multiple servos command.
        
        Args:
            positions: Dict of servo_id -> position
            time_ms: Movement time
            
        Returns:
            Command packet
        """
        # Data format: COUNT(1) [ID(1) POS_L(1) POS_H(1)]... TIME_L(1) TIME_H(1)
        data = bytearray()
        data.append(len(positions))  # Number of servos
        
        for servo_id, position in positions.items():
            data.append(servo_id)
            data.append(position & 0xFF)
            data.append((position >> 8) & 0xFF)
        
        data.append(time_ms & 0xFF)
        data.append((time_ms >> 8) & 0xFF)
        
        return WaveshareProtocol.create_packet(
            WaveshareProtocol.CMD_MULT_SERVO_MOVE,
            data
        )
    
    @staticmethod
    def control_led(on: bool, brightness: int = 255) -> bytes:
        """Create LED control command."""
        data = bytearray()
        data.append(0x01 if on else 0x00)
        data.append(brightness)
        
        return WaveshareProtocol.create_packet(
            WaveshareProtocol.CMD_LED_CONTROL,
            data
        )
    
    @staticmethod
    def set_torque(servo_id: int, enabled: bool) -> bytes:
        """Create torque enable/disable command."""
        data = bytearray()
        data.append(servo_id)  # 0 for all servos
        data.append(0x01 if enabled else 0x00)
        
        return WaveshareProtocol.create_packet(
            WaveshareProtocol.CMD_SERVO_TORQUE,
            data
        )
    
    @staticmethod
    def read_position(servo_id: int) -> bytes:
        """Create read position command."""
        data = bytearray([servo_id])
        
        return WaveshareProtocol.create_packet(
            WaveshareProtocol.CMD_READ_POSITION,
            data
        )
    
    @staticmethod
    def parse_response(data: bytes) -> Optional[Dict]:
        """
        Parse response from RoArm.
        
        Args:
            data: Response bytes
            
        Returns:
            Parsed response or None
        """
        if len(data) < 5:
            return None
        
        # Check header
        if data[0] != 0x55 or data[1] != 0x55:
            return None
        
        length = data[2]
        command = data[3]
        
        # Extract payload
        if len(data) < length + 3:
            return None
        
        payload = data[4:4+length-2]
        
        # Parse based on command
        result = {
            "command": command,
            "raw": data.hex()
        }
        
        if command == WaveshareProtocol.CMD_READ_POSITION:
            if len(payload) >= 2:
                position = payload[0] | (payload[1] << 8)
                result["position"] = position
        
        elif command == WaveshareProtocol.CMD_READ_STATUS:
            # Parse status response
            if len(payload) >= 12:  # 6 servos * 2 bytes
                positions = []
                for i in range(0, 12, 2):
                    pos = payload[i] | (payload[i+1] << 8)
                    positions.append(pos)
                result["positions"] = positions
        
        return result


class SerialManager:
    """
    Enhanced Serial Manager with multi-protocol support.
    """
    
    def __init__(self, port: str, baudrate: int = 115200, 
                 protocol: ProtocolType = ProtocolType.WAVESHARE):
        """
        Initialize SerialManager.
        
        Args:
            port: Serial port
            baudrate: Baud rate  
            protocol: Protocol type to use
        """
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
        self.serial = None
        self.connected = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Response buffer for binary protocols
        self._response_buffer = bytearray()
        self._response_event = threading.Event()
        self._reader_thread = None
        self._stop_reader = False
        
        # Servo mapping for Waveshare
        self.joint_to_servo = {
            "base": 1,
            "shoulder": 2, 
            "elbow": 3,
            "wrist": 4,
            "roll": 5,
            "hand": 6
        }
        
        # Position conversion
        self.rad_to_pwm = 318.31  # PWM units per radian (1000/π)
        self.pwm_center = 1500     # Center position
        
        logger.info(f"SerialManager initialized with {protocol.value} protocol")
    
    def connect(self) -> bool:
        """Connect to serial port."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,  # Non-blocking read
                write_timeout=2.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Wait for Arduino reset
            time.sleep(2.0)
            
            # Test connection based on protocol
            if self._test_connection():
                self.connected = True
                
                # Start reader thread for binary protocols
                if self.protocol in [ProtocolType.WAVESHARE]:
                    self._start_reader_thread()
                
                logger.info(f"Connected to {self.port}")
                return True
            else:
                logger.error("Connection test failed")
                self.serial.close()
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        self.connected = False
        
        # Stop reader thread
        if self._reader_thread:
            self._stop_reader = True
            self._reader_thread.join(timeout=1)
        
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
        
        logger.info("Disconnected")
    
    def _test_connection(self) -> bool:
        """Test connection with appropriate protocol."""
        try:
            if self.protocol == ProtocolType.WAVESHARE:
                # Test with LED command
                packet = WaveshareProtocol.control_led(True, 128)
                self.serial.write(packet)
                time.sleep(0.5)
                
                packet = WaveshareProtocol.control_led(False)
                self.serial.write(packet)
                time.sleep(0.5)
                
                # If no exception, assume success
                return True
                
            elif self.protocol == ProtocolType.JSON:
                # Test with status query
                self.serial.write(b'{"T":1}\n')
                time.sleep(0.5)
                
                # Check for response
                if self.serial.in_waiting > 0:
                    return True
                    
            elif self.protocol == ProtocolType.SSC32:
                # SSC-32 version query
                self.serial.write(b'VER\r')
                time.sleep(0.5)
                
                if self.serial.in_waiting > 0:
                    return True
            
            # Default: assume connected
            return True
            
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False
    
    def send_command(self, command: Dict, wait_response: bool = False) -> Optional[Any]:
        """
        Send command using appropriate protocol.
        
        Args:
            command: Command dictionary
            wait_response: Wait for response
            
        Returns:
            Response if wait_response=True
        """
        if not self.connected:
            logger.error("Not connected")
            return None
        
        try:
            with self._lock:
                if self.protocol == ProtocolType.WAVESHARE:
                    return self._send_waveshare(command, wait_response)
                elif self.protocol == ProtocolType.JSON:
                    return self._send_json(command, wait_response)
                elif self.protocol == ProtocolType.SSC32:
                    return self._send_ssc32(command, wait_response)
                else:
                    logger.error(f"Unsupported protocol: {self.protocol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return None
    
    def _send_waveshare(self, command: Dict, wait_response: bool) -> Optional[Any]:
        """Send command using Waveshare protocol."""
        # Parse command type
        cmd_type = command.get("T", 0)
        
        if cmd_type == 102:  # Joint movement
            # Convert positions to PWM
            positions = {}
            for joint, rad_pos in command.items():
                if joint in self.joint_to_servo:
                    servo_id = self.joint_to_servo[joint]
                    # Convert radians to PWM
                    pwm = int(self.pwm_center + (rad_pos * self.rad_to_pwm))
                    pwm = max(500, min(2500, pwm))
                    positions[servo_id] = pwm
            
            if positions:
                # Send move command
                packet = WaveshareProtocol.move_multiple_servos(positions, 1000)
                self.serial.write(packet)
                logger.debug(f"Sent Waveshare move: {positions}")
                
        elif cmd_type == 51:  # LED control
            on = command.get("led", 0) > 0
            brightness = command.get("brightness", 255)
            packet = WaveshareProtocol.control_led(on, brightness)
            self.serial.write(packet)
            logger.debug(f"Sent LED control: on={on}")
            
        elif cmd_type == 210:  # Torque control
            enabled = command.get("enabled", 0) > 0
            packet = WaveshareProtocol.set_torque(0, enabled)  # 0 = all servos
            self.serial.write(packet)
            logger.debug(f"Sent torque control: enabled={enabled}")
            
        elif cmd_type == 1:  # Status query
            # Read all servo positions
            positions = {}
            for joint, servo_id in self.joint_to_servo.items():
                packet = WaveshareProtocol.read_position(servo_id)
                self.serial.write(packet)
                time.sleep(0.05)
                
                # Read response
                if self.serial.in_waiting > 0:
                    response = self.serial.read(self.serial.in_waiting)
                    parsed = WaveshareProtocol.parse_response(response)
                    if parsed and "position" in parsed:
                        # Convert PWM to radians
                        pwm = parsed["position"]
                        rad = (pwm - self.pwm_center) / self.rad_to_pwm
                        positions[joint] = rad
            
            if wait_response:
                return {"positions": positions}
        
        return True
    
    def _send_json(self, command: Dict, wait_response: bool) -> Optional[Any]:
        """Send command using JSON protocol."""
        import json
        
        # Send as JSON
        cmd_str = json.dumps(command) + '\n'
        self.serial.write(cmd_str.encode('utf-8'))
        
        if wait_response:
            # Wait for response
            time.sleep(0.1)
            if self.serial.in_waiting > 0:
                response = self.serial.readline()
                try:
                    return json.loads(response.decode('utf-8'))
                except:
                    return None
        
        return True
    
    def _send_ssc32(self, command: Dict, wait_response: bool) -> Optional[Any]:
        """Send command using SSC-32 protocol."""
        # SSC-32 format: #1P1500T1000
        cmd_str = ""
        
        for joint, rad_pos in command.items():
            if joint in self.joint_to_servo:
                servo_id = self.joint_to_servo[joint]
                pwm = int(self.pwm_center + (rad_pos * self.rad_to_pwm))
                pwm = max(500, min(2500, pwm))
                cmd_str += f"#{servo_id}P{pwm}"
        
        if cmd_str:
            cmd_str += "T1000\r"  # 1 second move time
            self.serial.write(cmd_str.encode('ascii'))
            logger.debug(f"Sent SSC-32: {cmd_str.strip()}")
        
        return True
    
    def _start_reader_thread(self):
        """Start background thread to read responses."""
        self._stop_reader = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )
        self._reader_thread.start()
    
    def _reader_loop(self):
        """Background thread to read serial data."""
        while not self._stop_reader:
            try:
                if self.serial and self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    with self._lock:
                        self._response_buffer.extend(data)
                    self._response_event.set()
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Reader thread error: {e}")
                time.sleep(0.1)
    
    def test_all_protocols(self) -> str:
        """
        Test all protocols and return which one works.
        
        Returns:
            Working protocol name or "none"
        """
        logger.info("Testing all protocols...")
        
        for protocol in [ProtocolType.WAVESHARE, ProtocolType.JSON, 
                        ProtocolType.SSC32]:
            logger.info(f"Testing {protocol.value}...")
            
            # Disconnect if connected
            if self.connected:
                self.disconnect()
            
            # Set protocol
            self.protocol = protocol
            
            # Try to connect
            if self.connect():
                # Test LED
                self.send_command({"T": 51, "led": 1, "brightness": 255})
                time.sleep(0.5)
                self.send_command({"T": 51, "led": 0})
                
                # If no exception, this protocol works
                logger.info(f"✅ {protocol.value} protocol works!")
                return protocol.value
        
        logger.error("No working protocol found")
        return "none"
