"""
Thread-safe serial communication handler
Manages all low-level communication with the robot
"""

import serial
import json
import time
import threading
import logging
import re
from typing import Dict, Optional, Any
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class SerialConnection:
    """Thread-safe serial communication with the RoArm"""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False
        
        # Response parsing
        self._response_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        
    @property
    def is_connected(self) -> bool:
        """Check if serial connection is open"""
        return self._connected and self._serial and self._serial.is_open
    
    def connect(self) -> bool:
        """Establish serial connection"""
        with self._lock:
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()
                    
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                
                # Wait for connection to stabilize
                time.sleep(2.0)
                
                # Flush buffers
                self._flush_buffers()
                
                self._connected = True
                logger.info(f"✅ Connected to {self.port} @ {self.baudrate} baud")
                return True
                
            except serial.SerialException as e:
                logger.error(f"❌ Connection failed: {e}")
                self._connected = False
                return False
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                self._connected = False
                return False
    
    def disconnect(self):
        """Close serial connection"""
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                    logger.info("🔌 Serial connection closed")
                except Exception as e:
                    logger.error(f"Error closing serial: {e}")
            self._connected = False
    
    def send_command(self, command: Dict[str, Any], 
                    wait_time: float = 0.02, 
                    expect_response: bool = True) -> Optional[Dict]:
        """
        Send command and optionally wait for response
        Thread-safe implementation
        """
        if not self.is_connected:
            logger.warning("Not connected")
            return None
            
        with self._lock:
            try:
                # Convert command to JSON
                cmd_str = json.dumps(command, separators=(',', ':'))
                cmd_bytes = (cmd_str + '\n').encode('utf-8')
                
                # Clear input buffer before sending
                if self._serial.in_waiting > 0:
                    self._serial.read(self._serial.in_waiting)
                
                # Send command
                self._serial.write(cmd_bytes)
                self._serial.flush()
                
                # Small delay for command processing
                time.sleep(wait_time)
                
                # Read response if expected
                if expect_response:
                    return self._read_response()
                    
                return {"status": "sent"}
                
            except serial.SerialTimeoutException:
                logger.error("Serial write timeout")
                return None
            except Exception as e:
                logger.error(f"Command error: {e}")
                return None
    
    def _read_response(self, timeout: float = 0.5) -> Optional[Dict]:
        """Read and parse response from robot"""
        try:
            start_time = time.time()
            response_data = b""
            
            while (time.time() - start_time) < timeout:
                if self._serial.in_waiting > 0:
                    chunk = self._serial.read(self._serial.in_waiting)
                    response_data += chunk
                    
                    # Try to parse JSON from accumulated data
                    decoded = response_data.decode('utf-8', errors='ignore')
                    json_match = self._response_pattern.search(decoded)
                    
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except json.JSONDecodeError:
                            continue
                            
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        except Exception as e:
            logger.debug(f"Response read error: {e}")
            
        return None
    
    def _flush_buffers(self):
        """Flush input and output buffers"""
        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            time.sleep(0.1)
            
            # Clear any remaining data
            while self._serial.in_waiting > 0:
                self._serial.read(self._serial.in_waiting)
                time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Buffer flush error: {e}")
    
    def emergency_stop(self):
        """Send emergency stop command immediately"""
        if self._serial and self._serial.is_open:
            try:
                # Send emergency stop without acquiring lock (for true emergency)
                emergency_cmd = '{"T":0}\n'
                self._serial.write(emergency_cmd.encode('utf-8'))
                self._serial.flush()
                logger.critical("🛑 EMERGENCY STOP SENT")
            except Exception as e:
                logger.error(f"Emergency stop failed: {e}")
