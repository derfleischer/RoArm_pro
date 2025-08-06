# ============================================
# core/serial_comm.py
# ============================================
#!/usr/bin/env python3
"""
Serial Communication Manager für RoArm M3
Thread-safe serial communication with automatic reconnection.
"""

import serial
import json
import time
import threading
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SerialManager:
    """Thread-safe serial communication manager."""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """Establish serial connection."""
        try:
            with self._lock:
                self.serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                self.connected = True
                # Clear buffers
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
                return True
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close serial connection."""
        with self._lock:
            if self.serial and self.serial.is_open:
                self.serial.close()
            self.connected = False
    
    def send_command(self, command: Dict[str, Any], wait_response: bool = False) -> Optional[str]:
        """
        Send JSON command to RoArm.
        
        Args:
            command: Command dictionary
            wait_response: Wait for response
            
        Returns:
            Response string if wait_response=True
        """
        if not self.connected:
            logger.error("Not connected")
            return None
        
        try:
            with self._lock:
                # Convert to JSON and send
                json_str = json.dumps(command) + '\n'
                self.serial.write(json_str.encode('utf-8'))
                
                if wait_response:
                    response = self.serial.readline().decode('utf-8').strip()
                    return response
                    
                return None
                
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return None
    
    def read_response(self, timeout: Optional[float] = None) -> Optional[str]:
        """Read response from serial."""
        if not self.connected:
            return None
        
        try:
            with self._lock:
                if timeout:
                    old_timeout = self.serial.timeout
                    self.serial.timeout = timeout
                    response = self.serial.readline().decode('utf-8').strip()
                    self.serial.timeout = old_timeout
                else:
                    response = self.serial.readline().decode('utf-8').strip()
                
                return response if response else None
                
        except Exception as e:
            logger.error(f"Read response error: {e}")
            return None
