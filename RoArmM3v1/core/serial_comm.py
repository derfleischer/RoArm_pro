#!/usr/bin/env python3
"""
RoArm M3 Serial Communication Manager
Handles low-level serial communication with the robot
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class SerialManager:
    """
    Thread-safe serial communication manager for RoArm M3.
    Handles JSON-based protocol communication.
    """
    
    def __init__(self, port: str = "/dev/tty.usbserial-110",
                 baudrate: int = 115200,
                 timeout: float = 2.0):
        """
        Initialize Serial Manager.
        
        Args:
            port: Serial port name
            baudrate: Communication speed
            timeout: Read/write timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Serial connection
        self.serial_port = None
        self.connected = False
        
        # Threading
        self._lock = threading.Lock()
        self._read_thread = None
        self._write_queue = Queue()
        self._response_queue = Queue()
        self._running = False
        
        # Statistics
        self.bytes_sent = 0
        self.bytes_received = 0
        self.commands_sent = 0
        self.errors = 0
        
        logger.info(f"SerialManager initialized for {port} at {baudrate} baud")
    
    def connect(self) -> bool:
        """
        Establish serial connection.
        
        Returns:
            True if connection successful
        """
        with self._lock:
            if self.connected:
                logger.warning("Already connected")
                return True
            
            try:
                # Create serial connection
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                # Clear buffers
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
                
                # Start read thread
                self._running = True
                self._read_thread = threading.Thread(
                    target=self._read_loop,
                    daemon=True,
                    name="SerialReadThread"
                )
                self._read_thread.start()
                
                self.connected = True
                logger.info(f"✅ Connected to {self.port}")
                
                # Send initial query to verify connection
                time.sleep(0.5)
                test_cmd = {"T": 1}  # Status query
                if self.send_command(test_cmd, wait_response=True, timeout=1.0):
                    logger.info("Connection verified with status query")
                
                return True
                
            except serial.SerialException as e:
                logger.error(f"Serial connection failed: {e}")
                self.connected = False
                return False
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                self.connected = False
                return False
    
    def disconnect(self):
        """Disconnect serial connection."""
        with self._lock:
            if not self.connected:
                return
            
            try:
                # Stop read thread
                self._running = False
                if self._read_thread:
                    self._read_thread.join(timeout=2.0)
                
                # Close serial port
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()
                
                self.connected = False
                logger.info("Disconnected from serial port")
                
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
    
    def send_command(self, command: Dict[str, Any],
                    wait_response: bool = False,
                    timeout: float = 1.0) -> Optional[Dict]:
        """
        Send command to robot.
        
        Args:
            command: Command dictionary with at least 'T' field
            wait_response: Wait for response
            timeout: Response timeout in seconds
            
        Returns:
            Response dict if wait_response=True, None otherwise
        """
        if not self.connected:
            logger.error("Not connected")
            return None
        
        try:
            # Convert to JSON
            json_str = json.dumps(command)
            
            # Add newline if not present
            if not json_str.endswith('\n'):
                json_str += '\n'
            
            # Send command
            with self._lock:
                self.serial_port.write(json_str.encode('utf-8'))
                self.serial_port.flush()
                self.bytes_sent += len(json_str)
                self.commands_sent += 1
            
            logger.debug(f"Sent: {json_str.strip()}")
            
            # Wait for response if requested
            if wait_response:
                try:
                    response = self._response_queue.get(timeout=timeout)
                    return response
                except Empty:
                    logger.warning(f"Response timeout for command: {command}")
                    return None
            
            return None
            
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            self.errors += 1
            self.connected = False
            return None
        except Exception as e:
            logger.error(f"Send command error: {e}")
            self.errors += 1
            return None
    
    def _read_loop(self):
        """Background thread for reading serial data."""
        buffer = ""
        
        while self._running:
            try:
                if self.serial_port and self.serial_port.in_waiting:
                    # Read available data
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    self.bytes_received += len(data)
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            self._process_response(line)
                
                else:
                    time.sleep(0.01)  # Small delay to prevent CPU spinning
                    
            except serial.SerialException as e:
                logger.error(f"Serial read error: {e}")
                self.errors += 1
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                self.errors += 1
                time.sleep(0.1)
    
    def _process_response(self, line: str):
        """Process a response line from the robot."""
        try:
            # Parse JSON response
            response = json.loads(line)
            logger.debug(f"Received: {line}")
            
            # Put in response queue
            self._response_queue.put(response)
            
            # Handle specific response types
            if 'error' in response:
                logger.error(f"Robot error: {response['error']}")
                self.errors += 1
            elif 'warning' in response:
                logger.warning(f"Robot warning: {response['warning']}")
                
        except json.JSONDecodeError:
            # Not JSON - might be plain text response
            logger.debug(f"Non-JSON response: {line}")
        except Exception as e:
            logger.error(f"Response processing error: {e}")
    
    def flush_buffers(self):
        """Clear input and output buffers."""
        if self.connected and self.serial_port:
            with self._lock:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
            
            # Clear queues
            while not self._response_queue.empty():
                self._response_queue.get()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "connected": self.connected,
            "port": self.port,
            "baudrate": self.baudrate,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "commands_sent": self.commands_sent,
            "errors": self.errors
        }
    
    @staticmethod
    def list_available_ports() -> List[str]:
        """List all available serial ports."""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid
            })
        return ports
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
