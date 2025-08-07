#!/usr/bin/env python3
"""
RoArm M3 Serial Communication Manager
Non-blocking implementation für macOS
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
from typing import Optional, Dict, List
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class SerialManager:
    """
    Thread-safe Serial Communication Manager mit non-blocking reads.
    """
    
    def __init__(self, port: str = "/dev/tty.usbserial-110", 
                 baudrate: int = 115200, 
                 timeout: float = 0.1):  # Kurzer Timeout!
        """
        Initialisiert Serial Manager.
        
        Args:
            port: Serial port
            baudrate: Baud rate
            timeout: Read timeout (kurz für non-blocking)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        self.serial = None
        self.connected = False
        
        # Response handling
        self.response_queue = Queue()
        self.reader_thread = None
        self.running = False
        
        # Thread lock
        self._lock = threading.Lock()
        
        logger.info(f"SerialManager initialized for {port}")
    
    def connect(self) -> bool:
        """
        Verbindet mit dem Serial Port.
        
        Returns:
            True wenn erfolgreich
        """
        try:
            # Close if already open
            if self.serial and self.serial.is_open:
                self.serial.close()
            
            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,  # Wichtig: kurzer Timeout!
                write_timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Wait for Arduino to initialize
            time.sleep(2)  # Arduino braucht Zeit nach Serial-Verbindung
            
            self.connected = True
            
            # Start reader thread
            self._start_reader_thread()
            
            logger.info(f"✅ Connected to {self.port}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Trennt die Serial-Verbindung."""
        try:
            # Stop reader thread
            self._stop_reader_thread()
            
            # Close serial
            if self.serial and self.serial.is_open:
                self.serial.close()
            
            self.connected = False
            logger.info("Disconnected from serial port")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def send_command(self, command: Dict, wait_response: bool = False, 
                    timeout: float = 2.0) -> Optional[str]:
        """
        Sendet ein Command.
        
        Args:
            command: Command dictionary
            wait_response: Auf Antwort warten
            timeout: Response timeout
            
        Returns:
            Response string oder None
        """
        if not self.connected or not self.serial:
            logger.error("Not connected")
            return None
        
        try:
            # Convert to JSON
            json_str = json.dumps(command)
            
            # Clear response queue if waiting for response
            if wait_response:
                while not self.response_queue.empty():
                    try:
                        self.response_queue.get_nowait()
                    except Empty:
                        break
            
            # Send command
            with self._lock:
                self.serial.write(json_str.encode('utf-8'))
                self.serial.write(b'\n')  # Newline wichtig für Arduino!
                self.serial.flush()
            
            logger.debug(f"Sent: {json_str}")
            
            # Wait for response if requested
            if wait_response:
                try:
                    response = self.response_queue.get(timeout=timeout)
                    logger.debug(f"Received: {response}")
                    return response
                except Empty:
                    logger.warning(f"Response timeout for command: {command.get('T', 'unknown')}")
                    return None
            
            return ""  # Command sent successfully
            
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return None
    
    def send_raw(self, data: bytes) -> bool:
        """
        Sendet rohe Bytes.
        
        Args:
            data: Bytes to send
            
        Returns:
            True wenn erfolgreich
        """
        if not self.connected or not self.serial:
            return False
        
        try:
            with self._lock:
                self.serial.write(data)
                self.serial.flush()
            return True
        except Exception as e:
            logger.error(f"Send raw error: {e}")
            return False
    
    def read_line(self, timeout: float = 0.1) -> Optional[str]:
        """
        Liest eine Zeile (non-blocking).
        
        Args:
            timeout: Read timeout
            
        Returns:
            Line string oder None
        """
        try:
            response = self.response_queue.get(timeout=timeout)
            return response
        except Empty:
            return None
    
    def _start_reader_thread(self):
        """Startet den Reader Thread."""
        if not self.reader_thread:
            self.running = True
            self.reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True
            )
            self.reader_thread.start()
            logger.debug("Reader thread started")
    
    def _stop_reader_thread(self):
        """Stoppt den Reader Thread."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None
            logger.debug("Reader thread stopped")
    
    def _reader_loop(self):
        """
        Reader Thread Loop - liest kontinuierlich vom Serial Port.
        """
        buffer = ""
        
        while self.running:
            try:
                if self.serial and self.serial.is_open and self.serial.in_waiting > 0:
                    # Read available bytes
                    data = self.serial.read(self.serial.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            # Add to response queue
                            self.response_queue.put(line)
                            
                            # Log if debug
                            if line.startswith('{'):
                                logger.debug(f"Response: {line}")
                
                else:
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.01)
                    
            except serial.SerialException as e:
                logger.error(f"Serial read error: {e}")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Reader loop error: {e}")
                time.sleep(0.1)
    
    def flush_buffers(self):
        """Leert alle Puffer."""
        try:
            if self.serial and self.serial.is_open:
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
            
            # Clear response queue
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Flush error: {e}")
    
    def is_connected(self) -> bool:
        """Prüft ob verbunden."""
        return self.connected and self.serial and self.serial.is_open
    
    @staticmethod
    def list_ports() -> List[str]:
        """
        Listet verfügbare Serial Ports.
        
        Returns:
            Liste von Port-Namen
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
            logger.info(f"Found port: {port.device} - {port.description}")
        return ports
    
    @staticmethod
    def find_arduino_port() -> Optional[str]:
        """
        Sucht nach Arduino/USB Serial Port.
        
        Returns:
            Port name oder None
        """
        for port in serial.tools.list_ports.comports():
            # Check for common Arduino identifiers
            if any(x in port.description.lower() for x in ['arduino', 'usb', 'serial']):
                logger.info(f"Found Arduino port: {port.device}")
                return port.device
            
            # macOS specific
            if 'usbserial' in port.device.lower() or 'cu.' in port.device:
                logger.info(f"Found USB serial port: {port.device}")
                return port.device
        
        return None


# Test function
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find port
    port = SerialManager.find_arduino_port()
    if not port:
        print("No Arduino/USB serial port found!")
        print("\nAvailable ports:")
        for p in SerialManager.list_ports():
            print(f"  - {p}")
        sys.exit(1)
    
    print(f"Using port: {port}")
    
    # Test connection
    manager = SerialManager(port=port)
    
    if manager.connect():
        print("✅ Connected successfully!")
        
        # Test command
        print("\nSending test command...")
        response = manager.send_command({"T": 1}, wait_response=True, timeout=1.0)
        
        if response:
            print(f"Response: {response}")
        else:
            print("No response (timeout)")
        
        # Disconnect
        manager.disconnect()
    else:
        print("❌ Connection failed!")
