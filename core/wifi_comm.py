#!/usr/bin/env python3
"""
RoArm M3 WiFi Communication Manager (OPTIONAL)
Alternative Verbindung über WiFi zum ESP32
Platzieren Sie diese Datei in core/wifi_comm.py wenn Sie WiFi nutzen möchten
"""

import socket
import json
import time
import threading
from typing import Optional, Dict, List
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class WiFiManager:
    """
    WiFi Communication Manager für ESP32-basierte RoArm Steuerung.
    Kompatibel mit SerialManager Interface.
    """
    
    def __init__(self, host: str = "192.168.4.1",
                 port: int = 8080, 
                 timeout: float = 2.0):
        """
        Initialisiert WiFi Manager.
        
        Args:
            host: ESP32 IP-Adresse
            port: TCP Port
            timeout: Socket timeout
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.socket = None
        self.connected = False
        
        # Response handling
        self.response_queue = Queue()
        self.reader_thread = None
        self.running = False
        
        # Thread lock
        self._lock = threading.Lock()
        
        logger.info(f"WiFiManager initialized for {host}:{port}")
    
    def connect(self) -> bool:
        """Verbindet mit ESP32 über WiFi."""
        try:
            # Close existing
            if self.socket:
                self.socket.close()
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # Connect
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            
            self.connected = True
            
            # Start reader
            self._start_reader_thread()
            
            logger.info(f"✅ Connected via WiFi")
            return True
            
        except socket.timeout:
            logger.error(f"Connection timeout")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"WiFi error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Trennt WiFi-Verbindung."""
        try:
            self._stop_reader_thread()
            
            if self.socket:
                self.socket.close()
            
            self.connected = False
            logger.info("Disconnected")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def send_command(self, command: Dict, wait_response: bool = False, 
                    timeout: float = 2.0) -> Optional[str]:
        """Sendet Command (SerialManager kompatibel)."""
        if not self.connected:
            return None
        
        try:
            # Convert to JSON
            json_str = json.dumps(command)
            
            # Clear queue if waiting
            if wait_response:
                while not self.response_queue.empty():
                    try:
                        self.response_queue.get_nowait()
                    except Empty:
                        break
            
            # Send
            with self._lock:
                data = (json_str + '\n').encode('utf-8')
                self.socket.sendall(data)
            
            logger.debug(f"Sent: {json_str}")
            
            # Wait for response
            if wait_response:
                try:
                    response = self.response_queue.get(timeout=timeout)
                    return response
                except Empty:
                    return None
            
            return ""
            
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.connected = False
            return None
    
    def _start_reader_thread(self):
        """Startet Reader Thread."""
        if not self.reader_thread:
            self.running = True
            self.reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True
            )
            self.reader_thread.start()
    
    def _stop_reader_thread(self):
        """Stoppt Reader Thread."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None
    
    def _reader_loop(self):
        """Reader Loop."""
        buffer = ""
        
        while self.running:
            try:
                if self.socket:
                    self.socket.settimeout(0.1)
                    
                    try:
                        data = self.socket.recv(1024)
                        
                        if not data:
                            self.connected = False
                            break
                        
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if line:
                                self.response_queue.put(line)
                    
                    except socket.timeout:
                        pass
                
            except Exception as e:
                logger.error(f"Reader error: {e}")
                self.connected = False
                break
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected
    
    @staticmethod
    def scan_network(timeout: float = 3.0) -> List[str]:
        """Scan for ESP32 devices."""
        found = []
        
        # Try common IPs
        test_ips = [
            '192.168.4.1',  # ESP32 AP default
            '192.168.1.1',
            'roarm.local'
        ]
        
        for ip in test_ips:
            try:
                # Quick connection test
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                
                # Try common ports
                for port in [8080, 80, 23]:
                    result = sock.connect_ex((ip, port))
                    if result == 0:
                        found.append(ip)
                        break
                
                sock.close()
            except:
                pass
        
        return found
    
    @staticmethod
    def list_ports() -> List[str]:
        """Compatibility with SerialManager."""
        return WiFiManager.scan_network()
    
    @staticmethod
    def find_arduino_port() -> Optional[str]:
        """Compatibility with SerialManager."""
        devices = WiFiManager.scan_network()
        return devices[0] if devices else None
