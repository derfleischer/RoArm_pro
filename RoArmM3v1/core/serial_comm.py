#!/usr/bin/env python3
"""
Serial Communication Manager für RoArm M3
Basiert auf der FUNKTIONIERENDEN Kommunikation aus RoArm3v7.py
"""

import serial
import serial.tools.list_ports
import json
import time
import re
import logging

logger = logging.getLogger('RoArm.Serial')


class SerialManager:
    """Serial communication manager - based on working RoArm3v7.py"""
    
    def __init__(self, port="/dev/tty.usbserial-110", baudrate=115200, timeout=2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        
    def connect(self):
        """Connect to RoArm - EXACTLY like RoArm3v7.py"""
        try:
            if self.ser and self.ser.is_open:
                logger.info("Already connected")
                return True
            
            # Auto-detect if needed
            if "auto" in self.port.lower():
                self.port = self._auto_detect_port()
                logger.info(f"Auto-detected port: {self.port}")
            
            logger.info(f"🔌 Connecting to {self.port}...")
            
            # EXACT connection parameters from RoArm3v7.py
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Connection stabilization - EXACT timing
            time.sleep(2)
            self._flush_buffers()
            
            self.connected = True
            logger.info("✅ Connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Serial connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from RoArm"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                logger.info("🔌 Disconnected")
            except:
                pass
        self.connected = False
    
    def _flush_buffers(self):
        """Clean buffer management - EXACT from RoArm3v7.py"""
        if not self.ser:
            return
        
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            time.sleep(0.15)  # EXACT timing from RoArm3v7.py
            
            # Clear any remaining data
            while self.ser.in_waiting > 0:
                self.ser.read(self.ser.in_waiting)
                time.sleep(0.05)
                
        except Exception as e:
            logger.warning(f"Buffer flush error: {e}")
    
    def send_command(self, command_dict, wait_time=None, retries=3):
        """
        Send command EXACTLY like RoArm3v7.py's send_command_no_abort_check
        This is the version that WORKS!
        """
        if not self.ser or not self.ser.is_open:
            logger.error("❌ Not connected to RoArm")
            return None
        
        wait_time = wait_time or 0.05
        
        for attempt in range(retries):
            try:
                # Prepare command - EXACT format
                json_command = json.dumps(command_dict, separators=(',', ':'))
                command_bytes = (json_command + '\n').encode('utf-8')
                
                # Send command - EXACT sequence
                self._flush_buffers()
                self.ser.write(command_bytes)
                self.ser.flush()
                
                # Wait for completion
                time.sleep(wait_time)
                
                # Read response
                response = self._read_response()
                if response:
                    return response
                
                # Consider success for movement/LED commands
                if command_dict.get("T") in [102, 51]:  # Movement, LED
                    return {"status": "sent"}
                
            except Exception as e:
                logger.warning(f"Command attempt {attempt+1}/{retries} failed: {e}")
                
                if attempt < retries - 1:
                    time.sleep(0.2)
        
        logger.error(f"❌ Command failed after {retries} attempts")
        return None
    
    def _read_response(self):
        """Read and parse response - EXACT from RoArm3v7.py"""
        try:
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                return self._extract_json_response(response)
        except Exception as e:
            logger.debug(f"Response read error: {e}")
        
        return None
    
    def _extract_json_response(self, raw_data):
        """Extract valid JSON from response - EXACT from RoArm3v7.py"""
        if not raw_data:
            return None
        
        # Clean data
        clean_data = re.sub(r'[\x00-\x1f\x7f]', '', raw_data)
        
        # Find JSON patterns
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, clean_data)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # Look for valid response indicators
                if any(key in parsed for key in ['T', 'x', 'y', 'z', 'code', 'status']):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _auto_detect_port(self):
        """Auto-detect USB serial port"""
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
