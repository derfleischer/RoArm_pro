"""
Serial port detection utilities
Optimized for macOS but works cross-platform
"""

import sys
import serial.tools.list_ports
from typing import List, Optional

def get_default_port() -> str:
    """
    Auto-detect the most likely serial port for RoArm
    Priority order:
    1. USB serial ports with 'usbserial' in name
    2. Other /dev/cu.* ports (macOS)
    3. /dev/ttyUSB* (Linux)
    4. COM* ports (Windows)
    """
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        # Return platform-specific default
        if sys.platform == "darwin":  # macOS
            return "/dev/cu.usbserial-10"
        elif sys.platform.startswith("linux"):
            return "/dev/ttyUSB0"
        else:  # Windows
            return "COM3"
    
    # macOS: Look for usbserial devices first
    if sys.platform == "darwin":
        usb_ports = [p.device for p in ports if 'usbserial' in p.device.lower()]
        if usb_ports:
            return usb_ports[0]
            
        cu_ports = [p.device for p in ports if p.device.startswith('/dev/cu.')]
        if cu_ports:
            return cu_ports[0]
    
    # Linux: Look for ttyUSB devices
    elif sys.platform.startswith("linux"):
        tty_ports = [p.device for p in ports if 'ttyUSB' in p.device]
        if tty_ports:
            return tty_ports[0]
    
    # Windows: Look for COM ports
    else:
        com_ports = [p.device for p in ports if p.device.startswith('COM')]
        if com_ports:
            # Prefer higher COM ports (usually USB adapters)
            return sorted(com_ports)[-1]
    
    # Fallback to first available port
    return ports[0].device

def list_available_ports() -> List[dict]:
    """
    List all available serial ports with details
    Returns list of dicts with device, description, and hwid
    """
    ports = []
    
    for port in serial.tools.list_ports.comports():
        port_info = {
            'device': port.device,
            'description': port.description or 'Unknown',
            'hwid': port.hwid or 'Unknown',
            'is_usb': 'USB' in port.description or 'USB' in port.hwid
        }
        ports.append(port_info)
    
    # Sort by device name
    ports.sort(key=lambda x: x['device'])
    
    return ports

def find_roarm_port() -> Optional[str]:
    """
    Try to find a port that likely has RoArm connected
    This is a heuristic approach based on common patterns
    """
    ports = list_available_ports()
    
    # Look for specific patterns that indicate RoArm
    patterns = [
        'usbserial',    # Common macOS pattern
        'CH340',        # Common USB-Serial chip
        'CP210',        # Another common chip
        'FTDI',         # FTDI chips
        'Arduino',      # Sometimes shows as Arduino
    ]
    
    for port in ports:
        device_lower = port['device'].lower()
        desc_lower = port['description'].lower()
        
        for pattern in patterns:
            if pattern.lower() in device_lower or pattern.lower() in desc_lower:
                return port['device']
    
    # If no pattern matched, return None
    return None

def test_port(port: str, baudrate: int = 115200) -> bool:
    """
    Test if a port can be opened
    Returns True if successful, False otherwise
    """
    try:
        test_serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1.0
        )
        test_serial.close()
        return True
    except:
        return False
