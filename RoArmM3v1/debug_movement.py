#!/usr/bin/env python3
"""
RoArm M3 Movement Debug Script
Testet die Kommunikation und findet das richtige Command-Format
"""

import serial
import json
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ANSI Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.END}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


class RoArmDebugger:
    """Debug tool for RoArm communication."""
    
    def __init__(self, port="/dev/tty.usbserial-110", baudrate=115200):
        """Initialize debugger."""
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        
    def connect(self):
        """Connect to RoArm."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0,
                write_timeout=2.0
            )
            time.sleep(2)  # Wait for Arduino to reset
            print_success(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print_error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from RoArm."""
        if self.serial:
            self.serial.close()
            print_info("Disconnected")
    
    def send_raw(self, data):
        """Send raw data and print response."""
        if not self.serial:
            print_error("Not connected")
            return None
        
        try:
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Send data
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            print_info(f"Sending: {data}")
            self.serial.write(data)
            self.serial.flush()
            
            # Wait for response
            time.sleep(0.5)
            
            # Read response
            response = b''
            while self.serial.in_waiting:
                response += self.serial.read(self.serial.in_waiting)
                time.sleep(0.01)
            
            if response:
                print_success(f"Response: {response}")
                try:
                    # Try to decode as JSON
                    json_response = json.loads(response.decode('utf-8'))
                    print_info(f"Decoded: {json.dumps(json_response, indent=2)}")
                except:
                    # Just print raw
                    print_info(f"Raw: {response.decode('utf-8', errors='ignore')}")
            else:
                print_warning("No response received")
            
            return response
            
        except Exception as e:
            print_error(f"Send failed: {e}")
            return None
    
    def test_formats(self):
        """Test different command formats."""
        print_header("TESTING COMMAND FORMATS")
        
        # Different format variations to test
        test_commands = [
            # Format 1: Simple JSON
            '{"T":1}',
            
            # Format 2: JSON with newline
            '{"T":1}\n',
            
            # Format 3: JSON with carriage return
            '{"T":1}\r\n',
            
            # Format 4: Waveshare format (from their examples)
            '{"T":104,"x":0,"y":120,"z":28}\n',
            
            # Format 5: Status query variations
            '{"cmd":"status"}',
            '{"command":"status"}',
            '{"C":"status"}',
            
            # Format 6: Movement command variations
            '{"T":102,"base":0}',
            '{"T":102,"servo1":1500}',
            '{"T":102,"S1":1500}',
            
            # Format 7: Raw servo commands (if using servo IDs)
            '{"T":102,"1":1500,"2":1500,"3":1500,"4":1500,"5":1500,"6":1500}',
            
            # Format 8: Alternative movement format
            '{"cmd":"move","base":0,"shoulder":0}',
            
            # Format 9: G-code style (some versions use this)
            'G0 X0 Y120 Z28\n',
            
            # Format 10: Direct servo control
            '#1P1500T1000\r\n',  # SSC-32 format
        ]
        
        for i, cmd in enumerate(test_commands, 1):
            print(f"\n{Colors.BOLD}Test {i}: {cmd.strip()}{Colors.END}")
            response = self.send_raw(cmd)
            time.sleep(1)
        
        print_info("\nCheck which format got a meaningful response!")
    
    def test_servo_control(self):
        """Test direct servo control."""
        print_header("TESTING SERVO CONTROL")
        
        print_info("Testing individual servo movement...")
        
        # Test each servo
        servos = {
            1: "Base",
            2: "Shoulder", 
            3: "Elbow",
            4: "Wrist",
            5: "Roll",
            6: "Hand/Gripper"
        }
        
        for servo_id, name in servos.items():
            print(f"\n{Colors.BOLD}Testing {name} (Servo {servo_id}){Colors.END}")
            
            # Try different command formats
            commands = [
                # Format A: JSON with servo ID
                f'{{"T":102,"servo{servo_id}":1500}}\n',
                
                # Format B: JSON with joint name
                f'{{"T":102,"{name.lower()}":0}}\n',
                
                # Format C: Direct servo command
                f'#{servo_id}P1500T1000\r\n',
                
                # Format D: Array format
                f'{{"T":102,"servos":[{servo_id}],"positions":[1500]}}\n',
            ]
            
            for cmd in commands:
                print(f"  Trying: {cmd.strip()}")
                response = self.send_raw(cmd)
                
                if response and b'ok' in response.lower():
                    print_success(f"  → This format works for {name}!")
                    break
                
                time.sleep(0.5)
    
    def test_movement_sequence(self):
        """Test a simple movement sequence."""
        print_header("TESTING MOVEMENT SEQUENCE")
        
        print_info("Attempting to move base joint...")
        
        # Create test sequence
        sequence = [
            ("Enable torque", '{"T":210,"enabled":1}\n'),
            ("Query status", '{"T":1}\n'),
            ("Move base to center", '{"T":102,"base":0}\n'),
            ("Wait", None),
            ("Move base right", '{"T":102,"base":30}\n'),
            ("Wait", None),
            ("Move base left", '{"T":102,"base":-30}\n'),
            ("Wait", None),
            ("Move base center", '{"T":102,"base":0}\n'),
        ]
        
        for step, cmd in sequence:
            print(f"\n{Colors.BOLD}{step}{Colors.END}")
            
            if cmd is None:
                time.sleep(2)
            else:
                self.send_raw(cmd)
                time.sleep(1)
    
    def test_led(self):
        """Test LED control to verify communication."""
        print_header("TESTING LED CONTROL")
        
        led_commands = [
            ('{"T":51,"led":1,"brightness":255}\n', "LED ON (bright)"),
            ('{"T":51,"led":0}\n', "LED OFF"),
            ('{"T":51,"led":1,"brightness":128}\n', "LED ON (dim)"),
            ('{"T":51,"led":0}\n', "LED OFF"),
        ]
        
        for cmd, description in led_commands:
            print(f"\n{description}")
            self.send_raw(cmd)
            time.sleep(1)
        
        print_info("If LED blinked, communication is working!")
    
    def find_working_format(self):
        """Interactive format finder."""
        print_header("INTERACTIVE FORMAT FINDER")
        
        print_info("Let's find the correct format together...")
        print_info("I'll send commands and you tell me if the arm moves.\n")
        
        # Test basic communication first
        print("1️⃣  First, testing LED (should blink)...")
        self.send_raw('{"T":51,"led":1}\n')
        time.sleep(1)
        self.send_raw('{"T":51,"led":0}\n')
        
        led_works = input("Did the LED blink? (y/n): ").lower() == 'y'
        
        if not led_works:
            print_warning("Basic communication not working. Check:")
            print("  - Correct port? (current: {})".format(self.port))
            print("  - Correct baudrate? (current: {})".format(self.baudrate))
            print("  - Power supply connected?")
            print("  - RoArm firmware version?")
            return
        
        print_success("Communication confirmed!")
        
        # Test movement
        print("\n2️⃣  Testing movement commands...")
        print_warning("⚠️  Make sure arm has space to move!")
        input("Press ENTER when ready...")
        
        movement_formats = [
            ('{"T":102,"base":0.5}\n', "Format 1: Radians"),
            ('{"T":102,"base":30}\n', "Format 2: Degrees"),
            ('{"T":102,"servo1":1700}\n', "Format 3: Servo PWM"),
            ('{"T":102,"positions":{"base":0.5}}\n', "Format 4: Nested positions"),
            ('{"T":104,"x":100,"y":100,"z":100}\n', "Format 5: Cartesian"),
        ]
        
        for cmd, description in movement_formats:
            print(f"\nTrying: {description}")
            print(f"Command: {cmd.strip()}")
            self.send_raw(cmd)
            time.sleep(2)
            
            moved = input("Did the base rotate? (y/n): ").lower() == 'y'
            
            if moved:
                print_success(f"✅ FOUND WORKING FORMAT: {description}")
                print_info(f"Use this format: {cmd.strip()}")
                
                # Return to center
                print("Returning to center...")
                if "radians" in description.lower():
                    self.send_raw('{"T":102,"base":0}\n')
                elif "degrees" in description.lower():
                    self.send_raw('{"T":102,"base":0}\n')
                elif "pwm" in description.lower():
                    self.send_raw('{"T":102,"servo1":1500}\n')
                
                return cmd
        
        print_error("No working format found. The issue might be:")
        print("  - Wrong command IDs (T values)")
        print("  - Different firmware version")
        print("  - Custom protocol")
        print("\nTry checking the Waveshare wiki or firmware documentation.")
    
    def monitor_mode(self):
        """Monitor serial communication."""
        print_header("SERIAL MONITOR MODE")
        print_info("Monitoring serial port... (Ctrl+C to exit)")
        print_info("Type commands to send, or just watch responses.\n")
        
        try:
            while True:
                # Check for incoming data
                if self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    print(f"{Colors.GREEN}← {data.decode('utf-8', errors='ignore')}{Colors.END}")
                
                # Check for user input (non-blocking would be better)
                # For now, this is simplified
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nMonitor mode stopped.")


def main():
    """Main debug function."""
    print_header("ROARM M3 MOVEMENT DEBUGGER")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/tty.usbserial-110', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baudrate')
    parser.add_argument('--mode', choices=['auto', 'test', 'servo', 'led', 'monitor', 'find'], 
                       default='auto', help='Debug mode')
    args = parser.parse_args()
    
    # Create debugger
    debugger = RoArmDebugger(args.port, args.baud)
    
    # Connect
    if not debugger.connect():
        print_error("Failed to connect. Check:")
        print("  1. Is the port correct? List ports with: ls /dev/tty.*")
        print("  2. Is another program using the port?")
        print("  3. Is the RoArm powered on?")
        return 1
    
    try:
        if args.mode == 'auto':
            # Run all tests
            debugger.test_led()
            time.sleep(2)
            debugger.find_working_format()
            
        elif args.mode == 'test':
            debugger.test_formats()
            
        elif args.mode == 'servo':
            debugger.test_servo_control()
            
        elif args.mode == 'led':
            debugger.test_led()
            
        elif args.mode == 'monitor':
            debugger.monitor_mode()
            
        elif args.mode == 'find':
            debugger.find_working_format()
        
        print_header("DEBUG SESSION COMPLETE")
        print_info("Results:")
        print("  - If LED works but movement doesn't: Command format issue")
        print("  - If nothing works: Communication issue")
        print("  - If some formats work: Update controller.py with correct format")
        
    finally:
        debugger.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
