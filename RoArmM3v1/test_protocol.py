#!/usr/bin/env python3
"""
RoArm M3 Command ID Finder
Findet die richtigen Command IDs für deinen RoArm
"""

import serial
import json
import time
import sys

class CommandFinder:
    def __init__(self, port="/dev/tty.usbserial-110", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.found_commands = {}
        
    def connect(self):
        """Connect to RoArm."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
                write_timeout=2.0
            )
            time.sleep(2)  # Arduino reset
            
            # Clear buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            print(f"✅ Connected to {self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def send_and_read(self, command_dict):
        """Send command and read response."""
        try:
            # Send command
            cmd_str = json.dumps(command_dict) + '\n'
            self.serial.write(cmd_str.encode('utf-8'))
            
            # Wait for response
            time.sleep(0.3)
            
            # Read all available data
            response = ""
            while self.serial.in_waiting:
                chunk = self.serial.read(self.serial.in_waiting)
                response += chunk.decode('utf-8', errors='ignore')
                time.sleep(0.01)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def parse_response(self, response):
        """Parse JSON response."""
        if not response:
            return None
        
        # Try to find JSON in response
        try:
            # Response might have multiple JSON objects
            json_objects = []
            
            # Find all {...} patterns
            import re
            pattern = r'\{[^{}]*\}'
            matches = re.findall(pattern, response)
            
            for match in matches:
                try:
                    obj = json.loads(match)
                    json_objects.append(obj)
                except:
                    pass
            
            return json_objects
        except:
            return None
    
    def analyze_status_response(self):
        """Analyze the status response format."""
        print("\n" + "="*60)
        print("ANALYZING STATUS RESPONSE")
        print("="*60)
        
        # The response you showed has T:1051
        # Let's try that as status command
        test_commands = [
            {"T": 1051},  # From your response
            {"T": 1050},  # Maybe nearby
            {"T": 1052},
            {"T": 1000},
            {"T": 1},     # Original
            {"T": 104},   # Waveshare docs
        ]
        
        for cmd in test_commands:
            print(f"\nTrying: {cmd}")
            response = self.send_and_read(cmd)
            
            if response:
                print(f"Response: {response[:200]}...")  # First 200 chars
                
                parsed = self.parse_response(response)
                if parsed:
                    for obj in parsed:
                        if 'T' in obj:
                            print(f"  Parsed T:{obj['T']}")
                            if 'x' in obj or 'b' in obj:
                                print(f"  ✅ This looks like a status response!")
                                print(f"  Position data found:")
                                if 'b' in obj:
                                    print(f"    Base: {obj.get('b', 'N/A')}")
                                    print(f"    Shoulder: {obj.get('s', 'N/A')}")
                                    print(f"    Elbow: {obj.get('e', 'N/A')}")
                                if 'x' in obj:
                                    print(f"    X: {obj.get('x', 'N/A')}")
                                    print(f"    Y: {obj.get('y', 'N/A')}")
                                    print(f"    Z: {obj.get('z', 'N/A')}")
                                
                                self.found_commands['status'] = cmd['T']
    
    def find_movement_command(self):
        """Find the correct movement command."""
        print("\n" + "="*60)
        print("FINDING MOVEMENT COMMAND")
        print("="*60)
        print("\n⚠️  The arm will try to move - ensure it has space!")
        input("Press ENTER when ready...")
        
        # Get current position first
        print("\nGetting current position...")
        response = self.send_and_read({"T": 1051})
        current_pos = self.parse_response(response)
        
        if current_pos and len(current_pos) > 0:
            pos = current_pos[0]
            print(f"Current position: b={pos.get('b', 0):.3f}")
        
        # Try different movement commands
        test_commands = [
            # Most likely candidates based on Waveshare docs
            {"T": 104, "b": 0.1},  # T:104 is often move in Waveshare
            {"T": 102, "b": 0.1},  # Original guess
            {"T": 103, "b": 0.1},
            {"T": 105, "b": 0.1},
            
            # Different parameter names
            {"T": 104, "base": 0.1},
            {"T": 104, "servo1": 1600},
            {"T": 104, "s1": 1600},
            {"T": 104, "p1": 1600},
            
            # Array format
            {"T": 104, "servos": [1], "positions": [1600]},
            {"T": 104, "joints": {"base": 0.1}},
            
            # Cartesian movement
            {"T": 104, "x": 100, "y": 100, "z": 100},
            {"T": 105, "x": 100, "y": 100, "z": 100},
            
            # Other T values from 100-110
            {"T": 100, "b": 0.1},
            {"T": 101, "b": 0.1},
            {"T": 106, "b": 0.1},
            {"T": 107, "b": 0.1},
            {"T": 108, "b": 0.1},
            {"T": 109, "b": 0.1},
            {"T": 110, "b": 0.1},
        ]
        
        for cmd in test_commands:
            print(f"\nTrying: {cmd}")
            
            # Send command
            response = self.send_and_read(cmd)
            
            if response:
                print(f"Response: {response[:100]}...")
            
            # Wait a bit
            time.sleep(2)
            
            # Check if position changed
            response = self.send_and_read({"T": 1051})
            new_pos = self.parse_response(response)
            
            if new_pos and len(new_pos) > 0:
                new_b = new_pos[0].get('b', 0)
                
                moved = input(f"Did the base move? (y/n): ").lower() == 'y'
                
                if moved:
                    print(f"  ✅ FOUND MOVEMENT COMMAND: {cmd}")
                    self.found_commands['move'] = cmd['T']
                    
                    # Move back
                    cmd_back = cmd.copy()
                    if 'b' in cmd_back:
                        cmd_back['b'] = 0
                    elif 'base' in cmd_back:
                        cmd_back['base'] = 0
                    elif 'servo1' in cmd_back:
                        cmd_back['servo1'] = 1500
                    
                    self.send_and_read(cmd_back)
                    return cmd
            
            time.sleep(0.5)
        
        return None
    
    def find_led_command(self):
        """Find LED control command."""
        print("\n" + "="*60)
        print("FINDING LED COMMAND")
        print("="*60)
        
        # Try different LED commands
        test_commands = [
            # Standard guesses
            {"T": 51, "led": 1},
            {"T": 51, "led": 1, "brightness": 255},
            {"T": 50, "led": 1},
            {"T": 52, "led": 1},
            
            # Based on pattern (if status is 1051, LED might be 105X)
            {"T": 1050, "led": 1},
            {"T": 1051, "led": 1},
            {"T": 1052, "led": 1},
            {"T": 1053, "led": 1},
            
            # Other formats
            {"T": 7, "led": 1},  # Sometimes it's single digit
            {"T": 107, "led": 1},
            {"T": 207, "led": 1},
            
            # Different parameter names
            {"T": 51, "LED": 1},
            {"T": 51, "l": 1},
            {"T": 51, "light": 1},
            {"T": 51, "state": 1},
            
            # PWM/brightness variants
            {"T": 51, "pwm": 255},
            {"T": 51, "value": 255},
        ]
        
        for cmd in test_commands:
            print(f"\nTrying LED ON: {cmd}")
            self.send_and_read(cmd)
            time.sleep(0.5)
            
            # Try to turn off
            cmd_off = cmd.copy()
            for key in cmd_off:
                if key != 'T' and isinstance(cmd_off[key], (int, float)):
                    cmd_off[key] = 0
            
            print(f"Trying LED OFF: {cmd_off}")
            self.send_and_read(cmd_off)
            time.sleep(0.5)
            
            blinked = input("Did the LED blink? (y/n): ").lower() == 'y'
            
            if blinked:
                print(f"  ✅ FOUND LED COMMAND: T={cmd['T']}")
                self.found_commands['led'] = cmd['T']
                return cmd
        
        return None
    
    def find_torque_command(self):
        """Find torque enable/disable command."""
        print("\n" + "="*60)
        print("FINDING TORQUE COMMAND")
        print("="*60)
        
        test_commands = [
            {"T": 210, "enabled": 1},
            {"T": 210, "torque": 1},
            {"T": 208, "enabled": 1},
            {"T": 1210, "enabled": 1},
            {"T": 8, "enabled": 1},
            {"T": 108, "enabled": 1},
            {"T": 200, "enabled": 1},
            {"T": 201, "enabled": 1},
        ]
        
        for cmd in test_commands:
            print(f"\nTrying torque enable: {cmd}")
            self.send_and_read(cmd)
            time.sleep(0.5)
            
            # Try to move manually
            can_move = input("Try to move the arm manually. Is it stiff/locked? (y/n): ").lower() == 'y'
            
            if can_move:
                # Try disable
                cmd_off = cmd.copy()
                for key in cmd_off:
                    if key != 'T':
                        cmd_off[key] = 0
                
                print(f"Trying torque disable: {cmd_off}")
                self.send_and_read(cmd_off)
                time.sleep(0.5)
                
                can_move_now = input("Can you move it freely now? (y/n): ").lower() == 'y'
                
                if can_move_now:
                    print(f"  ✅ FOUND TORQUE COMMAND: T={cmd['T']}")
                    self.found_commands['torque'] = cmd['T']
                    
                    # Re-enable
                    self.send_and_read(cmd)
                    return cmd
        
        return None
    
    def save_findings(self):
        """Save found commands to file."""
        print("\n" + "="*60)
        print("COMMAND MAPPING RESULTS")
        print("="*60)
        
        if not self.found_commands:
            print("❌ No commands found")
            return
        
        print("\nFound commands:")
        for name, t_value in self.found_commands.items():
            print(f"  {name:10s}: T={t_value}")
        
        # Create Python constants file
        code = """# RoArm M3 Command IDs (Auto-detected)
# Generated by find_commands.py

COMMANDS = {
"""
        
        for name, t_value in self.found_commands.items():
            code += f'    "{name.upper()}": {t_value},\n'
        
        code += """}

# Example usage:
# {"T": COMMANDS["STATUS"]}  # Query status
# {"T": COMMANDS["MOVE"], "b": 0.5}  # Move base
"""
        
        filename = "roarm_commands.py"
        with open(filename, 'w') as f:
            f.write(code)
        
        print(f"\n✅ Saved to {filename}")
        
        # Also show how to update existing code
        print("\nUpdate your core/constants.py:")
        print("-" * 40)
        print("COMMANDS = {")
        for name, t_value in self.found_commands.items():
            print(f'    "{name.upper()}": {t_value},')
        print("}")
    
    def run_all(self):
        """Run all tests."""
        if not self.connect():
            return
        
        print("\n🔍 FINDING CORRECT COMMAND IDS")
        print("="*60)
        
        # Clear buffer first
        self.serial.reset_input_buffer()
        
        # 1. Analyze status
        self.analyze_status_response()
        
        # 2. Find LED
        self.find_led_command()
        
        # 3. Find movement
        self.find_movement_command()
        
        # 4. Find torque
        self.find_torque_command()
        
        # Save results
        self.save_findings()
        
        if self.serial:
            self.serial.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/tty.usbserial-110')
    parser.add_argument('--baud', type=int, default=115200)
    args = parser.parse_args()
    
    finder = CommandFinder(args.port, args.baud)
    finder.run_all()


if __name__ == "__main__":
    main()
