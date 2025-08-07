#!/usr/bin/env python3
"""
RoArm M3 Protocol Analyzer & Reverse Engineering Tool
Findet das tatsächliche Protokoll durch Analyse der Responses
"""

import serial
import json
import time
import sys
import re
from collections import defaultdict
import threading

class ProtocolAnalyzer:
    def __init__(self, port="/dev/tty.usbserial-110", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.responses = []
        self.monitoring = False
        self.monitor_thread = None
        
    def connect(self):
        """Connect to RoArm."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=2.0
            )
            time.sleep(2)
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            print(f"✅ Connected to {self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def analyze_continuous_stream(self, duration=5):
        """Analyze the continuous data stream from RoArm."""
        print("\n" + "="*60)
        print("ANALYZING CONTINUOUS DATA STREAM")
        print("="*60)
        print(f"Recording for {duration} seconds...\n")
        
        start_time = time.time()
        raw_data = ""
        json_objects = []
        
        while time.time() - start_time < duration:
            if self.serial.in_waiting:
                chunk = self.serial.read(self.serial.in_waiting)
                raw_data += chunk.decode('utf-8', errors='ignore')
            time.sleep(0.01)
        
        print(f"Collected {len(raw_data)} bytes of data")
        
        # Parse JSON objects
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, raw_data)
        
        for match in matches:
            try:
                obj = json.loads(match)
                json_objects.append(obj)
            except:
                pass
        
        print(f"Found {len(json_objects)} JSON objects")
        
        if json_objects:
            # Analyze structure
            print("\nData structure analysis:")
            print("-" * 40)
            
            # Get unique T values
            t_values = set(obj.get('T') for obj in json_objects if 'T' in obj)
            print(f"Unique T values: {sorted(t_values)}")
            
            # Analyze first object
            sample = json_objects[0]
            print(f"\nSample object keys: {list(sample.keys())}")
            print(f"\nSample data:")
            for key, value in sample.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            
            # Check if values change
            if len(json_objects) > 1:
                print("\nValue stability check:")
                for key in sample.keys():
                    if key == 'T':
                        continue
                    values = [obj.get(key) for obj in json_objects if key in obj]
                    if len(set(values)) == 1:
                        print(f"  {key}: STATIC (always {values[0]})")
                    else:
                        print(f"  {key}: DYNAMIC (range: {min(values):.3f} to {max(values):.3f})")
        
        return json_objects
    
    def test_interrupt_stream(self):
        """Test if we need to stop the stream before sending commands."""
        print("\n" + "="*60)
        print("TESTING STREAM INTERRUPTION")
        print("="*60)
        
        # Try different methods to stop the stream
        test_commands = [
            b'\x03',           # Ctrl+C
            b'\x04',           # Ctrl+D (EOF)
            b'\x1A',           # Ctrl+Z
            b'\r\n',           # Newline
            b'STOP\r\n',       # Text command
            b'STOP\n',
            b'stop\r\n',
            b'S\r\n',
            b'0\r\n',
            b'#\r\n',
            b'!\r\n',
            b'Q\r\n',          # Quit
            b'q\r\n',
            b'X\r\n',          # Exit
            b'x\r\n',
            b'\x00',           # Null byte
            b'\xFF\xFF\xFF',   # Common stop sequence
            b'+++',            # AT command escape
        ]
        
        for cmd in test_commands:
            print(f"\nTrying stop command: {cmd}")
            
            # Clear buffer
            self.serial.reset_input_buffer()
            
            # Send stop command
            self.serial.write(cmd)
            time.sleep(0.5)
            
            # Check if stream stopped
            data_before = self.serial.in_waiting
            time.sleep(1)
            data_after = self.serial.in_waiting
            
            if data_after == data_before or data_after < 50:
                print(f"  ✅ Stream might have stopped! (bytes waiting: {data_after})")
                
                # Try to send a test command
                test_cmd = json.dumps({"T": 1}) + '\n'
                self.serial.write(test_cmd.encode('utf-8'))
                time.sleep(0.5)
                
                response = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                print(f"  Response after test: {response[:100]}")
                
                if "1051" not in response:
                    print(f"  🎉 FOUND STOP COMMAND: {cmd}")
                    return cmd
            else:
                print(f"  ❌ Stream still active (bytes: {data_before} -> {data_after})")
        
        return None
    
    def test_command_formats(self, stop_cmd=None):
        """Test different command formats after stopping stream."""
        print("\n" + "="*60)
        print("TESTING COMMAND FORMATS")
        print("="*60)
        
        # Stop stream if we have a stop command
        if stop_cmd:
            print(f"Stopping stream with: {stop_cmd}")
            self.serial.write(stop_cmd)
            time.sleep(0.5)
            self.serial.reset_input_buffer()
        
        # Test different command formats
        test_formats = [
            # Standard JSON formats
            ('{"T":51,"led":1}', "JSON T:51 LED"),
            ('{"cmd":"led","state":1}', "JSON cmd:led"),
            ('{"command":"led","value":1}', "JSON command:led"),
            ('{"LED":1}', "JSON LED only"),
            ('{"led":1}', "JSON led only"),
            
            # Without T field
            ('{"led":1,"brightness":255}', "JSON no T field"),
            ('{"set":"led","value":1}', "JSON set:led"),
            
            # Text commands
            ('LED 1\r\n', "Text: LED 1"),
            ('LED ON\r\n', "Text: LED ON"),
            ('L1\r\n', "Text: L1"),
            ('led:1\r\n', "Text: led:1"),
            ('led=1\r\n', "Text: led=1"),
            (':LED1\r\n', "Text: :LED1"),
            
            # Single character commands
            ('L\r\n', "Single: L"),
            ('1\r\n', "Single: 1"),
            
            # Binary/Hex commands
            (b'\x01\x01', "Binary: 0x01 0x01"),
            (b'\x07\x01', "Binary: 0x07 0x01"),
            (b'L\x01', "Binary: L 0x01"),
        ]
        
        for cmd, description in test_formats:
            print(f"\nTesting: {description}")
            print(f"  Command: {cmd if isinstance(cmd, bytes) else cmd.strip()}")
            
            # Clear buffer
            self.serial.reset_input_buffer()
            
            # Send command
            if isinstance(cmd, bytes):
                self.serial.write(cmd)
            else:
                self.serial.write(cmd.encode('utf-8'))
            
            time.sleep(0.3)
            
            # Try OFF command
            if 'led' in str(cmd).lower() or 'LED' in str(cmd):
                if isinstance(cmd, str):
                    off_cmd = cmd.replace('1', '0').replace('ON', 'OFF')
                    self.serial.write(off_cmd.encode('utf-8'))
                
                time.sleep(0.3)
            
            worked = input("  Did the LED blink? (y/n): ").lower() == 'y'
            
            if worked:
                print(f"  🎉 WORKING FORMAT: {description}")
                return cmd
        
        return None
    
    def manual_test_mode(self):
        """Manual testing mode where user can type commands."""
        print("\n" + "="*60)
        print("MANUAL TEST MODE")
        print("="*60)
        print("Type commands to send (or 'quit' to exit)")
        print("Commands will be sent as-is with newline added")
        print("-" * 40)
        
        while True:
            cmd = input("\n> ")
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                break
            
            # Clear buffer
            self.serial.reset_input_buffer()
            
            # Send command
            if not cmd.endswith('\n'):
                cmd += '\n'
            
            self.serial.write(cmd.encode('utf-8'))
            
            # Wait for response
            time.sleep(0.5)
            
            if self.serial.in_waiting:
                response = self.serial.read(self.serial.in_waiting)
                print(f"Response: {response.decode('utf-8', errors='ignore')[:200]}")
            else:
                print("No response")
    
    def monitor_with_input(self):
        """Monitor mode that accepts input while showing stream."""
        print("\n" + "="*60)
        print("INTERACTIVE MONITOR MODE")
        print("="*60)
        print("Stream is shown above, type commands below")
        print("Commands: 'led on', 'led off', 'move base 0.1', etc.")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        # Start monitoring in thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Accept commands
        while True:
            try:
                cmd = input()
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Parse simple commands
                if 'led on' in cmd.lower():
                    self._try_led(True)
                elif 'led off' in cmd.lower():
                    self._try_led(False)
                elif 'move' in cmd.lower():
                    parts = cmd.split()
                    if len(parts) >= 3:
                        joint = parts[1]
                        value = float(parts[2])
                        self._try_move(joint, value)
                else:
                    # Send as-is
                    if not cmd.endswith('\n'):
                        cmd += '\n'
                    self.serial.write(cmd.encode('utf-8'))
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            if self.serial.in_waiting:
                data = self.serial.read(self.serial.in_waiting)
                print(f"[STREAM] {data.decode('utf-8', errors='ignore')}", end='')
            time.sleep(0.01)
    
    def _try_led(self, on):
        """Try various LED commands."""
        commands = [
            f'{{"T":51,"led":{1 if on else 0}}}',
            f'{{"led":{1 if on else 0}}}',
            f'LED {"ON" if on else "OFF"}',
            f'L{1 if on else 0}',
        ]
        
        for cmd in commands:
            self.serial.write((cmd + '\n').encode('utf-8'))
            time.sleep(0.1)
    
    def _try_move(self, joint, value):
        """Try various movement commands."""
        joint_map = {
            'base': 'b',
            'shoulder': 's',
            'elbow': 'e',
            'wrist': 't',
            'roll': 'r',
            'hand': 'g'
        }
        
        key = joint_map.get(joint, joint[0])
        
        commands = [
            f'{{"T":104,"{key}":{value}}}',
            f'{{"T":102,"{key}":{value}}}',
            f'{{"{key}":{value}}}',
            f'MOVE {joint.upper()} {value}',
        ]
        
        for cmd in commands:
            self.serial.write((cmd + '\n').encode('utf-8'))
            time.sleep(0.1)
    
    def run_complete_analysis(self):
        """Run complete protocol analysis."""
        if not self.connect():
            return
        
        print("\n" + "="*80)
        print("ROARM M3 COMPLETE PROTOCOL ANALYSIS")
        print("="*80)
        
        # Step 1: Analyze continuous stream
        print("\n[Step 1] Analyzing data stream...")
        json_objects = self.analyze_continuous_stream(3)
        
        # Step 2: Try to interrupt stream
        print("\n[Step 2] Testing stream interruption...")
        stop_cmd = self.test_interrupt_stream()
        
        if stop_cmd:
            print(f"\n✅ Found stop command: {stop_cmd}")
            
            # Step 3: Test commands with stopped stream
            print("\n[Step 3] Testing commands with stopped stream...")
            working_cmd = self.test_command_formats(stop_cmd)
            
            if working_cmd:
                print(f"\n🎉 SUCCESS! Working command format: {working_cmd}")
        else:
            print("\n⚠️ Could not stop stream, trying commands anyway...")
            
            # Test sending commands while stream is active
            working_cmd = self.test_command_formats()
        
        # Step 4: Manual testing
        print("\n[Step 4] Manual testing mode")
        manual = input("Enter manual test mode? (y/n): ").lower() == 'y'
        
        if manual:
            self.manual_test_mode()
        
        # Step 5: Interactive monitor
        monitor = input("\nTry interactive monitor mode? (y/n): ").lower() == 'y'
        
        if monitor:
            self.monitor_with_input()
        
        if self.serial:
            self.serial.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/tty.usbserial-110')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--mode', choices=['full', 'manual', 'monitor'], default='full')
    args = parser.parse_args()
    
    analyzer = ProtocolAnalyzer(args.port, args.baud)
    
    if args.mode == 'full':
        analyzer.run_complete_analysis()
    elif args.mode == 'manual':
        if analyzer.connect():
            analyzer.manual_test_mode()
    elif args.mode == 'monitor':
        if analyzer.connect():
            analyzer.monitor_with_input()


if __name__ == "__main__":
    main()
