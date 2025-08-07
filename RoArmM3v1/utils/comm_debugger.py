#!/usr/bin/env python3
"""
RoArm M3 Communication Debugger
Umfassendes System zum Debuggen der Arm-Kommunikation
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue
import hashlib
from datetime import datetime

class SignalType(Enum):
    """Signal-Typen für Analyse."""
    COMMAND = "CMD"      # Gesendeter Befehl
    RESPONSE = "RSP"     # Empfangene Antwort
    ERROR = "ERR"        # Fehler
    TIMEOUT = "TMO"      # Timeout
    INVALID = "INV"      # Ungültiges Signal


@dataclass
class Signal:
    """Ein Kommunikations-Signal."""
    timestamp: float
    direction: str  # 'TX' or 'RX'
    signal_type: SignalType
    data: Any
    raw_bytes: Optional[bytes] = None
    checksum: Optional[str] = None
    response_time: Optional[float] = None
    error_msg: Optional[str] = None


class CommunicationDebugger:
    """
    Hauptklasse für Kommunikations-Debugging.
    Überwacht, loggt und analysiert ALLE Signale.
    """
    
    def __init__(self, controller=None):
        """Initialize debugger."""
        self.controller = controller
        self.enabled = True
        
        # Signal storage
        self.signal_history: List[Signal] = []
        self.max_history = 1000
        
        # Statistics
        self.stats = {
            'commands_sent': 0,
            'responses_received': 0,
            'timeouts': 0,
            'errors': 0,
            'invalid_signals': 0,
            'avg_response_time': 0,
            'last_error': None
        }
        
        # Live monitoring
        self.monitor_queue = Queue()
        self.monitor_thread = None
        self.monitoring = False
        
        # Pattern detection
        self.expected_responses = {}  # cmd_id -> expected response
        self.response_patterns = {}   # Pattern matching
        
        # Error tracking
        self.error_patterns = []
        self.recurring_errors = {}
        
    def wrap_serial_manager(self, serial_manager):
        """
        Wraps SerialManager to intercept ALL communication.
        """
        original_send = serial_manager.send_command
        original_read = serial_manager.read_line
        
        def debug_send_command(command: Dict, wait_response: bool = False, timeout: float = 2.0):
            """Wrapped send_command with debugging."""
            
            # Log outgoing command
            signal = Signal(
                timestamp=time.time(),
                direction='TX',
                signal_type=SignalType.COMMAND,
                data=command,
                raw_bytes=json.dumps(command).encode() if command else None
            )
            
            # Calculate checksum
            if signal.raw_bytes:
                signal.checksum = hashlib.md5(signal.raw_bytes).hexdigest()[:8]
            
            self._log_signal(signal)
            
            # Show in console if monitoring
            if self.monitoring:
                self._print_signal(signal)
            
            # Send actual command
            start_time = time.time()
            try:
                response = original_send(command, wait_response, timeout)
                response_time = time.time() - start_time
                
                if wait_response:
                    if response:
                        # Log response
                        resp_signal = Signal(
                            timestamp=time.time(),
                            direction='RX',
                            signal_type=SignalType.RESPONSE,
                            data=response,
                            raw_bytes=response.encode() if response else None,
                            response_time=response_time
                        )
                        self._log_signal(resp_signal)
                        
                        # Verify response
                        self._verify_response(command, response)
                    else:
                        # Timeout
                        timeout_signal = Signal(
                            timestamp=time.time(),
                            direction='RX',
                            signal_type=SignalType.TIMEOUT,
                            data=None,
                            error_msg=f"Timeout after {timeout}s waiting for response to {command.get('T', '?')}"
                        )
                        self._log_signal(timeout_signal)
                        self.stats['timeouts'] += 1
                
                return response
                
            except Exception as e:
                # Log error
                error_signal = Signal(
                    timestamp=time.time(),
                    direction='TX',
                    signal_type=SignalType.ERROR,
                    data=command,
                    error_msg=str(e)
                )
                self._log_signal(error_signal)
                self.stats['errors'] += 1
                raise
        
        # Replace methods
        serial_manager.send_command = debug_send_command
        
        return serial_manager
    
    def _log_signal(self, signal: Signal):
        """Log a signal to history."""
        self.signal_history.append(signal)
        
        # Limit history size
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
        
        # Update stats
        if signal.signal_type == SignalType.COMMAND:
            self.stats['commands_sent'] += 1
        elif signal.signal_type == SignalType.RESPONSE:
            self.stats['responses_received'] += 1
            if signal.response_time:
                # Update average response time
                avg = self.stats['avg_response_time']
                n = self.stats['responses_received']
                self.stats['avg_response_time'] = (avg * (n-1) + signal.response_time) / n
        elif signal.signal_type == SignalType.ERROR:
            self.stats['last_error'] = signal.error_msg
    
    def _verify_response(self, command: Dict, response: str):
        """
        Verify if response matches expected pattern.
        """
        cmd_type = command.get('T', 0)
        
        # Define expected responses
        expected_patterns = {
            1: ['positions', 'status'],      # Status query
            51: ['led', 'ok'],               # LED control
            102: ['move', 'ok'],             # Joint control
            210: ['torque', 'enabled']       # Torque control
        }
        
        if cmd_type in expected_patterns:
            expected = expected_patterns[cmd_type]
            
            # Check if response contains expected keywords
            response_valid = False
            for keyword in expected:
                if keyword in response.lower():
                    response_valid = True
                    break
            
            if not response_valid:
                # Unexpected response
                signal = Signal(
                    timestamp=time.time(),
                    direction='RX',
                    signal_type=SignalType.INVALID,
                    data=response,
                    error_msg=f"Unexpected response for command {cmd_type}: {response[:50]}"
                )
                self._log_signal(signal)
                self.stats['invalid_signals'] += 1
    
    def _print_signal(self, signal: Signal):
        """Print signal in formatted way."""
        timestamp = datetime.fromtimestamp(signal.timestamp).strftime('%H:%M:%S.%f')[:-3]
        
        if signal.direction == 'TX':
            prefix = "→ TX"
            color = '\033[94m'  # Blue
        else:
            prefix = "← RX"
            color = '\033[92m'  # Green
        
        if signal.signal_type == SignalType.ERROR:
            color = '\033[91m'  # Red
        elif signal.signal_type == SignalType.TIMEOUT:
            color = '\033[93m'  # Yellow
        
        print(f"{color}[{timestamp}] {prefix} {signal.signal_type.value}: ", end='')
        
        if signal.data:
            if isinstance(signal.data, dict):
                print(json.dumps(signal.data, separators=(',', ':')))
            else:
                print(signal.data[:100] if len(str(signal.data)) > 100 else signal.data)
        
        if signal.error_msg:
            print(f"    ERROR: {signal.error_msg}")
        
        if signal.response_time:
            print(f"    Response time: {signal.response_time*1000:.1f}ms")
        
        print('\033[0m', end='')  # Reset color
    
    def start_live_monitor(self):
        """Start live monitoring in console."""
        if self.monitoring:
            return
        
        self.monitoring = True
        print("\n" + "="*60)
        print("📡 LIVE COMMUNICATION MONITOR")
        print("="*60)
        print("Showing all signals in real-time...")
        print("Press Ctrl+C to stop monitoring\n")
    
    def stop_live_monitor(self):
        """Stop live monitoring."""
        self.monitoring = False
    
    def show_statistics(self):
        """Show communication statistics."""
        print("\n" + "="*60)
        print("📊 COMMUNICATION STATISTICS")
        print("="*60)
        
        print(f"Commands sent:      {self.stats['commands_sent']}")
        print(f"Responses received: {self.stats['responses_received']}")
        print(f"Response rate:      {self.stats['responses_received']/max(1, self.stats['commands_sent'])*100:.1f}%")
        print(f"Timeouts:          {self.stats['timeouts']}")
        print(f"Errors:            {self.stats['errors']}")
        print(f"Invalid signals:   {self.stats['invalid_signals']}")
        print(f"Avg response time: {self.stats['avg_response_time']*1000:.1f}ms")
        
        if self.stats['last_error']:
            print(f"\nLast error: {self.stats['last_error']}")
        
        # Find most common command
        if self.signal_history:
            cmd_counts = {}
            for signal in self.signal_history:
                if signal.signal_type == SignalType.COMMAND and isinstance(signal.data, dict):
                    cmd_type = signal.data.get('T', 'unknown')
                    cmd_counts[cmd_type] = cmd_counts.get(cmd_type, 0) + 1
            
            if cmd_counts:
                most_common = max(cmd_counts, key=cmd_counts.get)
                print(f"\nMost common command: Type {most_common} ({cmd_counts[most_common]} times)")
    
    def analyze_patterns(self):
        """Analyze communication patterns for issues."""
        print("\n" + "="*60)
        print("🔍 PATTERN ANALYSIS")
        print("="*60)
        
        issues = []
        
        # Check for repeating timeouts
        timeout_streak = 0
        for signal in self.signal_history[-20:]:  # Last 20 signals
            if signal.signal_type == SignalType.TIMEOUT:
                timeout_streak += 1
            else:
                timeout_streak = 0
            
            if timeout_streak >= 3:
                issues.append("Multiple consecutive timeouts detected - Arduino may not be responding")
                break
        
        # Check for slow responses
        slow_responses = [s for s in self.signal_history 
                         if s.response_time and s.response_time > 0.5]
        if slow_responses:
            issues.append(f"{len(slow_responses)} slow responses (>500ms) detected")
        
        # Check for invalid signals
        invalid = [s for s in self.signal_history 
                  if s.signal_type == SignalType.INVALID]
        if invalid:
            issues.append(f"{len(invalid)} invalid/unexpected responses detected")
        
        # Check command/response ratio
        if self.stats['commands_sent'] > 10:
            response_rate = self.stats['responses_received'] / self.stats['commands_sent']
            if response_rate < 0.5:
                issues.append(f"Low response rate ({response_rate*100:.0f}%) - check Arduino firmware")
        
        if issues:
            print("⚠️  Issues found:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ No communication issues detected")
    
    def export_log(self, filename: str = "comm_debug.log"):
        """Export communication log to file."""
        with open(filename, 'w') as f:
            f.write("RoArm M3 Communication Debug Log\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            
            # Statistics
            f.write("STATISTICS:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Signal history
            f.write("SIGNAL HISTORY:\n")
            f.write("-"*60 + "\n")
            
            for signal in self.signal_history:
                timestamp = datetime.fromtimestamp(signal.timestamp).strftime('%H:%M:%S.%f')[:-3]
                f.write(f"[{timestamp}] {signal.direction} {signal.signal_type.value}\n")
                
                if signal.data:
                    f.write(f"  Data: {signal.data}\n")
                if signal.error_msg:
                    f.write(f"  Error: {signal.error_msg}\n")
                if signal.response_time:
                    f.write(f"  Response time: {signal.response_time*1000:.1f}ms\n")
                if signal.checksum:
                    f.write(f"  Checksum: {signal.checksum}\n")
                f.write("\n")
        
        print(f"✅ Log exported to {filename}")
    
    def test_communication(self):
        """Run communication test sequence."""
        print("\n" + "="*60)
        print("🧪 COMMUNICATION TEST")
        print("="*60)
        
        test_commands = [
            {"T": 1, "description": "Status query"},
            {"T": 51, "led": 1, "description": "LED on"},
            {"T": 51, "led": 0, "description": "LED off"},
            {"T": 255, "echo": 123, "description": "Echo test"},
        ]
        
        print("Running test sequence...\n")
        
        for cmd in test_commands:
            desc = cmd.pop('description', 'Unknown')
            print(f"Testing: {desc}")
            print(f"  Command: {cmd}")
            
            try:
                start = time.time()
                response = self.controller.serial.send_command(cmd, wait_response=True, timeout=1.0)
                elapsed = time.time() - start
                
                if response:
                    print(f"  ✅ Response: {response[:50]}")
                    print(f"  Time: {elapsed*1000:.1f}ms")
                else:
                    print(f"  ⚠️  No response (timeout)")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()
            time.sleep(0.5)
        
        print("Test complete - check statistics for results")


class DebugCommands:
    """Interactive debug commands for main menu."""
    
    def __init__(self, debugger: CommunicationDebugger):
        self.debugger = debugger
    
    def show_menu(self):
        """Show debug menu."""
        print("\n" + "="*50)
        print("🔍 DEBUG MENU")
        print("="*50)
        print("1. Start Live Monitor")
        print("2. Show Statistics")
        print("3. Analyze Patterns")
        print("4. Test Communication")
        print("5. Export Log")
        print("6. Send Custom Command")
        print("7. Simulate Errors")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self.debugger.start_live_monitor()
            input("Press ENTER to stop monitoring...")
            self.debugger.stop_live_monitor()
            
        elif choice == '2':
            self.debugger.show_statistics()
            
        elif choice == '3':
            self.debugger.analyze_patterns()
            
        elif choice == '4':
            self.debugger.test_communication()
            
        elif choice == '5':
            filename = input("Filename [comm_debug.log]: ").strip() or "comm_debug.log"
            self.debugger.export_log(filename)
            
        elif choice == '6':
            self._send_custom_command()
            
        elif choice == '7':
            self._simulate_errors()
    
    def _send_custom_command(self):
        """Send custom command for testing."""
        print("\nEnter command as JSON (e.g., {\"T\": 1})")
        cmd_str = input("Command: ").strip()
        
        try:
            command = json.loads(cmd_str)
            print(f"\nSending: {command}")
            
            response = self.debugger.controller.serial.send_command(
                command, 
                wait_response=True, 
                timeout=2.0
            )
            
            if response:
                print(f"Response: {response}")
            else:
                print("No response (timeout)")
                
        except json.JSONDecodeError:
            print("Invalid JSON format")
        except Exception as e:
            print(f"Error: {e}")
    
    def _simulate_errors(self):
        """Simulate various error conditions."""
        print("\n🐛 ERROR SIMULATION")
        print("-"*40)
        print("1. Timeout (no response)")
        print("2. Invalid response")
        print("3. Partial response")
        print("4. Corrupted data")
        print("5. Connection loss")
        
        choice = input("\n👉 Select error: ").strip()
        
        if choice == '1':
            print("Simulating timeout...")
            # Send command to non-existent register
            self.debugger.controller.serial.send_command(
                {"T": 999}, 
                wait_response=True, 
                timeout=1.0
            )
            
        # ... more error simulations ...
