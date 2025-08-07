#!/usr/bin/env python3
"""
RoArm M3 Professional Control System - COMPLETE VERSION
Mit integriertem Communication Debugger und allen Features
"""

import sys
import os
import time
import signal
import argparse
import traceback
import json
from pathlib import Path
from typing import Optional, Dict, Any
import threading
from datetime import datetime

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

# Core imports mit Fehlerbehandlung
try:
    from core.controller import RoArmController, RoArmConfig
    from core.constants import SERVO_LIMITS, HOME_POSITION
except ImportError as e:
    print(f"❌ Core module import failed: {e}")
    print("   Check if core/ directory exists with controller.py and constants.py")
    sys.exit(1)

# Communication Debugger
try:
    from utils.comm_debugger import CommunicationDebugger, DebugCommands
    DEBUGGER_AVAILABLE = True
except ImportError:
    print("⚠️ Communication debugger not available")
    DEBUGGER_AVAILABLE = False

# Pattern module
try:
    from patterns.scan_patterns import (
        RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
        TurntableScanPattern, CobwebScanPattern
    )
    PATTERNS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Pattern module not available: {e}")
    PATTERNS_AVAILABLE = False

# Teaching module
try:
    from teaching.recorder import TeachingRecorder, RecordingMode, TrajectoryType
    TEACHING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Teaching module not available: {e}")
    TEACHING_AVAILABLE = False

# Calibration module
try:
    from calibration.calibration_suite import CalibrationSuite, CalibrationType
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Calibration module not available: {e}")
    CALIBRATION_AVAILABLE = False

# Safety system
try:
    from safety.safety_system import SafetySystem, SafetyState, ShutdownReason
    SAFETY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Safety system not available: {e}")
    SAFETY_AVAILABLE = False

from utils.logger import setup_logger, get_logger
from utils.terminal import TerminalController

# Setup logging
setup_logger()
logger = get_logger(__name__)


class SystemDiagnostics:
    """Complete system diagnostics and debugging."""
    
    @staticmethod
    def run_full_diagnostics():
        """Run complete system diagnostics."""
        print("\n" + "="*60)
        print("🔍 FULL SYSTEM DIAGNOSTICS")
        print("="*60)
        
        issues = []
        warnings = []
        
        # 1. Python Version
        print("\n1️⃣ Python Version:")
        py_version = sys.version_info
        print(f"   Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
            issues.append("Python version too old (need 3.7+)")
        
        # 2. Required Modules
        print("\n2️⃣ Required Modules:")
        required = ['serial', 'yaml', 'numpy', 'core.controller', 'utils.logger']
        for module in required:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError as e:
                print(f"   ❌ {module}: {e}")
                issues.append(f"Missing: {module}")
        
        # 3. Optional Modules
        print("\n3️⃣ Optional Modules:")
        optional = {
            'comm_debugger': DEBUGGER_AVAILABLE,
            'patterns': PATTERNS_AVAILABLE,
            'teaching': TEACHING_AVAILABLE,
            'calibration': CALIBRATION_AVAILABLE,
            'safety': SAFETY_AVAILABLE
        }
        for module, available in optional.items():
            if available:
                print(f"   ✅ {module}")
            else:
                print(f"   ⚠️ {module} not available")
                warnings.append(f"Optional module {module} not available")
        
        # 4. Serial Ports
        print("\n4️⃣ Serial Ports:")
        try:
            from core.serial_comm import SerialManager
            ports = SerialManager.list_ports()
            
            # Filter out system ports
            real_ports = [p for p in ports 
                         if 'debug-console' not in p and 'Bluetooth' not in p]
            
            if real_ports:
                print(f"   ✅ Found {len(real_ports)} USB port(s):")
                for port in real_ports:
                    print(f"      - {port}")
            else:
                print("   ⚠️ No USB serial ports detected")
                warnings.append("RoArm not connected via USB")
        except Exception as e:
            print(f"   ❌ Serial check failed: {e}")
            issues.append(f"Serial error: {e}")
        
        # 5. WiFi Capability
        print("\n5️⃣ WiFi Module:")
        try:
            from core.wifi_comm import WiFiManager
            print("   ✅ WiFi module available")
            devices = WiFiManager.scan_network(timeout=2.0)
            if devices:
                print(f"   ✅ Found {len(devices)} ESP32 device(s)")
            else:
                print("   ⚠️ No ESP32 found on network")
        except ImportError:
            print("   ⚠️ WiFi module not installed")
            warnings.append("WiFi support not available")
        except Exception as e:
            print(f"   ❌ WiFi check error: {e}")
        
        # 6. Configuration
        print("\n6️⃣ Configuration:")
        config_path = Path("config.yaml")
        if config_path.exists():
            print(f"   ✅ config.yaml found ({config_path.stat().st_size} bytes)")
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                print(f"   ✅ Valid YAML with {len(cfg)} sections")
            except Exception as e:
                print(f"   ❌ Config error: {e}")
                issues.append("Invalid config.yaml")
        else:
            print("   ⚠️ config.yaml not found")
            warnings.append("Using default configuration")
        
        # 7. Controller Test
        print("\n7️⃣ Controller Creation:")
        try:
            config = RoArmConfig(auto_connect=False)
            controller = RoArmController(config)
            print("   ✅ Controller created successfully")
            
            # Check critical settings
            if config.auto_connect:
                warnings.append("auto_connect=True can cause blocking")
        except Exception as e:
            print(f"   ❌ Controller failed: {e}")
            issues.append(f"Controller error: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        if issues:
            print(f"\n❌ {len(issues)} Critical Issue(s):")
            for issue in issues:
                print(f"  • {issue}")
        
        if warnings:
            print(f"\n⚠️ {len(warnings)} Warning(s):")
            for warning in warnings:
                print(f"  • {warning}")
        
        if not issues:
            print("\n✅ System ready to operate!")
        else:
            print("\n❌ Fix critical issues before continuing")
        
        return len(issues) == 0


class RoArmCLI:
    """Main CLI class with complete functionality."""
    
    def __init__(self):
        self.controller = None
        self.teacher = None
        self.calibrator = None
        self.safety_system = None
        self.comm_debugger = None
        self.debug_commands = None
        self.terminal = TerminalController()
        self.running = True
        self.connection_type = None
        
        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n🛑 SHUTDOWN - Ctrl+C detected!")
        self.running = False
        
        if self.controller:
            try:
                self.controller.emergency_stop()
            except:
                pass
        
        # Export debug log if available
        if self.comm_debugger and self.comm_debugger.signal_history:
            print("Saving debug log...")
            self.comm_debugger.export_log("emergency_shutdown.log")
        
        sys.exit(0)
    
    def run(self, args):
        """Main run method."""
        try:
            # Start menu
            while self.running:
                choice = self._show_start_menu()
                
                if choice == '1':  # Quick Start
                    if self._establish_connection('auto'):
                        self._main_loop()
                        
                elif choice == '2':  # Connection Setup
                    conn_type = self._select_connection_method()
                    if self._establish_connection(conn_type):
                        self._main_loop()
                        
                elif choice == '3':  # System Diagnostics
                    SystemDiagnostics.run_full_diagnostics()
                    input("\nPress ENTER to continue...")
                    
                elif choice == '4':  # Communication Test
                    self._communication_test_mode()
                    
                elif choice == '0':  # Exit
                    break
                    
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            print(f"\n❌ CRITICAL ERROR: {e}")
            if args.debug:
                traceback.print_exc()
        finally:
            self._cleanup()
    
    def _show_start_menu(self) -> str:
        """Show start menu."""
        self._print_header()
        
        print("\n" + "="*50)
        print("START MENU")
        print("="*50)
        print("1. 🚀 Quick Start (Auto-detect)")
        print("2. 🔌 Connection Setup")
        print("3. 🔍 System Diagnostics")
        print("4. 📡 Communication Test Mode")
        print("0. 🚪 Exit")
        
        return input("\n👉 Select: ").strip()
    
    def _select_connection_method(self) -> str:
        """Select connection method."""
        print("\n🔌 CONNECTION METHOD")
        print("-" * 40)
        print("1. USB Serial (Cable)")
        print("2. WiFi (Wireless)")
        print("3. Auto-Detect")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            return 'serial'
        elif choice == '2':
            return 'wifi'
        elif choice == '3':
            return 'auto'
        else:
            return None
    
    def _establish_connection(self, conn_type: str) -> bool:
        """Establish connection with debugging."""
        if not conn_type:
            return False
            
        print("\n🔌 ESTABLISHING CONNECTION...")
        print("-" * 40)
        
        # Setup config
        config = RoArmConfig(
            port='auto',
            baudrate=115200,
            default_speed=1.0,
            debug=False,
            auto_connect=False  # CRITICAL!
        )
        
        connected = False
        
        # Try connections
        if conn_type == 'serial' or conn_type == 'auto':
            print("Trying USB Serial...")
            connected = self._try_serial_connection(config)
            
            if not connected and conn_type == 'auto':
                print("\nSerial failed, trying WiFi...")
                connected = self._try_wifi_connection(config)
        
        elif conn_type == 'wifi':
            print("Trying WiFi...")
            connected = self._try_wifi_connection(config)
        
        # Setup debugging if connected
        if connected:
            self._setup_debugging()
            print("\n✅ CONNECTION ESTABLISHED!")
            print(f"   Type: {self.connection_type}")
            
            # Initialize components
            self._initialize_components()
            return True
        else:
            print("\n❌ CONNECTION FAILED!")
            return False
    
    def _try_serial_connection(self, config: RoArmConfig) -> bool:
        """Try serial connection."""
        try:
            from core.serial_comm import SerialManager
            
            # Find port (exclude system ports)
            ports = SerialManager.list_ports()
            real_ports = [p for p in ports 
                         if 'debug-console' not in p and 'Bluetooth' not in p]
            
            if not real_ports:
                print("  ❌ No USB serial port found")
                return False
            
            config.port = real_ports[0]
            print(f"  Port: {config.port}")
            
            # Create controller
            self.controller = RoArmController(config)
            
            # Test connection
            if self.controller.serial.connected:
                self.connection_type = 'serial'
                return True
            
            return False
            
        except Exception as e:
            print(f"  Serial error: {e}")
            return False
    
    def _try_wifi_connection(self, config: RoArmConfig) -> bool:
        """Try WiFi connection."""
        try:
            from core.wifi_comm import WiFiManager
            
            print("  Scanning network...")
            devices = WiFiManager.scan_network(timeout=2.0)
            
            if not devices:
                print("  ❌ No ESP32 found")
                return False
            
            host = devices[0]
            print(f"  Found: {host}")
            
            wifi_mgr = WiFiManager(host=host)
            
            if wifi_mgr.connect():
                self.controller = RoArmController(config)
                self.controller.serial = wifi_mgr
                self.connection_type = 'wifi'
                return True
            
            return False
            
        except ImportError:
            print("  ❌ WiFi module not available")
            return False
        except Exception as e:
            print(f"  WiFi error: {e}")
            return False
    
    def _setup_debugging(self):
        """Setup communication debugging."""
        if DEBUGGER_AVAILABLE and self.controller:
            try:
                self.comm_debugger = CommunicationDebugger(self.controller)
                self.controller.serial = self.comm_debugger.wrap_serial_manager(
                    self.controller.serial
                )
                self.debug_commands = DebugCommands(self.comm_debugger)
                print("   🔍 Communication debugger active")
            except Exception as e:
                print(f"   ⚠️ Debugger setup failed: {e}")
    
    def _initialize_components(self):
        """Initialize optional components."""
        print("\n⚙️ INITIALIZING COMPONENTS...")
        
        if TEACHING_AVAILABLE:
            try:
                self.teacher = TeachingRecorder(self.controller)
                print("  ✅ Teaching system")
            except Exception as e:
                print(f"  ⚠️ Teaching: {e}")
        
        if CALIBRATION_AVAILABLE:
            try:
                self.calibrator = CalibrationSuite(self.controller)
                print("  ✅ Calibration system")
            except Exception as e:
                print(f"  ⚠️ Calibration: {e}")
        
        if SAFETY_AVAILABLE:
            try:
                self.safety_system = SafetySystem(self.controller)
                print("  ✅ Safety system")
            except Exception as e:
                print(f"  ⚠️ Safety: {e}")
    
    def _main_loop(self):
        """Main menu loop."""
        while self.running:
            try:
                self._show_main_menu()
                choice = input("\n👉 Select option: ").strip().upper()
                
                if not choice:
                    continue
                
                self._handle_main_menu(choice)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                logger.error(f"Menu error: {e}")
    
    def _print_header(self):
        """Print header."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n" + "="*60)
        print("🤖 RoArm M3 Professional Control System v2.0")
        print("="*60)
        print(f"📷 Scanner: Revopoint Mini2")
        print(f"🔌 Connection: {self.connection_type or 'Not connected'}")
        if self.comm_debugger:
            stats = self.comm_debugger.stats
            print(f"📊 Commands: {stats['commands_sent']} | Responses: {stats['responses_received']}")
        print("⚡ Press Ctrl+C for EMERGENCY STOP")
        print("="*60)
    
    def _show_main_menu(self):
        """Show main menu."""
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        
        if not self.controller or not self.controller.serial.connected:
            print("⚠️ NOT CONNECTED")
            print()
        
        print("1. 🎮 Manual Control")
        print("2. 📷 Scanner Patterns")
        print("3. 🎓 Teaching Mode")
        print("4. 📁 Load Sequence")
        print("5. 🏠 Move to Home")
        print("6. 🔧 Calibration")
        print("7. ⚙️  Settings")
        print("8. 📊 Status")
        print("9. 🔌 Reconnect")
        
        if self.comm_debugger:
            print("D. 🔍 Debug Communications")
        
        print("T. 🧪 Test Functions")
        print("0. 🚪 Exit")
    
    def _handle_main_menu(self, choice: str):
        """Handle menu choice."""
        # Check connection for commands that need it
        needs_connection = ['1', '2', '3', '4', '5', '6', 'T']
        
        if choice in needs_connection:
            if not self.controller or not self.controller.serial.connected:
                print("⚠️ Not connected! Use option 9 to reconnect.")
                return
        
        handlers = {
            '1': self._manual_control,
            '2': self._scanner_menu,
            '3': self._teaching_menu,
            '4': self._load_sequence,
            '5': self._move_home,
            '6': self._calibration_menu,
            '7': self._settings_menu,
            '8': self._show_status,
            '9': self._reconnect,
            'D': self._debug_communications,
            'T': self._test_functions,
            '0': self._exit
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid option")
    
    def _debug_communications(self):
        """Debug communications menu."""
        if not self.comm_debugger:
            print("⚠️ Debugger not available")
            return
        
        while True:
            print("\n" + "="*50)
            print("🔍 COMMUNICATION DEBUG")
            print("="*50)
            print("1. 📡 Live Monitor")
            print("2. 📊 Statistics")
            print("3. 🔍 Analyze Patterns")
            print("4. 🧪 Test Communication")
            print("5. 💾 Export Log")
            print("6. 📝 Send Custom Command")
            print("0. Back")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self._live_monitor()
            elif choice == '2':
                self.comm_debugger.show_statistics()
            elif choice == '3':
                self.comm_debugger.analyze_patterns()
            elif choice == '4':
                self.comm_debugger.test_communication()
            elif choice == '5':
                filename = input("Filename [debug.log]: ") or "debug.log"
                self.comm_debugger.export_log(filename)
            elif choice == '6':
                self._send_custom_command()
    
    def _live_monitor(self):
        """Live communication monitor."""
        print("\n📡 LIVE MONITOR - Press ENTER to stop")
        print("-" * 50)
        self.comm_debugger.start_live_monitor()
        
        # Show some test commands
        print("\nSending test commands...")
        self.controller.led_control(True, 128)
        time.sleep(0.5)
        self.controller.led_control(False)
        
        input("\nPress ENTER to stop monitoring...")
        self.comm_debugger.stop_live_monitor()
    
    def _send_custom_command(self):
        """Send custom command."""
        print("\nEnter command as JSON")
        print("Examples:")
        print('  {"T": 1}  - Status query')
        print('  {"T": 51, "led": 1}  - LED on')
        
        cmd_str = input("\nCommand: ").strip()
        
        try:
            command = json.loads(cmd_str)
            print(f"\nSending: {command}")
            
            response = self.controller.serial.send_command(
                command, 
                wait_response=True, 
                timeout=2.0
            )
            
            if response:
                print(f"Response: {response}")
            else:
                print("No response (timeout)")
                
        except json.JSONDecodeError:
            print("❌ Invalid JSON")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _test_functions(self):
        """Test various functions."""
        print("\n🧪 TEST FUNCTIONS")
        print("-" * 40)
        print("1. LED Blink Test")
        print("2. Status Query Test")
        print("3. Movement Test")
        print("4. Response Time Test")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            print("Blinking LED 3 times...")
            for i in range(3):
                self.controller.led_control(True, 255)
                time.sleep(0.3)
                self.controller.led_control(False)
                time.sleep(0.3)
            print("✅ Done")
            
        elif choice == '2':
            print("Querying status...")
            status = self.controller.query_status()
            if status:
                print(f"✅ Status: {json.dumps(status, indent=2)}")
            else:
                print("❌ No response")
                
        elif choice == '3':
            print("Small movement test...")
            pos = self.controller.current_position.copy()
            pos['base'] += 0.1
            if self.controller.move_joints(pos, speed=0.3):
                print("✅ Movement successful")
            else:
                print("❌ Movement failed")
                
        elif choice == '4':
            print("Testing response times...")
            times = []
            for i in range(5):
                start = time.time()
                self.controller.led_control(True, 128)
                elapsed = time.time() - start
                times.append(elapsed)
                time.sleep(0.1)
            
            avg_time = sum(times) / len(times)
            print(f"Average response: {avg_time*1000:.1f}ms")
    
    def _communication_test_mode(self):
        """Standalone communication test mode."""
        print("\n📡 COMMUNICATION TEST MODE")
        print("-" * 40)
        print("Testing without full connection...")
        
        # Try to create minimal connection
        from core.serial_comm import SerialManager
        
        ports = SerialManager.list_ports()
        real_ports = [p for p in ports 
                     if 'debug-console' not in p and 'Bluetooth' not in p]
        
        if not real_ports:
            print("❌ No USB ports found")
            return
        
        port = real_ports[0]
        print(f"Using port: {port}")
        
        serial_mgr = SerialManager(port=port)
        
        if serial_mgr.connect():
            print("✅ Port opened")
            
            # Test commands
            test_cmds = [
                {"T": 1, "name": "Status"},
                {"T": 51, "led": 1, "name": "LED on"},
                {"T": 51, "led": 0, "name": "LED off"},
            ]
            
            for cmd in test_cmds:
                name = cmd.pop('name', 'Unknown')
                print(f"\nTesting: {name}")
                print(f"  Command: {cmd}")
                
                response = serial_mgr.send_command(cmd, wait_response=True, timeout=1.0)
                
                if response:
                    print(f"  Response: {response}")
                else:
                    print("  No response")
            
            serial_mgr.disconnect()
        else:
            print("❌ Failed to open port")
    
    def _manual_control(self):
        """Manual control mode."""
        print("\n🎮 MANUAL CONTROL")
        print("-" * 40)
        print("Controls:")
        print("  q/a: Base left/right")
        print("  w/s: Shoulder up/down")
        print("  e/d: Elbow up/down")
        print("  x: Exit")
        
        print("\nFeature not fully implemented yet")
    
    def _scanner_menu(self):
        """Scanner patterns menu."""
        if not PATTERNS_AVAILABLE:
            print("⚠️ Pattern module not available")
            return
        
        print("\n📷 SCANNER PATTERNS")
        print("Feature implementation needed...")
    
    def _teaching_menu(self):
        """Teaching mode menu."""
        if not TEACHING_AVAILABLE:
            print("⚠️ Teaching module not available")
            return
        
        print("\n🎓 TEACHING MODE")
        print("Feature implementation needed...")
    
    def _load_sequence(self):
        """Load sequence menu."""
        print("\n📁 LOAD SEQUENCE")
        print("Feature implementation needed...")
    
    def _move_home(self):
        """Move to home position."""
        print("\n🏠 Moving to home position...")
        try:
            if self.controller.move_home(speed=0.5):
                print("✅ Home position reached")
            else:
                print("❌ Failed to reach home")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _calibration_menu(self):
        """Calibration menu."""
        if not CALIBRATION_AVAILABLE:
            print("⚠️ Calibration module not available")
            return
        
        print("\n🔧 CALIBRATION")
        print("Feature implementation needed...")
    
    def _settings_menu(self):
        """Settings menu."""
        print("\n⚙️ SETTINGS")
        print("-" * 40)
        
        if self.controller:
            print(f"Port: {self.controller.config.port}")
            print(f"Baudrate: {self.controller.config.baudrate}")
            print(f"Connection: {self.connection_type}")
            print(f"Speed: {self.controller.config.default_speed}")
            
            if self.comm_debugger:
                print(f"\nDebugger Statistics:")
                stats = self.comm_debugger.stats
                print(f"  Commands sent: {stats['commands_sent']}")
                print(f"  Responses: {stats['responses_received']}")
                print(f"  Timeouts: {stats['timeouts']}")
                print(f"  Errors: {stats['errors']}")
        else:
            print("Not connected")
    
    def _show_status(self):
        """Show system status."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        # Connection status
        if self.controller and self.controller.serial.connected:
            print(f"Connection: ✅ {self.connection_type}")
            
            # Try to get robot status
            print("\nQuerying robot...")
            status = self.controller.query_status()
            
            if status:
                print("\nPositions:")
                for joint, pos in status.get('positions', {}).items():
                    print(f"  {joint}: {pos:.3f} rad")
                
                if 'temperature' in status:
                    print(f"\nTemperature: {status['temperature']}°C")
                if 'voltage' in status:
                    print(f"Voltage: {status['voltage']}V")
            else:
                print("⚠️ No response from robot")
        else:
            print("Connection: ❌ Not connected")
        
        # Components status
        print("\nComponents:")
        print(f"  Teaching: {'✅' if self.teacher else '❌'}")
        print(f"  Calibration: {'✅' if self.calibrator else '❌'}")
        print(f"  Safety: {'✅' if self.safety_system else '❌'}")
        print(f"  Debugger: {'✅' if self.comm_debugger else '❌'}")
        
        # Debug info if available
        if self.comm_debugger:
            print("\nCommunication:")
            stats = self.comm_debugger.stats
            success_rate = stats['responses_received'] / max(1, stats['commands_sent']) * 100
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Avg response: {stats['avg_response_time']*1000:.1f}ms")
            
            if stats['last_error']:
                print(f"  Last error: {stats['last_error']}")
    
    def _reconnect(self):
        """Reconnect to robot."""
        print("\n🔌 RECONNECTING...")
        
        if self.controller:
            try:
                self.controller.disconnect()
            except:
                pass
        
        conn_type = self._select_connection_method()
        if conn_type and self._establish_connection(conn_type):
            print("✅ Reconnected successfully!")
        else:
            print("❌ Reconnection failed")
    
    def _exit(self):
        """Exit the program."""
        self.running = False
    
    def _cleanup(self):
        """Cleanup on exit."""
        print("\n🔌 Shutting down...")
        
        # Export debug log if available
        if self.comm_debugger and self.comm_debugger.signal_history:
            print("Saving communication log...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.comm_debugger.export_log(f"session_{timestamp}.log")
        
        # Disconnect
        if self.controller:
            try:
                print("Moving to safe position...")
                self.controller.move_home(speed=0.5)
                time.sleep(1)
            except:
                pass
            
            try:
                self.controller.disconnect()
            except:
                pass
        
        print("✅ Shutdown complete")
        print("Goodbye! 👋\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RoArm M3 Professional Control System with Debugger'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Start with live monitor'
    )
    
    args = parser.parse_args()
    
    # Start CLI
    cli = RoArmCLI()
    cli.run(args)


if __name__ == '__main__':
    main()
