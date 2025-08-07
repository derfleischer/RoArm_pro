#!/usr/bin/env python3
"""
RoArm M3 - Comprehensive Validation & Debug Tool
Findet und zeigt ALLE Probleme im System
"""

import sys
import os
import time
import traceback
import importlib.util
import threading
import serial
import serial.tools.list_ports
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def print_debug(text: str):
    print(f"   {Colors.CYAN}🔍 {text}{Colors.ENDC}")


class SystemValidator:
    """Comprehensive system validation and debugging."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.modules = {}
        self.serial_port = None
        
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print_header("RoArm M3 COMPREHENSIVE SYSTEM VALIDATION")
        print(f"Project Root: {self.project_root.absolute()}")
        print(f"Python Version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        all_ok = True
        
        # 1. Check Project Structure
        print_section("1. PROJECT STRUCTURE")
        if not self.check_project_structure():
            all_ok = False
        
        # 2. Check Dependencies
        print_section("2. PYTHON DEPENDENCIES")
        if not self.check_dependencies():
            all_ok = False
        
        # 3. Check Module Imports
        print_section("3. MODULE IMPORTS")
        if not self.check_module_imports():
            all_ok = False
        
        # 4. Check Serial Ports
        print_section("4. SERIAL PORTS")
        if not self.check_serial_ports():
            all_ok = False
        
        # 5. Check Core Components
        print_section("5. CORE COMPONENTS")
        if not self.check_core_components():
            all_ok = False
        
        # 6. Check Main.py Initialization
        print_section("6. MAIN.PY INITIALIZATION CHAIN")
        if not self.debug_main_initialization():
            all_ok = False
        
        # 7. Test Serial Communication
        print_section("7. SERIAL COMMUNICATION TEST")
        if not self.test_serial_communication():
            all_ok = False
        
        # Summary
        self.print_summary(all_ok)
        
        return all_ok
    
    def check_project_structure(self) -> bool:
        """Check if all required directories and files exist."""
        required_structure = {
            'core': ['controller.py', 'serial_comm.py', 'constants.py'],
            'motion': ['trajectory.py'],
            'patterns': ['scan_patterns.py'],
            'teaching': ['recorder.py'],
            'calibration': ['calibration_suite.py'],
            'safety': ['safety_system.py'],
            'utils': ['logger.py', 'terminal.py'],
        }
        
        all_ok = True
        
        for dir_name, files in required_structure.items():
            dir_path = self.project_root / dir_name
            
            if not dir_path.exists():
                print_error(f"Directory missing: {dir_name}/")
                all_ok = False
                continue
            
            print_success(f"Directory found: {dir_name}/")
            
            for file_name in files:
                file_path = dir_path / file_name
                if not file_path.exists():
                    print_error(f"  File missing: {dir_name}/{file_name}")
                    all_ok = False
                else:
                    size = file_path.stat().st_size
                    if size == 0:
                        print_warning(f"  File empty: {dir_name}/{file_name}")
                    else:
                        print_debug(f"  {file_name} ({size} bytes)")
        
        # Check main files
        main_files = ['main.py', 'config.yaml']
        for file_name in main_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                print_error(f"Main file missing: {file_name}")
                all_ok = False
            else:
                print_success(f"Main file found: {file_name}")
        
        return all_ok
    
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed."""
        required_packages = {
            'serial': 'pyserial',
            'yaml': 'pyyaml',
            'numpy': 'numpy',
        }
        
        all_ok = True
        
        for module_name, package_name in required_packages.items():
            try:
                __import__(module_name)
                print_success(f"{package_name} installed")
            except ImportError:
                print_error(f"{package_name} NOT installed - run: pip install {package_name}")
                all_ok = False
        
        # Check optional packages
        optional_packages = ['colorama', 'tqdm']
        for package in optional_packages:
            try:
                __import__(package)
                print_info(f"{package} installed (optional)")
            except ImportError:
                print_debug(f"{package} not installed (optional)")
        
        return all_ok
    
    def check_module_imports(self) -> bool:
        """Try to import all modules and check for errors."""
        all_ok = True
        
        # Add project root to path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        modules_to_check = [
            'core.constants',
            'core.serial_comm',
            'core.controller',
            'motion.trajectory',
            'patterns.scan_patterns',
            'teaching.recorder',
            'calibration.calibration_suite',
            'safety.safety_system',
            'utils.logger',
            'utils.terminal',
        ]
        
        for module_name in modules_to_check:
            try:
                # Try to import
                module = __import__(module_name, fromlist=[''])
                self.modules[module_name] = module
                print_success(f"Import OK: {module_name}")
                
                # Check for specific classes/functions
                self._check_module_contents(module_name, module)
                
            except ImportError as e:
                print_error(f"Import FAILED: {module_name}")
                print_debug(f"Error: {str(e)}")
                all_ok = False
            except Exception as e:
                print_error(f"Import ERROR: {module_name}")
                print_debug(f"Error: {str(e)}")
                traceback.print_exc()
                all_ok = False
        
        return all_ok
    
    def _check_module_contents(self, module_name: str, module):
        """Check if module has expected contents."""
        expected_contents = {
            'core.constants': ['SERVO_LIMITS', 'HOME_POSITION', 'COMMANDS'],
            'core.serial_comm': ['SerialManager'],
            'core.controller': ['RoArmController', 'RoArmConfig'],
            'motion.trajectory': ['TrajectoryGenerator', 'TrajectoryType'],
            'utils.logger': ['setup_logger', 'get_logger'],
        }
        
        if module_name in expected_contents:
            for item in expected_contents[module_name]:
                if hasattr(module, item):
                    print_debug(f"  Found: {item}")
                else:
                    print_warning(f"  Missing: {item}")
    
    def check_serial_ports(self) -> bool:
        """Check available serial ports."""
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print_error("No serial ports found!")
            return False
        
        print_success(f"Found {len(ports)} serial port(s):")
        
        arduino_port = None
        for port in ports:
            print_info(f"  {port.device}")
            print_debug(f"    Description: {port.description}")
            print_debug(f"    Hardware ID: {port.hwid}")
            
            # Check for Arduino/USB serial
            if any(x in port.description.lower() for x in ['arduino', 'usb', 'serial']):
                arduino_port = port.device
            elif 'usbserial' in port.device.lower() or 'cu.' in port.device:
                arduino_port = port.device
        
        if arduino_port:
            print_success(f"Likely Arduino port: {arduino_port}")
            self.serial_port = arduino_port
            return True
        else:
            print_warning("No obvious Arduino port found")
            if ports:
                self.serial_port = ports[0].device
                print_info(f"Will try: {self.serial_port}")
            return True
    
    def check_core_components(self) -> bool:
        """Check if core components can be instantiated."""
        all_ok = True
        
        # Check RoArmConfig
        try:
            from core.controller import RoArmConfig
            config = RoArmConfig()
            print_success("RoArmConfig instantiated")
            print_debug(f"  Default port: {config.port}")
            print_debug(f"  Default speed: {config.default_speed}")
        except Exception as e:
            print_error("RoArmConfig failed")
            print_debug(f"  Error: {str(e)}")
            all_ok = False
        
        # Check SerialManager
        try:
            from core.serial_comm import SerialManager
            # Don't actually connect, just create
            serial_mgr = SerialManager(port="dummy", baudrate=115200)
            print_success("SerialManager instantiated")
        except Exception as e:
            print_error("SerialManager failed")
            print_debug(f"  Error: {str(e)}")
            all_ok = False
        
        # Check SafetySystem
        try:
            from safety.safety_system import SafetySystem
            # Need a mock controller
            print_info("SafetySystem requires controller - skipping instantiation")
        except ImportError as e:
            print_warning("SafetySystem import failed (may be OK)")
            print_debug(f"  Error: {str(e)}")
        
        return all_ok
    
    def debug_main_initialization(self) -> bool:
        """Debug the main.py initialization chain."""
        print_info("Tracing main.py initialization...")
        
        try:
            # Import main components
            from main import RoArmCLI
            print_success("RoArmCLI imported")
            
            # Check __init__ method
            import inspect
            init_source = inspect.getsource(RoArmCLI.__init__)
            
            # Find potential blocking points
            blocking_keywords = ['query_status', 'wait', 'sleep', 'input', 'while True']
            found_issues = []
            
            for keyword in blocking_keywords:
                if keyword in init_source:
                    found_issues.append(keyword)
            
            if found_issues:
                print_warning(f"Potential blocking in __init__: {', '.join(found_issues)}")
            
            # Check run method
            run_source = inspect.getsource(RoArmCLI.run)
            
            # Find controller initialization
            if 'RoArmController' in run_source:
                print_info("RoArmController created in run()")
                
                # Check what happens after controller creation
                lines = run_source.split('\n')
                controller_line = -1
                for i, line in enumerate(lines):
                    if 'RoArmController' in line:
                        controller_line = i
                        break
                
                if controller_line >= 0:
                    print_debug("Code after controller creation:")
                    for i in range(controller_line + 1, min(controller_line + 6, len(lines))):
                        print_debug(f"    {lines[i].strip()}")
            
            # Check controller initialization
            from core.controller import RoArmController
            init_source = inspect.getsource(RoArmController.__init__)
            
            if 'auto_connect' in init_source and 'self.connect()' in init_source:
                print_warning("RoArmController auto-connects in __init__")
                
                # Check connect method
                connect_source = inspect.getsource(RoArmController.connect)
                if '_initialize_robot' in connect_source:
                    print_warning("connect() calls _initialize_robot()")
                    
                    # Check _initialize_robot
                    try:
                        init_robot_source = inspect.getsource(RoArmController._initialize_robot)
                        if 'query_status' in init_robot_source:
                            print_error("⚠️  BLOCKING FOUND: _initialize_robot() calls query_status()")
                            print_info("   This will block if Arduino doesn't respond!")
                        if 'set_torque' in init_robot_source:
                            print_warning("_initialize_robot() calls set_torque()")
                    except:
                        pass
            
            return True
            
        except Exception as e:
            print_error(f"Failed to analyze main.py: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_serial_communication(self) -> bool:
        """Test actual serial communication."""
        if not self.serial_port:
            print_warning("No serial port to test")
            return True
        
        print_info(f"Testing serial port: {self.serial_port}")
        
        try:
            # Test 1: Can we open the port?
            print_debug("Test 1: Opening port...")
            ser = serial.Serial(
                port=self.serial_port,
                baudrate=115200,
                timeout=0.1,  # Short timeout!
                write_timeout=0.1
            )
            print_success("Port opened")
            
            # Test 2: Clear buffers
            print_debug("Test 2: Clearing buffers...")
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print_success("Buffers cleared")
            
            # Test 3: Send a test command
            print_debug("Test 3: Sending test command...")
            test_cmd = '{"T":51,"led":1,"brightness":128}\n'
            ser.write(test_cmd.encode())
            ser.flush()
            print_success("Command sent")
            
            # Test 4: Check for response (non-blocking)
            print_debug("Test 4: Checking for response (0.5s)...")
            time.sleep(0.5)
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                print_success(f"Got response: {response}")
            else:
                print_warning("No response (Arduino might not send feedback)")
            
            # Test 5: Close
            ser.close()
            print_success("Port closed")
            
            return True
            
        except serial.SerialException as e:
            print_error(f"Serial error: {str(e)}")
            if "PermissionError" in str(e):
                print_info("   Try: sudo chmod 666 {self.serial_port}")
            elif "could not open" in str(e):
                print_info("   Port might be in use by another program")
            return False
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            return False
    
    def print_summary(self, all_ok: bool):
        """Print summary of validation."""
        print_header("VALIDATION SUMMARY")
        
        if all_ok:
            print_success("All checks passed!")
        else:
            print_error("Some checks failed!")
        
        if self.errors:
            print(f"\n{Colors.FAIL}Errors ({len(self.errors)}):{Colors.ENDC}")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n{Colors.WARNING}Warnings ({len(self.warnings)}):{Colors.ENDC}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Specific recommendations
        print_header("RECOMMENDATIONS TO FIX MAIN.PY")
        
        print(f"\n{Colors.BOLD}The main.py hangs because:{Colors.ENDC}")
        print("1. RoArmController.__init__() calls connect() if auto_connect=True")
        print("2. connect() calls _initialize_robot()")
        print("3. _initialize_robot() calls query_status() which waits for response")
        print("4. Arduino doesn't respond → BLOCKS FOREVER")
        
        print(f"\n{Colors.BOLD}Quick Fix:{Colors.ENDC}")
        print("In core/controller.py, modify _initialize_robot():")
        print(f"{Colors.CYAN}")
        print("def _initialize_robot(self):")
        print('    """Initialize robot after connection."""')
        print("    try:")
        print("        # LED blink")
        print("        self.led_control(True, brightness=128)")
        print("        time.sleep(0.5)")
        print("        self.led_control(False)")
        print("        ")
        print("        # COMMENT OUT OR MAKE OPTIONAL:")
        print("        # self.query_status()  # <-- THIS BLOCKS!")
        print("        ")
        print("        # OR use short timeout:")
        print("        # status = self.query_status_with_timeout(0.5)")
        print("        ")
        print("        logger.info('Robot initialized')")
        print("    except Exception as e:")
        print("        logger.error(f'Init error: {e}')")
        print(f"{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Better Fix:{Colors.ENDC}")
        print("Set auto_connect=False in main.py and connect manually:")
        print(f"{Colors.CYAN}")
        print("config = RoArmConfig(")
        print("    port=args.port,")
        print("    baudrate=args.baudrate,")
        print("    auto_connect=False  # <-- ADD THIS")
        print(")")
        print("controller = RoArmController(config)")
        print("if controller.connect():")
        print("    print('Connected!')")
        print(f"{Colors.ENDC}")


def main():
    """Main entry point."""
    print(f"{Colors.BOLD}RoArm M3 System Validator v2.0{Colors.ENDC}")
    print("This tool will identify why main.py hangs\n")
    
    # Get project root
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    # Run validation
    validator = SystemValidator(project_root)
    success = validator.run_all_checks()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
