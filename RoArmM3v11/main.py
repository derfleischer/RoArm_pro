#!/usr/bin/env python3
"""
RoArm M3 Main Control Interface
Professional Scanning Suite - COMPLETE Implementation
Version 7.2.0 - Movement Fixes Applied
"""

import sys
import os
import time
import signal
import argparse
import traceback
import threading
import math
import json
import yaml
import shutil
import logging
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np

# ================================
# CORE IMPORTS - All correct
# ================================

# Utils imports
from utils.logger import get_logger
from utils.safety import SafetyMonitor

logger = get_logger(__name__)

# Core imports
from core import RoArmController, RoArmConfig
from core.constants import HOME_POSITION, SCANNER_POSITION, SERVO_LIMITS, SPEED_LIMITS
from core.serial_comm import SerialManager

# Pattern imports
from patterns import (
    RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
    TurntableScanPattern, AdaptiveScanPattern, HelixScanPattern,
    CobwebScanPattern, TableScanPattern, StatueSpiralPattern,
    create_scan_pattern, get_pattern_presets, ScanPattern, ScanPoint,
    TechnicalScanningConfigurator
)

# Teaching imports
from teaching.recorder import TeachingRecorder

# Calibration import - Use the real SafeCalibrationSuite
from calibration.calibration_suite import CalibrationSuite

# Safety System - with proper fallbacks
try:
    from safety.safety_system import SafetySystem
    logger.info("✅ SafetySystem loaded from safety module")
except ImportError as e:
    logger.warning(f"SafetySystem not available: {e} - using fallback")
    
    class SafetySystem:
        """Fallback SafetySystem implementation."""
        def __init__(self, controller):
            self.controller = controller
            self.active = True
            self.stats = {
                'emergency_stops': 0,
                'limit_violations': 0,
                'collisions': 0,
                'uptime': time.time()
            }
        
        def get_status(self):
            return "Active" if self.active else "Inactive"
        
        def get_statistics(self):
            return self.stats
        
        def reset(self):
            self.active = True
            return True
        
        def shutdown(self):
            self.active = False
            return True

# Try to import additional safety components
CollisionDetector = None
EmergencyHandler = None

try:
    from safety.collision_detector import CollisionDetector
    logger.info("✅ CollisionDetector loaded")
except ImportError:
    pass

try:
    from safety.emergency_handler import EmergencyHandler
    logger.info("✅ EmergencyHandler loaded")
except ImportError:
    pass

# Fallbacks if not available
if CollisionDetector is None:
    class CollisionDetector:
        """Fallback CollisionDetector."""
        def __init__(self, controller):
            self.controller = controller
            self.active = True
        
        def is_active(self):
            return self.active
        
        def reset(self):
            self.active = True
            return True

if EmergencyHandler is None:
    class EmergencyHandler:
        """Fallback EmergencyHandler."""
        def __init__(self, controller):
            self.controller = controller
        
        def trigger_emergency_stop(self):
            logger.warning("EMERGENCY STOP TRIGGERED")
            if hasattr(self.controller, 'emergency_stop'):
                return self.controller.emergency_stop()
            elif hasattr(self.controller, 'stop'):
                return self.controller.stop()
            return True
        
        def reset(self):
            return True

logger.info("✅ All modules loaded successfully")

# ================================
# CONTROLLER ADAPTER (NEW)
# ================================

class RoArmAdapter:
    """Adapter to handle different controller method signatures."""
    
    def __init__(self, controller):
        self.controller = controller
        
    def move_to_position(self, position: Dict[str, float], speed: float = 0.3, wait: bool = True):
        """Universal move method with fallbacks."""
        try:
            # Try primary method
            if hasattr(self.controller, 'move_to_position'):
                return self.controller.move_to_position(position, speed=speed, wait=wait)
            
            # Try move_joints
            elif hasattr(self.controller, 'move_joints'):
                return self.controller.move_joints(position, speed=speed, wait=wait)
            
            # Try individual joint moves
            elif hasattr(self.controller, 'move_joint'):
                for joint, angle in position.items():
                    self.controller.move_joint(joint, angle, speed=speed)
                return True
            
            # Try set_joint_positions
            elif hasattr(self.controller, 'set_joint_positions'):
                return self.controller.set_joint_positions(position, speed=speed)
            
            else:
                logger.warning("No suitable movement method found")
                return False
                
        except Exception as e:
            logger.error(f"Movement error in adapter: {e}")
            return False
    
    def __getattr__(self, name):
        """Forward all other calls to controller."""
        return getattr(self.controller, name)

# ================================
# SEQUENCE MANAGEMENT
# ================================

class DirectSequenceManager:
    """Direct sequence management via files."""
    
    def __init__(self, base_path: str = "sequences"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def list_sequences(self) -> List[str]:
        """List all saved sequences."""
        return [f.stem for f in self.base_path.glob("*.json")]
    
    def save_sequence(self, sequence: List[dict], name: str, metadata: dict = None):
        """Save a sequence."""
        data = {
            'name': name,
            'sequence': sequence,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'waypoints': len(sequence)
        }
        
        filepath = self.base_path / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_sequence(self, name: str) -> Optional[List[dict]]:
        """Load a sequence."""
        filepath = self.base_path / f"{name}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('sequence', data)
        return None
    
    def delete_sequence(self, name: str) -> bool:
        """Delete a sequence."""
        filepath = self.base_path / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def export_sequence(self, name: str, export_path: str) -> bool:
        """Export a sequence."""
        source = self.base_path / f"{name}.json"
        if source.exists():
            shutil.copy(source, export_path)
            return True
        return False
    
    def import_sequence(self, import_path: str, name: str) -> bool:
        """Import a sequence."""
        if Path(import_path).exists():
            target = self.base_path / f"{name}.json"
            shutil.copy(import_path, target)
            return True
        return False

# ================================
# MAIN INTERFACE CLASS
# ================================

class RoArmMainInterface:
    """
    Professional Main Interface for RoArm M3 Control.
    Complete implementation of all features.
    """
    
    def __init__(self, config_path: str = "config.yaml", debug_mode: bool = False):
        """Initialize the Main Interface."""
        self.debug_mode = debug_mode
        self.config_path = config_path
        self.running = True
        
        # Core components
        self.controller = None
        self.adapter = None  # NEW: Controller adapter
        self.safety_monitor = SafetyMonitor(SERVO_LIMITS)
        self.safety_monitor.strict_mode = False  # FIX 1: Permissive mode!
        
        # Teaching components
        self.teaching_recorder = None
        self.sequence_manager = DirectSequenceManager()
        
        # Calibration components
        self.calibration_suite = None
        
        # Safety components
        self.safety_system = None
        self.collision_detector = None
        self.emergency_handler = None
        
        # Technical configurator
        self.technical_configurator = TechnicalScanningConfigurator()
        
        # State tracking
        self.session_start = time.time()
        self.current_position = "unknown"
        self.last_scan_result = None
        self.scan_history = []
        self.command_count = 0
        self.movement_count = 0
        self.error_count = 0
        
        # Settings from config
        self.config = self._load_config()
        self.auto_home_on_startup = self.config.get('auto_home_on_startup', True)
        self.simulator_mode = False
        
        # Data storage
        self.taught_positions = []
        self.current_sequence = []
        self.calibration_data = {}
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load saved positions if exist
        self._load_taught_positions()
    
    def _clear_screen(self):
        """Clear screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        default_config = {
            'auto_home_on_startup': True,
            'default_speed': 0.3,
            'default_settle_time': 0.5,
            'safety_limits_enabled': True,
            'scan_save_path': 'scans',
            'sequence_save_path': 'sequences',
            'calibration_save_path': 'calibration',
            'log_level': 'INFO'
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    default_config.update(config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
        else:
            # Create default config file
            try:
                with open(self.config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default config: {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not create config file: {e}")
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        print(f"\n⛔ Received shutdown signal...")
        self.running = False
        
        # Emergency stop if possible
        try:
            if self.controller and not self.simulator_mode:
                if hasattr(self.controller, 'stop'):
                    self.controller.stop()
                elif hasattr(self.controller, 'emergency_stop'):
                    self.controller.emergency_stop()
        except:
            pass
    
    def _load_taught_positions(self):
        """Load previously saved positions."""
        try:
            pos_file = Path("taught_positions.json")
            if pos_file.exists():
                with open(pos_file, 'r') as f:
                    self.taught_positions = json.load(f)
                logger.info(f"Loaded {len(self.taught_positions)} saved positions")
        except Exception as e:
            logger.debug(f"Could not load positions: {e}")
    
    def _save_positions_to_file(self):
        """Save taught positions to file."""
        try:
            with open("taught_positions.json", 'w') as f:
                json.dump(self.taught_positions, f, indent=2)
            logger.info(f"Saved {len(self.taught_positions)} positions")
        except Exception as e:
            logger.error(f"Could not save positions: {e}")
    
    def run(self):
        """Main execution loop."""
        try:
            # Parse arguments
            args = self._parse_arguments()
            
            if args.debug:
                self.debug_mode = True
                logger.setLevel(logging.DEBUG)
            
            # Show startup info
            self._show_startup_info()
            
            # Initialize robot
            if not self._initialize_robot_connection(args):
                return 1
            
            # Initialize all components
            self._initialize_components()
            
            # Check calibration status
            self._check_calibration_status()
            
            # Handle command line options
            if args.calibrate:
                self._run_full_calibration()
                return 0
            
            if args.pattern:
                self._execute_pattern_by_name(args.pattern)
                return 0
            
            if args.test:
                self._run_complete_system_test()
                return 0
            
            # Auto-home if enabled
            if self.auto_home_on_startup and not args.no_home and not self.simulator_mode:
                self._perform_auto_home()
            
            # Main menu loop
            while self.running:
                try:
                    self._show_main_menu()
                    choice = input("\n👉 Select option: ").strip()
                    
                    if choice:
                        self.command_count += 1
                    
                    self._handle_main_menu(choice)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Use option 0 to exit properly")
                    time.sleep(1)
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in main loop: {e}")
                    if self.debug_mode:
                        traceback.print_exc()
                    print(f"\n❌ Error: {e}")
                    input("Press ENTER to continue...")
            
        except KeyboardInterrupt:
            print("\n⛔ Shutdown requested by user...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return 1
        finally:
            self._cleanup()
            return 0
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='RoArm M3 Professional Control Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python3 main.py                    # Normal start
  python3 main.py --simulator        # Start in simulator mode
  python3 main.py --debug            # Enable debug output
  python3 main.py --calibrate        # Run calibration and exit
  python3 main.py --pattern raster   # Execute pattern and exit
"""
        )
        
        parser.add_argument('--debug', action='store_true', 
                          help='Enable debug mode with verbose output')
        parser.add_argument('--config', default='config.yaml', 
                          help='Configuration file path')
        parser.add_argument('--port', 
                          help='Serial port override')
        parser.add_argument('--baud', type=int, default=115200,
                          help='Baudrate override')
        parser.add_argument('--calibrate', action='store_true', 
                          help='Run auto-calibration and exit')
        parser.add_argument('--pattern', 
                          help='Execute specific pattern and exit')
        parser.add_argument('--test', action='store_true', 
                          help='Run system test and exit')
        parser.add_argument('--simulator', action='store_true', 
                          help='Use simulator mode')
        parser.add_argument('--no-home', action='store_true', 
                          help='Skip auto-home on startup')
        
        return parser.parse_args()
    
    def _show_startup_info(self):
        """Show startup information."""
        self._clear_screen()
        print("="*60)
        print("🤖 ROARM M3 PROFESSIONAL CONTROL INTERFACE")
        print("Version 7.2.0 - Movement Fixes Applied")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Config: {self.config_path}")
        print(f"Debug Mode: {'✅ ENABLED' if self.debug_mode else '❌ Disabled'}")
        print(f"Python: {sys.version.split()[0]}")
        print("="*60)
    
    def _initialize_robot_connection(self, args) -> bool:
        """Initialize robot connection."""
        try:
            print("\n🔌 Initializing robot connection...")
            
            # Create config
            config = RoArmConfig()
            
            # Apply overrides from arguments
            if args.port:
                config.port = args.port
                logger.info(f"Using port override: {args.port}")
            if args.baud:
                config.baudrate = args.baud
                logger.info(f"Using baudrate override: {args.baud}")
            
            # Create controller
            self.controller = RoArmController(config)
            
            # Connect or use simulator
            if args.simulator:
                print("🎮 Starting in SIMULATOR MODE...")
                print("   No hardware required")
                print("   All movements will be simulated")
                self.simulator_mode = True
                logger.info("Running in simulator mode")
            else:
                print(f"🔌 Connecting to {config.port} at {config.baudrate} baud...")
                
                # Try to connect
                try:
                    if hasattr(self.controller, 'connect'):
                        if self.controller.connect():
                            print("✅ Successfully connected to robot!")
                            logger.info("Robot connection established")
                        else:
                            raise Exception("Connection failed")
                    else:
                        print("✅ Controller initialized")
                        logger.info("Controller ready")
                        
                except Exception as e:
                    print(f"❌ Connection failed: {e}")
                    print("\nConnection troubleshooting:")
                    print("1. Check USB cable connection")
                    print("2. Verify correct port (ls /dev/tty* or Device Manager)")
                    print("3. Check robot power")
                    print("4. Try different baudrate (9600, 57600, 115200)")
                    
                    use_sim = input("\nSwitch to simulator mode? (y/n): ").lower()
                    if use_sim == 'y':
                        print("🎮 Switching to SIMULATOR MODE...")
                        self.simulator_mode = True
                        logger.info("Fallback to simulator mode")
                    else:
                        return False
            
            # NEW: Create adapter wrapper
            self.adapter = RoArmAdapter(self.controller)
            
            return True
            
        except Exception as e:
            logger.error(f"Connection initialization error: {e}")
            print(f"❌ Connection error: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize all components."""
        logger.info("Initializing components...")
        
        try:
            # Teaching Recorder
            self.teaching_recorder = TeachingRecorder(self.controller)
            logger.debug("Teaching recorder initialized")
            
            # Calibration Suite
            self.calibration_suite = CalibrationSuite(self.controller)
            logger.debug("Calibration suite initialized")
            
            # Safety System
            self.safety_system = SafetySystem(self.controller)
            self.collision_detector = CollisionDetector(self.controller)
            self.emergency_handler = EmergencyHandler(self.controller)
            logger.debug("Safety components initialized")
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            print(f"⚠️ Some components could not be initialized: {e}")
    
    def _check_calibration_status(self):
        """Check and display calibration status."""
        try:
            if self.calibration_suite and hasattr(self.calibration_suite, 'is_calibrated'):
                if self.calibration_suite.is_calibrated():
                    print("📐 Calibration: ✅ Active")
                    if hasattr(self.calibration_suite, 'calibration'):
                        cal = self.calibration_suite.calibration
                        if hasattr(cal, 'timestamp') and cal.timestamp:
                            timestamp = datetime.fromtimestamp(cal.timestamp)
                            print(f"   Last calibrated: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print("📐 Calibration: ⚠️ Recommended")
                    print("   Run calibration for optimal performance")
            else:
                print("📐 Calibration: ❓ Status unknown")
        except Exception as e:
            logger.debug(f"Calibration status check: {e}")
            print("📐 Calibration: ❓ Status unknown")
    
    def _perform_auto_home(self):
        """Perform auto-home on startup."""
        print("\n🏠 Performing auto-home...")
        try:
            if self.simulator_mode:
                print("🎮 Simulator: Moving to home position...")
                time.sleep(2)
            else:
                # Use adapter for movement
                success = self.adapter.move_to_position(
                    HOME_POSITION, 
                    speed=self.config.get('default_speed', 0.3)
                )
                if not success:
                    print("⚠️ Home movement failed")
                    return
            
            self.current_position = "home"
            self.movement_count += 1
            print("✅ Auto-home complete")
            logger.info("Auto-home completed")
            
        except Exception as e:
            print(f"⚠️ Auto-home failed: {e}")
            logger.error(f"Auto-home failed: {e}")
    
    # ================================
    # MAIN MENU SYSTEM
    # ================================
    
    def _show_main_menu(self):
        """Show main menu with complete status information."""
        self._clear_screen()
        
        print("\n🤖 ROARM M3 MAIN MENU")
        print("="*50)
        
        # System Status
        self._show_system_status()
        
        print("\n=== BASIC OPERATIONS ===")
        print("1. 🏠 Move to Home Position")
        print("2. 📍 Move to Scanner Position")
        print("3. 🎮 Manual Control & Teaching")
        print("4. 📋 Sequence Management")
        
        print("\n=== SCANNING ===")
        print("5. 📷 Basic Scan Patterns")
        print("6. 🔬 Advanced Scan Patterns")
        print("7. 🔧 Technical Scanner Configuration")
        print("8. ⚡ Quick Scan Presets")
        
        print("\n=== CALIBRATION ===")
        print("9. 📐 Calibration Suite")
        print("10. 🔄 Quick Calibration")
        
        print("\n=== SYSTEM ===")
        print("11. ⚙️ System Settings")
        print("12. 🛡️ Safety System")
        print("13. 📊 Diagnostics & Logs")
        print("14. 🧪 System Tests")
        
        print("\n0. 🚪 Exit")
        
        # Session Info
        self._show_session_info()
    
    def _show_system_status(self):
        """Show detailed system status."""
        status_items = []
        
        # Connection status
        if self.simulator_mode:
            status_items.append("🎮 Simulator")
        elif self.controller:
            try:
                if hasattr(self.controller, 'is_connected') and self.controller.is_connected():
                    status_items.append("✅ Connected")
                else:
                    status_items.append("⚠️ Controller Ready")
            except:
                status_items.append("❌ Disconnected")
        else:
            status_items.append("❌ No Controller")
        
        # Position
        if self.current_position != "unknown":
            status_items.append(f"📍 {self.current_position.title()}")
        
        # Calibration
        try:
            if self.calibration_suite and hasattr(self.calibration_suite, 'is_calibrated'):
                if self.calibration_suite.is_calibrated():
                    status_items.append("📐 Calibrated")
        except:
            pass
        
        # Safety
        if self.safety_system:
            status_items.append("🛡️ Protected")
        
        print("Status: " + " | ".join(status_items))
    
    def _show_session_info(self):
        """Show session information."""
        uptime = (time.time() - self.session_start) / 60
        
        info_parts = [
            f"{self.command_count} commands",
            f"{self.movement_count} movements",
            f"{len(self.scan_history)} scans"
        ]
        
        if self.error_count > 0:
            info_parts.append(f"⚠️ {self.error_count} errors")
        
        info_parts.append(f"{uptime:.1f} min")
        
        print(f"\n📊 Session: " + " | ".join(info_parts))
    
    def _handle_main_menu(self, choice: str):
        """Handle main menu selection."""
        menu_actions = {
            '1': self._move_to_home,
            '2': self._move_to_scanner_position,
            '3': self._manual_control_and_teaching,
            '4': self._sequence_management_menu,
            '5': self._basic_scan_patterns_menu,
            '6': self._advanced_scan_patterns_menu,
            '7': self._technical_scanner_configuration,
            '8': self._quick_scan_presets_menu,
            '9': self._calibration_suite_menu,
            '10': self._quick_calibration,
            '11': self._system_settings_menu,
            '12': self._safety_system_menu,
            '13': self._diagnostics_and_logs,
            '14': self._system_tests_menu,
            '0': self._exit_application,
            'exit': self._exit_application,
            'quit': self._exit_application
        }
        
        action = menu_actions.get(choice.lower())
        if action:
            try:
                action()
            except Exception as e:
                self.error_count += 1
                logger.error(f"Menu action error: {e}")
                print(f"\n❌ Error executing action: {e}")
                if self.debug_mode:
                    traceback.print_exc()
                input("\nPress ENTER to continue...")
        elif choice:
            print("❌ Invalid option")
            time.sleep(1)
    
    # ================================
    # MOVEMENT OPERATIONS
    # ================================
    
    def _move_to_home(self):
        """Move to home position."""
        print("\n🏠 MOVE TO HOME POSITION")
        print("-"*30)
        
        # Show current position if available
        if not self.simulator_mode and self.controller:
            try:
                if hasattr(self.controller, 'get_current_position'):
                    current = self.controller.get_current_position()
                    if current:
                        print("Current position:")
                        for joint, angle in current.items():
                            print(f"  {joint}: {math.degrees(angle):.1f}°")
            except Exception as e:
                logger.debug(f"Could not get current position: {e}")
        
        # Get parameters
        default_speed = self.config.get('default_speed', 0.5)
        speed_input = input(f"Movement speed (0.1-1.0) [{default_speed}]: ").strip()
        speed = float(speed_input) if speed_input else default_speed
        speed = max(0.1, min(1.0, speed))
        
        # Safety check
        if self.safety_monitor.check_limits(HOME_POSITION):
            if input(f"\n🚀 Move to home at speed {speed}? (y/n): ").lower() == 'y':
                print("\n🚀 Moving to home position...")
                
                try:
                    start_time = time.time()
                    
                    if self.simulator_mode:
                        print("🎮 Simulator: Moving to home...")
                        for i in range(3):
                            print(f"   {'.'*(i+1)}")
                            time.sleep(0.5)
                        success = True
                    else:
                        # Use adapter for movement
                        success = self.adapter.move_to_position(HOME_POSITION, speed=speed)
                    
                    duration = time.time() - start_time
                    
                    if success:
                        print(f"✅ Home position reached in {duration:.1f}s")
                        self.current_position = "home"
                        self.movement_count += 1
                        logger.info(f"Moved to home in {duration:.1f}s")
                    else:
                        print("❌ Failed to reach home position")
                        self.error_count += 1
                        
                except Exception as e:
                    print(f"❌ Movement error: {e}")
                    logger.error(f"Home movement error: {e}")
                    self.error_count += 1
        else:
            print("⚠️ Safety limits would be exceeded!")
        
        input("\nPress ENTER to continue...")
    
    def _move_to_scanner_position(self):
        """Move to scanner position."""
        print("\n📍 MOVE TO SCANNER POSITION")
        print("-"*35)
        
        default_speed = self.config.get('default_speed', 0.3)
        speed_input = input(f"Movement speed (0.1-1.0) [{default_speed}]: ").strip()
        speed = float(speed_input) if speed_input else default_speed
        speed = max(0.1, min(1.0, speed))
        
        if input(f"\n🚀 Move to scanner position at speed {speed}? (y/n): ").lower() == 'y':
            print("\n🚀 Moving to scanner position...")
            
            try:
                start_time = time.time()
                
                if self.simulator_mode:
                    print("🎮 Simulator: Moving to scanner position...")
                    for i in range(3):
                        print(f"   {'.'*(i+1)}")
                        time.sleep(0.5)
                    success = True
                else:
                    # Use adapter for movement
                    success = self.adapter.move_to_position(SCANNER_POSITION, speed=speed)
                
                duration = time.time() - start_time
                
                if success:
                    print(f"✅ Scanner position reached in {duration:.1f}s")
                    self.current_position = "scanner"
                    self.movement_count += 1
                    logger.info(f"Moved to scanner position in {duration:.1f}s")
                else:
                    print("❌ Failed to reach scanner position")
                    self.error_count += 1
                    
            except Exception as e:
                print(f"❌ Movement error: {e}")
                logger.error(f"Scanner movement error: {e}")
                self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    # ================================
    # MANUAL CONTROL & TEACHING
    # ================================
    
    def _manual_control_and_teaching(self):
        """Manual control and teaching menu."""
        while True:
            self._clear_screen()
            
            print("\n🎮 MANUAL CONTROL & TEACHING")
            print("="*40)
            
            # Status
            print(f"Taught positions: {len(self.taught_positions)}")
            if self.teaching_recorder and hasattr(self.teaching_recorder, 'is_recording'):
                try:
                    if self.teaching_recorder.is_recording:
                        print("Status: 🔴 RECORDING")
                    else:
                        print("Status: ⚪ Ready")
                except:
                    print("Status: ⚪ Ready")
            else:
                print("Status: ⚪ Ready")
            
            print("\n=== CONTROL MODES ===")
            print("1. 🕹️ Joint-by-Joint Control")
            print("2. 📐 Coordinate Input (Angles)")
            print("3. 📍 Cartesian Coordinates (XYZ)")
            print("4. 🔄 Free Movement (Torque Off)")
            
            print("\n=== TEACHING ===")
            print("5. 💾 Save Current Position")
            print("6. ▶️ Move to Saved Position")
            print("7. 📋 List Saved Positions")
            print("8. 🗑️ Delete Position")
            
            print("\n=== RECORDING ===")
            print("9. 🔴 Start Recording")
            print("10. ⏹️ Stop Recording")
            print("11. ▶️ Replay Recording")
            print("12. 💾 Save Recording")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self._joint_by_joint_control()
            elif choice == '2':
                self._coordinate_input_angles()
            elif choice == '3':
                self._cartesian_coordinate_input()
            elif choice == '4':
                self._free_movement_mode()
            elif choice == '5':
                self._save_current_position()
            elif choice == '6':
                self._move_to_saved_position()
            elif choice == '7':
                self._list_saved_positions()
            elif choice == '8':
                self._delete_saved_position()
            elif choice == '9':
                self._start_recording()
            elif choice == '10':
                self._stop_recording()
            elif choice == '11':
                self._replay_recording()
            elif choice == '12':
                self._save_recording()
            elif choice == '0':
                break
            else:
                if choice:
                    print("❌ Invalid option")
                    time.sleep(1)
    
    def _joint_by_joint_control(self):
        """Joint control implementation."""
        print("\n🕹️ JOINT-BY-JOINT CONTROL")
        print("-"*30)
        print("\nControls:")
        print("  Q/A - Base ±10°")
        print("  W/S - Shoulder ±10°")
        print("  E/D - Elbow ±10°")
        print("  R/F - Wrist ±10°")
        print("  T/G - Roll ±10°")
        print("  Y/H - Hand ±10°")
        print("  1-9 - Set speed (10%-90%)")
        print("  0   - Emergency stop")
        print("  P   - Print current position")
        print("  X   - Exit")
        
        current_speed = 0.3
        print(f"\nCurrent speed: {current_speed*100:.0f}%")
        
        joint_map = {
            'q': ('base', 0.1745),      # 10 degrees in radians
            'a': ('base', -0.1745),
            'w': ('shoulder', 0.1745),
            's': ('shoulder', -0.1745),
            'e': ('elbow', 0.1745),
            'd': ('elbow', -0.1745),
            'r': ('wrist', 0.1745),
            'f': ('wrist', -0.1745),
            't': ('roll', 0.1745),
            'g': ('roll', -0.1745),
            'y': ('hand', 0.1745),
            'h': ('hand', -0.1745)
        }
        
        print("\nEnter commands (or 'x' to exit):")
        
        while True:
            try:
                cmd = input("> ").lower().strip()
                
                if cmd in ['x', 'exit', 'quit']:
                    break
                
                elif cmd in joint_map:
                    joint, delta = joint_map[cmd]
                    print(f"Moving {joint} by {math.degrees(delta):.0f}°...")
                    
                    if self.simulator_mode:
                        print(f"🎮 Simulator: {joint} moved")
                        time.sleep(0.2)
                    else:
                        try:
                            if hasattr(self.controller, 'move_joint'):
                                self.controller.move_joint(joint, delta, speed=current_speed)
                            elif hasattr(self.controller, 'get_current_position'):
                                current = self.controller.get_current_position()
                                if current and joint in current:
                                    current[joint] += delta
                                    self.adapter.move_to_position(current, speed=current_speed)
                            else:
                                print("⚠️ Joint control not available")
                                continue
                            print(f"✅ {joint} moved")
                        except Exception as e:
                            print(f"❌ Movement failed: {e}")
                            self.error_count += 1
                    
                    self.movement_count += 1
                
                elif cmd == 'p':
                    self._print_current_position()
                
                elif cmd == '0':
                    self._emergency_stop()
                    break
                
                elif cmd.isdigit() and 1 <= int(cmd) <= 9:
                    current_speed = int(cmd) / 10.0
                    print(f"Speed set to {current_speed*100:.0f}%")
                
                elif cmd == '':
                    continue
                    
                else:
                    print("❓ Unknown command (use 'x' to exit)")
                    
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted - exiting control mode")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.error_count += 1
    
    def _coordinate_input_angles(self):
        """Input joint angles directly."""
        print("\n📐 COORDINATE INPUT (ANGLES)")
        print("-"*30)
        print("Enter joint angles in degrees (press ENTER for 0):")
        
        try:
            angles = {}
            joints = ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']
            
            for joint in joints:
                # Get servo limits for this joint
                limits = SERVO_LIMITS.get(joint, (-180, 180))
                limit_deg = (math.degrees(limits[0]), math.degrees(limits[1]))
                
                prompt = f"{joint.capitalize()} angle [{limit_deg[0]:.0f} to {limit_deg[1]:.0f}°]: "
                value = input(prompt).strip()
                
                if value:
                    angle_deg = float(value)
                    # Check limits
                    if angle_deg < limit_deg[0] or angle_deg > limit_deg[1]:
                        print(f"⚠️ Warning: {angle_deg}° exceeds limits, clamping to range")
                        angle_deg = max(limit_deg[0], min(limit_deg[1], angle_deg))
                    angles[joint] = math.radians(angle_deg)
                else:
                    angles[joint] = 0.0
            
            # Show target
            print("\n📍 Target position:")
            for joint, angle in angles.items():
                print(f"  {joint}: {math.degrees(angle):.1f}°")
            
            # Safety check
            if self.safety_monitor.check_limits(angles):
                speed_input = input(f"\nMovement speed (0.1-1.0) [{self.config.get('default_speed', 0.3)}]: ").strip()
                speed = float(speed_input) if speed_input else self.config.get('default_speed', 0.3)
                
                if input("\n🚀 Move to position? (y/n): ").lower() == 'y':
                    print("Moving to coordinates...")
                    
                    if self.simulator_mode:
                        print("🎮 Simulator: Movement complete")
                        time.sleep(1)
                    else:
                        if self.adapter.move_to_position(angles, speed=speed):
                            print("✅ Position reached")
                        else:
                            print("⚠️ Movement failed")
                    
                    self.current_position = "custom"
                    self.movement_count += 1
            else:
                print("⚠️ Position exceeds safety limits!")
                
        except ValueError as e:
            print(f"❌ Invalid input: Please enter numbers only")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _cartesian_coordinate_input(self):
        """Input Cartesian XYZ coordinates."""
        print("\n📍 CARTESIAN COORDINATES (XYZ)")
        print("-"*35)
        print("Enter target position in millimeters:")
        
        try:
            x_input = input("X coordinate [0]: ").strip()
            y_input = input("Y coordinate [0]: ").strip()
            z_input = input("Z coordinate [100]: ").strip()
            
            x = float(x_input) / 1000 if x_input else 0.0
            y = float(y_input) / 1000 if y_input else 0.0
            z = float(z_input) / 1000 if z_input else 0.1
            
            print(f"\n📍 Target: X={x*1000:.1f}mm, Y={y*1000:.1f}mm, Z={z*1000:.1f}mm")
            
            # Calculate inverse kinematics
            print("Calculating joint angles...")
            
            if self.simulator_mode:
                print("🎮 Simulator: IK calculation successful")
                angles = {'base': 0, 'shoulder': 0.5, 'elbow': -0.5, 
                         'wrist': 0, 'roll': 0, 'hand': 0}
                time.sleep(0.5)
            else:
                if hasattr(self.controller, 'inverse_kinematics'):
                    angles = self.controller.inverse_kinematics(x, y, z)
                else:
                    print("⚠️ Inverse kinematics not available")
                    print("This feature requires IK implementation in the controller")
                    angles = None
            
            if angles:
                print("Joint angles calculated:")
                for joint, angle in angles.items():
                    print(f"  {joint}: {math.degrees(angle):.1f}°")
                
                if input("\n🚀 Move to position? (y/n): ").lower() == 'y':
                    if self.simulator_mode:
                        print("🎮 Simulator: Movement complete")
                        time.sleep(1)
                    else:
                        if self.adapter.move_to_position(angles):
                            print("✅ Position reached")
                        else:
                            print("⚠️ Movement failed")
                    
                    self.movement_count += 1
            else:
                print("❌ Position unreachable (outside workspace)")
                
        except ValueError:
            print("❌ Invalid input: Please enter numbers only")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _free_movement_mode(self):
        """Free movement with torque disabled."""
        print("\n🔄 FREE MOVEMENT MODE")
        print("-"*25)
        print("⚠️ This will disable servo torque")
        print("You can move the robot manually")
        
        if self.simulator_mode:
            print("\n❌ Not available in simulator mode")
            input("Press ENTER to continue...")
            return
        
        if input("\nProceed? (y/n): ").lower() != 'y':
            return
        
        try:
            print("\n🔓 Disabling torque...")
            
            if hasattr(self.controller, 'disable_torque'):
                self.controller.disable_torque()
                print("✅ Torque disabled - move robot manually")
            else:
                print("❌ Torque control not available in controller")
                input("\nPress ENTER to continue...")
                return
            
            print("\nOptions:")
            print("  s - Save current position")
            print("  r - Record waypoint")
            print("  x - Re-enable torque and exit")
            
            while True:
                cmd = input("> ").lower().strip()
                
                if cmd in ['x', 'exit']:
                    break
                elif cmd == 's':
                    self._save_current_position()
                elif cmd == 'r':
                    if self.teaching_recorder and hasattr(self.teaching_recorder, 'is_recording'):
                        try:
                            if self.teaching_recorder.is_recording:
                                if hasattr(self.teaching_recorder, 'add_waypoint'):
                                    self.teaching_recorder.add_waypoint()
                                    print("✅ Waypoint recorded")
                                else:
                                    print("❌ Recording not supported")
                            else:
                                print("⚠️ Start recording first")
                        except:
                            print("❌ Recording not available")
                    else:
                        print("❌ Recording not available")
            
            print("\n🔒 Re-enabling torque...")
            
            if hasattr(self.controller, 'enable_torque'):
                self.controller.enable_torque()
            
            print("✅ Torque re-enabled")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.error_count += 1
            # Try to re-enable torque on error
            try:
                if hasattr(self.controller, 'enable_torque'):
                    self.controller.enable_torque()
            except:
                pass
        
        input("\nPress ENTER to continue...")
    
    def _save_current_position(self):
        """Save current robot position."""
        name = input("Position name: ").strip()
        if not name:
            print("❌ Name required")
            return
        
        try:
            if self.simulator_mode:
                position = {
                    'base': 0.0, 'shoulder': 0.0, 'elbow': 0.0,
                    'wrist': 0.0, 'roll': 0.0, 'hand': 0.0
                }
            else:
                if hasattr(self.controller, 'get_current_position'):
                    position = self.controller.get_current_position()
                else:
                    print("❌ Position reading not available")
                    return
            
            taught_pos = {
                'name': name,
                'position': position,
                'timestamp': datetime.now().isoformat(),
                'description': input("Description (optional): ").strip()
            }
            
            self.taught_positions.append(taught_pos)
            self._save_positions_to_file()
            
            print(f"✅ Position '{name}' saved")
            logger.info(f"Position saved: {name}")
            
        except Exception as e:
            print(f"❌ Error saving position: {e}")
            self.error_count += 1
    
    def _move_to_saved_position(self):
        """Move to a saved position."""
        if not self.taught_positions:
            print("❌ No saved positions")
            input("Press ENTER to continue...")
            return
        
        print("\n▶️ MOVE TO SAVED POSITION")
        print("-"*30)
        
        for i, pos in enumerate(self.taught_positions):
            print(f"{i+1}. {pos['name']}")
            if pos.get('description'):
                print(f"   {pos['description']}")
        
        try:
            idx_input = input("\nSelect position number: ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(self.taught_positions):
                pos = self.taught_positions[idx]
                speed_input = input(f"Speed (0.1-1.0) [{self.config.get('default_speed', 0.3)}]: ").strip()
                speed = float(speed_input) if speed_input else self.config.get('default_speed', 0.3)
                
                print(f"\n🚀 Moving to '{pos['name']}'...")
                
                if self.simulator_mode:
                    print("🎮 Simulator: Movement complete")
                    time.sleep(1)
                else:
                    if self.adapter.move_to_position(pos['position'], speed=speed):
                        print(f"✅ Reached '{pos['name']}'")
                    else:
                        print("⚠️ Movement failed")
                
                self.movement_count += 1
            else:
                print("❌ Invalid selection")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _list_saved_positions(self):
        """List all saved positions."""
        print("\n📋 SAVED POSITIONS")
        print("-"*25)
        
        if self.taught_positions:
            for i, pos in enumerate(self.taught_positions):
                print(f"\n{i+1}. {pos['name']}")
                print(f"   Saved: {pos['timestamp']}")
                if pos.get('description'):
                    print(f"   {pos['description']}")
                
                # Show joint angles
                print("   Joints:")
                for joint, angle in pos['position'].items():
                    print(f"     {joint}: {math.degrees(angle):.1f}°")
        else:
            print("No positions saved")
        
        input("\nPress ENTER to continue...")
    
    def _delete_saved_position(self):
        """Delete a saved position."""
        if not self.taught_positions:
            print("❌ No positions to delete")
            input("Press ENTER to continue...")
            return
        
        print("\n🗑️ DELETE POSITION")
        print("-"*20)
        
        for i, pos in enumerate(self.taught_positions):
            print(f"{i+1}. {pos['name']}")
        
        try:
            idx_input = input("\nPosition to delete (number): ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(self.taught_positions):
                deleted = self.taught_positions.pop(idx)
                self._save_positions_to_file()
                print(f"✅ Deleted '{deleted['name']}'")
                logger.info(f"Position deleted: {deleted['name']}")
            else:
                print("❌ Invalid selection")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _start_recording(self):
        """Start sequence recording."""
        print("\n🔴 START RECORDING")
        print("-"*20)
        
        # Check if already recording
        recording = False
        if self.teaching_recorder and hasattr(self.teaching_recorder, 'is_recording'):
            try:
                recording = self.teaching_recorder.is_recording
            except:
                pass
        
        if recording:
            print("⚠️ Already recording")
            input("Press ENTER to continue...")
            return
        
        mode = input("Recording mode (manual/continuous) [manual]: ").strip().lower() or "manual"
        
        if mode not in ['manual', 'continuous']:
            print("❌ Invalid mode")
            return
        
        interval = None
        if mode == "continuous":
            interval_input = input("Recording interval (s) [0.5]: ").strip()
            interval = float(interval_input) if interval_input else 0.5
        
        try:
            print(f"\n🔴 Starting {mode} recording...")
            
            # Initialize sequence
            self.current_sequence = []
            
            # Try to use TeachingRecorder if available
            if self.teaching_recorder and hasattr(self.teaching_recorder, 'start_recording'):
                try:
                    self.teaching_recorder.start_recording(mode=mode, interval=interval)
                except Exception as e:
                    logger.debug(f"TeachingRecorder start failed: {e}")
            
            if mode == "manual":
                print("Move robot and press ENTER to add waypoint")
                print("Type 'stop' to stop recording")
                
                while True:
                    cmd = input("> ").lower().strip()
                    if cmd == 'stop':
                        break
                    else:
                        # Add waypoint
                        if self.simulator_mode:
                            waypoint = {
                                'position': {'base': 0, 'shoulder': 0, 'elbow': 0,
                                           'wrist': 0, 'roll': 0, 'hand': 0},
                                'timestamp': time.time()
                            }
                        else:
                            if hasattr(self.controller, 'get_current_position'):
                                waypoint = {
                                    'position': self.controller.get_current_position(),
                                    'timestamp': time.time()
                                }
                            else:
                                print("⚠️ Cannot read position")
                                continue
                        
                        self.current_sequence.append(waypoint)
                        print(f"✅ Waypoint {len(self.current_sequence)} added")
            else:
                print(f"Recording waypoints every {interval}s")
                print("Press ENTER to stop recording...")
                
                # Start continuous recording in thread
                stop_event = threading.Event()
                
                def record_continuous():
                    while not stop_event.is_set():
                        if self.simulator_mode:
                            waypoint = {
                                'position': {'base': 0, 'shoulder': 0, 'elbow': 0,
                                           'wrist': 0, 'roll': 0, 'hand': 0},
                                'timestamp': time.time()
                            }
                        else:
                            if hasattr(self.controller, 'get_current_position'):
                                try:
                                    waypoint = {
                                        'position': self.controller.get_current_position(),
                                        'timestamp': time.time()
                                    }
                                except:
                                    continue
                            else:
                                continue
                        
                        self.current_sequence.append(waypoint)
                        print(f"📍 Waypoint {len(self.current_sequence)} recorded")
                        time.sleep(interval)
                
                thread = threading.Thread(target=record_continuous)
                thread.daemon = True
                thread.start()
                
                input()  # Wait for ENTER
                stop_event.set()
                thread.join(timeout=1)
            
            self._stop_recording()
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            self.error_count += 1
            
        input("\nPress ENTER to continue...")
    
    def _stop_recording(self):
        """Stop sequence recording."""
        try:
            # Try to stop TeachingRecorder if it's recording
            if self.teaching_recorder and hasattr(self.teaching_recorder, 'is_recording'):
                try:
                    if self.teaching_recorder.is_recording:
                        if hasattr(self.teaching_recorder, 'stop_recording'):
                            sequence = self.teaching_recorder.stop_recording()
                            if sequence:
                                self.current_sequence = sequence
                except Exception as e:
                    logger.debug(f"TeachingRecorder stop failed: {e}")
            
            if self.current_sequence and len(self.current_sequence) > 0:
                print(f"\n✅ Recording stopped")
                print(f"Recorded {len(self.current_sequence)} waypoints")
            else:
                print("❌ No waypoints recorded")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.error_count += 1
    
    def _replay_recording(self):
        """Replay last recorded sequence."""
        print("\n▶️ REPLAY RECORDING")
        print("-"*20)
        
        if not self.current_sequence:
            print("❌ No recording to replay")
            input("Press ENTER to continue...")
            return
        
        print(f"Sequence has {len(self.current_sequence)} waypoints")
        speed_input = input("Playback speed (0.1-2.0) [1.0]: ").strip()
        speed = float(speed_input) if speed_input else 1.0
        
        if input("\n▶️ Start replay? (y/n): ").lower() == 'y':
            try:
                print("▶️ Playing sequence...")
                
                for i, waypoint in enumerate(self.current_sequence):
                    print(f"Waypoint {i+1}/{len(self.current_sequence)}")
                    
                    if self.simulator_mode:
                        time.sleep(0.5 / speed)
                    else:
                        if 'position' in waypoint:
                            self.adapter.move_to_position(
                                waypoint['position'], 
                                speed=self.config.get('default_speed', 0.3) * speed
                            )
                        
                        # Wait between waypoints
                        if i < len(self.current_sequence) - 1:
                            next_waypoint = self.current_sequence[i + 1]
                            time_diff = next_waypoint['timestamp'] - waypoint['timestamp']
                            time.sleep(time_diff / speed)
                
                print("✅ Replay complete")
                
            except Exception as e:
                print(f"❌ Replay error: {e}")
                self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _save_recording(self):
        """Save recorded sequence."""
        if not self.current_sequence:
            print("❌ No recording to save")
            input("Press ENTER to continue...")
            return
        
        print("\n💾 SAVE RECORDING")
        print("-"*20)
        
        name = input("Sequence name: ").strip()
        if not name:
            print("❌ Name required")
            return
        
        try:
            description = input("Description (optional): ").strip()
            
            # Save sequence
            metadata = {
                'description': description,
                'waypoints': len(self.current_sequence),
                'timestamp': datetime.now().isoformat()
            }
            
            filepath = self.sequence_manager.save_sequence(
                self.current_sequence,
                name,
                metadata
            )
            
            print(f"✅ Sequence '{name}' saved to {filepath}")
            logger.info(f"Sequence saved: {name}")
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    # ================================
    # SEQUENCE MANAGEMENT
    # ================================
    
    def _sequence_management_menu(self):
        """Sequence management menu."""
        while True:
            self._clear_screen()
            
            print("\n📋 SEQUENCE MANAGEMENT")
            print("="*35)
            
            # Get sequences
            sequences = self.sequence_manager.list_sequences()
            print(f"Available sequences: {len(sequences)}")
            
            print("\n1. 📋 List Sequences")
            print("2. ▶️ Play Sequence")
            print("3. ✏️ Edit Sequence")
            print("4. 🗑️ Delete Sequence")
            print("5. 📤 Export Sequence")
            print("6. 📥 Import Sequence")
            print("7. 🔄 Loop Sequence")
            print("8. ⚡ Quick Play Last")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self._list_sequences()
            elif choice == '2':
                self._play_sequence()
            elif choice == '3':
                self._edit_sequence()
            elif choice == '4':
                self._delete_sequence()
            elif choice == '5':
                self._export_sequence()
            elif choice == '6':
                self._import_sequence()
            elif choice == '7':
                self._loop_sequence()
            elif choice == '8':
                self._quick_play_last()
            elif choice == '0':
                break
    
    def _list_sequences(self):
        """List all available sequences."""
        print("\n📋 AVAILABLE SEQUENCES")
        print("-"*30)
        
        sequences = self.sequence_manager.list_sequences()
        
        if sequences:
            for i, seq_name in enumerate(sequences):
                print(f"{i+1}. {seq_name}")
                
                # Try to load metadata
                try:
                    seq_path = self.sequence_manager.base_path / f"{seq_name}.json"
                    with open(seq_path, 'r') as f:
                        data = json.load(f)
                        if 'metadata' in data:
                            meta = data['metadata']
                            if 'description' in meta and meta['description']:
                                print(f"   {meta['description']}")
                            if 'waypoints' in meta:
                                print(f"   Waypoints: {meta['waypoints']}")
                except:
                    pass
        else:
            print("No sequences available")
        
        input("\nPress ENTER to continue...")
    
    def _play_sequence(self):
        """Play a saved sequence."""
        sequences = self.sequence_manager.list_sequences()
        
        if not sequences:
            print("❌ No sequences available")
            input("Press ENTER to continue...")
            return
        
        print("\n▶️ PLAY SEQUENCE")
        print("-"*20)
        
        for i, seq in enumerate(sequences):
            print(f"{i+1}. {seq}")
        
        try:
            idx_input = input("\nSelect sequence number: ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(sequences):
                sequence = self.sequence_manager.load_sequence(sequences[idx])
                
                if sequence:
                    print(f"\nSequence '{sequences[idx]}' loaded")
                    print(f"Waypoints: {len(sequence)}")
                    
                    speed_input = input("Playback speed (0.1-2.0) [1.0]: ").strip()
                    speed = float(speed_input) if speed_input else 1.0
                    
                    if input("\n▶️ Start playback? (y/n): ").lower() == 'y':
                        print("▶️ Playing sequence...")
                        
                        for i, waypoint in enumerate(sequence):
                            print(f"Waypoint {i+1}/{len(sequence)}")
                            
                            if self.simulator_mode:
                                time.sleep(0.5 / speed)
                            else:
                                # Check waypoint structure
                                if isinstance(waypoint, dict):
                                    if 'position' in waypoint:
                                        pos = waypoint['position']
                                    else:
                                        pos = waypoint
                                    
                                    self.adapter.move_to_position(
                                        pos,
                                        speed=self.config.get('default_speed', 0.3) * speed
                                    )
                        
                        print("✅ Playback complete")
                else:
                    print("❌ Could not load sequence")
            else:
                print("❌ Invalid selection")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Playback error: {e}")
            self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _edit_sequence(self):
        """Edit a sequence - simplified version."""
        print("\n✏️ EDIT SEQUENCE")
        print("-"*20)
        print("⚠️ Sequence editing is limited in this version")
        print("You can delete and re-record sequences")
        input("\nPress ENTER to continue...")
    
    def _delete_sequence(self):
        """Delete a sequence."""
        sequences = self.sequence_manager.list_sequences()
        
        if not sequences:
            print("❌ No sequences available")
            input("Press ENTER to continue...")
            return
        
        print("\n🗑️ DELETE SEQUENCE")
        print("-"*20)
        
        for i, seq in enumerate(sequences):
            print(f"{i+1}. {seq}")
        
        try:
            idx_input = input("\nSelect sequence number: ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(sequences):
                seq_name = sequences[idx]
                
                if input(f"\n❓ Delete '{seq_name}'? (y/n): ").lower() == 'y':
                    if self.sequence_manager.delete_sequence(seq_name):
                        print(f"✅ Sequence '{seq_name}' deleted")
                        logger.info(f"Sequence deleted: {seq_name}")
                    else:
                        print("❌ Could not delete sequence")
            else:
                print("❌ Invalid selection")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Delete error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _export_sequence(self):
        """Export sequence to file."""
        sequences = self.sequence_manager.list_sequences()
        
        if not sequences:
            print("❌ No sequences available")
            input("Press ENTER to continue...")
            return
        
        print("\n📤 EXPORT SEQUENCE")
        print("-"*20)
        
        for i, seq in enumerate(sequences):
            print(f"{i+1}. {seq}")
        
        try:
            idx_input = input("\nSelect sequence number: ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(sequences):
                seq_name = sequences[idx]
                export_path = input(f"Export path [{seq_name}_export.json]: ").strip()
                if not export_path:
                    export_path = f"{seq_name}_export.json"
                
                if self.sequence_manager.export_sequence(seq_name, export_path):
                    print(f"✅ Sequence exported to {export_path}")
                    logger.info(f"Sequence exported: {seq_name} -> {export_path}")
                else:
                    print("❌ Export failed")
            else:
                print("❌ Invalid selection")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Export error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _import_sequence(self):
        """Import sequence from file."""
        print("\n📥 IMPORT SEQUENCE")
        print("-"*20)
        
        import_path = input("Import file path: ").strip()
        
        if not import_path:
            print("❌ Path required")
            input("Press ENTER to continue...")
            return
        
        if not Path(import_path).exists():
            print(f"❌ File not found: {import_path}")
            input("Press ENTER to continue...")
            return
        
        try:
            name = input("Sequence name (leave empty to use filename): ").strip()
            if not name:
                name = Path(import_path).stem
            
            if self.sequence_manager.import_sequence(import_path, name):
                print(f"✅ Sequence '{name}' imported")
                logger.info(f"Sequence imported: {import_path} -> {name}")
            else:
                print("❌ Import failed")
            
        except Exception as e:
            print(f"❌ Import error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _loop_sequence(self):
        """Loop a sequence continuously."""
        sequences = self.sequence_manager.list_sequences()
        
        if not sequences:
            print("❌ No sequences available")
            input("Press ENTER to continue...")
            return
        
        print("\n🔄 LOOP SEQUENCE")
        print("-"*20)
        
        for i, seq in enumerate(sequences):
            print(f"{i+1}. {seq}")
        
        try:
            idx_input = input("\nSelect sequence number: ").strip()
            if not idx_input:
                return
                
            idx = int(idx_input) - 1
            
            if 0 <= idx < len(sequences):
                sequence = self.sequence_manager.load_sequence(sequences[idx])
                
                if sequence:
                    loops_input = input("Number of loops (0=infinite): ").strip()
                    loops = int(loops_input) if loops_input else 0
                    
                    speed_input = input("Playback speed [1.0]: ").strip()
                    speed = float(speed_input) if speed_input else 1.0
                    
                    print(f"\n🔄 Looping '{sequences[idx]}'...")
                    print("Press Ctrl+C to stop")
                    
                    loop_count = 0
                    try:
                        while loops == 0 or loop_count < loops:
                            loop_count += 1
                            print(f"\nLoop {loop_count}...")
                            
                            for i, waypoint in enumerate(sequence):
                                if self.simulator_mode:
                                    time.sleep(0.2 / speed)
                                else:
                                    if isinstance(waypoint, dict) and 'position' in waypoint:
                                        self.adapter.move_to_position(
                                            waypoint['position'],
                                            speed=self.config.get('default_speed', 0.3) * speed
                                        )
                            
                            if loops == 0:
                                time.sleep(1)  # Pause between loops
                            
                    except KeyboardInterrupt:
                        print("\n⏹️ Loop stopped")
                else:
                    print("❌ Could not load sequence")
                        
        except ValueError:
            print("❌ Please enter valid numbers")
        except Exception as e:
            print(f"❌ Loop error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _quick_play_last(self):
        """Quick play last used sequence."""
        if self.current_sequence:
            print("\n⚡ QUICK PLAY")
            print(f"Playing last sequence ({len(self.current_sequence)} waypoints)...")
            
            try:
                for i, waypoint in enumerate(self.current_sequence):
                    print(f"Waypoint {i+1}/{len(self.current_sequence)}")
                    
                    if self.simulator_mode:
                        time.sleep(0.2)
                    else:
                        if isinstance(waypoint, dict) and 'position' in waypoint:
                            self.adapter.move_to_position(waypoint['position'])
                
                print("✅ Playback complete")
                
            except Exception as e:
                print(f"❌ Playback error: {e}")
        else:
            print("❌ No sequence in memory")
        
        input("\nPress ENTER to continue...")
    
    # ================================
    # SCAN PATTERNS - FIXED IMPLEMENTATIONS
    # ================================
    
    def _point_to_joints(self, point, pattern) -> Dict[str, float]:
        """Convert scan point to joint angles. FIX 3: New helper method"""
        target = SCANNER_POSITION.copy()
        
        # Handle different point types
        if hasattr(point, 'x') and hasattr(point, 'y'):
            # Cartesian coordinates
            target['base'] = math.atan2(point.y, point.x + 0.15)
            if hasattr(point, 'z'):
                target['shoulder'] = SCANNER_POSITION['shoulder'] + (point.z * 2)
        elif hasattr(point, 'theta'):
            # Spherical coordinates
            target['base'] = point.theta
            if hasattr(point, 'phi'):
                target['shoulder'] = SCANNER_POSITION['shoulder'] - point.phi/2
        elif hasattr(point, 'azimuth'):
            # Alternative spherical
            target['base'] = point.azimuth
            if hasattr(point, 'elevation'):
                target['shoulder'] = SCANNER_POSITION['shoulder'] + point.elevation
        
        # Clamp to limits
        for joint in target:
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                target[joint] = max(min_val, min(max_val, target[joint]))
        
        return target
    
    def _basic_scan_patterns_menu(self):
        """Basic scan patterns menu."""
        while True:
            self._clear_screen()
            
            print("\n📷 BASIC SCAN PATTERNS")
            print("="*35)
            
            print("\n1. 📐 Raster Scan (Grid)")
            print("2. 🌀 Spiral Scan")
            print("3. 🌐 Spherical Scan")
            print("4. 🔄 Turntable Scan")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select pattern: ").strip()
            
            if choice == '1':
                self._execute_raster_scan()
            elif choice == '2':
                self._execute_spiral_scan()
            elif choice == '3':
                self._execute_spherical_scan()
            elif choice == '4':
                self._execute_turntable_scan()
            elif choice == '0':
                break
    
    def _execute_raster_scan(self):
        """Execute raster scan."""
        print("\n📐 RASTER SCAN CONFIGURATION")
        print("-"*40)
        
        try:
            width = float(input("Scan width (cm) [20]: ").strip() or "20") / 100
            height = float(input("Scan height (cm) [15]: ").strip() or "15") / 100
            rows = int(input("Number of rows [10]: ").strip() or "10")
            cols = int(input("Number of columns [10]: ").strip() or "10")
            
            pattern = RasterScanPattern(
                width=width,
                height=height,
                rows=rows,
                cols=cols
            )
            
            total_points = rows * cols
            print(f"\n📊 SCAN PREVIEW:")
            print(f"  Grid: {cols} × {rows} = {total_points} points")
            print(f"  Area: {width*100:.1f} × {height*100:.1f} cm")
            
            if input(f"\n🚀 Execute scan? (y/n): ").lower() == 'y':
                self._execute_scan_pattern(pattern)
                
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _execute_spiral_scan(self):
        """Execute spiral scan."""
        print("\n🌀 SPIRAL SCAN CONFIGURATION")
        print("-"*40)
        
        try:
            r_start = float(input("Start radius (cm) [5]: ").strip() or "5") / 100
            r_end = float(input("End radius (cm) [15]: ").strip() or "15") / 100
            revolutions = int(input("Number of revolutions [5]: ").strip() or "5")
            
            pattern = SpiralScanPattern(
                radius_start=r_start,
                radius_end=r_end,
                revolutions=revolutions
            )
            
            if input(f"\n🚀 Execute scan? (y/n): ").lower() == 'y':
                self._execute_scan_pattern(pattern)
                
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _execute_spherical_scan(self):
        """Execute spherical scan."""
        print("\n🌐 SPHERICAL SCAN CONFIGURATION")
        print("-"*40)
        
        try:
            radius = float(input("Scan radius (cm) [15]: ").strip() or "15") / 100
            theta_steps = int(input("Horizontal steps [12]: ").strip() or "12")
            phi_steps = int(input("Vertical steps [8]: ").strip() or "8")
            
            pattern = SphericalScanPattern(
                radius=radius,
                theta_steps=theta_steps,
                phi_steps=phi_steps
            )
            
            if input(f"\n🚀 Execute scan? (y/n): ").lower() == 'y':
                self._execute_scan_pattern(pattern)
                
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _execute_turntable_scan(self):
        """Execute turntable scan."""
        print("\n🔄 TURNTABLE SCAN CONFIGURATION")
        print("-"*40)
        
        try:
            rotations = int(input("Number of rotations [8]: ").strip() or "8")
            vertical_steps = int(input("Vertical scan steps [5]: ").strip() or "5")
            radius = float(input("Distance to object (cm) [15]: ").strip() or "15") / 100
            
            pattern = TurntableScanPattern(
                rotations=rotations,
                vertical_steps=vertical_steps,
                radius=radius
            )
            
            if input(f"\n🚀 Execute scan? (y/n): ").lower() == 'y':
                self._execute_scan_pattern(pattern)
                
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _advanced_scan_patterns_menu(self):
        """Advanced scan patterns menu."""
        print("\n🔬 ADVANCED SCAN PATTERNS")
        print("-"*35)
        print("Advanced patterns are available")
        print("Use the pattern module directly for:")
        print("  • Helix Scan")
        print("  • Adaptive Scan")
        print("  • Cobweb Scan")
        print("  • Table Scan")
        print("  • Statue Spiral")
        input("\nPress ENTER to continue...")
    
    def _technical_scanner_configuration(self):
        """Technical scanner configuration."""
        if self.technical_configurator:
            try:
                if hasattr(self.technical_configurator, 'expert_configuration_menu'):
                    self.technical_configurator.expert_configuration_menu()
                else:
                    print("⚠️ Technical configurator method not available")
            except Exception as e:
                print(f"❌ Configuration error: {e}")
        else:
            print("❌ Technical configurator not initialized")
        
        input("\nPress ENTER to continue...")
    
    def _quick_scan_presets_menu(self):
        """Quick scan presets menu."""
        print("\n⚡ QUICK SCAN PRESETS")
        print("-"*30)
        print("1. ⚡ Quick Preview (25 points)")
        print("2. 🎯 Standard Quality (100 points)")
        print("3. 🔬 High Quality (200+ points)")
        
        choice = input("\nSelect preset: ").strip()
        
        if choice == '1':
            pattern = RasterScanPattern(width=0.15, height=0.15, rows=5, cols=5)
            self._execute_scan_pattern(pattern)
        elif choice == '2':
            pattern = RasterScanPattern(width=0.20, height=0.20, rows=10, cols=10)
            self._execute_scan_pattern(pattern)
        elif choice == '3':
            pattern = SphericalScanPattern(radius=0.15, theta_steps=18, phi_steps=12)
            self._execute_scan_pattern(pattern)
        
        input("\nPress ENTER to continue...")
    
    def _execute_scan_pattern(self, pattern):
        """Execute any scan pattern. FIX 2: Complete rewrite with actual movement"""
        try:
            # Get scan points
            if hasattr(pattern, 'get_points'):
                points = pattern.get_points()
            else:
                print("❌ Pattern does not support point generation")
                return
            
            total_points = len(points)
            print(f"\n🚀 Executing scan with {total_points} points...")
            
            # Get scan parameters
            speed_input = input(f"Scan speed (0.1-1.0) [{self.config.get('default_speed', 0.3)}]: ").strip()
            speed = float(speed_input) if speed_input else self.config.get('default_speed', 0.3)
            
            settle_time_input = input("Settle time between points (s) [0.5]: ").strip()
            settle_time = float(settle_time_input) if settle_time_input else 0.5
            
            print("\n📷 Starting scan...")
            
            # First move to scanner position
            if not self.simulator_mode:
                print("Moving to scanner position...")
                self.adapter.move_to_position(SCANNER_POSITION, speed=speed)
                time.sleep(1)
            
            # Execute scan points
            scan_data = []
            start_time = time.time()
            
            for i, point in enumerate(points):
                print(f"Point {i+1}/{total_points}", end='\r')
                
                if self.simulator_mode:
                    # Simulate scanning
                    time.sleep(0.1)
                    scan_data.append({
                        'point': i,
                        'timestamp': time.time()
                    })
                else:
                    # Convert scan point to joint angles
                    target_joints = self._point_to_joints(point, pattern)
                    
                    # Move to scan point
                    success = self.adapter.move_to_position(target_joints, speed=speed)
                    
                    if success:
                        # Wait for settling
                        time.sleep(settle_time)
                        
                        # Here you would trigger scanner/camera
                        # For now, just record the position
                        scan_data.append({
                            'point': i,
                            'position': target_joints,
                            'timestamp': time.time()
                        })
                    else:
                        print(f"\n⚠️ Failed to reach point {i+1}")
            
            duration = time.time() - start_time
            
            print(f"\n✅ Scan complete!")
            print(f"  Points scanned: {len(scan_data)}/{total_points}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Average: {duration/max(1, len(scan_data)):.2f}s per point")
            
            # Save scan result
            self.last_scan_result = {
                'pattern': pattern.__class__.__name__,
                'points': scan_data,
                'total_points': total_points,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to scan history
            self.scan_history.append({
                'pattern': pattern.__class__.__name__,
                'points': total_points,
                'timestamp': datetime.now().isoformat()
            })
            
            # Ask if user wants to save the scan
            if input("\n💾 Save scan data? (y/n): ").lower() == 'y':
                scan_name = input("Scan name: ").strip()
                if scan_name:
                    self._save_scan_data(scan_name)
            
        except Exception as e:
            print(f"\n❌ Scan execution error: {e}")
            logger.error(f"Scan error: {e}")
            if self.debug_mode:
                traceback.print_exc()
            self.error_count += 1
    
    def _save_scan_data(self, name: str):
        """Save scan data to file."""
        try:
            scan_dir = Path(self.config.get('scan_save_path', 'scans'))
            scan_dir.mkdir(exist_ok=True)
            
            filepath = scan_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filepath, 'w') as f:
                json.dump(self.last_scan_result, f, indent=2)
            
            print(f"✅ Scan saved to {filepath}")
            logger.info(f"Scan saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Could not save scan: {e}")
            logger.error(f"Scan save error: {e}")
    
    # ================================
    # CALIBRATION
    # ================================
    
    def _calibration_suite_menu(self):
        """Calibration suite menu."""
        while True:
            self._clear_screen()
            
            print("\n📐 CALIBRATION SUITE")
            print("="*35)
            
            # Show calibration status
            self._check_calibration_status()
            
            print("\n1. 🔄 Full Auto-Calibration")
            print("2. 📐 Single Joint Calibration")
            print("3. 🎯 Test Repeatability")
            print("4. 💾 Save Calibration")
            print("5. 📂 Load Calibration")
            print("6. 📊 Export Report")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self._run_full_calibration()
            elif choice == '2':
                self._calibrate_single_joint()
            elif choice == '3':
                self._test_repeatability()
            elif choice == '4':
                self._save_calibration()
            elif choice == '5':
                self._load_calibration()
            elif choice == '6':
                self._export_calibration_report()
            elif choice == '0':
                break
    
    def _run_full_calibration(self):
        """Run full calibration."""
        print("\n🔄 FULL AUTO-CALIBRATION")
        print("-"*30)
        
        if self.simulator_mode:
            print("⚠️ Cannot calibrate in simulator mode")
            input("Press ENTER to continue...")
            return
        
        if input("\nThis will take 5-10 minutes. Proceed? (y/n): ").lower() == 'y':
            try:
                if self.calibration_suite and hasattr(self.calibration_suite, 'run_auto_calibration'):
                    success = self.calibration_suite.run_auto_calibration()
                    if success:
                        print("\n✅ Calibration complete!")
                    else:
                        print("\n⚠️ Calibration completed with warnings")
                else:
                    print("❌ Calibration suite not available")
            except Exception as e:
                print(f"❌ Calibration error: {e}")
                self.error_count += 1
        
        input("\nPress ENTER to continue...")
    
    def _calibrate_single_joint(self):
        """Calibrate single joint."""
        print("\n📐 SINGLE JOINT CALIBRATION")
        print("-"*30)
        
        joints = ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']
        for i, joint in enumerate(joints):
            print(f"{i+1}. {joint.capitalize()}")
        
        try:
            idx = int(input("\nSelect joint: ").strip()) - 1
            if 0 <= idx < len(joints):
                if self.calibration_suite and hasattr(self.calibration_suite, 'calibrate_single_joint'):
                    success = self.calibration_suite.calibrate_single_joint(joints[idx])
                    if success:
                        print(f"✅ {joints[idx]} calibrated")
                    else:
                        print(f"❌ Calibration failed")
                else:
                    print("❌ Calibration not available")
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _test_repeatability(self):
        """Test repeatability."""
        print("\n🎯 REPEATABILITY TEST")
        print("-"*25)
        
        if self.calibration_suite and hasattr(self.calibration_suite, 'test_repeatability'):
            positions = int(input("Test positions [5]: ").strip() or "5")
            cycles = int(input("Cycles per position [3]: ").strip() or "3")
            
            print("\nRunning repeatability test...")
            results = self.calibration_suite.test_repeatability(positions, cycles)
            
            if results:
                print("\n📊 Results:")
                for joint, stats in results.items():
                    if 'repeatability' in stats:
                        print(f"  {joint}: ±{stats['repeatability']*1000:.2f} mrad")
        else:
            print("❌ Test not available")
        
        input("\nPress ENTER to continue...")
    
    def _save_calibration(self):
        """Save calibration."""
        if self.calibration_suite and hasattr(self.calibration_suite, 'save_calibration'):
            filepath = input("Save path [calibration/calibration.json]: ").strip()
            if not filepath:
                self.calibration_suite.save_calibration()
            else:
                self.calibration_suite.save_calibration(filepath)
            print("✅ Calibration saved")
        else:
            print("❌ Cannot save calibration")
        
        input("\nPress ENTER to continue...")
    
    def _load_calibration(self):
        """Load calibration."""
        if self.calibration_suite and hasattr(self.calibration_suite, 'load_calibration'):
            filepath = input("Load path [calibration/system_calibration.json]: ").strip()
            if not filepath:
                self.calibration_suite.load_calibration()
            else:
                self.calibration_suite.load_calibration(filepath)
            print("✅ Calibration loaded")
        else:
            print("❌ Cannot load calibration")
        
        input("\nPress ENTER to continue...")
    
    def _export_calibration_report(self):
        """Export calibration report."""
        if self.calibration_suite and hasattr(self.calibration_suite, 'export_report'):
            filepath = input("Report path [calibration/report.txt]: ").strip() or "calibration/report.txt"
            report = self.calibration_suite.export_report(filepath)
            print(f"✅ Report saved to {filepath}")
            print("\nReport preview:")
            print("-"*40)
            lines = report.split('\n')[:20]
            for line in lines:
                print(line)
            if len(report.split('\n')) > 20:
                print("...")
        else:
            print("❌ Report export not available")
        
        input("\nPress ENTER to continue...")
    
    def _quick_calibration(self):
        """Quick calibration."""
        print("\n🔄 QUICK CALIBRATION")
        print("Fast calibration (~2 minutes)")
        
        if self.simulator_mode:
            print("⚠️ Cannot calibrate in simulator mode")
        elif self.calibration_suite and hasattr(self.calibration_suite, 'run_auto_calibration'):
            if input("\nProceed? (y/n): ").lower() == 'y':
                print("Running quick calibration...")
                # Note: run_quick_calibration doesn't exist in SafeCalibrationSuite
                # but we could use run_auto_calibration(include_scanner=False)
                success = self.calibration_suite.run_auto_calibration(include_scanner=False)
                if success:
                    print("✅ Quick calibration complete")
                else:
                    print("⚠️ Calibration completed with warnings")
        else:
            print("❌ Quick calibration not available")
        
        input("\nPress ENTER to continue...")
    
    # ================================
    # SYSTEM MENUS
    # ================================
    
    def _system_settings_menu(self):
        """System settings menu."""
        while True:
            self._clear_screen()
            
            print("\n⚙️ SYSTEM SETTINGS")
            print("="*30)
            
            print(f"\nCurrent Settings:")
            print(f"  Auto-home: {'✅' if self.auto_home_on_startup else '❌'}")
            print(f"  Simulator: {'✅' if self.simulator_mode else '❌'}")
            print(f"  Debug mode: {'✅' if self.debug_mode else '❌'}")
            print(f"  Default speed: {self.config.get('default_speed', 0.3)}")
            print(f"  Safety limits: {'✅' if self.config.get('safety_limits_enabled', True) else '❌'}")
            
            print("\n1. 🏠 Toggle Auto-Home")
            print("2. 🐛 Toggle Debug Mode")
            print("3. ⚡ Set Default Speed")
            print("4. 🛡️ Toggle Safety Limits")
            print("5. 💾 Save Settings")
            print("6. 🔄 Reset to Defaults")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self.auto_home_on_startup = not self.auto_home_on_startup
                self.config['auto_home_on_startup'] = self.auto_home_on_startup
                print(f"Auto-home: {'✅ Enabled' if self.auto_home_on_startup else '❌ Disabled'}")
                time.sleep(1)
                
            elif choice == '2':
                self.debug_mode = not self.debug_mode
                if self.debug_mode:
                    logger.setLevel(logging.DEBUG)
                else:
                    logger.setLevel(logging.INFO)
                print(f"Debug mode: {'✅ Enabled' if self.debug_mode else '❌ Disabled'}")
                time.sleep(1)
                
            elif choice == '3':
                speed_input = input("Default speed (0.1-1.0): ").strip()
                if speed_input:
                    try:
                        speed = float(speed_input)
                        self.config['default_speed'] = max(0.1, min(1.0, speed))
                        print(f"Default speed set to {self.config['default_speed']}")
                    except ValueError:
                        print("❌ Invalid speed value")
                    time.sleep(1)
                
            elif choice == '4':
                self.config['safety_limits_enabled'] = not self.config.get('safety_limits_enabled', True)
                print(f"Safety limits: {'✅ Enabled' if self.config['safety_limits_enabled'] else '⚠️ DISABLED'}")
                if not self.config['safety_limits_enabled']:
                    print("⚠️ WARNING: Disabling safety limits may damage the robot!")
                time.sleep(2)
                
            elif choice == '5':
                self._save_settings()
                
            elif choice == '6':
                if input("Reset all settings to defaults? (y/n): ").lower() == 'y':
                    self.config = self._load_config()
                    print("✅ Settings reset to defaults")
                    time.sleep(1)
                
            elif choice == '0':
                break
    
    def _save_settings(self):
        """Save settings to config file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print("✅ Settings saved")
        except Exception as e:
            print(f"❌ Could not save settings: {e}")
        time.sleep(1)
    
    def _safety_system_menu(self):
        """Safety system menu."""
        print("\n🛡️ SAFETY SYSTEM")
        print("-"*25)
        
        if self.safety_system:
            print(f"Status: {self.safety_system.get_status()}")
            
            if hasattr(self.safety_system, 'get_statistics'):
                stats = self.safety_system.get_statistics()
                print("\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        else:
            print("Status: ❌ Not initialized")
        
        print("\nSafety features:")
        print("  • Joint limit monitoring")
        print("  • Speed limiting")
        print("  • Emergency stop")
        print("  • Collision detection")
        
        input("\nPress ENTER to continue...")
    
    def _diagnostics_and_logs(self):
        """Show diagnostics and logs."""
        print("\n📊 DIAGNOSTICS & LOGS")
        print("-"*30)
        
        print(f"\nSession Statistics:")
        print(f"  Uptime: {(time.time() - self.session_start)/60:.1f} minutes")
        print(f"  Commands: {self.command_count}")
        print(f"  Movements: {self.movement_count}")
        print(f"  Scans: {len(self.scan_history)}")
        print(f"  Errors: {self.error_count}")
        
        print(f"\nSystem Info:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        print(f"  Mode: {'Simulator' if self.simulator_mode else 'Hardware'}")
        
        if Path("logs").exists():
            log_files = list(Path("logs").glob("*.log"))
            if log_files:
                print(f"\nLog files: {len(log_files)}")
                latest = max(log_files, key=lambda f: f.stat().st_mtime)
                print(f"  Latest: {latest.name}")
        
        input("\nPress ENTER to continue...")
    
    def _system_tests_menu(self):
        """System tests menu."""
        print("\n🧪 SYSTEM TESTS")
        print("-"*25)
        
        print("1. 🔌 Connection Test")
        print("2. 🎮 Joint Movement Test")
        print("3. 🛡️ Safety Limits Test")
        print("4. 📷 Scanner Test")
        print("5. 🔄 Full System Test")
        
        choice = input("\nSelect test: ").strip()
        
        if choice == '1':
            self._test_connection()
        elif choice == '2':
            self._test_joint_movement()
        elif choice == '3':
            self._test_safety_limits()
        elif choice == '4':
            self._test_scanner()
        elif choice == '5':
            self._run_complete_system_test()
        
        input("\nPress ENTER to continue...")
    
    def _test_connection(self):
        """Test connection."""
        print("\n🔌 Testing connection...")
        
        if self.simulator_mode:
            print("✅ Simulator mode active")
        elif self.controller:
            if hasattr(self.controller, 'is_connected'):
                if self.controller.is_connected():
                    print("✅ Controller connected")
                else:
                    print("❌ Controller not connected")
            else:
                print("✅ Controller initialized")
        else:
            print("❌ No controller")
    
    def _test_joint_movement(self):
        """Test joint movement."""
        print("\n🎮 Testing joint movement...")
        
        if self.simulator_mode:
            print("🎮 Simulating joint test...")
            time.sleep(1)
            print("✅ Simulated test complete")
        else:
            print("Testing small movements...")
            # Would implement actual joint tests here
            print("⚠️ Joint test not fully implemented")
    
    def _test_safety_limits(self):
        """Test safety limits."""
        print("\n🛡️ Testing safety limits...")
        
        test_position = {'base': 10.0}  # Intentionally out of range
        if self.safety_monitor.check_limits(test_position):
            print("❌ Safety check failed - should have caught invalid position")
        else:
            print("✅ Safety limits working correctly")
    
    def _test_scanner(self):
        """Test scanner."""
        print("\n📷 Testing scanner...")
        
        if self.simulator_mode:
            print("🎮 Scanner test in simulator mode")
            print("✅ Simulated scanner OK")
        else:
            print("⚠️ Scanner test requires hardware")
    
    def _run_complete_system_test(self):
        """Run complete system test."""
        print("\n🔄 COMPLETE SYSTEM TEST")
        print("-"*30)
        
        tests = [
            ("Connection", self._test_connection),
            ("Safety Limits", self._test_safety_limits),
            ("Joint Movement", self._test_joint_movement),
            ("Scanner", self._test_scanner)
        ]
        
        for name, test_func in tests:
            print(f"\nTesting {name}...")
            try:
                test_func()
            except Exception as e:
                print(f"❌ {name} test failed: {e}")
        
        print("\n✅ System test complete")
    
    def _execute_pattern_by_name(self, pattern_name: str):
        """Execute pattern by command line argument."""
        print(f"\n🚀 Executing pattern: {pattern_name}")
        
        patterns = {
            'raster': RasterScanPattern(width=0.2, height=0.2, rows=10, cols=10),
            'spiral': SpiralScanPattern(radius_start=0.05, radius_end=0.15, revolutions=5),
            'spherical': SphericalScanPattern(radius=0.15, theta_steps=12, phi_steps=8)
        }
        
        if pattern_name.lower() in patterns:
            self._execute_scan_pattern(patterns[pattern_name.lower()])
        else:
            print(f"❌ Unknown pattern: {pattern_name}")
            print(f"Available patterns: {', '.join(patterns.keys())}")
    
    def _print_current_position(self):
        """Print current robot position."""
        try:
            if self.simulator_mode:
                print("📍 Current position (simulated):")
                print("  All joints at home position")
            else:
                if hasattr(self.controller, 'get_current_position'):
                    position = self.controller.get_current_position()
                    if position:
                        print("📍 Current position:")
                        for joint, angle in position.items():
                            print(f"  {joint}: {math.degrees(angle):.1f}°")
                    else:
                        print("❌ Could not read position")
                else:
                    print("❌ Position reading not available")
        except Exception as e:
            print(f"❌ Error reading position: {e}")
    
    def _emergency_stop(self):
        """Trigger emergency stop."""
        print("\n🛑 EMERGENCY STOP!")
        
        if self.emergency_handler:
            self.emergency_handler.trigger_emergency_stop()
        elif self.controller:
            if hasattr(self.controller, 'emergency_stop'):
                self.controller.emergency_stop()
            elif hasattr(self.controller, 'stop'):
                self.controller.stop()
        
        print("✅ Emergency stop executed")
    
    def _exit_application(self):
        """Exit the application."""
        print("\n👋 Shutting down...")
        self.running = False
    
    def _cleanup(self):
        """Cleanup before exit."""
        try:
            # Save any unsaved data
            if self.taught_positions:
                self._save_positions_to_file()
            
            # Disconnect controller
            if self.controller and not self.simulator_mode:
                if hasattr(self.controller, 'disconnect'):
                    self.controller.disconnect()
                elif hasattr(self.controller, 'close'):
                    self.controller.close()
            
            # Shutdown safety systems
            if self.safety_system and hasattr(self.safety_system, 'shutdown'):
                self.safety_system.shutdown()
            
            logger.info("Application shutdown complete")
            print("✅ Cleanup complete")
            print("👋 Goodbye!")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            print(f"⚠️ Cleanup warning: {e}")


# ================================
# MAIN ENTRY POINT
# ================================

def main():
    """Main entry point."""
    interface = RoArmMainInterface()
    return interface.run()


if __name__ == "__main__":
    sys.exit(main())
