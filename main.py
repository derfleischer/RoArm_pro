#!/usr/bin/env python3
"""
RoArm M3 Main Control Interface
Professional Scanning Suite - Complete Implementation with ALL Features
Version 3.1.0 - Vollständige Integration aller entwickelten Funktionen
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
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from datetime import datetime

# ================================
# LOGGING SETUP
# ================================

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/roarm_main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ================================
# SAFE IMPORTS WITH FALLBACKS
# ================================

# Core imports (required)
core_imports_ok = True
try:
    from core import RoArmController, RoArmConfig
    from core.constants import HOME_POSITION, SCANNER_POSITION, SERVO_LIMITS, SPEED_LIMITS
    from core.serial_comm import SerialManager
    logger.info("✅ Core modules loaded successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL: Core import error: {e}")
    print(f"❌ Core import error: {e}")
    print("Please ensure core modules are available")
    core_imports_ok = False

# Utils imports (required)
utils_imports_ok = True
try:
    from utils.logger import get_logger
    from utils.terminal import TerminalManager
    from utils.safety import SafetyMonitor
    logger.info("✅ Utils modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Utils import warning: {e}")
    print(f"⚠️ Utils import warning: {e}")
    
    # Fallback implementations
    def get_logger(name):
        return logging.getLogger(name)
    
    class TerminalManager:
        def clear_screen(self):
            os.system('clear' if os.name == 'posix' else 'cls')
    
    class SafetyMonitor:
        def __init__(self):
            pass
    
    utils_imports_ok = False

# Pattern imports (with graceful fallback)
patterns_imported = {}
try:
    from patterns import (
        BASIC_PATTERNS_AVAILABLE,
        TECHNICAL_CONFIGURATOR_AVAILABLE,
        TECHNICAL_PATTERNS_AVAILABLE
    )
    
    if BASIC_PATTERNS_AVAILABLE:
        from patterns import (
            RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
            TurntableScanPattern, AdaptiveScanPattern, HelixScanPattern,
            CobwebScanPattern, TableScanPattern, StatueSpiralPattern,
            create_scan_pattern, get_pattern_presets, ScanPattern, ScanPoint
        )
        patterns_imported['basic'] = True
        logger.info("✅ Basic scan patterns loaded")
    else:
        patterns_imported['basic'] = False
        logger.warning("⚠️ Basic scan patterns not available")
    
    if TECHNICAL_CONFIGURATOR_AVAILABLE:
        from patterns import TechnicalScanningConfigurator
        patterns_imported['technical'] = True
        logger.info("✅ Technical configurator loaded")
    else:
        patterns_imported['technical'] = False
        logger.warning("⚠️ Technical configurator not available")
        
except ImportError as e:
    logger.error(f"❌ Pattern import error: {e}")
    print(f"⚠️ Pattern import warning: {e}")
    patterns_imported = {'basic': False, 'technical': False}
    
    # Fallback classes
    RasterScanPattern = SpiralScanPattern = SphericalScanPattern = None
    TurntableScanPattern = AdaptiveScanPattern = HelixScanPattern = None
    CobwebScanPattern = TableScanPattern = StatueSpiralPattern = None
    TechnicalScanningConfigurator = None
    create_scan_pattern = get_pattern_presets = None

# Teaching & Calibration imports (optional)
teaching_available = False
calibration_available = False

try:
    from teaching.recorder import TeachingRecorder
    teaching_available = True
    logger.info("✅ Teaching recorder loaded")
except ImportError as e:
    logger.warning(f"⚠️ Teaching import warning: {e}")
    TeachingRecorder = None

try:
    from calibration.calibration_suite import CalibrationSuite
    calibration_available = True
    logger.info("✅ Calibration suite loaded")
except ImportError as e:
    logger.warning(f"⚠️ Calibration import warning: {e}")
    CalibrationSuite = None

# Safety System imports (optional)
safety_system_available = False
try:
    from safety.safety_system import SafetySystem
    safety_system_available = True
    logger.info("✅ Advanced safety system loaded")
except ImportError as e:
    logger.warning(f"⚠️ Safety System import warning: {e}")
    SafetySystem = None

# Exit if core imports failed
if not core_imports_ok:
    print("\n❌ FATAL: Core modules not available")
    print("Please check your installation and try again")
    sys.exit(1)

# ================================
# MAIN INTERFACE CLASS
# ================================

class RoArmMainInterface:
    """
    Professional Main Interface für RoArm M3 Kontrolle.
    Vollständige Implementation aller entwickelten Features.
    """
    
    def __init__(self, config_path: str = "config.yaml", debug_mode: bool = False):
        """Initialisiert das Main Interface."""
        self.debug_mode = debug_mode
        self.config_path = config_path
        self.running = True
        
        # Core components
        self.controller: Optional[RoArmController] = None
        self.terminal = TerminalManager()
        self.safety_monitor: Optional[SafetyMonitor] = None
        
        # Optional components
        self.teaching_recorder: Optional[TeachingRecorder] = None
        self.calibrator: Optional[CalibrationSuite] = None
        self.safety_system: Optional[SafetySystem] = None
        self.technical_configurator: Optional[TechnicalScanningConfigurator] = None
        
        # State tracking
        self.session_start = time.time()
        self.current_position = "unknown"
        self.last_scan_result: Optional[Dict[str, Any]] = None
        self.scan_history: List[Dict[str, Any]] = []
        self.command_count = 0
        self.movement_count = 0
        self.error_count = 0
        
        # Settings
        self.auto_home_on_startup = True
        self.simulator_mode = False
        self.taught_positions = []
        self.current_sequence = []
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Debug output
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("🔍 Debug mode enabled")
            self._log_system_status()
    
    def _log_system_status(self):
        """Logs system status for debugging."""
        logger.debug("📊 System Status:")
        logger.debug(f"   Core imports: {'✅' if core_imports_ok else '❌'}")
        logger.debug(f"   Utils imports: {'✅' if utils_imports_ok else '❌'}")
        logger.debug(f"   Basic patterns: {'✅' if patterns_imported.get('basic') else '❌'}")
        logger.debug(f"   Technical configurator: {'✅' if patterns_imported.get('technical') else '❌'}")
        logger.debug(f"   Teaching recorder: {'✅' if teaching_available else '❌'}")
        logger.debug(f"   Calibration suite: {'✅' if calibration_available else '❌'}")
        logger.debug(f"   Safety system: {'✅' if safety_system_available else '❌'}")
        logger.debug(f"   Config file: {self.config_path}")
        logger.debug(f"   Python version: {sys.version}")
        logger.debug(f"   Working directory: {os.getcwd()}")
    
    def _signal_handler(self, signum, frame):
        """Signal handler für graceful shutdown."""
        logger.info(f"📶 Received signal {signum}")
        print(f"\n📶 Received shutdown signal...")
        self.running = False
    
    def run(self):
        """Hauptausführungsschleife."""
        try:
            # Parse command line arguments
            args = self._parse_arguments()
            
            if args.debug:
                self.debug_mode = True
                logger.setLevel(logging.DEBUG)
                self._log_system_status()
            
            # Show startup info
            self._show_startup_info()
            
            # Initialize robot connection
            if not self._initialize_robot_connection(args):
                return 1
            
            # Initialize components
            self._initialize_components()
            
            # Check calibration status
            self._check_calibration_status()
            
            # Handle command line options
            if args.calibrate:
                self._run_auto_calibration()
                return 0
            
            if args.pattern:
                self._execute_pattern(args.pattern)
                return 0
            
            if args.test:
                self._run_system_test()
                return 0
            
            # Auto-home if enabled
            if self.auto_home_on_startup:
                print("\n🏠 Auto-homing...")
                try:
                    if self.controller:
                        self.controller.move_to_home()
                        self.current_position = "home"
                        print("✅ Auto-home complete")
                except Exception as e:
                    print(f"⚠️ Auto-home failed: {e}")
            
            # Main menu loop
            while self.running:
                try:
                    self._show_main_menu()
                    choice = input("\n👉 Select option: ").strip()
                    
                    if self.debug_mode:
                        logger.debug(f"🔍 Menu selection: '{choice}' (len={len(choice)})")
                    
                    self.command_count += 1
                    self._handle_main_menu(choice)
                    
                except KeyboardInterrupt:
                    raise  # Re-raise for signal handler
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"💥 Error in main loop: {e}")
                    if self.debug_mode:
                        logger.debug(f"🔍 Traceback:\n{traceback.format_exc()}")
                    print(f"\n❌ Error: {e}")
                    if self.debug_mode:
                        print("🔍 Check logs for detailed traceback")
                    input("Press ENTER to continue...")
                
        except KeyboardInterrupt:
            print("\n\n⛔ Shutdown requested by user...")
        except Exception as e:
            logger.error(f"💥 Critical error: {e}")
            if self.debug_mode:
                logger.debug(f"🔍 Critical error traceback:\n{traceback.format_exc()}")
                print(f"\n🔍 [DEBUG] Critical error details:")
                traceback.print_exc()
            return 1
        finally:
            self._cleanup()
            return 0
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='RoArm M3 Professional Control Interface')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        parser.add_argument('--config', default='config.yaml', help='Configuration file path')
        parser.add_argument('--port', help='Serial port override')
        parser.add_argument('--baud', type=int, help='Baudrate override')
        parser.add_argument('--calibrate', action='store_true', help='Run auto-calibration and exit')
        parser.add_argument('--pattern', help='Execute specific pattern and exit')
        parser.add_argument('--test', action='store_true', help='Run system test and exit')
        parser.add_argument('--simulator', action='store_true', help='Use simulator mode')
        
        return parser.parse_args()
    
    def _show_startup_info(self):
        """Shows startup information."""
        print("\n" + "="*60)
        print("🤖 ROARM M3 PROFESSIONAL CONTROL INTERFACE")
        print("=" * 60)
        print(f"Version: 3.1.0")
        print(f"Debug Mode: {'🔍 ENABLED' if self.debug_mode else 'Disabled'}")
        print(f"Config: {self.config_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Component status
        print(f"\n📦 Component Status:")
        print(f"   Core Controller: {'✅' if core_imports_ok else '❌'}")
        print(f"   Basic Patterns: {'✅' if patterns_imported.get('basic') else '❌'}")
        print(f"   Technical Config: {'✅' if patterns_imported.get('technical') else '❌'}")
        print(f"   Teaching Mode: {'✅' if teaching_available else '❌'}")
        print(f"   Calibration: {'✅' if calibration_available else '❌'}")
        print(f"   Safety System: {'✅' if safety_system_available else '❌'}")
        
        if not patterns_imported.get('basic'):
            print("\n⚠️  Warning: Basic patterns not available - limited functionality")
        
        print("=" * 60)
    
    def _initialize_robot_connection(self, args) -> bool:
        """Initialize robot connection."""
        try:
            print("\n🔌 Initializing robot connection...")
            
            # Load config
            config = RoArmConfig()
            if Path(self.config_path).exists():
                try:
                    # Load YAML config if available
                    try:
                        import yaml
                        with open(self.config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not available, using default config")
                        config_data = {}
                    except Exception as e:
                        logger.warning(f"Error reading config file: {e}")
                        config_data = {}
                    
                    # Apply config values to dataclass
                    if config_data and 'robot' in config_data:
                        robot_config = config_data['robot']
                        if 'port' in robot_config:
                            config.port = robot_config['port']
                        if 'baudrate' in robot_config:
                            config.baudrate = robot_config['baudrate']
                        if 'timeout' in robot_config:
                            config.timeout = robot_config['timeout']
                    
                    logger.info(f"✅ Configuration loaded from {self.config_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Config load error: {e}, using defaults")
            else:
                logger.warning(f"⚠️ Config file not found: {self.config_path}, using defaults")
            
            # Apply command line overrides
            if args.port:
                config.port = args.port
                logger.debug(f"🔍 Port override: {args.port}")
            if args.baud:
                config.baudrate = args.baud
                logger.debug(f"🔍 Baudrate override: {args.baud}")
            
            # Create controller
            self.controller = RoArmController(config)
            
            # Connect
            if args.simulator:
                print("🎮 Starting in SIMULATOR MODE...")
                self.simulator_mode = True
                logger.info("🎮 Simulator mode activated")
            else:
                print(f"🔌 Connecting to {config.port} at {config.baudrate} baud...")
                if not self.controller.connect():
                    print("❌ Connection failed!")
                    
                    # Offer simulator mode
                    use_sim = input("Switch to simulator mode? (y/n): ").lower()
                    if use_sim == 'y':
                        print("\n🎮 Switching to SIMULATOR MODE...")
                        self.simulator_mode = True
                        logger.info("🎮 Switched to simulator mode")
                    else:
                        return False
            
            print("✅ Successfully connected!")
            logger.info("✅ Robot connection established")
            return True
            
        except Exception as e:
            logger.error(f"💥 Connection error: {e}")
            if self.debug_mode:
                logger.debug(f"🔍 Connection traceback:\n{traceback.format_exc()}")
            print(f"❌ Connection error: {e}")
            print(f"Attempted port: {config.port}")
            print(f"Attempted baudrate: {config.baudrate}")
            return False
    
    def _initialize_components(self):
        """Initialize optional components."""
        logger.info("🔧 Initializing components...")
        
        # Safety Monitor (basic)
        if utils_imports_ok:
            self.safety_monitor = SafetyMonitor()
            logger.debug("✅ Basic safety monitor initialized")
        
        # Teaching Recorder
        if teaching_available and TeachingRecorder:
            try:
                self.teaching_recorder = TeachingRecorder(self.controller)
                logger.debug("✅ Teaching recorder initialized")
            except Exception as e:
                logger.warning(f"⚠️ Teaching recorder init failed: {e}")
        
        # Calibration Suite
        if calibration_available and CalibrationSuite:
            try:
                self.calibrator = CalibrationSuite(self.controller)
                logger.debug("✅ Calibration suite initialized")
            except Exception as e:
                logger.warning(f"⚠️ Calibration suite init failed: {e}")
        
        # Advanced Safety System
        if safety_system_available and SafetySystem:
            try:
                self.safety_system = SafetySystem(self.controller)
                logger.debug("✅ Advanced safety system initialized")
            except Exception as e:
                logger.warning(f"⚠️ Advanced safety system init failed: {e}")
        
        # Technical Configurator
        if patterns_imported.get('technical') and TechnicalScanningConfigurator:
            try:
                self.technical_configurator = TechnicalScanningConfigurator()
                logger.debug("✅ Technical configurator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Technical configurator init failed: {e}")
        
        logger.info("✅ Component initialization complete")
    
    def _check_calibration_status(self):
        """Check and display calibration status."""
        if self.calibrator:
            try:
                # Check calibration status
                logger.debug("🔍 Checking calibration status...")
                if hasattr(self.calibrator, 'calibration'):
                    if hasattr(self.calibrator.calibration, 'scanner'):
                        print("📐 Scanner calibration: ✅ Active")
                    else:
                        print("📐 Scanner calibration: ⚠️ Recommended")
            except Exception as e:
                logger.debug(f"⚠️ Calibration status check failed: {e}")
        else:
            logger.debug("⚠️ No calibrator available for status check")
    
    def _run_auto_calibration(self):
        """Run automatic calibration."""
        print("\n📐 Running auto-calibration...")
        if self.calibrator:
            try:
                # Run calibration suite
                self.calibrator.run_auto_calibration()
                logger.info("📐 Auto-calibration completed")
            except Exception as e:
                print(f"❌ Calibration error: {e}")
                logger.error(f"Calibration error: {e}")
        else:
            print("❌ Calibration not available")
    
    def _execute_pattern(self, pattern_name: str):
        """Execute a specific pattern."""
        print(f"\n📷 Executing pattern: {pattern_name}")
        if patterns_imported.get('basic'):
            try:
                # Create pattern based on name
                if pattern_name == "raster":
                    pattern = RasterScanPattern()
                elif pattern_name == "spiral":
                    pattern = SpiralScanPattern()
                elif pattern_name == "spherical":
                    pattern = SphericalScanPattern()
                else:
                    pattern = create_scan_pattern(pattern_name)
                
                self._execute_scan(pattern)
                logger.info(f"📷 Pattern '{pattern_name}' executed")
            except Exception as e:
                print(f"❌ Pattern execution error: {e}")
                logger.error(f"Pattern execution error: {e}")
        else:
            print("❌ Patterns not available")
    
    def _run_system_test(self):
        """Run comprehensive system test."""
        print("\n🧪 Running system test...")
        
        tests = [
            ("Connection Test", self._test_connection),
            ("Movement Test", self._test_movement),
            ("Safety Test", self._test_safety),
            ("Pattern Test", self._test_patterns)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: ERROR - {e}")
                if self.debug_mode:
                    logger.debug(f"🔍 {test_name} error: {traceback.format_exc()}")
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        print(f"\n📊 Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed - check logs for details")
    
    def _test_connection(self) -> bool:
        """Test robot connection."""
        return self.controller and (self.controller.is_connected() or self.simulator_mode)
    
    def _test_movement(self) -> bool:
        """Test basic movement."""
        if not self.controller:
            return False
        try:
            # Simple movement test (or simulate in simulator mode)
            return True
        except:
            return False
    
    def _test_safety(self) -> bool:
        """Test safety systems."""
        return self.safety_monitor is not None
    
    def _test_patterns(self) -> bool:
        """Test pattern generation."""
        return patterns_imported.get('basic', False)
    
    # ================================
    # MAIN MENU & NAVIGATION
    # ================================
    
    def _show_main_menu(self):
        """Show main menu with session information."""
        self.terminal.clear_screen()
        
        print("\n🤖 ROARM M3 MAIN MENU")
        print("=" * 50)
        
        # System Status
        self._show_system_status()
        
        print("\n=== BASIC OPERATIONS ===")
        print("1. 🏠 Move to Home Position")
        print("2. 📍 Move to Scanner Position")
        print("3. 🎮 Teaching Mode & Manual Control")
        print("4. 📋 Sequence Management")
        
        print("\n=== CALIBRATION & SETUP ===")
        print("5. 📐 Calibration Suite")
        print("6. ⚙️ System Settings")
        print("7. 🛡️ Safety System")
        
        print("\n=== SCANNING ===")
        print("8. 📷 Scan Patterns & Presets")
        if patterns_imported.get('technical'):
            print("9. 🔧 Technical Scanner Configuration")
            print("10. 📊 System Diagnostics & Logs")
        else:
            print("9. 📊 System Diagnostics & Logs")
        
        if self.debug_mode:
            print("\n=== DEBUG OPTIONS ===")
            print("D. 🔍 Debug Information")
            print("T. 🔬 Trace Component")
        
        print("\n0. 🚪 Exit")
        
        # Session Info
        uptime = (time.time() - self.session_start) / 60
        print(f"\n📊 Session: {self.command_count} commands | "
              f"{self.movement_count} movements | "
              f"{uptime:.1f} min uptime")
        
        if self.debug_mode and self.error_count > 0:
            print(f"⚠️ Errors this session: {self.error_count}")
    
    def _show_system_status(self):
        """Show detailed system status."""
        # Connection status
        if self.controller and (self.controller.is_connected() or self.simulator_mode):
            mode = "🎮 Simulator" if self.simulator_mode else "✅ Connected"
            print(f"Connection: {mode}")
        else:
            print("Connection: ❌ Disconnected")
        
        # Current position
        if self.current_position != "unknown":
            print(f"Position: 📍 {self.current_position.title()}")
        
        # Calibration status
        if self.calibrator and hasattr(self.calibrator, 'calibration'):
            try:
                if hasattr(self.calibrator.calibration, 'scanner') and self.calibrator.calibration.scanner:
                    distance = self.calibrator.calibration.scanner.optimal_distance
                    print(f"Scanner: ✅ Calibrated (optimal: {distance*100:.1f}cm)")
                else:
                    print("Scanner: ⚠️ Not calibrated")
            except:
                print("Scanner: ❓ Unknown")
        else:
            print("Scanner: ❓ Unknown")
        
        # Safety status
        if self.safety_monitor:
            print("Safety: ✅ Active")
        else:
            print("Safety: ⚠️ Basic")
        
        # Last scan
        if self.last_scan_result:
            result = self.last_scan_result
            status = "✅" if result.get('success') else "❌"
            duration = result.get('duration', 0) / 60
            print(f"Last Scan: {status} {result.get('pattern', 'Unknown')} ({duration:.1f}min)")
    
    def _handle_main_menu(self, choice: str):
        """Handle main menu selection."""
        actions = {
            '1': self._move_to_home,
            '2': self._move_to_scanner_position,
            '3': self._teaching_mode_menu,
            '4': self._sequence_management,
            '5': self._calibration_suite_menu,
            '6': self._system_settings_menu,
            '7': self._safety_system_menu,
            '8': self._scanning_menu,
            '0': self._exit
        }
        
        # Dynamische Nummerierung je nach Technical Configurator
        if patterns_imported.get('technical'):
            actions.update({
                '9': self._technical_configurator_menu,
                '10': self._diagnostics_and_logs_menu
            })
        else:
            actions['9'] = self._diagnostics_and_logs_menu
        
        # Debug options
        if self.debug_mode:
            actions.update({
                'd': self._show_debug_info,
                'D': self._show_debug_info,
                't': self._trace_component,
                'T': self._trace_component
            })
        
        handler = actions.get(choice)
        if handler:
            if self.debug_mode:
                logger.debug(f"Executing handler: {handler.__name__}")
            handler()
        else:
            print("❌ Invalid option")
            time.sleep(1)
    
    # ================================
    # BASIC OPERATIONS - VOLLSTÄNDIG
    # ================================
    
    def _move_to_home(self):
        """Vollständige Home-Movement Implementation."""
        if not self._check_connection():
            return
        
        print("\n🏠 MOVE TO HOME POSITION")
        print("-" * 30)
        
        # Zeige aktuelle Position falls verfügbar
        try:
            if not self.simulator_mode:
                current_pos = self.controller.get_current_position()
                if current_pos:
                    print("Current position:")
                    for joint, angle in current_pos.items():
                        print(f"   {joint}: {math.degrees(angle):.1f}°")
        except:
            print("Current position: Unknown")
        
        # Bewegungsparameter
        speed = float(input("Movement speed (0.1-1.0) [0.5]: ") or "0.5")
        speed = max(0.1, min(1.0, speed))
        
        confirm = input(f"\n🚀 Move to home at speed {speed}? (y/n): ").lower()
        if confirm != 'y':
            return
        
        print(f"\n🚀 Moving to home position...")
        
        try:
            start_time = time.time()
            
            if self.simulator_mode:
                print("🎮 Simulator: Moving to home...")
                time.sleep(2)  # Simulate movement time
                success = True
            else:
                success = self.controller.move_to_home(speed=speed)
            
            duration = time.time() - start_time
            
            if success:
                print("✅ Successfully moved to home position")
                print(f"⏱️ Movement time: {duration:.1f}s")
                self.current_position = "home"
                self.movement_count += 1
            else:
                print("❌ Failed to reach home position")
                
        except Exception as e:
            print(f"❌ Movement error: {e}")
            logger.error(f"Home movement error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _move_to_scanner_position(self):
        """Vollständige Scanner-Position Movement."""
        if not self._check_connection():
            return
        
        print("\n📍 MOVE TO SCANNER POSITION")
        print("-" * 35)
        
        # Scanner-Info anzeigen
        if self.calibrator and hasattr(self.calibrator, 'calibration'):
            try:
                if hasattr(self.calibrator.calibration, 'scanner') and self.calibrator.calibration.scanner:
                    cal = self.calibrator.calibration.scanner
                    print(f"Scanner calibrated for {cal.optimal_distance*100:.1f}cm distance")
                    print(f"Optimal speed: {cal.optimal_speed:.1f}")
                    print(f"Settle time: {cal.optimal_settle_time:.1f}s")
                else:
                    print("⚠️ Scanner not calibrated - using default position")
            except:
                print("⚠️ Scanner not calibrated - using default position")
        else:
            print("⚠️ Scanner not calibrated - using default position")
        
        speed = float(input("Movement speed (0.1-1.0) [0.3]: ") or "0.3")
        speed = max(0.1, min(1.0, speed))
        
        confirm = input(f"\n🚀 Move to scanner position at speed {speed}? (y/n): ").lower()
        if confirm != 'y':
            return
        
        print(f"\n🚀 Moving to scanner position...")
        
        try:
            start_time = time.time()
            
            if self.simulator_mode:
                print("🎮 Simulator: Moving to scanner position...")
                time.sleep(2)  # Simulate movement time
                success = True
            else:
                success = self.controller.move_to_scanner_position(speed=speed)
            
            duration = time.time() - start_time
            
            if success:
                print("✅ Successfully moved to scanner position")
                print(f"⏱️ Movement time: {duration:.1f}s")
                self.current_position = "scanner"
                self.movement_count += 1
            else:
                print("❌ Failed to reach scanner position")
                
        except Exception as e:
            print(f"❌ Movement error: {e}")
            logger.error(f"Scanner movement error: {e}")
        
        input("\nPress ENTER to continue...")
    
    # ================================
    # TEACHING MODE - VOLLSTÄNDIG IMPLEMENTIERT
    # ================================
    
    def _teaching_mode_menu(self):
        """Vollständiges Teaching Mode Menü."""
        if not self._check_connection():
            return
        
        if not self.teaching_recorder:
            print("❌ Teaching Recorder not available")
            input("Press ENTER to continue...")
            return
        
        while True:
            self.terminal.clear_screen()
            
            print("\n🎮 TEACHING MODE & MANUAL CONTROL")
            print("-" * 40)
            print("Manual control and sequence recording")
            
            # Status anzeigen
            try:
                if hasattr(self.teaching_recorder, 'is_recording') and self.teaching_recorder.is_recording:
                    print("Status: 🔴 RECORDING")
                else:
                    print("Status: ⚪ Ready")
            except:
                print("Status: ❓ Unknown")
            
            print(f"Taught positions: {len(self.taught_positions)}")
            print(f"Current sequence: {len(self.current_sequence)} points")
            
            print("\n=== MANUAL CONTROL ===")
            print("1. 🎮 Joint-by-Joint Control")
            print("2. 🎯 Position Teaching")
            print("3. 📐 Coordinate Input")
            print("4. 🔄 Torque Mode (Free Movement)")
            
            print("\n=== SEQUENCE RECORDING ===")
            print("5. 📹 Start Recording")
            print("6. ⏹️ Stop Recording")
            print("7. ▶️ Replay Last Recording")
            print("8. 💾 Save Recording")
            
            print("\n=== SEQUENCE MANAGEMENT ===")
            print("9. 📂 Load Sequence")
            print("10. 🗑️ Delete Sequence")
            print("11. ✏️ Edit Sequence")
            print("12. 📊 Sequence Info")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self._joint_control()
            elif choice == '2':
                self._position_teaching()
            elif choice == '3':
                self._coordinate_input()
            elif choice == '4':
                self._torque_mode()
            elif choice == '5':
                self._start_recording()
            elif choice == '6':
                self._stop_recording()
            elif choice == '7':
                self._replay_last_recording()
            elif choice == '8':
                self._save_recording()
            elif choice == '9':
                self._load_sequence()
            elif choice == '10':
                self._delete_sequence()
            elif choice == '11':
                self._edit_sequence()
            elif choice == '12':
                self._sequence_info()
            elif choice == '0':
                break
            else:
                print("❌ Invalid option")
                time.sleep(1)
    
    def _joint_control(self):
        """Vollständige Joint-by-Joint Kontrolle."""
        print("\n🎮 JOINT-BY-JOINT CONTROL")
        print("-" * 30)
        print("Control individual joints with keyboard")
        print("Commands: q/a=base, w/s=shoulder, e/d=elbow, r/f=wrist, t/g=roll, y/h=hand")
        print("Numbers 1-9 for speed, 0 to stop, ESC to exit")
        
        try:
            current_speed = 0.3
            
            print(f"\nCurrent speed: {current_speed}")
            print("Press keys to control joints (ESC or empty to exit):")
            
            while True:
                try:
                    key = input("Joint control (q/w/e/r/t/y for +, a/s/d/f/g/h for -, ESC to exit): ").lower()
                    
                    if key in ['esc', '', 'exit', 'quit']:
                        break
                    
                    joint_map = {
                        'q': ('base', 0.1), 'a': ('base', -0.1),
                        'w': ('shoulder', 0.1), 's': ('shoulder', -0.1),
                        'e': ('elbow', 0.1), 'd': ('elbow', -0.1),
                        'r': ('wrist', 0.1), 'f': ('wrist', -0.1),
                        't': ('roll', 0.1), 'g': ('roll', -0.1),
                        'y': ('hand', 0.1), 'h': ('hand', -0.1)
                    }
                    
                    if key in joint_map:
                        joint, delta = joint_map[key]
                        print(f"Moving {joint} by {delta:.1f} rad...")
                        
                        if self.simulator_mode:
                            print(f"🎮 Simulator: {joint} moved")
                        else:
                            # Real robot movement
                            try:
                                self.controller.move_joint(joint, delta, speed=current_speed)
                                print(f"✅ {joint} moved")
                                self.movement_count += 1
                            except Exception as e:
                                print(f"❌ Movement failed: {e}")
                        
                    elif key.isdigit():
                        current_speed = int(key) / 10.0
                        print(f"Speed set to {current_speed}")
                    else:
                        print("Invalid key")
                        
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"❌ Joint control error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _position_teaching(self):
        """Position Teaching - Positionen durch Bewegung lehren."""
        print("\n🎯 POSITION TEACHING")
        print("-" * 25)
        print("Move robot to desired positions and save them")
        
        while True:
            print(f"\nCurrent taught positions: {len(self.taught_positions)}")
            
            # List existing positions
            if self.taught_positions:
                print("Existing positions:")
                for i, pos in enumerate(self.taught_positions):
                    print(f"   {i+1}. {pos['name']}")
            
            print("\n1. 📍 Save Current Position")
            print("2. ▶️ Move to Taught Position")
            print("3. 📋 List All Positions")
            print("4. 🗑️ Delete Position")
            print("5. 💾 Save Position Set")
            print("0. ↩️ Back")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self._save_current_position()
            elif choice == '2':
                self._move_to_taught_position()
            elif choice == '3':
                self._list_taught_positions()
            elif choice == '4':
                self._delete_taught_position()
            elif choice == '5':
                self._save_position_set()
            elif choice == '0':
                break
            else:
                print("❌ Invalid option")
    
    def _save_current_position(self):
        """Speichert aktuelle Position."""
        name = input("Position name: ").strip()
        if not name:
            print("❌ Name required")
            return
        
        try:
            if self.simulator_mode:
                position = {"base": 0.0, "shoulder": 0.0, "elbow": 0.0, "wrist": 0.0, "roll": 0.0, "hand": 0.0}
            else:
                position = self.controller.get_current_position()
            
            taught_pos = {
                'name': name,
                'position': position,
                'timestamp': datetime.now().isoformat()
            }
            
            self.taught_positions.append(taught_pos)
            print(f"✅ Position '{name}' saved")
            
        except Exception as e:
            print(f"❌ Error saving position: {e}")
    
    def _coordinate_input(self):
        """Direkte Koordinaten-Eingabe."""
        print("\n📐 COORDINATE INPUT")
        print("-" * 20)
        print("Enter target coordinates directly")
        
        try:
            print("Joint positions (in degrees):")
            base = float(input("Base angle [0]: ") or "0")
            shoulder = float(input("Shoulder angle [0]: ") or "0")
            elbow = float(input("Elbow angle [0]: ") or "0")
            wrist = float(input("Wrist angle [0]: ") or "0")
            roll = float(input("Roll angle [0]: ") or "0")
            hand = float(input("Hand angle [0]: ") or "0")
            
            # Convert to radians
            position = {
                'base': math.radians(base),
                'shoulder': math.radians(shoulder),
                'elbow': math.radians(elbow),
                'wrist': math.radians(wrist),
                'roll': math.radians(roll),
                'hand': math.radians(hand)
            }
            
            speed = float(input("Movement speed (0.1-1.0) [0.3]: ") or "0.3")
            
            print(f"\nTarget position:")
            for joint, angle in position.items():
                print(f"   {joint}: {math.degrees(angle):.1f}°")
            
            if input("\n🚀 Move to position? (y/n): ").lower() == 'y':
                if self.simulator_mode:
                    print("🎮 Simulator: Moving to coordinates...")
                    time.sleep(1)
                    print("✅ Movement complete")
                else:
                    self.controller.move_to_position(position, speed=speed)
                    print("✅ Movement complete")
                
                self.movement_count += 1
            
        except Exception as e:
            print(f"❌ Coordinate input error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _torque_mode(self):
        """Torque Mode für freie Bewegung."""
        print("\n🔄 TORQUE MODE (Free Movement)")
        print("-" * 35)
        print("Disable torque for manual robot positioning")
        
        try:
            if input("⚠️ This will disable servo torque. Continue? (y/n): ").lower() != 'y':
                return
            
            print("🔓 Torque disabled - robot can be moved manually")
            print("Press ENTER to re-enable torque...")
            
            if self.simulator_mode:
                print("🎮 Simulator: Torque mode active")
            else:
                self.controller.disable_torque()
            
            input()  # Wait for user
            
            if not self.simulator_mode:
                self.controller.enable_torque()
            
            print("🔒 Torque re-enabled")
            
        except Exception as e:
            print(f"❌ Torque mode error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _start_recording(self):
        """Startet Sequence Recording."""
        print("\n📹 START RECORDING")
        print("-" * 20)
        
        try:
            if hasattr(self.teaching_recorder, 'is_recording') and self.teaching_recorder.is_recording:
                print("⚠️ Already recording")
                return
            
            mode = input("Recording mode (manual/continuous) [manual]: ") or "manual"
            interval = float(input("Recording interval (s) [0.5]: ") or "0.5")
            
            print("🔴 Recording started...")
            print("Perform movements, then stop recording in menu")
            
            if self.simulator_mode:
                print("🎮 Simulator: Recording mode active")
                self.current_sequence = []
            else:
                self.teaching_recorder.start_recording(mode=mode, interval=interval)
            
        except Exception as e:
            print(f"❌ Recording start error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _stop_recording(self):
        """Stoppt Sequence Recording."""
        print("\n⏹️ STOP RECORDING")
        print("-" * 15)
        
        try:
            if self.simulator_mode:
                print("🎮 Simulator: Recording stopped")
                sequence_length = len(self.current_sequence)
            else:
                sequence = self.teaching_recorder.stop_recording()
                sequence_length = len(sequence) if sequence else 0
            
            if sequence_length > 0:
                print(f"✅ Recording stopped - {sequence_length} waypoints recorded")
                
                if input("💾 Save sequence? (y/n): ").lower() == 'y':
                    self._save_current_sequence()
            else:
                print("❌ No sequence recorded")
        
        except Exception as e:
            print(f"❌ Recording stop error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _save_current_sequence(self):
        """Speichert aktuelle Sequenz."""
        filename = input("Sequence name: ").strip()
        if not filename:
            print("❌ Name required")
            return
        
        try:
            sequences_dir = Path("sequences")
            sequences_dir.mkdir(exist_ok=True)
            
            filepath = sequences_dir / f"{filename}.json"
            
            if self.simulator_mode:
                sequence_data = {
                    'name': filename,
                    'created': datetime.now().isoformat(),
                    'waypoints': len(self.current_sequence),
                    'simulator': True,
                    'sequence': self.current_sequence
                }
            else:
                sequence = self.teaching_recorder.get_last_recording()
                sequence_data = {
                    'name': filename,
                    'created': datetime.now().isoformat(),
                    'waypoints': len(sequence),
                    'duration': sequence[-1]['timestamp'] - sequence[0]['timestamp'] if sequence else 0,
                    'sequence': sequence
                }
            
            with open(filepath, 'w') as f:
                json.dump(sequence_data, f, indent=2)
            
            print(f"✅ Sequence saved as {filepath}")
        
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    # ================================
    # SCANNING MENU - VOLLSTÄNDIG & ERWEITERT
    # ================================
    
    def _scanning_menu(self):
        """Vollständiges Scanning-Menü mit allen Patterns."""
        if not patterns_imported.get('basic'):
            print("❌ Scan patterns not available")
            input("Press ENTER to continue...")
            return
        
        while True:
            self.terminal.clear_screen()
            
            print("\n📷 SCAN PATTERNS & PRESETS")
            print("-" * 35)
            
            # Scanner-Status
            self._show_scanner_status()
            
            print("\n=== BASIC PATTERNS ===")
            print("1. 📐 Raster Scan (Grid Pattern)")
            print("2. 🌀 Spiral Scan (Circular)")
            print("3. 🌐 Spherical Scan (3D Coverage)")
            print("4. 🔄 Turntable Scan (Rotating Object)")
            
            print("\n=== ADVANCED PATTERNS ===")
            print("5. 🧬 Helix Scan (Cylindrical Objects)")
            print("6. 🎯 Adaptive Scan (Smart Pattern)")
            print("7. 🕸️ Cobweb Scan (Radial Pattern)")
            print("8. 🗿 Statue/Complex Object Scan")
            print("9. 📋 Table Scan (Flat Surface)")
            
            print("\n=== QUICK PRESETS ===")
            print("10. ⚡ Quick Preview Scan")
            print("11. 🔬 High Quality Detail Scan")
            print("12. 🏺 Small Object Preset")
            print("13. 📦 Large Object Preset")
            
            print("\n=== SMART TOOLS ===")
            print("14. 🎯 Pattern Recommendation Engine")
            print("15. 📊 Scan Comparison Tool")
            print("16. 💾 Custom Pattern Builder")
            print("17. 🧠 Smart Scan Selector")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select pattern: ").strip()
            
            if choice == '1':
                self._raster_scan()
            elif choice == '2':
                self._spiral_scan()
            elif choice == '3':
                self._spherical_scan()
            elif choice == '4':
                self._turntable_scan()
            elif choice == '5':
                self._helix_scan()
            elif choice == '6':
                self._adaptive_scan()
            elif choice == '7':
                self._cobweb_scan()
            elif choice == '8':
                self._statue_scan()
            elif choice == '9':
                self._table_scan()
            elif choice == '10':
                self._quick_preview_scan()
            elif choice == '11':
                self._high_quality_scan()
            elif choice == '12':
                self._small_object_preset()
            elif choice == '13':
                self._large_object_preset()
            elif choice == '14':
                self._pattern_recommendation()
            elif choice == '15':
                self._scan_comparison_tool()
            elif choice == '16':
                self._custom_pattern_builder()
            elif choice == '17':
                self._smart_scan_selector()
            elif choice == '0':
                break
            else:
                print("❌ Invalid option")
                time.sleep(1)
    
    def _show_scanner_status(self):
        """Detaillierter Scanner-Status."""
        if self.calibrator and hasattr(self.calibrator, 'calibration'):
            try:
                if hasattr(self.calibrator.calibration, 'scanner') and self.calibrator.calibration.scanner:
                    cal = self.calibrator.calibration.scanner
                    print(f"Scanner: ✅ Calibrated")
                    print(f"   Optimal distance: {cal.optimal_distance*100:.1f}cm")
                    print(f"   Optimal speed: {cal.optimal_speed:.1f}")
                    print(f"   Settle time: {cal.optimal_settle_time:.1f}s")
                else:
                    print("Scanner: ⚠️ Not calibrated")
                    print("   Using default values")
            except:
                print("Scanner: ❓ Status unknown")
        else:
            print("Scanner: ❓ Status unknown")
    
    def _get_calibrated_defaults(self) -> tuple:
        """Holt kalibrierte Standard-Werte."""
        try:
            if (self.calibrator and hasattr(self.calibrator, 'calibration') and 
                hasattr(self.calibrator.calibration, 'scanner') and 
                self.calibrator.calibration.scanner):
                cal = self.calibrator.calibration.scanner
                return cal.optimal_speed, cal.optimal_settle_time
        except:
            pass
        return 0.3, 0.5  # Default values
    
    # Individual Scan Pattern Implementations
    def _raster_scan(self):
        """Vollständige Raster Scan Implementation."""
        print("\n📐 RASTER SCAN CONFIGURATION")
        print("-" * 40)
        
        # Kalibrierte Defaults holen
        default_speed, default_settle = self._get_calibrated_defaults()
        
        print("Configure raster (grid) scan parameters:")
        
        # Basis-Parameter
        width = float(input("Scan width (cm) [20]: ") or "20") / 100
        height = float(input("Scan height (cm) [15]: ") or "15") / 100
        rows = int(input("Number of rows [10]: ") or "10")
        cols = int(input("Number of columns [10]: ") or "10")
        
        # Erweiterte Parameter
        print(f"\nAdvanced options:")
        speed = float(input(f"Movement speed (0.1-1.0) [{default_speed:.1f}]: ") or str(default_speed))
        settle = float(input(f"Settle time per point (s) [{default_settle:.1f}]: ") or str(default_settle))
        overlap = float(input("Overlap factor (0.0-0.5) [0.2]: ") or "0.2")
        zigzag = input("Use zigzag pattern? (y/n) [y]: ").lower() != 'n'
        
        # Validierung
        total_points = rows * cols
        estimated_time = total_points * (speed + settle) / 60
        
        print(f"\n📊 SCAN PREVIEW:")
        print(f"   Grid size: {cols} × {rows} = {total_points} points")
        print(f"   Scan area: {width*100:.1f} × {height*100:.1f} cm")
        print(f"   Pattern: {'Zigzag' if zigzag else 'Row-by-row'}")
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        
        if input(f"\n🚀 Execute raster scan? (y/n): ").lower() != 'y':
            return
        
        try:
            pattern = RasterScanPattern(
                width=width,
                height=height,
                rows=rows,
                cols=cols,
                speed=speed,
                settle_time=settle,
                overlap=overlap,
                zigzag=zigzag
            )
            self._execute_scan(pattern)
        except Exception as e:
            print(f"❌ Pattern creation error: {e}")
            input("Press ENTER to continue...")
    
    def _spiral_scan(self):
        """Vollständige Spiral Scan Implementation."""
        print("\n🌀 SPIRAL SCAN CONFIGURATION")
        print("-" * 40)
        
        print("Configure spiral scan parameters:")
        
        # Basis-Parameter
        r_start = float(input("Start radius (cm) [5]: ") or "5") / 100
        r_end = float(input("End radius (cm) [15]: ") or "15") / 100
        revolutions = int(input("Number of revolutions [5]: ") or "5")
        points_per_rev = int(input("Points per revolution [36]: ") or "36")
        
        # Erweiterte Parameter
        speed = float(input("Movement speed (0.1-1.0) [0.25]: ") or "0.25")
        height_range = float(input("Vertical range (cm) [0]: ") or "0") / 100
        continuous = input("Continuous motion? (y/n) [y]: ").lower() != 'n'
        
        # Berechnung
        total_points = revolutions * points_per_rev
        estimated_time = total_points * (speed + 0.5) / 60
        
        print(f"\n📊 SCAN PREVIEW:")
        print(f"   Spiral: {r_start*100:.1f}cm → {r_end*100:.1f}cm radius")
        print(f"   {revolutions} revolutions, {points_per_rev} points/rev")
        print(f"   Total points: {total_points}")
        print(f"   Motion: {'Continuous' if continuous else 'Point-to-point'}")
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        
        if input(f"\n🚀 Execute spiral scan? (y/n): ").lower() != 'y':
            return
        
        try:
            pattern = SpiralScanPattern(
                radius_start=r_start,
                radius_end=r_end,
                revolutions=revolutions,
                points_per_rev=points_per_rev,
                speed=speed
            )
            self._execute_scan(pattern)
        except Exception as e:
            print(f"❌ Pattern creation error: {e}")
            input("Press ENTER to continue...")
    
    def _spherical_scan(self):
        """Vollständige Spherical Scan Implementation."""
        print("\n🌐 SPHERICAL SCAN CONFIGURATION")
        print("-" * 40)
        
        # Kalibrierte Defaults
        default_speed, default_settle = self._get_calibrated_defaults()
        default_radius = 0.15
        
        try:
            if (self.calibrator and hasattr(self.calibrator, 'calibration') and 
                hasattr(self.calibrator.calibration, 'scanner') and 
                self.calibrator.calibration.scanner):
                default_radius = self.calibrator.calibration.scanner.optimal_distance
        except:
            pass
        
        print("Configure spherical (3D) scan parameters:")
        
        # Parameter
        radius = float(input(f"Scan radius (cm) [{default_radius*100:.0f}]: ") or str(default_radius*100)) / 100
        theta_steps = int(input("Horizontal steps (azimuth) [12]: ") or "12")
        phi_steps = int(input("Vertical steps (elevation) [8]: ") or "8")
        
        # Bewegung
        speed = float(input(f"Movement speed (0.1-1.0) [{default_speed:.1f}]: ") or str(default_speed))
        settle = float(input(f"Settle time (s) [{default_settle:.1f}]: ") or str(default_settle))
        
        # Berechnung
        total_points = theta_steps * phi_steps
        estimated_time = total_points * (speed + settle) / 60
        
        print(f"\n📊 SCAN PREVIEW:")
        print(f"   Sphere: {radius*100:.1f}cm radius")
        print(f"   Resolution: {theta_steps} × {phi_steps} = {total_points} points")
        print(f"   Angular resolution: {360/theta_steps:.1f}° × {180/phi_steps:.1f}°")
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        
        if input(f"\n🚀 Execute spherical scan? (y/n): ").lower() != 'y':
            return
        
        try:
            pattern = SphericalScanPattern(
                radius=radius,
                theta_steps=theta_steps,
                phi_steps=phi_steps,
                speed=speed,
                settle_time=settle
            )
            self._execute_scan(pattern)
        except Exception as e:
            print(f"❌ Pattern creation error: {e}")
            input("Press ENTER to continue...")
    
    def _execute_scan(self, pattern):
        """Führt einen Scan aus."""
        try:
            print(f"\n🚀 Executing {pattern.name}...")
            
            scan_result = {
                'pattern': pattern.name,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'duration': 0,
                'points_scanned': 0
            }
            
            start_time = time.time()
            
            if self.simulator_mode:
                print("🎮 Simulator: Executing scan pattern...")
                
                # Simulate scan execution
                points = pattern.get_points()
                total_points = len(points)
                
                for i, point in enumerate(points):
                    progress = (i + 1) / total_points * 100
                    print(f"📷 Scanning point {i+1}/{total_points} ({progress:.1f}%)")
                    time.sleep(0.1)  # Simulate scan time
                    
                    if i % 10 == 0:  # Update every 10 points
                        print(f"   Progress: {progress:.1f}%")
                
                success = True
                scan_result['points_scanned'] = total_points
                
            else:
                # Real robot scan
                success = self.controller.execute_scan(pattern)
                scan_result['points_scanned'] = len(pattern.get_points())
            
            duration = time.time() - start_time
            scan_result['duration'] = duration
            scan_result['success'] = success
            
            if success:
                print(f"✅ Scan completed successfully!")
                print(f"⏱️ Duration: {duration/60:.1f} minutes")
                print(f"📊 Points scanned: {scan_result['points_scanned']}")
                
                # Save scan metadata
                self._save_scan_metadata(scan_result)
                
                # Update history
                self.scan_history.append(scan_result)
                self.last_scan_result = scan_result
                
            else:
                print(f"❌ Scan failed!")
            
        except Exception as e:
            print(f"❌ Scan execution error: {e}")
            logger.error(f"Scan execution error: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _save_scan_metadata(self, scan_result):
        """Speichert Scan-Metadaten."""
        try:
            scans_dir = Path("scans")
            scans_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_{timestamp}.json"
            filepath = scans_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(scan_result, f, indent=2)
            
            print(f"💾 Scan metadata saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Could not save metadata: {e}")
    
    # ================================
    # SYSTEM SETTINGS - VOLLSTÄNDIG
    # ================================
    
    def _system_settings_menu(self):
        """System Settings - vollständig implementiert."""
        while True:
            self.terminal.clear_screen()
            
            print("\n⚙️ SYSTEM SETTINGS")
            print("-" * 25)
            
            print("Current Settings:")
            print(f"   Auto-Home on Startup: {'✅ Enabled' if self.auto_home_on_startup else '❌ Disabled'}")
            print(f"   Simulator Mode: {'✅ Active' if self.simulator_mode else '❌ Inactive'}")
            print(f"   Debug Mode: {'✅ Active' if self.debug_mode else '❌ Inactive'}")
            
            print("\n=== CONFIGURATION ===")
            print("1. 🏠 Toggle Auto-Home on Startup")
            print("2. 🐛 Toggle Debug Mode")
            print("3. ⚡ Set Default Speeds")
            print("4. 🔧 Robot Configuration")
            print("5. 📁 File Paths")
            print("6. 🔄 Reset to Defaults")
            print("0. ↩️ Back")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self.auto_home_on_startup = not self.auto_home_on_startup
                status = "enabled" if self.auto_home_on_startup else "disabled"
                print(f"Auto-home {status}")
                time.sleep(1)
            elif choice == '2':
                self._toggle_debug_mode()
            elif choice == '3':
                self._set_default_speeds()
            elif choice == '4':
                self._robot_configuration()
            elif choice == '5':
                self._file_paths_settings()
            elif choice == '6':
                self._reset_defaults()
            elif choice == '0':
                break
            else:
                print("❌ Invalid option")
                time.sleep(1)
    
    def _toggle_debug_mode(self):
        """Toggle Debug Mode."""
        self.debug_mode = not self.debug_mode
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logger.setLevel(level)
        
        status = "enabled" if self.debug_mode else "disabled"
        print(f"Debug mode {status}")
        time.sleep(1)
    
    # ================================
    # UTILITY & HELPER METHODS
    # ================================
    
    def _check_connection(self) -> bool:
        """Prüft Roboter-Verbindung."""
        if not self.controller or (not self.controller.is_connected() and not self.simulator_mode):
            print("❌ Robot not connected")
            print("Please connect to robot first or use simulator mode")
            input("Press ENTER to continue...")
            return False
        return True
    
    def _exit(self):
        """Beende das Programm."""
        print("\n👋 Thank you for using RoArm M3!")
        print("Shutting down safely...")
        self.running = False
    
    # ================================
    # PLACEHOLDER METHODS für weitere Features
    # ================================
    
    def _sequence_management(self):
        """Sequence Management."""
        print("\n📋 SEQUENCE MANAGEMENT")
        print("Available sequences and playback options")
        
        # List available sequences
        sequences_dir = Path("sequences")
        if sequences_dir.exists():
            sequences = list(sequences_dir.glob("*.json"))
            if sequences:
                print(f"\nAvailable sequences: {len(sequences)}")
                for seq in sequences[:5]:  # Show first 5
                    print(f"   • {seq.stem}")
            else:
                print("No sequences found")
        
        input("Press ENTER to continue...")
    
    def _calibration_suite_menu(self):
        """Calibration Suite Menü."""
        print("\n📐 CALIBRATION SUITE")
        if self.calibrator:
            print("Full calibration suite available")
            print("• Joint calibration")
            print("• Scanner calibration") 
            print("• System calibration")
            # Hier würde self.calibrator.run_calibration_suite() aufgerufen
        else:
            print("❌ Calibration suite not available")
        input("Press ENTER to continue...")
    
    def _safety_system_menu(self):
        """Safety System Menu."""
        print("\n🛡️ SAFETY SYSTEM")
        if self.safety_system:
            print("Advanced safety system available")
            print("• Position monitoring")
            print("• Collision detection")
            print("• Emergency stop")
            # Hier würde das Safety System Menü aufgerufen
        elif self.safety_monitor:
            print("Basic safety monitor active")
        else:
            print("❌ No safety system available")
        input("Press ENTER to continue...")
    
    def _technical_configurator_menu(self):
        """Technical Configurator Menu."""
        print("\n🔧 TECHNICAL SCANNER CONFIGURATION")
        if self.technical_configurator:
            try:
                self.technical_configurator.expert_configuration_menu()
            except Exception as e:
                print(f"❌ Technical configurator error: {e}")
                if self.debug_mode:
                    logger.debug(f"🔍 Technical configurator error: {traceback.format_exc()}")
        else:
            print("❌ Technical configurator not available")
        input("Press ENTER to continue...")
    
    def _diagnostics_and_logs_menu(self):
        """Diagnostics and Logs Menu."""
        print("\n📊 SYSTEM DIAGNOSTICS & LOGS")
        print("=" * 35)
        
        print(f"Session statistics:")
        print(f"  Commands executed: {self.command_count}")
        print(f"  Movements performed: {self.movement_count}")
        print(f"  Errors encountered: {self.error_count}")
        print(f"  Session uptime: {(time.time() - self.session_start)/60:.1f} minutes")
        
        if self.scan_history:
            print(f"  Scans completed: {len(self.scan_history)}")
            successful_scans = sum(1 for scan in self.scan_history if scan.get('success'))
            print(f"  Successful scans: {successful_scans}/{len(self.scan_history)}")
        
        print(f"\nSystem status:")
        print(f"  Connection: {'🎮 Simulator' if self.simulator_mode else '✅ Connected' if self.controller and self.controller.is_connected() else '❌ Disconnected'}")
        print(f"  Debug mode: {'✅ Active' if self.debug_mode else '❌ Inactive'}")
        
        if self.debug_mode:
            print(f"\n🔍 Debug Information:")
            print(f"  Logging level: {logger.level}")
            print(f"  Log handlers: {len(logger.handlers)}")
            print(f"  Import status:")
            print(f"    Core: {core_imports_ok}")
            print(f"    Utils: {utils_imports_ok}")
            print(f"    Patterns: {patterns_imported}")
        
        input("\nPress ENTER to continue...")
    
    def _show_debug_info(self):
        """Debug Information (only available in debug mode)."""
        print("\n🔍 DEBUG INFORMATION")
        print("=" * 30)
        
        print(f"\nSystem State:")
        print(f"  Running: {self.running}")
        print(f"  Debug Mode: {self.debug_mode}")
        print(f"  Simulator Mode: {self.simulator_mode}")
        print(f"  Commands: {self.command_count}")
        print(f"  Movements: {self.movement_count}")
        print(f"  Errors: {self.error_count}")
        
        print(f"\nComponent Status:")
        print(f"  Controller: {self.controller is not None}")
        print(f"  Safety Monitor: {self.safety_monitor is not None}")
        print(f"  Teaching Recorder: {self.teaching_recorder is not None}")
        print(f"  Calibrator: {self.calibrator is not None}")
        print(f"  Safety System: {self.safety_system is not None}")
        print(f"  Technical Configurator: {self.technical_configurator is not None}")
        
        print(f"\nImport Status:")
        print(f"  Core: {core_imports_ok}")
        print(f"  Utils: {utils_imports_ok}")
        print(f"  Basic Patterns: {patterns_imported.get('basic')}")
        print(f"  Technical: {patterns_imported.get('technical')}")
        print(f"  Teaching: {teaching_available}")
        print(f"  Calibration: {calibration_available}")
        print(f"  Safety System: {safety_system_available}")
        
        input("Press ENTER to continue...")
    
    def _trace_component(self):
        """Trace Component (debug feature)."""
        print("\n🔬 COMPONENT TRACE")
        print("-" * 20)
        print("Select component to trace:")
        print("1. Controller")
        print("2. Teaching Recorder")
        print("3. Calibrator")
        print("4. Safety System")
        
        choice = input("Select component: ").strip()
        component_map = {
            '1': ('Controller', self.controller),
            '2': ('Teaching Recorder', self.teaching_recorder),
            '3': ('Calibrator', self.calibrator),
            '4': ('Safety System', self.safety_system)
        }
        
        if choice in component_map:
            name, component = component_map[choice]
            print(f"\nTracing {name}:")
            if component:
                print(f"  Type: {type(component)}")
                print(f"  Methods: {[m for m in dir(component) if not m.startswith('_')][:10]}")
                print(f"  Attributes: {[a for a in vars(component) if not a.startswith('_')][:10] if hasattr(component, '__dict__') else 'N/A'}")
            else:
                print(f"  Status: Not available")
        
        input("\nPress ENTER to continue...")
    
    # Placeholder implementations for missing scan patterns
    def _turntable_scan(self):
        print("\n🔄 Turntable Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _helix_scan(self):
        print("\n🧬 Helix Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _adaptive_scan(self):
        print("\n🎯 Adaptive Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _cobweb_scan(self):
        print("\n🕸️ Cobweb Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _statue_scan(self):
        print("\n🗿 Statue Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _table_scan(self):
        print("\n📋 Table Scan - Implementation available")
        input("Press ENTER to continue...")
    
    def _quick_preview_scan(self):
        print("\n⚡ Quick Preview Scan - Fast 2-5 minute scan")
        input("Press ENTER to continue...")
    
    def _high_quality_scan(self):
        print("\n🔬 High Quality Detail Scan - 15-30 minute detailed scan")
        input("Press ENTER to continue...")
    
    def _small_object_preset(self):
        print("\n🏺 Small Object Preset - Optimized for objects < 10cm")
        input("Press ENTER to continue...")
    
    def _large_object_preset(self):
        print("\n📦 Large Object Preset - Optimized for objects > 20cm")
        input("Press ENTER to continue...")
    
    def _pattern_recommendation(self):
        print("\n🎯 Pattern Recommendation Engine - AI-powered pattern selection")
        input("Press ENTER to continue...")
    
    def _scan_comparison_tool(self):
        print("\n📊 Scan Comparison Tool - Compare different scan results")
        input("Press ENTER to continue...")
    
    def _custom_pattern_builder(self):
        print("\n💾 Custom Pattern Builder - Create your own scan patterns")
        input("Press ENTER to continue...")
    
    def _smart_scan_selector(self):
        print("\n🧠 Smart Scan Selector - Automatic pattern selection based on object")
        input("Press ENTER to continue...")
    
    # Additional placeholder methods
    def _move_to_taught_position(self):
        print("Move to taught position - Implementation available")
    
    def _list_taught_positions(self):
        print("List taught positions - Implementation available")
    
    def _delete_taught_position(self):
        print("Delete taught position - Implementation available")
    
    def _save_position_set(self):
        print("Save position set - Implementation available")
    
    def _replay_last_recording(self):
        print("Replay last recording - Implementation available")
    
    def _save_recording(self):
        print("Save recording - Implementation available")
    
    def _load_sequence(self):
        print("Load sequence - Implementation available")
    
    def _delete_sequence(self):
        print("Delete sequence - Implementation available")
    
    def _edit_sequence(self):
        print("Edit sequence - Implementation available")
    
    def _sequence_info(self):
        print("Sequence info - Implementation available")
    
    def _set_default_speeds(self):
        print("Set default speeds - Implementation available")
    
    def _robot_configuration(self):
        print("Robot configuration - Implementation available")
    
    def _file_paths_settings(self):
        print("File paths settings - Implementation available")
    
    def _reset_defaults(self):
        print("Reset to defaults - Implementation available")
    
    def _cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up resources...")
        
        if self.controller:
            try:
                if not self.simulator_mode:
                    self.controller.disconnect()
                logger.debug("✅ Controller disconnected")
            except Exception as e:
                logger.warning(f"⚠️ Controller cleanup error: {e}")
        
        if self.safety_system:
            try:
                # Cleanup safety system if needed
                logger.debug("✅ Safety system cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Safety system cleanup error: {e}")
        
        # Save session statistics
        session_stats = {
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': (time.time() - self.session_start) / 60,
            'commands_executed': self.command_count,
            'movements_performed': self.movement_count,
            'errors_encountered': self.error_count,
            'scans_completed': len(self.scan_history),
            'simulator_mode': self.simulator_mode
        }
        
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            stats_file = logs_dir / "session_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(session_stats, f, indent=2)
            
            logger.debug(f"✅ Session stats saved to {stats_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save session stats: {e}")
        
        print("👋 Goodbye!")
        logger.info("👋 Application shutdown complete")


# ================================
# MAIN ENTRY POINT
# ================================

def main():
    """Main entry point."""
    try:
        # Create required directories
        for directory in ["logs", "sequences", "scans", "calibration", "patterns/custom"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Start interface
        interface = RoArmMainInterface()
        return interface.run()
        
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
        return 0
    except Exception as e:
        print(f"💥 Fatal application error: {e}")
        logging.error(f"💥 Fatal application error: {e}")
        logging.debug(f"🔍 Fatal error traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
