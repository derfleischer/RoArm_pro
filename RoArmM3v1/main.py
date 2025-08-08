#!/usr/bin/env python3
"""
RoArm M3 Professional Control System
Hauptprogramm mit Command Line Interface
Optimiert für macOS M4 mit Revopoint Mini2 Scanner
Version 3.1.0 - Mit erweitertem Debug-Support
"""

import sys
import time
import signal
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

from core.controller import RoArmController, RoArmConfig
from core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION, COMMANDS, DEFAULT_SPEED
from patterns.scan_patterns import (
    RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
    TurntableScanPattern, CobwebScanPattern, AdaptiveScanPattern,
    HelixScanPattern, TableScanPattern, StatueSpiralPattern,
    create_scan_pattern, get_pattern_presets)

from teaching.recorder import TeachingRecorder, RecordingMode, TrajectoryType
from calibration.calibration_suite import CalibrationSuite, CalibrationType
from safety.safety_system import SafetySystem, SafetyState, ShutdownReason
from utils.logger import setup_logger, get_logger
from utils.terminal import TerminalController

# Logger wird in main() initialisiert
logger = None

# Version Info
VERSION = "3.1.0"
BUILD_DATE = "2024-01-15"


class RoArmCLI:
    """Command Line Interface für RoArm Control."""
    
    def __init__(self, debug_mode: bool = False, trace_mode: bool = False):
        self.controller = None
        self.teacher = None
        self.calibrator = None
        self.safety_system = None
        self.terminal = TerminalController()
        self.running = True
        
        # Debug Settings
        self.debug_mode = debug_mode
        self.trace_mode = trace_mode
        
        # Session Statistics
        self.session_start = time.time()
        self.command_count = 0
        self.movement_count = 0
        self.error_count = 0
        
        # Signal Handler für Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Logger für diese Klasse
        global logger
        if logger is None:
            logger = get_logger(__name__)
        
        if self.debug_mode:
            logger.debug(f"RoArmCLI initialized - Debug: {debug_mode}, Trace: {trace_mode}")
    
    def _signal_handler(self, signum, frame):
        """Handler für Ctrl+C - Emergency Stop."""
        print("\n\n🚨 EMERGENCY STOP - Ctrl+C detected!")
        logger.warning("Emergency stop triggered by user (Ctrl+C)")
        
        if self.controller:
            self.controller.emergency_stop()
        
        if self.safety_system:
            self.safety_system.emergency_stop("User interrupt (Ctrl+C)")
        
        self.running = False
        sys.exit(0)
    
    def run(self, args):
        """Startet die CLI."""
        try:
            # Debug-Info
            if self.debug_mode:
                logger.debug(f"Starting CLI with args: {args}")
                logger.debug(f"Python: {sys.version}")
                logger.debug(f"Working directory: {Path.cwd()}")
                logger.debug(f"Script location: {Path(__file__).absolute()}")
            
            # Header
            self._print_header()
            
            # Bei Debug-Modus, zeige zusätzliche Info
            if self.debug_mode:
                print("🔍 DEBUG MODE ACTIVE - Verbose logging enabled")
                if self.trace_mode:
                    print("🔬 TRACE MODE ACTIVE - Maximum verbosity")
                print("-" * 60)
            
            # Controller Setup
            config = RoArmConfig(
                port=args.port if not args.simulate else "SIMULATOR",
                baudrate=args.baudrate,
                default_speed=args.speed,
                debug=self.debug_mode
            )
            
            # Bei Simulator-Modus, Mock verwenden
            if args.simulate:
                print("🎮 SIMULATOR MODE ACTIVATED")
                print("  No hardware connection required")
                print("  All commands will be simulated")
                print("-" * 60)
                
                # Patch the controller to use mock serial
                from core.mock_serial import MockSerialManager
                
                # Create mock controller
                self.controller = RoArmController(config)
                # Replace serial manager with mock
                self.controller.serial = MockSerialManager(config.port, config.baudrate)
                self.controller.serial.connect()
                
                logger.info("Simulator mode activated - using MockSerialManager")
            else:
                # Normale Hardware-Verbindung
                print(f"🔌 Connecting to RoArm on {config.port}...")
                self.controller = RoArmController(config)
                
                if not self.controller.serial.connected:
                    print("❌ Failed to connect to RoArm")
                    print("   Check cable and port settings")
                    print("\n💡 TIP: Use --simulate to run without hardware")
                    logger.error("Failed to connect to RoArm")
                    
                    # Frage ob Simulator-Modus verwendet werden soll
                    use_sim = input("\nStart in simulator mode instead? (y/n): ").lower()
                    if use_sim == 'y':
                        print("\n🎮 Switching to SIMULATOR MODE...")
                        from core.mock_serial import MockSerialManager
                        self.controller.serial = MockSerialManager("SIMULATOR", config.baudrate)
                        self.controller.serial.connect()
                    else:
                        return
                
                print("✅ Successfully connected!")
            logger.info("Successfully connected to RoArm")
            
            # Komponenten initialisieren
            self._initialize_components()
            
            # Prüfe Kalibrierungsstatus
            self._check_calibration_status()
            
            # Auto-Calibration wenn gewünscht
            if args.calibrate:
                self._run_auto_calibration()
                return
            
            # Auto-Start Pattern wenn angegeben
            if args.pattern:
                self._execute_pattern(args.pattern)
                return
            
            # Main Menu Loop
            while self.running:
                try:
                    self._show_main_menu()
                    choice = input("\n👉 Select option: ").strip()
                    
                    if self.debug_mode:
                        logger.debug(f"Menu selection: '{choice}' (len={len(choice)})")
                    
                    self.command_count += 1
                    self._handle_main_menu(choice)
                    
                except KeyboardInterrupt:
                    raise  # Re-raise für Signal Handler
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in main loop: {e}")
                    if self.debug_mode:
                        logger.debug(f"Traceback:\n{traceback.format_exc()}")
                    print(f"\n❌ Error: {e}")
                    input("Press ENTER to continue...")
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            if self.debug_mode:
                print(f"\n[DEBUG] Critical error details:")
                traceback.print_exc()
        finally:
            self._cleanup()
    
    def _initialize_components(self):
        """Initialisiert alle Komponenten mit Debug-Output."""
        print("\nInitializing components...")
        
        components_to_init = [
            ("Teaching Recorder", self._init_teacher),
            ("Calibration Suite", self._init_calibrator),
            ("Safety System", self._init_safety_system)
        ]
        
        for name, init_func in components_to_init:
            try:
                if self.debug_mode:
                    logger.debug(f"Initializing {name}...")
                
                init_func()
                print(f"  ✅ {name}")
                
                if self.debug_mode:
                    logger.debug(f"{name} initialized successfully")
                    
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                logger.error(f"Failed to initialize {name}: {e}")
                if self.debug_mode:
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
        
        print("All components initialized!")
    
    def _init_teacher(self):
        """Initialisiert Teaching Recorder."""
        self.teacher = TeachingRecorder(self.controller)
    
    def _init_calibrator(self):
        """Initialisiert Calibration Suite."""
        self.calibrator = CalibrationSuite(self.controller)
    
    def _init_safety_system(self):
        """Initialisiert Safety System."""
        self.safety_system = SafetySystem(self.controller)
    
    def _print_header(self):
        """Zeigt den Header."""
        print("\n" + "="*70)
        print(f"🤖 RoArm M3 Professional Control System v{VERSION}")
        print("="*70)
        print("📷 Optimized for Revopoint Mini2 Scanner")
        print("🍎 macOS M4 Edition")
        print("🔧 Professional Calibration Suite")
        
        if self.debug_mode:
            print(f"🔍 Debug Mode: ON | Build: {BUILD_DATE}")
        
        print("\n⚡ Press Ctrl+C anytime for EMERGENCY STOP")
        print("="*70 + "\n")
    
    def _check_calibration_status(self):
        """Prüft und zeigt Kalibrierungsstatus."""
        try:
            if self.calibrator and self.calibrator.calibration.calibration_valid:
                age_days = (time.time() - self.calibrator.calibration.timestamp) / 86400
                accuracy = self.calibrator.calibration.overall_accuracy * 1000  # in mrad
                
                if age_days < 7:
                    status_icon = "✅"
                    status_text = "GOOD"
                elif age_days < 30:
                    status_icon = "⚠️"
                    status_text = "OK (aging)"
                else:
                    status_icon = "⚠️"
                    status_text = "OLD"
                
                print(f"{status_icon} Calibration Status: {status_text}")
                print(f"   Age: {age_days:.0f} days")
                print(f"   Accuracy: ±{accuracy:.1f} mrad")
            else:
                print("⚠️ No valid calibration found")
                print("   Run calibration for best accuracy (Option 6)")
            print()
        except Exception as e:
            logger.error(f"Error checking calibration status: {e}")
            if self.debug_mode:
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
    
    def _show_main_menu(self):
        """Zeigt das Hauptmenü."""
        print("\n" + "="*50)
        print("MAIN MENU")
        
        # Debug-Info wenn aktiviert
        if self.debug_mode:
            print(f"[DEBUG MODE]")
            print(f"  Components: C:{self._check_component('controller')} "
                  f"T:{self._check_component('teacher')} "
                  f"Cal:{self._check_component('calibrator')} "
                  f"S:{self._check_component('safety_system')}")
        
        print("="*50)
        print("1. 🎮 Manual Control")
        print("2. 📷 Scanner Patterns")
        print("3. 🎓 Teaching Mode")
        print("4. 📁 Load & Play Sequence")
        print("5. 🏠 Move to Home")
        print("6. 🔧 Calibration Suite")
        print("7. ⚙️  Settings")
        print("8. 📊 Status")
        print("9. 🧪 System Test")
        
        if self.debug_mode:
            print("D. 🔍 Debug Information")
            print("T. 🔬 Trace Component")
        
        print("0. 🚪 Exit")
        print("-" * 50)
        
        # Session Info
        uptime = (time.time() - self.session_start) / 60
        print(f"📊 Session: {self.command_count} commands | "
              f"{self.movement_count} movements | "
              f"{uptime:.1f} min uptime")
        
        if self.debug_mode and self.error_count > 0:
            print(f"⚠️ Errors this session: {self.error_count}")
    
    def _check_component(self, name: str) -> str:
        """Prüft ob Komponente existiert."""
        return "✓" if hasattr(self, name) and getattr(self, name) else "✗"
    
    def _handle_main_menu(self, choice: str):
        """Verarbeitet Hauptmenü-Auswahl mit verbessertem Error Handling."""
        handlers = {
            '1': self._manual_control,
            '2': self._scanner_menu,
            '3': self._teaching_menu,
            '4': self._load_sequence,
            '5': self._move_home,
            '6': self._calibration_menu,
            '7': self._settings_menu,
            '8': self._show_status,
            '9': self._system_test,
            '0': self._exit,
            'd': self._show_debug_info if self.debug_mode else None,
            'D': self._show_debug_info if self.debug_mode else None,
            't': self._trace_component if self.debug_mode else None,
            'T': self._trace_component if self.debug_mode else None
        }
        
        # Debug-Logging
        if self.debug_mode:
            logger.debug(f"Processing menu choice: '{choice}'")
            logger.debug(f"Available handlers: {[k for k, v in handlers.items() if v]}")
        
        handler = handlers.get(choice)
        
        if handler:
            handler_name = handler.__name__ if handler else "None"
            
            if self.debug_mode:
                logger.debug(f"Executing handler: {handler_name}")
                start_time = time.time()
            
            try:
                handler()
                
                if self.debug_mode:
                    elapsed = time.time() - start_time
                    logger.debug(f"Handler {handler_name} completed in {elapsed:.2f}s")
                    
            except AttributeError as e:
                self.error_count += 1
                logger.error(f"AttributeError in {handler_name}: {e}")
                
                if self.debug_mode:
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                    print(f"\n[DEBUG] AttributeError Details:")
                    print(f"  Missing: {e}")
                    print(f"  Handler: {handler_name}")
                    traceback.print_exc()
                else:
                    print(f"❌ Component Error: {e}")
                
                input("\nPress ENTER to continue...")
                
            except ImportError as e:
                self.error_count += 1
                logger.error(f"ImportError in {handler_name}: {e}")
                
                if self.debug_mode:
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                    print(f"\n[DEBUG] Import Error:")
                    print(f"  Missing module: {e}")
                    traceback.print_exc()
                else:
                    print(f"❌ Missing module: {e}")
                
                input("\nPress ENTER to continue...")
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Unexpected error in {handler_name}: {e}")
                
                if self.debug_mode:
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                    print(f"\n[DEBUG] Unexpected Error in {handler_name}:")
                    print(f"  Type: {type(e).__name__}")
                    print(f"  Message: {e}")
                    traceback.print_exc()
                else:
                    print(f"❌ Error: {e}")
                
                input("\nPress ENTER to continue...")
        else:
            if self.debug_mode:
                logger.debug(f"Invalid menu option: '{choice}' (ord={[ord(c) for c in choice]})")
            print(f"❌ Invalid option: '{choice}'")
    
    # ============== DEBUG FUNCTIONS ==============
    
    def _show_debug_info(self):
        """Zeigt detaillierte Debug-Informationen."""
        print("\n" + "="*60)
        print("🔍 DEBUG INFORMATION")
        print("="*60)
        
        # System Info
        print("\n📊 SYSTEM:")
        print(f"  Python: {sys.version}")
        print(f"  Platform: {sys.platform}")
        print(f"  Script: {Path(__file__).absolute()}")
        print(f"  CWD: {Path.cwd()}")
        print(f"  Version: {VERSION}")
        print(f"  Build: {BUILD_DATE}")
        
        # Session Stats
        print("\n📈 SESSION:")
        uptime = time.time() - self.session_start
        print(f"  Uptime: {uptime:.1f}s ({uptime/60:.1f} min)")
        print(f"  Commands: {self.command_count}")
        print(f"  Movements: {self.movement_count}")
        print(f"  Errors: {self.error_count}")
        
        # Component Status
        print("\n🔧 COMPONENTS:")
        components = {
            'Controller': self.controller,
            'Serial': self.controller.serial if self.controller else None,
            'Teacher': self.teacher,
            'Calibrator': self.calibrator,
            'Safety System': self.safety_system,
            'Terminal': self.terminal
        }
        
        for name, component in components.items():
            if component:
                status = "✅ OK"
                # Check if it's mock serial
                if name == 'Serial' and hasattr(component, 'robot_state'):
                    details = "(MockSerialManager - SIMULATOR)"
                else:
                    details = f"({component.__class__.__name__})"
            else:
                status = "❌ Missing"
                details = ""
            print(f"  {name:15s}: {status} {details}")
        
        # Show simulator status if active
        if self.controller and hasattr(self.controller.serial, 'robot_state'):
            print("\n🎮 SIMULATOR STATUS:")
            mock = self.controller.serial
            print(f"  Commands sent: {len(mock.command_history)}")
            print(f"  Current positions:")
            for joint, pos in mock.robot_state.positions.items():
                print(f"    {joint:10s}: {pos:+.3f} rad")
            print(f"  Torque: {'ON' if mock.robot_state.torque_enabled else 'OFF'}")
            print(f"  LED: {'ON' if mock.robot_state.led_on else 'OFF'}")
        
        # Configuration
        print("\n⚙️ CONFIGURATION:")
        if self.controller and self.controller.config:
            config = self.controller.config
            print(f"  Port: {config.port}")
            print(f"  Baudrate: {config.baudrate}")
            print(f"  Speed: {config.default_speed}")
            print(f"  Scanner Weight: {config.scanner_weight}kg")
            print(f"  Debug: {config.debug}")
            print(f"  Auto-connect: {config.auto_connect}")
        
        # Imports Check
        print("\n📦 MODULE STATUS:")
        modules_to_check = [
            ('serial', 'Serial Communication'),
            ('numpy', 'Numerical Computing'),
            ('yaml', 'Configuration'),
            ('core.controller', 'Main Controller'),
            ('core.constants', 'Constants'),
            ('patterns.scan_patterns', 'Scan Patterns'),
            ('teaching.recorder', 'Teaching Mode'),
            ('calibration.calibration_suite', 'Calibration'),
            ('safety.safety_system', 'Safety System'),
            ('utils.logger', 'Logging'),
            ('utils.terminal', 'Terminal Control')
        ]
        
        for module_name, description in modules_to_check:
            try:
                if '.' in module_name:
                    # Local module
                    module = sys.modules.get(module_name)
                    if not module:
                        __import__(module_name)
                        module = sys.modules.get(module_name)
                else:
                    # External module
                    module = __import__(module_name)
                
                if module:
                    version = getattr(module, '__version__', 'N/A')
                    status = f"✅ {version}" if version != 'N/A' else "✅"
                else:
                    status = "⚠️ Imported but None"
                    
            except ImportError as e:
                status = f"❌ {str(e)[:30]}"
            except Exception as e:
                status = f"⚠️ {str(e)[:30]}"
            
            print(f"  {module_name:30s}: {status}")
        
        # Constants Check
        print("\n📐 CONSTANTS CHECK:")
        try:
            from core.constants import SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION
            print(f"  SERVO_LIMITS: ✅ ({len(SERVO_LIMITS)} joints)")
            print(f"  HOME_POSITION: ✅ ({len(HOME_POSITION)} joints)")
            print(f"  SCANNER_POSITION: ✅ ({len(SCANNER_POSITION)} joints)")
            
            if self.trace_mode:
                print("\n  Joint Limits:")
                for joint, (min_val, max_val) in SERVO_LIMITS.items():
                    print(f"    {joint:10s}: [{min_val:+.2f}, {max_val:+.2f}] rad")
                    
        except ImportError as e:
            print(f"  Constants import failed: ❌ {e}")
        
        # Memory Usage
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"\n💾 MEMORY:")
            print(f"  RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
            print(f"  VMS: {mem_info.vms / 1024 / 1024:.1f} MB")
        except ImportError:
            pass
        
        input("\nPress ENTER to continue...")
    
    def _trace_component(self):
        """Trace-Modus für einzelne Komponenten."""
        print("\n🔬 TRACE COMPONENT")
        print("-" * 40)
        print("Select component to trace:")
        print("1. Controller")
        print("2. Serial Communication")
        print("3. Teaching Recorder")
        print("4. Calibration Suite")
        print("5. Safety System")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1' and self.controller:
            self._trace_controller()
        elif choice == '2' and self.controller:
            self._trace_serial()
        elif choice == '3' and self.teacher:
            self._trace_teacher()
        elif choice == '4' and self.calibrator:
            self._trace_calibrator()
        elif choice == '5' and self.safety_system:
            self._trace_safety()
        elif choice != '0':
            print("Component not available or invalid choice")
        
        input("\nPress ENTER to continue...")
    
    def _trace_controller(self):
        """Detaillierte Controller-Informationen."""
        print("\n📋 CONTROLLER TRACE:")
        c = self.controller
        print(f"  Config Port: {c.config.port}")
        print(f"  Connected: {c.serial.connected}")
        print(f"  Current Position: {c.current_position}")
        print(f"  Current Speed: {c.current_speed}")
        print(f"  Torque Enabled: {c.torque_enabled}")
        print(f"  Emergency Stop: {c.emergency_stop_flag}")
        print(f"  Scanner Mounted: {c.scanner_mounted}")
        print(f"  Queue Size: {c.command_queue.qsize()}")
    
    def _trace_serial(self):
        """Detaillierte Serial-Informationen."""
        print("\n📡 SERIAL TRACE:")
        s = self.controller.serial
        print(f"  Port: {s.port}")
        print(f"  Baudrate: {s.baudrate}")
        print(f"  Connected: {s.connected}")
        print(f"  Timeout: {s.timeout}")
        if hasattr(s, 'serial') and s.serial:
            print(f"  Is Open: {s.serial.is_open}")
            print(f"  In Waiting: {s.serial.in_waiting}")
    
    def _trace_teacher(self):
        """Detaillierte Teaching-Informationen."""
        print("\n🎓 TEACHING TRACE:")
        t = self.teacher
        print(f"  Recording: {t.is_recording}")
        print(f"  Mode: {t.recording_mode}")
        print(f"  Waypoint Count: {t.waypoint_count}")
        print(f"  Current Speed: {t.current_speed}")
        print(f"  Current Acceleration: {t.current_acceleration}")
        print(f"  Current Trajectory: {t.current_trajectory}")
    
    def _trace_calibrator(self):
        """Detaillierte Calibration-Informationen."""
        print("\n🔧 CALIBRATION TRACE:")
        c = self.calibrator
        print(f"  Valid: {c.calibration.calibration_valid}")
        print(f"  Accuracy: {c.calibration.overall_accuracy*1000:.2f} mrad")
        print(f"  Joints Calibrated: {len(c.calibration.joints)}")
        print(f"  Scanner Calibrated: {c.calibration.scanner is not None}")
        print(f"  Is Calibrating: {c.is_calibrating}")
        print(f"  Progress: {c.calibration_progress:.1f}%")
    
    def _trace_safety(self):
        """Detaillierte Safety-Informationen."""
        print("\n🛡️ SAFETY TRACE:")
        s = self.safety_system
        print(f"  State: {s.safety_state}")
        print(f"  Emergency Active: {s.emergency_active}")
        print(f"  Shutdown in Progress: {s.shutdown_in_progress}")
        print(f"  Watchdog Enabled: {s.watchdog_enabled}")
        print(f"  Event Count: {len(s.events)}")
        print(f"  Last Heartbeat: {time.time() - s.last_heartbeat:.1f}s ago")
    
    # ============== CALIBRATION SUITE ==============
    
    def _calibration_menu(self):
        """Erweiterte Kalibrierungsoptionen."""
        try:
            if not self.calibrator:
                print("❌ Calibration Suite not initialized!")
                input("\nPress ENTER to continue...")
                return
            
            print("\n🔧 CALIBRATION SUITE")
            print("-" * 40)
            
            # Status anzeigen
            if self.calibrator.calibration.calibration_valid:
                print(f"Current calibration: VALID ✅")
                print(f"Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad")
            else:
                print("Current calibration: NONE ⚠️")
            
            print("\n1. 🚀 Auto Calibration (Full)")
            print("2. 🎯 Single Joint Calibration")
            print("3. 📷 Scanner Position Calibration")
            print("4. 🔄 Backlash Compensation")
            print("5. ⚖️  Weight Compensation")
            print("6. 🎯 Test Repeatability")
            print("7. ✅ Verify Calibration")
            print("8. 📊 Export Calibration Report")
            print("9. 💾 Save/Load Calibration")
            print("0. Back")
            
            choice = input("\n👉 Select option: ").strip()
            
            if self.debug_mode:
                logger.debug(f"Calibration menu choice: '{choice}'")
            
            if choice == '1':
                self._run_auto_calibration()
            elif choice == '2':
                self._calibrate_single_joint()
            elif choice == '3':
                self._calibrate_scanner_position()
            elif choice == '4':
                self._calibrate_backlash()
            elif choice == '5':
                self._calibrate_weight()
            elif choice == '6':
                self._test_repeatability()
            elif choice == '7':
                self._verify_calibration()
            elif choice == '8':
                self._export_calibration_report()
            elif choice == '9':
                self._save_load_calibration()
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in calibration menu: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    def _run_auto_calibration(self):
        """Führt automatische Kalibrierung durch."""
        print("\n🚀 AUTO CALIBRATION")
        print("="*50)
        print("This will perform a complete system calibration.")
        print("The process takes about 10-15 minutes.")
        print("\nSteps:")
        print("1. Find endstops")
        print("2. Calibrate each joint")
        print("3. Measure backlash")
        print("4. Weight compensation")
        print("5. Scanner alignment")
        print("6. Repeatability test")
        
        confirm = input("\nStart calibration? (y/n): ").lower()
        if confirm != 'y':
            print("Calibration cancelled")
            return
        
        # Scanner-Kalibrierung einschließen?
        include_scanner = input("Include scanner calibration? (y/n): ").lower() == 'y'
        
        print("\n" + "="*50)
        print("CALIBRATION IN PROGRESS...")
        print("DO NOT INTERRUPT!")
        print("="*50 + "\n")
        
        # Starte Kalibrierung
        success = self.calibrator.run_auto_calibration(include_scanner=include_scanner)
        
        if success:
            print("\n✅ CALIBRATION SUCCESSFUL!")
            print(f"Overall accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad")
            
            # Report anzeigen
            report = self.calibrator.export_report()
            print("\nCalibration Report:")
            print(report)
        else:
            print("\n❌ CALIBRATION FAILED")
            print("Please check the log for details")
    
    def _calibrate_single_joint(self):
        """Kalibriert einzelnes Gelenk."""
        print("\n🎯 SINGLE JOINT CALIBRATION")
        print("-" * 40)
        print("Select joint to calibrate:")
        print("1. Base")
        print("2. Shoulder")
        print("3. Elbow")
        print("4. Wrist")
        print("5. Roll")
        print("6. Hand (Gripper)")
        print("0. Cancel")
        
        joints = ['', 'base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']
        choice = input("\n👉 Select joint: ").strip()
        
        try:
            idx = int(choice)
            if 1 <= idx <= 6:
                joint = joints[idx]
                print(f"\nCalibrating {joint}...")
                
                success = self.calibrator.calibrate_single_joint(joint)
                
                if success:
                    print(f"✅ {joint} calibrated successfully")
                    if joint in self.calibrator.calibration.joints:
                        cal = self.calibrator.calibration.joints[joint]
                        print(f"   Offset: {cal.offset*1000:.2f} mrad")
                        print(f"   Accuracy: ±{cal.accuracy*1000:.2f} mrad")
                else:
                    print(f"❌ Failed to calibrate {joint}")
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _calibrate_scanner_position(self):
        """Optimiert Scanner-Position."""
        print("\n📷 SCANNER POSITION CALIBRATION")
        print("-" * 40)
        print("This will find the optimal scanner mounting position.")
        print("You need a calibration object (sphere or cube).")
        
        confirm = input("\nReady to start? (y/n): ").lower()
        if confirm == 'y':
            success = self.calibrator.calibrate_scanner_position()
            if success:
                print("✅ Scanner position optimized")
            else:
                print("❌ Scanner calibration failed")
    
    def _calibrate_backlash(self):
        """Misst und kompensiert Spiel."""
        print("\n🔄 BACKLASH COMPENSATION")
        print("-" * 40)
        print("Measuring mechanical play in joints...")
        
        self.calibrator._measure_backlash()
        
        print("\nBacklash measurements:")
        for joint_name, joint in self.calibrator.calibration.joints.items():
            if joint.backlash > 0:
                print(f"  {joint_name:10s}: {joint.backlash*1000:.2f} mrad")
        
        self.calibrator.save_calibration()
        print("\n✅ Backlash compensation updated")
    
    def _calibrate_weight(self):
        """Kalibriert Gewichtskompensation."""
        print("\n⚖️ WEIGHT COMPENSATION")
        print("-" * 40)
        
        weight = float(input("Scanner weight (g) [200]: ") or "200") / 1000
        self.controller.config.scanner_weight = weight
        
        print("Calibrating gravity compensation...")
        self.calibrator._calibrate_weight_compensation()
        
        print("✅ Weight compensation calibrated")
        print(f"   Scanner weight: {weight*1000:.0f}g")
        print(f"   Compensation factors updated")
    
    def _test_repeatability(self):
        """Testet Wiederholgenauigkeit."""
        print("\n🎯 REPEATABILITY TEST")
        print("-" * 40)
        
        positions = int(input("Number of test positions [10]: ") or "10")
        cycles = int(input("Cycles per position [3]: ") or "3")
        
        print(f"\nTesting {positions} positions with {cycles} cycles each...")
        print("This will take approximately {:.0f} minutes".format(positions * cycles * 0.5))
        
        confirm = input("\nStart test? (y/n): ").lower()
        if confirm != 'y':
            return
        
        print("\nRunning repeatability test...")
        stats = self.calibrator.test_repeatability(positions, cycles)
        
        print("\n" + "="*50)
        print("REPEATABILITY TEST RESULTS")
        print("="*50)
        
        overall_repeatability = []
        for joint, stat in stats.items():
            repeat = stat['repeatability'] * 1000  # in mrad
            overall_repeatability.append(repeat)
            
            status = "✅" if repeat < 2.0 else "⚠️"
            print(f"{joint:10s}: ±{repeat:.2f} mrad (3σ) {status}")
        
        if overall_repeatability:
            mean_repeat = sum(overall_repeatability) / len(overall_repeatability)
            print("-"*50)
            print(f"Overall: ±{mean_repeat:.2f} mrad")
    
    def _verify_calibration(self):
        """Verifiziert aktuelle Kalibrierung."""
        print("\n✅ VERIFYING CALIBRATION")
        print("-" * 40)
        
        if self.calibrator.verify_calibration():
            print("✅ Calibration is VALID")
            print(f"   Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad")
            
            age_days = (time.time() - self.calibrator.calibration.timestamp) / 86400
            print(f"   Age: {age_days:.0f} days")
            
            if age_days > 30:
                print("   ⚠️ Consider recalibration (>30 days old)")
        else:
            print("❌ Calibration verification FAILED")
            print("   Recalibration recommended")
    
    def _export_calibration_report(self):
        """Exportiert Kalibrierungsbericht."""
        print("\n📊 EXPORT CALIBRATION REPORT")
        
        filepath = input("Report filename [calibration_report.txt]: ").strip()
        if not filepath:
            filepath = "calibration/calibration_report.txt"
        
        report = self.calibrator.export_report(filepath)
        print(f"\n✅ Report saved to {filepath}")
        
        show = input("Display report? (y/n): ").lower()
        if show == 'y':
            print("\n" + report)
    
    def _save_load_calibration(self):
        """Speichert oder lädt Kalibrierung."""
        print("\n💾 SAVE/LOAD CALIBRATION")
        print("-" * 40)
        print("1. Save current calibration")
        print("2. Load calibration from file")
        print("3. Export to backup")
        print("4. Reset to defaults")
        print("0. Cancel")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            filepath = input("Save as [system_calibration.json]: ").strip()
            if not filepath:
                filepath = "calibration/system_calibration.json"
            self.calibrator.save_calibration(filepath)
            print(f"✅ Calibration saved to {filepath}")
            
        elif choice == '2':
            filepath = input("Load from [system_calibration.json]: ").strip()
            if not filepath:
                filepath = "calibration/system_calibration.json"
            self.calibrator.load_calibration(filepath)
            
            if self.calibrator.calibration.calibration_valid:
                print("✅ Calibration loaded successfully")
            else:
                print("⚠️ Loaded file contains no valid calibration")
                
        elif choice == '3':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"calibration/backup_{timestamp}.json"
            self.calibrator.save_calibration(filepath)
            print(f"✅ Backup saved to {filepath}")
            
        elif choice == '4':
            confirm = input("Reset calibration to defaults? (y/n): ").lower()
            if confirm == 'y':
                from calibration.calibration_suite import SystemCalibration
                self.calibrator.calibration = SystemCalibration()
                print("✅ Calibration reset to defaults")
    
    # ============== SYSTEM TEST ==============
    
    def _system_test(self):
        """Führt Systemtest durch."""
        try:
            print("\n🧪 SYSTEM TEST")
            print("-" * 40)
            print("1. Communication Test")
            print("2. Joint Movement Test")
            print("3. Speed Test")
            print("4. Emergency Stop Test")
            print("5. Full System Check")
            print("0. Back")
            
            choice = input("\n👉 Select test: ").strip()
            
            if choice == '1':
                self._test_communication()
            elif choice == '2':
                self._test_joints()
            elif choice == '3':
                self._test_speed()
            elif choice == '4':
                self._test_emergency_stop()
            elif choice == '5':
                self._full_system_check()
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in system test: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    def _test_communication(self):
        """Testet serielle Kommunikation."""
        print("\n📡 COMMUNICATION TEST")
        print("-" * 40)
        
        tests_passed = 0
        tests_total = 5
        
        # Test 1: Connection
        print("1. Testing connection... ", end='')
        if self.controller.serial.connected:
            print("✅")
            tests_passed += 1
        else:
            print("❌")
        
        # Test 2: Query status
        print("2. Testing status query... ", end='')
        status = self.controller.query_status()
        if status:
            print("✅")
            tests_passed += 1
        else:
            print("❌")
        
        # Test 3: LED control
        print("3. Testing LED control... ", end='')
        if self.controller.led_control(True, 128):
            time.sleep(0.5)
            self.controller.led_control(False)
            print("✅")
            tests_passed += 1
        else:
            print("❌")
        
        # Test 4: Response time
        print("4. Testing response time... ", end='')
        start = time.time()
        self.controller.query_status()
        response_time = (time.time() - start) * 1000
        if response_time < 100:  # <100ms
            print(f"✅ ({response_time:.0f}ms)")
            tests_passed += 1
        else:
            print(f"⚠️ ({response_time:.0f}ms - slow)")
        
        # Test 5: Command queue
        print("5. Testing command queue... ", end='')
        try:
            for _ in range(10):
                self.controller.command_queue.put({"T": 1})
            time.sleep(0.5)
            print("✅")
            tests_passed += 1
        except:
            print("❌")
        
        # Results
        print("-" * 40)
        print(f"Results: {tests_passed}/{tests_total} tests passed")
        
        if tests_passed == tests_total:
            print("✅ Communication working perfectly")
        elif tests_passed >= 3:
            print("⚠️ Communication mostly working")
        else:
            print("❌ Communication problems detected")
    
    def _test_joints(self):
        """Testet alle Gelenke."""
        print("\n🦾 JOINT MOVEMENT TEST")
        print("-" * 40)
        print("Testing each joint individually...")
        
        for joint in ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']:
            print(f"\nTesting {joint}... ", end='')
            
            # Small movement
            pos = self.controller.current_position.copy()
            pos[joint] += 0.2
            
            if self.controller.move_joints(pos, speed=0.5):
                print("forward ✓", end='')
                time.sleep(1)
                
                # Back
                pos[joint] -= 0.4
                if self.controller.move_joints(pos, speed=0.5):
                    print(", backward ✓", end='')
                    time.sleep(1)
                    
                    # Return
                    pos[joint] += 0.2
                    self.controller.move_joints(pos, speed=0.5)
                    print(" ✅")
                else:
                    print(" ❌")
            else:
                print("❌")
        
        # Return home
        print("\nReturning to home position...")
        self.controller.move_home(speed=0.5)
        print("✅ Joint test complete")
    
    def _test_speed(self):
        """Testet verschiedene Geschwindigkeiten."""
        print("\n⚡ SPEED TEST")
        print("-" * 40)
        
        speeds = [0.2, 0.5, 1.0, 1.5, 2.0]
        
        for speed in speeds:
            print(f"Testing speed {speed:.1f}... ", end='')
            
            # Move with speed
            pos = self.controller.current_position.copy()
            pos['base'] = 0.5
            
            start = time.time()
            self.controller.move_joints(pos, speed=speed)
            duration = time.time() - start
            
            # Back to center
            pos['base'] = 0.0
            self.controller.move_joints(pos, speed=speed)
            
            print(f"✅ ({duration:.1f}s)")
        
        print("\n✅ Speed test complete")
    
    def _test_emergency_stop(self):
        """Testet Emergency Stop."""
        print("\n🚨 EMERGENCY STOP TEST")
        print("-" * 40)
        print("⚠️  WARNING: This will test the emergency stop function")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm != 'y':
            return
        
        print("\nStarting movement...")
        # Start slow movement
        pos = self.controller.current_position.copy()
        pos['base'] = 3.0  # Long movement
        self.controller.move_joints(pos, speed=0.2, wait=False)
        
        time.sleep(1)
        print("Triggering EMERGENCY STOP!")
        self.controller.emergency_stop()
        
        print("Checking if movement stopped... ", end='')
        time.sleep(0.5)
        
        # Check if stopped
        status1 = self.controller.query_status()
        time.sleep(0.5)
        status2 = self.controller.query_status()
        
        if status1 and status2:
            pos1 = status1['positions']['base']
            pos2 = status2['positions']['base']
            
            if abs(pos2 - pos1) < 0.01:
                print("✅ Movement stopped")
                
                # Reset emergency
                print("Resetting emergency state...")
                self.controller.reset_emergency()
                self.controller.set_torque(True)
                
                # Test movement after reset
                print("Testing movement after reset... ", end='')
                if self.controller.move_home(speed=0.5):
                    print("✅")
                    print("\n✅ Emergency stop test PASSED")
                else:
                    print("❌")
                    print("\n⚠️ Emergency stop works but reset failed")
            else:
                print("❌ Movement NOT stopped!")
                print("\n❌ Emergency stop test FAILED")
        else:
            print("❌ Could not verify")
    
    def _full_system_check(self):
        """Kompletter Systemcheck."""
        print("\n🔍 FULL SYSTEM CHECK")
        print("="*50)
        
        checks = {
            "Serial Connection": False,
            "Calibration": False,
            "Joint Limits": False,
            "Home Position": False,
            "Gripper": False,
            "LED": False,
            "Emergency Stop": False,
            "Teaching Mode": False,
            "Weight Compensation": False
        }
        
        # Import SERVO_LIMITS here if needed
        try:
            from core.constants import SERVO_LIMITS
        except ImportError:
            logger.error("Failed to import SERVO_LIMITS")
            SERVO_LIMITS = {}
        
        # 1. Serial
        print("Checking serial connection... ", end='')
        if self.controller.serial.connected:
            checks["Serial Connection"] = True
            print("✅")
        else:
            print("❌")
        
        # 2. Calibration
        print("Checking calibration... ", end='')
        if self.calibrator.calibration.calibration_valid:
            checks["Calibration"] = True
            print("✅")
        else:
            print("⚠️ (not calibrated)")
        
        # 3. Joint Limits
        print("Checking joint limits... ", end='')
        all_good = True
        for joint, limits in SERVO_LIMITS.items():
            if limits[0] >= limits[1]:
                all_good = False
        checks["Joint Limits"] = all_good
        print("✅" if all_good else "❌")
        
        # 4. Home Position
        print("Checking home position... ", end='')
        if self.controller.move_home(speed=1.0):
            checks["Home Position"] = True
            print("✅")
        else:
            print("❌")
        
        # 5. Gripper
        print("Checking gripper... ", end='')
        if self.controller.gripper_control(0.5):
            checks["Gripper"] = True
            print("✅")
        else:
            print("❌")
        
        # 6. LED
        print("Checking LED... ", end='')
        if self.controller.led_control(True, 128):
            time.sleep(0.2)
            self.controller.led_control(False)
            checks["LED"] = True
            print("✅")
        else:
            print("❌")
        
        # 7. Emergency Stop
        print("Checking emergency stop... ", end='')
        self.controller.emergency_stop()
        self.controller.reset_emergency()
        checks["Emergency Stop"] = True
        print("✅")
        
        # 8. Teaching Mode
        print("Checking teaching mode... ", end='')
        if self.teacher.start_recording("test", RecordingMode.MANUAL):
            self.teacher.stop_recording()
            checks["Teaching Mode"] = True
            print("✅")
        else:
            print("❌")
        
        # 9. Weight Compensation
        print("Checking weight compensation... ", end='')
        if self.controller.config.enable_weight_compensation:
            checks["Weight Compensation"] = True
            print("✅")
        else:
            print("⚠️ (disabled)")
        
        # Results
        print("\n" + "="*50)
        print("SYSTEM CHECK RESULTS")
        print("="*50)
        
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{check:25s}: {status}")
        
        print("-"*50)
        print(f"Overall: {passed}/{total} checks passed")
        
        if passed == total:
            print("\n✅ System is fully operational!")
        elif passed >= total * 0.8:
            print("\n⚠️ System mostly operational")
        else:
            print("\n❌ System has issues - check failed items")
    
    # ============== MANUAL CONTROL ==============
    
    def _manual_control(self):
        """Manuelle Steuerung."""
        try:
            print("\n🎮 MANUAL CONTROL MODE")
            print("-" * 40)
            print("Controls:")
            print("  q/a: Base left/right")
            print("  w/s: Shoulder up/down")
            print("  e/d: Elbow up/down")
            print("  r/f: Wrist up/down")
            print("  t/g: Roll left/right")
            print("  y/h: Gripper open/close")
            print("  +/-: Speed up/down")
            print("  space: Emergency stop")
            print("  x: Exit manual control")
            print("-" * 40)
            
            speed = 1.0
            step = 0.1  # Radians per keypress
            
            # Apply calibration if available
            if self.calibrator.calibration.calibration_valid:
                print("📐 Using calibrated limits")
                step = 0.05  # Smaller steps with calibration
            
            print(f"\nCurrent speed: {speed:.1f}")
            print("Ready for input...")
            
            while True:
                key = self.terminal.get_key()
                if not key:
                    continue
                
                if key == 'x':
                    break
                elif key == ' ':
                    self.controller.emergency_stop()
                    print("🚨 EMERGENCY STOP")
                elif key == '+':
                    speed = min(2.0, speed + 0.1)
                    print(f"Speed: {speed:.1f}")
                elif key == '-':
                    speed = max(0.1, speed - 0.1)
                    print(f"Speed: {speed:.1f}")
                else:
                    # Joint control
                    joint_map = {
                        'q': ('base', -step),
                        'a': ('base', step),
                        'w': ('shoulder', step),
                        's': ('shoulder', -step),
                        'e': ('elbow', step),
                        'd': ('elbow', -step),
                        'r': ('wrist', step),
                        'f': ('wrist', -step),
                        't': ('roll', step),
                        'g': ('roll', -step),
                        'y': ('hand', -step),
                        'h': ('hand', step)
                    }
                    
                    if key in joint_map:
                        joint, delta = joint_map[key]
                        current = self.controller.current_position[joint]
                        new_pos = {joint: current + delta}
                        
                        # Apply calibration if available
                        if self.calibrator.calibration.calibration_valid:
                            if joint in self.calibrator.calibration.joints:
                                cal = self.calibrator.calibration.joints[joint]
                                # Use calibrated limits
                                new_pos[joint] = max(cal.safe_min, min(cal.safe_max, new_pos[joint]))
                        
                        self.controller.move_joints(
                            new_pos,
                            speed=speed,
                            trajectory_type=TrajectoryType.LINEAR,
                            wait=False
                        )
                        self.movement_count += 1
                        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in manual control: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    # ============== SCANNER PATTERNS ==============
    
    def _scanner_menu(self):
        """Scanner Pattern Menü - ERWEITERT."""
        print("\n📷 SCANNER PATTERNS")
        print("-" * 40)
        
        # Zeige Scanner-Status
        if self.calibrator.calibration.scanner:
            print(f"Scanner calibrated: ✅")
            print(f"Optimal distance: {self.calibrator.calibration.scanner.optimal_distance*100:.1f}cm")
        else:
            print("Scanner calibration: ⚠️ Not calibrated")
        
        print("\n=== BASIC PATTERNS ===")
        print("1. 📐 Raster Scan (Grid)")
        print("2. 🌀 Spiral Scan")
        print("3. 🌐 Spherical Scan (3D)")
        print("4. 🔄 Turntable Scan")
        print("5. 🕸️ Cobweb Scan")
        
        print("\n=== ADVANCED PATTERNS ===")
        print("6. 🎯 Adaptive Scan (Smart)")
        print("7. 🧬 Helix Scan (Cylindrical)")
        print("8. 🗿 Statue Spiral Scan")
        print("9. 📋 Table Scan (Flat)")
        
        print("\n=== QUICK PRESETS ===")
        print("10. ⚡ Quick Scan (Fast)")
        print("11. 🔬 Detailed Scan (Slow)")
        print("12. 🏺 Small Object Preset")
        print("13. 📦 Large Object Preset")
        print("14. 🎯 Smart Scan Selector")
        
        print("\n0. ↩️ Back")
        
        choice = input("\n👉 Select pattern: ").strip()
        
        patterns = {
            '1': self._raster_scan,
            '2': self._spiral_scan,
            '3': self._spherical_scan,
            '4': self._turntable_scan,
            '5': self._cobweb_scan,
            '6': self._adaptive_scan,
            '7': self._helix_scan,
            '8': self._statue_scan,
            '9': self._table_scan,
            '10': self._quick_preset_scan,
            '11': self._detailed_preset_scan,
            '12': self._small_object_preset,
            '13': self._large_object_preset,
            '14': self._smart_scan_selector
        }
        
        if choice in patterns:
            patterns[choice]()
        elif choice != '0':
            print("❌ Invalid option")

    
    def _raster_scan(self):
        """Raster Scan ausführen."""
        print("\n📐 RASTER SCAN SETUP")
        print("-" * 40)
        
        # Use calibrated values if available
        if self.calibrator.calibration.scanner:
            default_speed = self.calibrator.calibration.scanner.optimal_speed
            default_settle = self.calibrator.calibration.scanner.optimal_settle_time
        else:
            default_speed = 0.3
            default_settle = 0.5
        
        # Parameter abfragen
        width = float(input("Width (cm) [20]: ") or "20") / 100
        height = float(input("Height (cm) [15]: ") or "15") / 100
        rows = int(input("Rows [10]: ") or "10")
        cols = int(input("Columns [10]: ") or "10")
        speed = float(input(f"Speed (0.1-1.0) [{default_speed:.1f}]: ") or str(default_speed))
        settle = float(input(f"Settle time (s) [{default_settle:.1f}]: ") or str(default_settle))
        
        # Pattern erstellen
        pattern = RasterScanPattern(
            width=width,
            height=height,
            rows=rows,
            cols=cols,
            speed=speed,
            settle_time=settle
        )
        
        # Ausführen
        self._execute_scan(pattern)
    
    def _spiral_scan(self):
        """Spiral Scan ausführen."""
        print("\n🌀 SPIRAL SCAN SETUP")
        print("-" * 40)
        
        r_start = float(input("Start radius (cm) [5]: ") or "5") / 100
        r_end = float(input("End radius (cm) [15]: ") or "15") / 100
        revs = int(input("Revolutions [5]: ") or "5")
        points = int(input("Points per revolution [36]: ") or "36")
        speed = float(input("Speed (0.1-1.0) [0.25]: ") or "0.25")
        continuous = input("Continuous motion? (y/n) [y]: ").lower() != 'n'
        
        pattern = SpiralScanPattern(
            radius_start=r_start,
            radius_end=r_end,
            revolutions=revs,
            points_per_rev=points,
            speed=speed,
            continuous=continuous
        )
        
        self._execute_scan(pattern)
    
    def _spherical_scan(self):
        """Spherical Scan ausführen."""
        print("\n🌐 SPHERICAL SCAN SETUP")
        print("-" * 40)
        
        # Use calibrated optimal distance if available
        if self.calibrator.calibration.scanner:
            default_radius = self.calibrator.calibration.scanner.optimal_distance
        else:
            default_radius = 0.15
        
        radius = float(input(f"Radius (cm) [{default_radius*100:.0f}]: ") or str(default_radius*100)) / 100
        h_steps = int(input("Horizontal steps [12]: ") or "12")
        v_steps = int(input("Vertical steps [8]: ") or "8")
        speed = float(input("Speed (0.1-1.0) [0.3]: ") or "0.3")
        
        pattern = SphericalScanPattern(
            radius=radius,
            theta_steps=h_steps,
            phi_steps=v_steps,
            speed=speed
        )
        
        self._execute_scan(pattern)
    
    def _turntable_scan(self):
        """Turntable Scan ausführen."""
        print("\n🔄 TURNTABLE SCAN SETUP")
        print("-" * 40)
        
        steps = int(input("Rotation steps [36]: ") or "36")
        levels = int(input("Height levels [1]: ") or "1")
        settle = float(input("Settle time (s) [1.0]: ") or "1.0")
        
        pattern = TurntableScanPattern(
            steps=steps,
            height_levels=levels,
            settle_time=settle
        )
        
        self._execute_scan(pattern)
    
    def _cobweb_scan(self):
        """Cobweb Scan ausführen."""
        print("\n🕸️ COBWEB SCAN SETUP")
        print("-" * 40)
        
        lines = int(input("Radial lines [8]: ") or "8")
        circles = int(input("Circles [5]: ") or "5")
        radius = float(input("Max radius (cm) [15]: ") or "15") / 100
        
        pattern = CobwebScanPattern(
            radial_lines=lines,
            circles=circles,
            max_radius=radius
        )
        
        self._execute_scan(pattern)
    
    def _adaptive_scan(self):
        """Adaptiver Scan ausführen."""
        print("\n🎯 ADAPTIVE SCAN SETUP")
        print("-" * 40)
        print("Intelligent scan that adapts to object geometry")
        
        initial_points = int(input("Initial scan points [20]: ") or "20")
        refinement = float(input("Refinement threshold (0.01-0.1) [0.05]: ") or "0.05")
        
        pattern = AdaptiveScanPattern(
            initial_points=initial_points,
            refinement_threshold=refinement,
            center_position=SCANNER_POSITION
        )
        
        self._execute_scan(pattern)
    
    def _helix_scan(self):
        """Helix Scan für zylindrische Objekte."""
        print("\n🧬 HELIX SCAN SETUP")
        print("-" * 40)
        print("Optimal for cylindrical objects (bottles, cans, tubes)")
        
        radius = float(input("Object radius (cm) [12]: ") or "12") / 100
        height = float(input("Object height (cm) [20]: ") or "20") / 100
        turns = int(input("Number of turns [5]: ") or "5")
        speed = float(input("Speed (0.1-1.0) [0.3]: ") or "0.3")
        
        pattern = HelixScanPattern(
            radius=radius,
            height=height,
            turns=turns,
            speed=speed,
            center_position=SCANNER_POSITION
        )
        
        self._execute_scan(pattern)
    
    def _statue_scan(self):
        """Statue Spiral Scan ausführen."""
        print("\n🗿 STATUE SPIRAL SCAN SETUP")
        print("-" * 40)
        print("Optimized for vertical objects with details")
        
        height = float(input("Statue height (cm) [25]: ") or "25") / 100
        radius_start = float(input("Start radius (cm) [8]: ") or "8") / 100
        radius_end = float(input("End radius (cm) [12]: ") or "12") / 100
        revolutions = int(input("Revolutions [8]: ") or "8")
        
        pattern = StatueSpiralPattern(
            height_range=height,
            radius_start=radius_start,
            radius_end=radius_end,
            revolutions=revolutions,
            center_position=SCANNER_POSITION
        )
        
        self._execute_scan(pattern)
    
    def _table_scan(self):
        """Table/Flat surface scan."""
        print("\n📋 TABLE SCAN SETUP")
        print("-" * 40)
        print("Optimized for flat surfaces and documents")
        
        # Essentially a turntable scan with single level
        pattern = TableScanPattern(
            steps=36,
            height_levels=1,
            radius=0.15,
            center_position=SCANNER_POSITION
        )
        
        self._execute_scan(pattern)
    
    def _quick_preset_scan(self):
        """Quick Scan Preset - schneller Übersichtsscan."""
        print("\n⚡ QUICK SCAN PRESET")
        print("-" * 40)
        print("Fast overview scan - 5x5 grid, high speed")
        
        confirm = input("Start quick scan? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            pattern = RasterScanPattern(
                rows=5,
                cols=5,
                speed=0.5,
                settle_time=0.3,
                center_position=SCANNER_POSITION
            )
            self._execute_scan(pattern)
    
    def _detailed_preset_scan(self):
        """Detailed Scan Preset - langsamer Detailscan."""
        print("\n🔬 DETAILED SCAN PRESET")
        print("-" * 40)
        print("High-resolution scan - 15x15 grid, slow speed")
        
        confirm = input("Start detailed scan? This takes ~10 minutes (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            pattern = RasterScanPattern(
                rows=15,
                cols=15,
                speed=0.2,
                settle_time=0.8,
                overlap=0.35,
                center_position=SCANNER_POSITION
            )
            self._execute_scan(pattern)
    
    def _small_object_preset(self):
        """Small Object Preset - für kleine Objekte."""
        print("\n🏺 SMALL OBJECT PRESET")
        print("-" * 40)
        print("Optimized for objects < 10cm")
        
        print("Select object type:")
        print("1. Round/Cylindrical")
        print("2. Flat/Rectangular")
        print("3. Complex/Irregular")
        
        obj_type = input("Type (1-3): ").strip()
        
        if obj_type == '1':
            pattern = SpiralScanPattern(
                radius_start=0.05,
                radius_end=0.10,
                revolutions=4,
                points_per_rev=30,
                speed=0.2,
                center_position=SCANNER_POSITION
            )
        elif obj_type == '2':
            pattern = RasterScanPattern(
                width=0.12,
                height=0.12,
                rows=8,
                cols=8,
                speed=0.25,
                center_position=SCANNER_POSITION
            )
        else:
            pattern = AdaptiveScanPattern(
                initial_points=25,
                refinement_threshold=0.03,
                center_position=SCANNER_POSITION
            )
        
        self._execute_scan(pattern)
    
    def _large_object_preset(self):
        """Large Object Preset - für große Objekte."""
        print("\n📦 LARGE OBJECT PRESET")
        print("-" * 40)
        print("Optimized for objects > 20cm")
        
        print("Select scan strategy:")
        print("1. Full 3D coverage (spherical)")
        print("2. Turntable style (rotating)")
        print("3. Multi-level (horizontal layers)")
        
        strategy = input("Strategy (1-3): ").strip()
        
        if strategy == '1':
            pattern = SphericalScanPattern(
                radius=0.20,
                theta_steps=16,
                phi_steps=10,
                speed=0.3,
                center_position=SCANNER_POSITION
            )
        elif strategy == '2':
            pattern = TurntableScanPattern(
                steps=48,
                height_levels=3,
                height_range=0.20,
                speed=0.35,
                center_position=SCANNER_POSITION
            )
        else:
            # Multi-level raster
            pattern = RasterScanPattern(
                width=0.25,
                height=0.25,
                rows=12,
                cols=12,
                speed=0.3,
                center_position=SCANNER_POSITION
            )
        
        self._execute_scan(pattern)

    # ============== 6. HAUPT-MENÜ ANPASSUNG (optional) ==============
    # In der _show_main_menu Methode, füge diese Zeile nach den Scanner-Optionen ein:

    def _show_main_menu_addition(self):
        """Zusätzliche Menüpunkte für erweiterte Scanner-Features."""
        print("14. 🎯 Smart Scan (Auto-select pattern)")
        print("15. 📊 Compare scan patterns")
        print("16. 💾 Save/Load scan configuration")

    # ============== 7. SMART SCAN HELPER (Bonus) ==============
    # Intelligente Pattern-Auswahl basierend auf Objekttyp:

    def _smart_scan_selector(self):
        """Hilft bei der Auswahl des optimalen Scan-Patterns."""
        print("\n🎯 SMART SCAN SELECTOR")
        print("-" * 40)
        print("Answer a few questions to get the optimal scan pattern")
        
        print("\nObject shape:")
        print("1. Flat/Planar (paper, PCB, relief)")
        print("2. Cylindrical (bottle, can, tube)")
        print("3. Spherical (ball, head)")
        print("4. Boxy (cube, box)")
        print("5. Complex/Irregular (statue, toy)")
        
        shape = input("Shape (1-5): ").strip()
        
        print("\nObject size:")
        print("1. Tiny (<5cm)")
        print("2. Small (5-15cm)")
        print("3. Medium (15-25cm)")
        print("4. Large (>25cm)")
        
        size = input("Size (1-4): ").strip()
        
        print("\nDesired quality:")
        print("1. Quick preview")
        print("2. Standard quality")
        print("3. High detail")
        
        quality = input("Quality (1-3): ").strip()
        
        # Pattern selection logic
        if shape == '1':  # Flat
            pattern = RasterScanPattern(
                rows=5 if quality == '1' else 10 if quality == '2' else 15,
                cols=5 if quality == '1' else 10 if quality == '2' else 15,
                zigzag=True,
                speed=0.5 if quality == '1' else 0.3 if quality == '2' else 0.2
            )
        elif shape == '2':  # Cylindrical
            pattern = HelixScanPattern(
                turns=3 if quality == '1' else 5 if quality == '2' else 8,
                points_per_turn=20 if quality == '1' else 30 if quality == '2' else 40
            )
        elif shape == '3':  # Spherical
            pattern = SphericalScanPattern(
                theta_steps=8 if quality == '1' else 12 if quality == '2' else 16,
                phi_steps=6 if quality == '1' else 8 if quality == '2' else 12
            )
        elif shape == '4':  # Boxy
            pattern = TurntableScanPattern(
                steps=24 if quality == '1' else 36 if quality == '2' else 48,
                height_levels=1 if quality == '1' else 2 if quality == '2' else 3
            )
        else:  # Complex
            pattern = AdaptiveScanPattern(
                initial_points=15 if quality == '1' else 25 if quality == '2' else 40
            )
        
        print(f"\n✨ Recommended: {pattern.name}")
        confirm = input("Execute this scan? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            self._execute_scan(pattern)

    
    def _execute_scan(self, pattern):
        """
        Universelle Scan-Ausführung für alle Pattern-Typen.
        Erweiterte Version mit Fehlerbehandlung und Optimierungen.
        """
        # Validierung
        if not pattern:
            print("❌ No pattern provided")
            return False
        
        print(f"\n🚀 Starting {pattern.name}...")
        print("Press Ctrl+C to abort\n")
        
        # Progress tracking
        start_time = time.time()
        
        try:
            # 1. SCANNER POSITION ANFAHREN
            print("📍 Moving to scanner position...")
            if not self.controller.move_to_scanner_position(speed=0.5):
                print("⚠️ Could not reach scanner position, continuing anyway...")
            time.sleep(2)
            
            # 2. KALIBRIERUNG ANWENDEN (falls vorhanden)
            if hasattr(self.calibrator, 'calibration') and self.calibrator.calibration.scanner:
                print("📐 Applying calibrated scanner parameters")
                
                # Übernehme optimale Parameter aus Kalibrierung
                if hasattr(pattern, 'settle_time'):
                    original_settle = pattern.settle_time
                    pattern.settle_time = max(
                        pattern.settle_time,
                        self.calibrator.calibration.scanner.optimal_settle_time
                    )
                    if pattern.settle_time != original_settle:
                        print(f"   Settle time adjusted: {original_settle:.2f}s → {pattern.settle_time:.2f}s")
                
                # Falls Pattern keine center_position hat, setze Scanner-Position
                if not hasattr(pattern, 'center_position') or pattern.center_position is None:
                    pattern.center_position = SCANNER_POSITION.copy()
            
            # 3. PATTERN-SPEZIFISCHE VORBEREITUNG
            pattern_points = None
            if hasattr(pattern, 'generate_points'):
                print(f"📊 Generating {pattern.name} points...")
                pattern_points = pattern.generate_points()
                
                if pattern_points:
                    # Schätze Scan-Dauer
                    total_points = len(pattern_points)
                    avg_time_per_point = getattr(pattern, 'speed', 0.3) + getattr(pattern, 'settle_time', 0.5)
                    estimated_duration = total_points * avg_time_per_point
                    
                    print(f"   Points to scan: {total_points}")
                    print(f"   Estimated duration: {estimated_duration/60:.1f} minutes")
                    
                    # Bei langen Scans Bestätigung einholen
                    if estimated_duration > 300:  # > 5 Minuten
                        confirm = input(f"⏱️ This scan will take ~{estimated_duration/60:.1f} minutes. Continue? (y/n): ")
                        if confirm.lower() not in ['y', 'yes']:
                            print("❌ Scan cancelled by user")
                            return False
            
            # 4. LED AKTIVIEREN (für bessere Sichtbarkeit)
            print("💡 Activating LED...")
            self.controller.led_control(True, brightness=200)
            
            # 5. PATTERN AUSFÜHREN
            print(f"🔄 Executing {pattern.name}...")
            print("-" * 40)
            
            # Progress callback für lange Scans
            def progress_callback(current, total):
                if current % 10 == 0 or current == total:
                    progress = (current / total) * 100
                    elapsed = time.time() - start_time
                    if current > 0:
                        remaining = (elapsed / current) * (total - current)
                        print(f"   Progress: {progress:.1f}% ({current}/{total}) - ETA: {remaining/60:.1f} min")
                    else:
                        print(f"   Progress: {progress:.1f}% ({current}/{total})")
            
            # Setze Progress-Callback wenn möglich
            if hasattr(pattern, 'set_progress_callback'):
                pattern.set_progress_callback(progress_callback)
            
            # HAUPTAUSFÜHRUNG
            success = self.controller.execute_pattern(pattern)
            
            # 6. NACHBEREITUNG
            print("-" * 40)
            
            # LED ausschalten
            self.controller.led_control(False)
            
            # Scan-Statistik
            elapsed_time = time.time() - start_time
            
            if success:
                print(f"\n✅ {pattern.name} completed successfully!")
                print(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
                
                # Scan-Zusammenfassung
                if pattern_points:
                    print(f"📊 Scanned {len(pattern_points)} points")
                    
                # Optional: Speichere Scan-Konfiguration für Wiederverwendung
                save_config = input("\n💾 Save this scan configuration? (y/n): ").strip().lower()
                if save_config in ['y', 'yes']:
                    self._save_scan_configuration(pattern)
            else:
                print(f"\n❌ {pattern.name} failed or was aborted")
                print(f"⏱️ Stopped after: {elapsed_time/60:.1f} minutes")
                
                # Fehleranalyse
                if elapsed_time < 5:  # Sehr schneller Abbruch
                    print("💡 Tip: Check robot connection and servo power")
                elif pattern_points and hasattr(self.controller, 'last_error'):
                    print(f"🔍 Last error: {self.controller.last_error}")
            
            return success
            
        except KeyboardInterrupt:
            print("\n\n🛑 Scan interrupted by user!")
            self.controller.emergency_stop()
            self.controller.led_control(False)
            return False
            
        except Exception as e:
            print(f"\n❌ Scan error: {e}")
            logger.error(f"Scan execution error: {e}", exc_info=True)
            self.controller.led_control(False)
            return False
        
        finally:
            # Sicherstellen dass LED aus ist
            try:
                self.controller.led_control(False)
            except:
                pass


    def _save_scan_configuration(self, pattern):
        """Speichert Scan-Konfiguration für spätere Verwendung."""
        try:
            config_name = input("Configuration name: ").strip()
            if not config_name:
                return
            
            # Erstelle Konfigurations-Dict
            config = {
                'name': config_name,
                'pattern_type': pattern.__class__.__name__,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {}
            }
            
            # Sammle Pattern-Parameter
            for attr in ['width', 'height', 'rows', 'cols', 'radius', 'speed', 
                         'settle_time', 'revolutions', 'steps', 'turns']:
                if hasattr(pattern, attr):
                    config['parameters'][attr] = getattr(pattern, attr)
            
            # Speichere in Datei
            config_dir = Path("scan_configurations")
            config_dir.mkdir(exist_ok=True)
            
            filename = config_dir / f"{config_name.replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save configuration: {e}")


    def _load_scan_configuration(self):
        """Lädt gespeicherte Scan-Konfiguration."""
        try:
            config_dir = Path("scan_configurations")
            if not config_dir.exists():
                print("📁 No saved configurations found")
                return None
            
            configs = list(config_dir.glob("*.json"))
            if not configs:
                print("📁 No saved configurations found")
                return None
            
            print("\n💾 SAVED CONFIGURATIONS:")
            for i, config_file in enumerate(configs, 1):
                # Lade und zeige Info
                with open(config_file, 'r') as f:
                    data = json.load(f)
                print(f"{i}. {data['name']} ({data['pattern_type']}) - {data['timestamp']}")
            
            choice = input("\nSelect configuration (number) or 0 to cancel: ").strip()
            
            if choice == '0':
                return None
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    with open(configs[idx], 'r') as f:
                        config = json.load(f)
                    
                    # Rekonstruiere Pattern
                    pattern_type = config['pattern_type']
                    params = config['parameters']
                    
                    # Erstelle Pattern basierend auf Typ
                    if pattern_type == 'RasterScanPattern':
                        pattern = RasterScanPattern(**params)
                    elif pattern_type == 'SpiralScanPattern':
                        pattern = SpiralScanPattern(**params)
                    elif pattern_type == 'SphericalScanPattern':
                        pattern = SphericalScanPattern(**params)
                    elif pattern_type == 'HelixScanPattern':
                        pattern = HelixScanPattern(**params)
                    # ... weitere Pattern-Typen
                    else:
                        print(f"❌ Unknown pattern type: {pattern_type}")
                        return None
                    
                    print(f"✅ Loaded configuration: {config['name']}")
                    return pattern
                    
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return None
                
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return None
            
    
    # ============== TEACHING MODE ==============
    
    def _teaching_menu(self):
        """Teaching Mode Menü."""
        try:
            print("\n🎓 TEACHING MODE")
            print("-" * 40)
            print("1. Start Recording (Manual)")
            print("2. Start Recording (Continuous)")
            print("3. Set Recording Parameters")
            print("4. Save Current Sequence")
            print("5. Load Sequence")
            print("0. Back")
            
            choice = input("\n👉 Select option: ").strip()
            
            if choice == '1':
                self._start_teaching(RecordingMode.MANUAL)
            elif choice == '2':
                self._start_teaching(RecordingMode.CONTINUOUS)
            elif choice == '3':
                self._set_teaching_parameters()
            elif choice == '4':
                self.teacher.save_sequence()
            elif choice == '5':
                self._load_teaching_sequence()
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in teaching menu: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    def _start_teaching(self, mode: RecordingMode):
        """Startet Teaching Mode."""
        name = input("Sequence name: ").strip()
        if not name:
            print("❌ Name required")
            return
        
        desc = input("Description (optional): ").strip()
        
        # Start recording
        if self.teacher.start_recording(name, mode, desc):
            print(f"\n🔴 Recording '{name}' started ({mode.value} mode)")
            
            if mode == RecordingMode.MANUAL:
                print("\nControls:")
                print("  SPACE: Record waypoint")
                print("  +/-: Change speed")
                print("  a/z: Change acceleration")
                print("  j: Change jerk")
                print("  t: Change trajectory type")
                print("  s: Stop recording")
                print("\nMove robot to positions and press SPACE to record")
                
                while self.teacher.is_recording:
                    key = self.terminal.get_key()
                    
                    if key == ' ':
                        if self.teacher.record_waypoint():
                            print(f"✓ Waypoint {self.teacher.waypoint_count} recorded")
                            print(f"   Speed: {self.teacher.current_speed:.1f}")
                            print(f"   Accel: {self.teacher.current_acceleration:.1f}")
                    elif key == '+':
                        self.teacher.current_speed = min(2.0, self.teacher.current_speed + 0.1)
                        print(f"Speed: {self.teacher.current_speed:.1f}")
                    elif key == '-':
                        self.teacher.current_speed = max(0.1, self.teacher.current_speed - 0.1)
                        print(f"Speed: {self.teacher.current_speed:.1f}")
                    elif key == 'a':
                        self.teacher.current_acceleration = min(5.0, self.teacher.current_acceleration + 0.5)
                        print(f"Acceleration: {self.teacher.current_acceleration:.1f}")
                    elif key == 'z':
                        self.teacher.current_acceleration = max(0.5, self.teacher.current_acceleration - 0.5)
                        print(f"Acceleration: {self.teacher.current_acceleration:.1f}")
                    elif key == 'j':
                        self.teacher.current_jerk = min(10.0, self.teacher.current_jerk + 1.0)
                        print(f"Jerk: {self.teacher.current_jerk:.1f}")
                    elif key == 't':
                        # Cycle through trajectory types
                        types = [TrajectoryType.LINEAR, TrajectoryType.S_CURVE, 
                                TrajectoryType.TRAPEZOIDAL, TrajectoryType.SINUSOIDAL]
                        current_idx = types.index(self.teacher.current_trajectory)
                        self.teacher.current_trajectory = types[(current_idx + 1) % len(types)]
                        print(f"Trajectory: {self.teacher.current_trajectory.value}")
                    elif key == 's':
                        break
            
            else:  # Continuous mode
                print("Recording continuously...")
                print("Press 's' to stop")
                
                while self.teacher.is_recording:
                    key = self.terminal.get_key()
                    if key == 's':
                        break
                    time.sleep(0.1)
            
            # Stop and save
            sequence = self.teacher.stop_recording()
            if sequence:
                print(f"\n✅ Recording complete: {len(sequence.waypoints)} waypoints")
                save = input("Save sequence? (y/n): ").lower()
                if save == 'y':
                    self.teacher.save_sequence()
    
    def _set_teaching_parameters(self):
        """Setzt Teaching Parameter."""
        print("\n⚙️ TEACHING PARAMETERS")
        print("-" * 40)
        
        speed = float(input(f"Speed [{self.teacher.current_speed:.1f}]: ") or self.teacher.current_speed)
        acc = float(input(f"Acceleration [{self.teacher.current_acceleration:.1f}]: ") or self.teacher.current_acceleration)
        jerk = float(input(f"Jerk [{self.teacher.current_jerk:.1f}]: ") or self.teacher.current_jerk)
        
        print("\nTrajectory types:")
        print("1. Linear")
        print("2. Trapezoidal")
        print("3. S-Curve")
        print("4. Sinusoidal")
        
        traj_choice = input("Select [3]: ").strip() or '3'
        trajectories = {
            '1': TrajectoryType.LINEAR,
            '2': TrajectoryType.TRAPEZOIDAL,
            '3': TrajectoryType.S_CURVE,
            '4': TrajectoryType.SINUSOIDAL
        }
        
        trajectory = trajectories.get(traj_choice, TrajectoryType.S_CURVE)
        
        settle = float(input("Settle time (s) [0.0]: ") or "0.0")
        
        self.teacher.set_parameters(
            speed=speed,
            acceleration=acc,
            jerk=jerk,
            trajectory_type=trajectory,
            settle_time=settle
        )
        
        print("✅ Parameters updated")
    
    def _load_teaching_sequence(self):
        """Lädt eine Teaching Sequenz."""
        # Liste verfügbare Sequenzen
        seq_dir = Path("sequences")
        if not seq_dir.exists():
            print("No sequences found")
            return
        
        sequences = list(seq_dir.glob("*.json"))
        if not sequences:
            print("No sequences found")
            return
        
        print("\nAvailable sequences:")
        for i, seq in enumerate(sequences, 1):
            print(f"{i}. {seq.name}")
        
        choice = input("\nSelect sequence: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                sequence = self.teacher.load_sequence(str(sequences[idx]))
                if sequence:
                    print(f"✅ Loaded '{sequence.name}'")
                    # Play option
                    play = input("Play sequence? (y/n): ").lower()
                    if play == 'y':
                        self._play_sequence(sequence)
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def _play_sequence(self, sequence):
        """Spielt eine Sequenz ab."""
        speed_factor = float(input("Speed factor [1.0]: ") or "1.0")
        loop = input("Loop? (y/n): ").lower() == 'y'
        
        print(f"\n▶️ Playing '{sequence.name}'...")
        print("Press Ctrl+C to stop\n")
        
        # TODO: Implement sequence playback
        # This would iterate through waypoints and execute movements
        
    def _load_sequence(self):
        """Wrapper für Load & Play Sequence."""
        self._load_teaching_sequence()
    
    # ============== OTHER FUNCTIONS ==============
    
    def _move_home(self):
        """Bewegt zur Home Position."""
        print("\n🏠 Moving to home position...")
        if self.controller.move_home(speed=0.5):
            print("✅ Home position reached")
            self.movement_count += 1
        else:
            print("❌ Failed to reach home")
    
    def _settings_menu(self):
        """Einstellungen."""
        try:
            print("\n⚙️ SETTINGS")
            print("-" * 40)
            print(f"Port: {self.controller.config.port}")
            print(f"Baudrate: {self.controller.config.baudrate}")
            print(f"Default Speed: {self.controller.config.default_speed}")
            print(f"Scanner Weight: {self.controller.config.scanner_weight}kg")
            print(f"Weight Compensation: {self.controller.config.enable_weight_compensation}")
            
            # Show calibration status
            if self.calibrator.calibration.calibration_valid:
                print(f"Calibration: VALID (±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad)")
            else:
                print("Calibration: NONE")
            
            print("-" * 40)
            print("1. Change Speed")
            print("2. Toggle Weight Compensation")
            print("3. Set Scanner Weight")
            print("4. Reload Configuration")
            print("0. Back")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                speed = float(input("New speed (0.1-2.0): "))
                self.controller.config.default_speed = max(0.1, min(2.0, speed))
                self.controller.current_speed = self.controller.config.default_speed
                print(f"✅ Speed set to {self.controller.config.default_speed}")
            elif choice == '2':
                self.controller.config.enable_weight_compensation = not self.controller.config.enable_weight_compensation
                state = "enabled" if self.controller.config.enable_weight_compensation else "disabled"
                print(f"✅ Weight compensation {state}")
            elif choice == '3':
                weight = float(input("Scanner weight (g): ")) / 1000
                self.controller.config.scanner_weight = weight
                print(f"✅ Scanner weight set to {weight*1000:.0f}g")
            elif choice == '4':
                # Reload config from file
                import yaml
                try:
                    with open('config.yaml', 'r') as f:
                        config = yaml.safe_load(f)
                    print("✅ Configuration reloaded")
                except Exception as e:
                    print(f"❌ Failed to reload config: {e}")
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in settings menu: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    def _show_status(self):
        """Zeigt Status an."""
        try:
            print("\n📊 SYSTEM STATUS")
            print("-" * 40)
            
            # Check if simulator mode
            is_simulator = hasattr(self.controller.serial, 'robot_state')
            
            if is_simulator:
                print("🎮 MODE: SIMULATOR (No hardware connected)")
                print("-" * 40)
            
            status = self.controller.query_status()
            if status:
                print("Joint Positions (rad):")
                for joint, pos in status['positions'].items():
                    # Show calibrated value if available
                    if self.calibrator.calibration.calibration_valid:
                        if joint in self.calibrator.calibration.joints:
                            cal = self.calibrator.calibration.joints[joint]
                            calibrated = cal.apply_calibration(pos)
                            print(f"  {joint:10s}: {pos:+.3f} (cal: {calibrated:+.3f})")
                        else:
                            print(f"  {joint:10s}: {pos:+.3f}")
                    else:
                        print(f"  {joint:10s}: {pos:+.3f}")
                
                print(f"\nTorque Enabled: {status['torque_enabled']}")
                print(f"Temperature: {status.get('temperature', 'N/A')}°C")
                print(f"Voltage: {status.get('voltage', 'N/A')}V")
                
                # Show simulator flag if present
                if status.get('simulator'):
                    print("\n⚠️ This is SIMULATED data")
                
                # Calibration info
                if self.calibrator.calibration.calibration_valid:
                    print(f"\nCalibration:")
                    print(f"  Status: VALID ✅")
                    print(f"  Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad")
                    age_days = (time.time() - self.calibrator.calibration.timestamp) / 86400
                    print(f"  Age: {age_days:.0f} days")
                
                # If simulator, show command log
                if is_simulator:
                    print("\n📝 SIMULATOR COMMAND LOG (last 5):")
                    commands = self.controller.serial.get_command_log()
                    for cmd in commands[-5:]:
                        cmd_type = cmd.get('T', '?')
                        print(f"  Command T={cmd_type}: {cmd}")
                        
            else:
                print("❌ Failed to query status")
                
            input("\nPress ENTER to continue...")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error showing status: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print(f"❌ Error: {e}")
            input("\nPress ENTER to continue...")
    
    def _exit(self):
        """Beendet das Programm."""
        self.running = False
    
    def _cleanup(self):
        """Aufräumen beim Beenden."""
        print("\n🔌 Shutting down...")
        
        if self.controller:
            # Save calibration if changed
            if self.calibrator and self.calibrator.calibration.calibration_valid:
                print("Saving calibration...")
                self.calibrator.save_calibration()
            
            # Graceful shutdown via safety system
            if self.safety_system:
                print("Initiating graceful shutdown...")
                self.safety_system.graceful_shutdown(ShutdownReason.USER_REQUEST)
            else:
                # Manual shutdown
                print("Moving to safe position...")
                self.controller.move_home(speed=0.5)
                time.sleep(1)
                
                # Disconnect
                self.controller.disconnect()
        
        # Session statistics
        uptime = (time.time() - self.session_start) / 60
        print(f"\nSession Statistics:")
        print(f"  Duration: {uptime:.1f} minutes")
        print(f"  Commands: {self.command_count}")
        print(f"  Movements: {self.movement_count}")
        print(f"  Errors: {self.error_count}")
        
        print("\n✅ Shutdown complete")
        print("Goodbye! 👋\n")
    
    def _execute_pattern(self, pattern_name: str):
        """Führt ein Pattern direkt aus (für CLI-Argumente)."""
        patterns = {
            'raster': RasterScanPattern(),
            'spiral': SpiralScanPattern(),
            'spherical': SphericalScanPattern(),
            'turntable': TurntableScanPattern(),
            'cobweb': CobwebScanPattern()
        }
        
        if pattern_name in patterns:
            print(f"Executing {pattern_name} pattern...")
            self.controller.move_to_scanner_position()
            time.sleep(2)
            self.controller.execute_pattern(patterns[pattern_name])
        else:
            print(f"Unknown pattern: {pattern_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f'RoArm M3 Professional Control System v{VERSION}'
    )
    
    parser.add_argument(
        '--port',
        default='/dev/tty.usbserial-110',
        help='Serial port'
    )
    
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Baudrate'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Default speed factor (0.1-2.0)'
    )
    
    parser.add_argument(
        '--pattern',
        choices=['raster', 'spiral', 'spherical', 'turntable', 'cobweb'],
        help='Execute pattern directly'
    )
    
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run auto calibration and exit'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Enable trace mode (maximum verbosity)'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run in simulator mode (no hardware required)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    
    if args.debug:
        log_level = logging.DEBUG
    elif args.trace:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level)
    
    # Setup logger mit korrekten Parametern basierend auf utils.logger
    # Nutze die setup_logger Funktion wie sie definiert ist
    setup_logger(level=logging.getLevelName(log_level))
    logger = get_logger(__name__)
    
    # Debug info
    if args.debug or args.trace:
        print("\n" + "="*60)
        print(f"🔍 {'TRACE' if args.trace else 'DEBUG'} MODE ACTIVATED")
        print("="*60)
        print(f"Log Level: {logging.getLevelName(log_level)}")
        print(f"Python: {sys.version}")
        print("="*60 + "\n")
        
        if logger:
            logger.debug("Debug mode activated")
            logger.debug(f"Python version: {sys.version}")
            logger.debug(f"Working directory: {Path.cwd()}")
    
    # Import statements for constants
    global SERVO_LIMITS
    try:
        from core.constants import SERVO_LIMITS
        if args.debug:
            logger.debug(f"SERVO_LIMITS imported: {len(SERVO_LIMITS)} joints")
    except ImportError as e:
        print(f"❌ Critical: Failed to import constants: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)
    
    # Start CLI
    try:
        cli = RoArmCLI(debug_mode=args.debug, trace_mode=args.trace)
        cli.run(args)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if args.debug or args.trace:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
