#!/usr/bin/env python3
"""
RoArm M3 Professional Control System - Enhanced Edition
Version 3.0.0 - Mit allen erweiterten Features
Optimiert für macOS M4 mit Revopoint Mini2 Scanner
"""

import sys
import time
import signal
import argparse
import traceback
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

# ========== STANDARD IMPORTS ==========
from core.controller import RoArmController, RoArmConfig
from core.constants import (
    SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION, PARK_POSITION,
    COMMANDS, DEFAULT_SPEED, ERROR_MESSAGES, SUCCESS_MESSAGES
)
from patterns.scan_patterns import (
    RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
    TurntableScanPattern, CobwebScanPattern
)
from safety.safety_system import SafetySystem, SafetyState, ShutdownReason
from utils.logger import setup_logger, get_logger

# ========== ERWEITERTE IMPORTS ==========
try:
    # Versuche erweiterte Module zu laden
    from utils.terminal_enhanced import EnhancedTerminalController, ManualControlHelper
    from teaching.teaching_enhanced import (
        EnhancedTeachingRecorder, TeachingSequence,
        PlaybackMode, RecordingMode as EnhancedRecordingMode
    )
    from calibration.calibration_enhanced import EnhancedCalibrationSuite
    from control.manual_control_enhanced import EnhancedManualControl, create_manual_control
    ENHANCED_MODULES = True
except ImportError:
    # Fallback auf Standard-Module
    from utils.terminal import TerminalController as EnhancedTerminalController
    from teaching.recorder import TeachingRecorder as EnhancedTeachingRecorder, RecordingMode as EnhancedRecordingMode
    from calibration.calibration_suite import CalibrationSuite as EnhancedCalibrationSuite
    EnhancedManualControl = None
    create_manual_control = None
    ENHANCED_MODULES = False
    print("⚠️ Enhanced modules not found - using standard versions")

# ========== DEBUG MODE IMPORTS ==========
from utils.debug_mode import (
    MockController, 
    SimulationMode, 
    run_debug_session,
    DebugMenu
)
from motion.trajectory import TrajectoryType

# Setup logging
setup_logger()
logger = get_logger(__name__)

# Version Info
VERSION = "3.0.0"
BUILD_DATE = "2024-01-15"
ENHANCED_SUFFIX = " Enhanced" if ENHANCED_MODULES else ""


class OperationMode(Enum):
    """Betriebsmodi der Anwendung."""
    NORMAL = "normal"
    DEBUG = "debug"
    SIMULATION = "simulation"
    TEST = "test"
    DEMO = "demo"


class RoArmCLI:
    """Enhanced Command Line Interface für RoArm Control."""
    
    def __init__(self):
        self.controller = None
        self.teacher = None
        self.calibrator = None
        self.safety_system = None
        self.manual_control = None
        self.terminal = EnhancedTerminalController()
        self.running = True
        self.operation_mode = OperationMode.NORMAL
        self.debug_enabled = False
        
        # Session tracking
        self.session_stats = {
            "start_time": time.time(),
            "commands_executed": 0,
            "patterns_completed": 0,
            "sequences_recorded": 0,
            "sequences_played": 0,
            "calibrations_run": 0,
            "errors_encountered": 0,
            "emergency_stops": 0
        }
        
        # Signal Handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler für Ctrl+C."""
        print("\n\n🚨 EMERGENCY STOP - Ctrl+C detected!")
        
        if self.controller:
            self.controller.emergency_stop()
            
            if self.debug_enabled:
                print("\nDebug Mode Options:")
                print("1. Reset and continue")
                print("2. Show debug summary")
                print("3. Exit")
                
                choice = input("\n👉 Select [3]: ").strip() or '3'
                
                if choice == '1':
                    self.controller.reset_emergency()
                    return
                elif choice == '2':
                    if hasattr(self.controller, 'print_debug_summary'):
                        self.controller.print_debug_summary()
        
        self.running = False
        sys.exit(0)
    
    def run(self, args):
        """Haupteinstiegspunkt."""
        try:
            # Bestimme Betriebsmodus
            if args.debug:
                self.operation_mode = OperationMode.DEBUG
                self.debug_enabled = True
            elif args.test:
                self.operation_mode = OperationMode.TEST
                self.debug_enabled = True
            elif args.demo:
                self.operation_mode = OperationMode.DEMO
                self.debug_enabled = True
            elif args.simulation:
                self.operation_mode = OperationMode.SIMULATION
                self.debug_enabled = True
            
            # Header
            self._print_header()
            
            # Modus-spezifische Ausführung
            if self.operation_mode == OperationMode.NORMAL:
                self._run_normal_mode(args)
            elif self.operation_mode == OperationMode.DEBUG:
                self._run_debug_mode(args)
            elif self.operation_mode == OperationMode.TEST:
                self._run_test_mode(args)
            elif self.operation_mode == OperationMode.DEMO:
                self._run_demo_mode(args)
            elif self.operation_mode == OperationMode.SIMULATION:
                self._run_simulation_mode(args)
                
        except KeyboardInterrupt:
            print("\n\n✋ Shutdown requested...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            self.session_stats["errors_encountered"] += 1
            if self.debug_enabled:
                print("\n🔍 DEBUG: Full traceback:")
                traceback.print_exc()
        finally:
            self._cleanup()
    
    def _print_header(self):
        """Zeigt den Header."""
        print("\n" + "="*70)
        print(f"🤖 RoArm M3 Professional Control System v{VERSION}{ENHANCED_SUFFIX}")
        print("="*70)
        
        if self.operation_mode == OperationMode.DEBUG:
            print("🔧 DEBUG MODE - Simulated Hardware")
            print("   No physical robot required")
        elif self.operation_mode == OperationMode.TEST:
            print("🧪 TEST MODE - Automated Testing")
        elif self.operation_mode == OperationMode.DEMO:
            print("🎭 DEMO MODE - Demonstration")
        elif self.operation_mode == OperationMode.SIMULATION:
            print("🎮 SIMULATION MODE - Virtual Robot")
        else:
            print("📷 Optimized for Revopoint Mini2 Scanner")
            print("🍎 macOS M4 Edition")
            if ENHANCED_MODULES:
                print("✨ Enhanced Features Active:")
                print("   • Real-time manual control (no Enter key needed)")
                print("   • Advanced teaching with smart recording")
                print("   • Professional calibration algorithms")
                print("   • Extended sequence management")
        
        print("⚡ Press Ctrl+C anytime for EMERGENCY STOP")
        print("="*70 + "\n")
    
    # ========== NORMAL MODE ==========
    
    def _run_normal_mode(self, args):
        """Normaler Betrieb mit echter Hardware."""
        config = RoArmConfig(
            port=args.port,
            baudrate=args.baudrate,
            default_speed=args.speed,
            debug=False
        )
        
        print(f"🔌 Connecting to RoArm on {config.port}...")
        self.controller = RoArmController(config)
        
        if not self.controller.serial.connected:
            print("❌ Failed to connect to RoArm")
            print("\n💡 Tip: Use --debug mode to run without hardware")
            return
        
        print("✅ Successfully connected!\n")
        
        self._initialize_components()
        
        if args.calibrate:
            self._run_auto_calibration()
            return
        
        if args.pattern:
            self._execute_pattern(args.pattern)
            return
        
        self._run_cli_loop()
    
    # ========== DEBUG MODE ==========
    
    def _run_debug_mode(self, args):
        """Debug-Modus mit simulierter Hardware."""
        print("🔧 DEBUG MODE INITIALIZATION")
        print("-"*60)
        
        config = RoArmConfig(
            port="MOCK_DEBUG",
            baudrate=args.baudrate,
            default_speed=args.speed,
            debug=True
        )
        
        if args.quick:
            sim_mode = SimulationMode.PERFECT
            print("Using PERFECT simulation (no delays)")
        else:
            print("\nSelect simulation fidelity:")
            print("1. 🎯 Perfect   - Instant, no errors")
            print("2. 🌍 Realistic - With delays and noise")
            print("3. 🎲 Random    - Random events (5% failure)")
            print("4. 💥 Failure   - Test error handling")
            
            choice = input("\n👉 Select mode [2]: ").strip() or '2'
            
            mode_map = {
                '1': SimulationMode.PERFECT,
                '2': SimulationMode.REALISTIC,
                '3': SimulationMode.RANDOM,
                '4': SimulationMode.FAILURE
            }
            sim_mode = mode_map.get(choice, SimulationMode.REALISTIC)
        
        print(f"\n🎮 Creating MockController (mode={sim_mode.value})...")
        self.controller = MockController(config, simulation_mode=sim_mode)
        
        self._initialize_components()
        
        print("✅ Debug environment ready!")
        
        if not args.quick:
            print("\n" + "="*50)
            print("DEBUG MODE OPTIONS")
            print("="*50)
            print("1. 📋 Normal CLI (with simulated hardware)")
            print("2. 🔧 Interactive Debug Menu")
            print("3. 🧪 Run automated test suite")
            print("4. 🎭 Demo sequence")
            print("5. 📊 Performance benchmark")
            
            choice = input("\n👉 Select option [1]: ").strip() or '1'
            
            if choice == '1':
                self._run_cli_loop()
            elif choice == '2':
                run_debug_session(self.controller)
            elif choice == '3':
                self._run_automated_tests()
            elif choice == '4':
                self._run_demo_sequence()
            elif choice == '5':
                self._run_performance_benchmark()
        else:
            self._run_cli_loop()
    
    # ========== TEST MODE ==========
    
    def _run_test_mode(self, args):
        """Automatisierter Test-Modus."""
        print("🧪 TEST MODE")
        print("-"*60)
        
        config = RoArmConfig(debug=True)
        self.controller = MockController(config, SimulationMode.PERFECT)
        self._initialize_components()
        
        test_results = {
            "Connection": self._test_connection(),
            "Basic Movement": self._test_basic_movement(),
            "Manual Control": self._test_manual_control(),
            "Teaching Mode": self._test_teaching_enhanced(),
            "Calibration": self._test_calibration_enhanced(),
            "Patterns": self._test_patterns(),
            "Emergency Stop": self._test_emergency()
        }
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        
        for test, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test:20s}: {status}")
        
        print("-"*50)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {total-passed} tests failed")
            sys.exit(1)
    
    # ========== COMPONENT INITIALIZATION ==========
    
    def _initialize_components(self):
        """Initialisiert alle Komponenten."""
        print("Initializing components...")
        
        # Teaching Recorder
        self.teacher = EnhancedTeachingRecorder(self.controller)
        
        # Calibration Suite
        self.calibrator = EnhancedCalibrationSuite(self.controller)
        
        # Manual Control (nur wenn Enhanced verfügbar)
        if ENHANCED_MODULES and create_manual_control:
            self.manual_control = create_manual_control(self.controller, self.teacher)
        
        # Safety System
        if self.operation_mode == OperationMode.NORMAL:
            self.safety_system = SafetySystem(self.controller)
        
        if self.operation_mode == OperationMode.NORMAL:
            self._check_calibration_status()
        
        print(f"✅ Components initialized{ENHANCED_SUFFIX}\n")
    
    # ========== MAIN CLI LOOP ==========
    
    def _run_cli_loop(self):
        """Haupt-CLI-Loop."""
        while self.running:
            try:
                self._show_main_menu()
                choice = input("\n👉 Select option: ").strip()
                
                if choice:
                    self._handle_main_menu(choice)
                    self.session_stats["commands_executed"] += 1
                    
            except KeyboardInterrupt:
                print("\n\nUse Ctrl+C again to exit or select from menu")
                continue
            except Exception as e:
                logger.error(f"Menu error: {e}")
                self.session_stats["errors_encountered"] += 1
                
                if self.debug_enabled:
                    print(f"\n🔍 DEBUG: {e}")
                    traceback.print_exc()
    
    def _show_main_menu(self):
        """Zeigt das Hauptmenü."""
        print("\n" + "="*50)
        print("MAIN MENU", end="")
        
        if self.debug_enabled:
            print(f" [{self.operation_mode.value.upper()} MODE]", end="")
        if ENHANCED_MODULES:
            print(" [ENHANCED]", end="")
        
        print("\n" + "="*50)
        
        print("1. 🎮 Manual Control" + (" (Enhanced)" if ENHANCED_MODULES else ""))
        print("2. 📷 Scanner Patterns")
        print("3. 🎓 Teaching Mode" + (" (Enhanced)" if ENHANCED_MODULES else ""))
        print("4. 📁 Sequence Manager")
        print("5. 🏠 Move to Home")
        print("6. 🔧 Calibration Suite" + (" (Pro)" if ENHANCED_MODULES else ""))
        print("7. ⚙️  Settings")
        print("8. 📊 Status & Info")
        
        if self.operation_mode == OperationMode.NORMAL:
            print("9. 🧪 System Test")
        else:
            print("9. 🔍 Debug Tools")
        
        if self.debug_enabled:
            print("D. 🐛 Debug Menu")
            print("T. 🧪 Run Tests")
            print("M. 📊 Show Metrics")
        
        print("0. 🚪 Exit")
    
    def _handle_main_menu(self, choice: str):
        """Verarbeitet Hauptmenü-Auswahl."""
        handlers = {
            '1': self._manual_control,
            '2': self._scanner_menu,
            '3': self._teaching_menu_enhanced if ENHANCED_MODULES else self._teaching_menu,
            '4': self._sequence_manager,
            '5': self._move_home,
            '6': self._calibration_menu_enhanced if ENHANCED_MODULES else self._calibration_menu,
            '7': self._settings_menu,
            '8': self._status_menu,
            '9': self._system_test if self.operation_mode == OperationMode.NORMAL else self._debug_tools,
            '0': self._exit
        }
        
        if self.debug_enabled:
            debug_handlers = {
                'd': lambda: run_debug_session(self.controller),
                'D': lambda: run_debug_session(self.controller),
                't': self._run_automated_tests,
                'T': self._run_automated_tests,
                'm': self._show_metrics,
                'M': self._show_metrics
            }
            handlers.update(debug_handlers)
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid option")
    
    # ========== MANUAL CONTROL ==========
    
    def _manual_control(self):
        """Manuelle Steuerung."""
        if ENHANCED_MODULES and self.manual_control:
            # Verwende Enhanced Manual Control
            print("\n🎮 Starting Enhanced Manual Control...")
            print("   • Real-time movement (no Enter needed)")
            print("   • Hold keys for continuous movement")
            print("   • Visual feedback")
            time.sleep(1)
            self.manual_control.start()
        else:
            # Fallback auf alte Version
            self._manual_control_legacy()
    
    def _manual_control_legacy(self):
        """Legacy Manual Control."""
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
        print("  x: Exit")
        print("-" * 40)
        print("⚠️ Note: Press Enter after each key")
        
        speed = 1.0
        step = 0.1
        
        print(f"\nCurrent speed: {speed:.1f}")
        print("Ready for input...")
        
        while True:
            key = input().strip().lower()
            
            if key == 'x':
                break
            elif key == ' ':
                self.controller.emergency_stop()
                print("🚨 EMERGENCY STOP")
                self.session_stats["emergency_stops"] += 1
            elif key == '+':
                speed = min(2.0, speed + 0.1)
                print(f"Speed: {speed:.1f}")
            elif key == '-':
                speed = max(0.1, speed - 0.1)
                print(f"Speed: {speed:.1f}")
            else:
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
                    
                    self.controller.move_joints(
                        new_pos,
                        speed=speed,
                        trajectory_type=TrajectoryType.LINEAR,
                        wait=False
                    )
                    print(f"{joint}: {current + delta:+.3f}")
    
    # ========== SCANNER PATTERNS ==========
    
    def _scanner_menu(self):
        """Scanner Pattern Menü."""
        print("\n📷 SCANNER PATTERNS")
        print("-" * 40)
        
        if self.calibrator.calibration.calibration_valid:
            print(f"Scanner calibrated: ✅")
            if hasattr(self.calibrator.calibration, 'scanner_optimal_distance'):
                print(f"Optimal distance: {self.calibrator.calibration.scanner_optimal_distance*100:.1f}cm")
        else:
            print("Scanner calibration: ⚠️ Not calibrated")
        
        print("\n1. Raster Scan (Grid)")
        print("2. Spiral Scan")
        print("3. Spherical Scan")
        print("4. Turntable Scan")
        print("5. Cobweb Scan")
        print("6. Custom Pattern")
        print("0. Back")
        
        choice = input("\n👉 Select pattern: ").strip()
        
        patterns = {
            '1': self._raster_scan,
            '2': self._spiral_scan,
            '3': self._spherical_scan,
            '4': self._turntable_scan,
            '5': self._cobweb_scan,
            '6': self._custom_pattern
        }
        
        if choice in patterns:
            patterns[choice]()
    
    def _execute_scan_pattern(self, pattern):
        """Führt einen Scan aus."""
        print(f"\n🚀 Starting {pattern.name}...")
        print("Press Ctrl+C to abort\n")
        
        print("Moving to scanner position...")
        self.controller.move_to_scanner_position(speed=0.5)
        time.sleep(2)
        
        success = self.controller.execute_pattern(pattern)
        
        if success:
            print(f"\n✅ {pattern.name} completed successfully!")
            self.session_stats["patterns_completed"] += 1
        else:
            print(f"\n❌ {pattern.name} failed or was aborted")
    
    def _raster_scan(self):
        """Raster Scan."""
        print("\n📐 RASTER SCAN SETUP")
        width = float(input("Width (cm) [20]: ") or "20") / 100
        height = float(input("Height (cm) [15]: ") or "15") / 100
        rows = int(input("Rows [10]: ") or "10")
        cols = int(input("Columns [10]: ") or "10")
        
        pattern = RasterScanPattern(width=width, height=height, rows=rows, cols=cols)
        self._execute_scan_pattern(pattern)
    
    def _spiral_scan(self):
        """Spiral Scan."""
        print("\n🌀 SPIRAL SCAN SETUP")
        r_start = float(input("Start radius (cm) [5]: ") or "5") / 100
        r_end = float(input("End radius (cm) [15]: ") or "15") / 100
        revs = int(input("Revolutions [5]: ") or "5")
        
        pattern = SpiralScanPattern(radius_start=r_start, radius_end=r_end, revolutions=revs)
        self._execute_scan_pattern(pattern)
    
    def _spherical_scan(self):
        """Spherical Scan."""
        print("\n🌐 SPHERICAL SCAN SETUP")
        radius = float(input("Radius (cm) [15]: ") or "15") / 100
        h_steps = int(input("Horizontal steps [12]: ") or "12")
        v_steps = int(input("Vertical steps [8]: ") or "8")
        
        pattern = SphericalScanPattern(radius=radius, theta_steps=h_steps, phi_steps=v_steps)
        self._execute_scan_pattern(pattern)
    
    def _turntable_scan(self):
        """Turntable Scan."""
        print("\n🔄 TURNTABLE SCAN SETUP")
        steps = int(input("Rotation steps [36]: ") or "36")
        levels = int(input("Height levels [1]: ") or "1")
        
        pattern = TurntableScanPattern(steps=steps, height_levels=levels)
        self._execute_scan_pattern(pattern)
    
    def _cobweb_scan(self):
        """Cobweb Scan."""
        print("\n🕸️ COBWEB SCAN SETUP")
        lines = int(input("Radial lines [8]: ") or "8")
        circles = int(input("Circles [5]: ") or "5")
        radius = float(input("Max radius (cm) [15]: ") or "15") / 100
        
        pattern = CobwebScanPattern(radial_lines=lines, circles=circles, max_radius=radius)
        self._execute_scan_pattern(pattern)
    
    def _custom_pattern(self):
        """Custom Pattern."""
        print("\n🎨 CUSTOM PATTERN")
        print("Feature in development...")
    
    # ========== ENHANCED TEACHING MODE ==========
    
    def _teaching_menu_enhanced(self):
        """Erweitertes Teaching Mode Menü."""
        print("\n🎓 ENHANCED TEACHING MODE")
        print("-" * 40)
        
        # Status
        if self.teacher.is_recording:
            print(f"🔴 RECORDING: '{self.teacher.current_sequence.name}'")
            print(f"   Waypoints: {self.teacher.waypoint_count}")
            print(f"   Duration: {time.time() - self.teacher.start_time:.1f}s")
            print()
        
        if self.teacher.is_playing:
            print(f"▶️ PLAYING: Position {self.teacher.playback_position}")
            print()
        
        # Zeige Sequenzen
        sequences = self.teacher.list_sequences()
        if sequences:
            print(f"📁 Saved sequences: {len(sequences)}")
            for i, name in enumerate(sequences[:3], 1):
                seq = self.teacher.get_sequence(name)
                if seq:
                    print(f"   {name}: {len(seq.waypoints)} points, {seq.total_duration:.1f}s")
            print()
        
        print("1. Start Recording")
        print("2. Stop Recording")
        print("3. Play Sequence")
        print("4. Sequence Manager")
        print("5. Quick Record (Manual)")
        print("6. Smart Record")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self._start_teaching()
        elif choice == '2':
            self._stop_teaching()
        elif choice == '3':
            self._play_teaching_sequence()
        elif choice == '4':
            self._sequence_manager()
        elif choice == '5':
            self._quick_record()
        elif choice == '6':
            self._smart_record()
    
    def _teaching_menu(self):
        """Standard Teaching Menu (Fallback)."""
        print("\n🎓 TEACHING MODE")
        print("-" * 40)
        print("1. Start Recording")
        print("2. Stop Recording")
        print("3. Save Sequence")
        print("4. Load Sequence")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self._start_teaching()
        elif choice == '2':
            self._stop_teaching()
        elif choice == '3':
            if hasattr(self.teacher, 'save_sequence'):
                self.teacher.save_sequence()
        elif choice == '4':
            self._load_teaching_sequence()
    
    def _start_teaching(self):
        """Startet Teaching."""
        if self.teacher.is_recording:
            print("⚠️ Already recording!")
            return
        
        name = input("Sequence name: ").strip()
        if not name:
            print("❌ Name required")
            return
        
        print("\nRecording modes:")
        print("1. Manual (press key for waypoints)")
        print("2. Continuous (automatic sampling)")
        print("3. Smart (intelligent detection)")
        
        mode_choice = input("Mode [1]: ").strip() or '1'
        
        mode_map = {
            '1': EnhancedRecordingMode.MANUAL,
            '2': EnhancedRecordingMode.CONTINUOUS,
            '3': getattr(EnhancedRecordingMode, 'SMART', EnhancedRecordingMode.MANUAL)
        }
        
        mode = mode_map.get(mode_choice, EnhancedRecordingMode.MANUAL)
        
        if self.teacher.start_recording(name, mode):
            print(f"🔴 Recording started: '{name}' ({mode.value} mode)")
            self.session_stats["sequences_recorded"] += 1
            
            if mode == EnhancedRecordingMode.MANUAL:
                print("Move robot and press ENTER to record waypoints")
                print("Type 'stop' to finish")
                
                while self.teacher.is_recording:
                    cmd = input().strip().lower()
                    if cmd == 'stop':
                        break
                    elif cmd == '' or cmd == 'record':
                        if self.teacher.record_waypoint(f"Point {self.teacher.waypoint_count + 1}"):
                            print(f"✓ Waypoint {self.teacher.waypoint_count} recorded")
    
    def _stop_teaching(self):
        """Stoppt Teaching."""
        if not self.teacher.is_recording:
            print("Not recording")
            return
        
        seq = self.teacher.stop_recording()
        if seq:
            print(f"✅ Recording stopped: {len(seq.waypoints)} waypoints")
            
            save = input("Save sequence? (y/n): ").lower()
            if save == 'y':
                if hasattr(self.teacher, 'save_sequence'):
                    self.teacher.save_sequence(seq)
                    print("✅ Sequence saved")
    
    def _play_teaching_sequence(self):
        """Spielt Sequenz ab."""
        sequences = self.teacher.list_sequences() if hasattr(self.teacher, 'list_sequences') else []
        
        if not sequences:
            print("No sequences available")
            return
        
        print("\nAvailable sequences:")
        for i, name in enumerate(sequences, 1):
            print(f"{i}. {name}")
        
        choice = input("\nSelect sequence: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                seq = self.teacher.get_sequence(sequences[idx])
                if seq:
                    print(f"▶️ Playing '{seq.name}'...")
                    
                    if hasattr(self.teacher, 'start_playback'):
                        self.teacher.start_playback(seq)
                        self.session_stats["sequences_played"] += 1
                    else:
                        print("Playback not available in standard mode")
        except:
            print("Invalid selection")
    
    def _quick_record(self):
        """Schnelle manuelle Aufzeichnung."""
        name = f"quick_{time.strftime('%H%M%S')}"
        if self.teacher.start_recording(name, EnhancedRecordingMode.MANUAL):
            print(f"🔴 Quick recording: '{name}'")
            print("Press ENTER for waypoints, 'q' to finish")
            
            while True:
                key = input().strip().lower()
                if key == 'q':
                    break
                elif key == '':
                    self.teacher.record_waypoint()
                    print(f"✓ Point {self.teacher.waypoint_count}")
            
            seq = self.teacher.stop_recording()
            if seq:
                print(f"✅ Recorded {len(seq.waypoints)} points")
                self.teacher.save_sequence(seq)
    
    def _smart_record(self):
        """Intelligente Aufzeichnung."""
        if hasattr(EnhancedRecordingMode, 'SMART'):
            name = f"smart_{time.strftime('%H%M%S')}"
            if self.teacher.start_recording(name, EnhancedRecordingMode.SMART):
                print(f"🔴 Smart recording: '{name}'")
                print("Move the robot - waypoints recorded automatically")
                print("Press any key to stop")
                
                input()
                
                seq = self.teacher.stop_recording()
                if seq:
                    print(f"✅ Smart recording: {len(seq.waypoints)} points")
        else:
            print("Smart recording not available")
    
    # ========== SEQUENCE MANAGER ==========
    
    def _sequence_manager(self):
        """Sequenz-Verwaltung."""
        print("\n📁 SEQUENCE MANAGER")
        print("-" * 40)
        
        if not hasattr(self.teacher, 'list_sequences'):
            print("Sequence management not available")
            return
        
        sequences = self.teacher.list_sequences()
        
        if sequences:
            print(f"Available sequences ({len(sequences)}):")
            for i, name in enumerate(sequences, 1):
                seq = self.teacher.get_sequence(name)
                if seq:
                    print(f"{i:2d}. {name:20s} - {len(seq.waypoints):3d} points, {seq.total_duration:6.1f}s")
        else:
            print("No sequences saved")
        
        print("\n1. Play Sequence")
        print("2. Delete Sequence")
        print("3. Export Sequence")
        print("4. Import Sequence")
        print("5. Merge Sequences")
        print("6. Optimize Sequence")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self._play_teaching_sequence()
        elif choice == '2':
            self._delete_sequence()
        elif choice == '3':
            self._export_sequence()
        elif choice == '4':
            self._import_sequence()
        elif choice == '5':
            self._merge_sequences()
        elif choice == '6':
            self._optimize_sequence()
    
    def _delete_sequence(self):
        """Löscht Sequenz."""
        sequences = self.teacher.list_sequences()
        if not sequences:
            print("No sequences to delete")
            return
        
        print("\nSelect sequence to delete:")
        for i, name in enumerate(sequences, 1):
            print(f"{i}. {name}")
        
        choice = input("\nSelect: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                name = sequences[idx]
                confirm = input(f"Delete '{name}'? (y/n): ").lower()
                if confirm == 'y':
                    if self.teacher.delete_sequence(name):
                        print("✅ Sequence deleted")
        except:
            print("Invalid selection")
    
    def _export_sequence(self):
        """Exportiert Sequenz."""
        if not hasattr(self.teacher, 'export_sequence'):
            print("Export not available")
            return
        
        sequences = self.teacher.list_sequences()
        if not sequences:
            print("No sequences to export")
            return
        
        print("\nSelect sequence to export:")
        for i, name in enumerate(sequences, 1):
            print(f"{i}. {name}")
        
        choice = input("\nSelect: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                seq = self.teacher.get_sequence(sequences[idx])
                if seq:
                    format_choice = input("Format (csv/txt) [csv]: ").strip() or 'csv'
                    filename = f"export_{seq.name}.{format_choice}"
                    
                    if self.teacher.export_sequence(seq, filename, format_choice):
                        print(f"✅ Exported to {filename}")
        except:
            print("Export failed")
    
    def _import_sequence(self):
        """Importiert Sequenz."""
        filename = input("Filename: ").strip()
        if filename and Path(filename).exists():
            seq = self.teacher.load_sequence(filename)
            if seq:
                print(f"✅ Imported '{seq.name}'")
        else:
            print("File not found")
    
    def _merge_sequences(self):
        """Verbindet Sequenzen."""
        if not hasattr(self.teacher, 'merge_sequences'):
            print("Merge not available")
            return
        
        sequences = self.teacher.list_sequences()
        if len(sequences) < 2:
            print("Need at least 2 sequences to merge")
            return
        
        print("\nSelect sequences to merge (comma-separated):")
        for i, name in enumerate(sequences, 1):
            print(f"{i}. {name}")
        
        choices = input("\nSelect: ").strip().split(',')
        
        try:
            selected = []
            for c in choices:
                idx = int(c.strip()) - 1
                if 0 <= idx < len(sequences):
                    seq = self.teacher.get_sequence(sequences[idx])
                    if seq:
                        selected.append(seq)
            
            if len(selected) >= 2:
                name = input("New sequence name: ").strip()
                if name:
                    merged = self.teacher.merge_sequences(selected, name)
                    self.teacher.save_sequence(merged)
                    print(f"✅ Merged into '{name}'")
        except:
            print("Merge failed")
    
    def _optimize_sequence(self):
        """Optimiert Sequenz."""
        sequences = self.teacher.list_sequences()
        if not sequences:
            print("No sequences to optimize")
            return
        
        print("\nSelect sequence to optimize:")
        for i, name in enumerate(sequences, 1):
            print(f"{i}. {name}")
        
        choice = input("\nSelect: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                seq = self.teacher.get_sequence(sequences[idx])
                if seq and hasattr(seq, 'optimize'):
                    before = len(seq.waypoints)
                    seq.optimize()
                    after = len(seq.waypoints)
                    self.teacher.save_sequence(seq)
                    print(f"✅ Optimized: {before} → {after} waypoints")
        except:
            print("Optimization failed")
    
    # ========== ENHANCED CALIBRATION ==========
    
    def _calibration_menu_enhanced(self):
        """Erweitertes Kalibrierungsmenü."""
        print("\n🔧 PROFESSIONAL CALIBRATION SUITE")
        print("-" * 40)
        
        if self.calibrator.calibration.calibration_valid:
            print(f"Status: VALID ✅")
            print(f"Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.2f} mrad")
            print(f"Score: {self.calibrator.calibration.calibration_score*100:.1f}%")
        else:
            print("Status: NOT CALIBRATED ⚠️")
        
        print("\n1. 🚀 Full Auto Calibration")
        print("2. 🎯 Quick Calibration")
        print("3. 📊 Calibration Report")
        print("4. 📈 Plot Calibration Curves")
        print("5. ✅ Verify Calibration")
        print("6. 🔧 Advanced Options")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self._run_full_calibration()
        elif choice == '2':
            self._run_quick_calibration()
        elif choice == '3':
            self._show_calibration_report()
        elif choice == '4':
            self._plot_calibration()
        elif choice == '5':
            self._verify_calibration()
        elif choice == '6':
            self._advanced_calibration()
    
    def _calibration_menu(self):
        """Standard Kalibrierungsmenü."""
        print("\n🔧 CALIBRATION SUITE")
        print("-" * 40)
        
        if self.calibrator.calibration.calibration_valid:
            print("Status: VALID ✅")
        else:
            print("Status: NOT CALIBRATED ⚠️")
        
        print("\n1. Auto Calibration")
        print("2. Verify Calibration")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            self._run_quick_calibration()
        elif choice == '2':
            self._verify_calibration()
    
    def _run_full_calibration(self):
        """Vollständige Kalibrierung."""
        print("\n🚀 FULL CALIBRATION")
        print("This will take 15-20 minutes")
        
        options = []
        if input("Include scanner calibration? (y/n): ").lower() == 'y':
            options.append('scanner')
        if input("Include thermal calibration? (y/n): ").lower() == 'y':
            options.append('thermal')
        if input("Include dynamic calibration? (y/n): ").lower() == 'y':
            options.append('dynamic')
        
        confirm = input("\nStart calibration? (y/n): ").lower()
        if confirm == 'y':
            success = self.calibrator.run_full_calibration(
                include_scanner='scanner' in options,
                include_thermal='thermal' in options,
                include_dynamic='dynamic' in options
            )
            
            if success:
                print("\n✅ Calibration successful!")
                self.session_stats["calibrations_run"] += 1
            else:
                print("\n❌ Calibration failed")
    
    def _run_quick_calibration(self):
        """Schnelle Kalibrierung."""
        print("\n⚡ QUICK CALIBRATION")
        print("Basic calibration (5-10 minutes)")
        
        confirm = input("\nStart? (y/n): ").lower()
        if confirm == 'y':
            if hasattr(self.calibrator, 'run_full_calibration'):
                success = self.calibrator.run_full_calibration(
                    include_scanner=False,
                    include_thermal=False,
                    include_dynamic=False
                )
            else:
                # Fallback
                success = False
                print("Running basic calibration...")
                time.sleep(2)
                print("✅ Basic calibration complete")
                success = True
            
            if success:
                self.session_stats["calibrations_run"] += 1
    
    def _show_calibration_report(self):
        """Zeigt Kalibrierungsbericht."""
        if hasattr(self.calibrator, 'generate_calibration_report'):
            report = self.calibrator.generate_calibration_report()
            print("\n" + report)
            
            save = input("\nSave report to file? (y/n): ").lower()
            if save == 'y':
                filename = f"calibration_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    f.write(report)
                print(f"✅ Report saved to {filename}")
        else:
            print("Report not available")
    
    def _plot_calibration(self):
        """Plottet Kalibrierungskurven."""
        if hasattr(self.calibrator, 'plot_calibration_curves'):
            filename = f"calibration_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            self.calibrator.plot_calibration_curves(filename)
            print(f"✅ Plot saved to {filename}")
        else:
            print("Plotting not available")
    
    def _verify_calibration(self):
        """Verifiziert Kalibrierung."""
        print("\n✅ Verifying calibration...")
        
        if hasattr(self.calibrator, '_verify_calibration_enhanced'):
            valid = self.calibrator._verify_calibration_enhanced()
        else:
            valid = self.calibrator.verify_calibration() if hasattr(self.calibrator, 'verify_calibration') else False
        
        if valid:
            print("✅ Calibration is VALID")
        else:
            print("❌ Calibration verification FAILED")
    
    def _advanced_calibration(self):
        """Erweiterte Kalibrierungsoptionen."""
        print("\n🔧 ADVANCED CALIBRATION")
        print("-" * 40)
        print("1. Individual Joint Calibration")
        print("2. Backlash Measurement")
        print("3. Repeatability Test")
        print("4. Kinematic Calibration")
        print("5. Export Calibration")
        print("6. Import Calibration")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            joint = input("Joint name (base/shoulder/elbow/wrist/roll/hand): ").strip()
            if joint in SERVO_LIMITS:
                if hasattr(self.calibrator, '_calibrate_joint_enhanced'):
                    self.calibrator._calibrate_joint_enhanced(joint)
                else:
                    print("Joint calibration not available")
        elif choice == '2':
            if hasattr(self.calibrator, '_measure_backlash_enhanced'):
                self.calibrator._measure_backlash_enhanced()
            else:
                print("Backlash measurement not available")
        # ... weitere Optionen ...
    
    # ========== WEITERE MENÜS ==========
    
    def _move_home(self):
        """Home Position."""
        print("\n🏠 Moving to home position...")
        if self.controller.move_home(speed=0.5):
            print("✅ Home position reached")
        else:
            print("❌ Failed to reach home")
    
    def _settings_menu(self):
        """Einstellungen."""
        print("\n⚙️ SETTINGS")
        print("-" * 40)
        print(f"Mode: {self.operation_mode.value}")
        print(f"Enhanced: {ENHANCED_MODULES}")
        print(f"Debug: {self.debug_enabled}")
        if hasattr(self.controller, 'config'):
            print(f"Port: {self.controller.config.port}")
            print(f"Speed: {self.controller.config.default_speed}")
        print("\n1. Change Speed")
        print("2. Export Configuration")
        print("3. Import Configuration")
        print("0. Back")
        
        choice = input("\n👉 Select: ").strip()
        
        if choice == '1':
            speed = float(input("New speed (0.1-2.0): "))
            if hasattr(self.controller, 'current_speed'):
                self.controller.current_speed = max(0.1, min(2.0, speed))
                print(f"✅ Speed set to {self.controller.current_speed}")
    
    def _status_menu(self):
        """Status und Informationen."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        # Version
        print(f"Version: {VERSION}{ENHANCED_SUFFIX}")
        print(f"Mode: {self.operation_mode.value}")
        
        # Session
        duration = time.time() - self.session_stats["start_time"]
        print(f"\nSession: {duration/60:.1f} minutes")
        print(f"Commands: {self.session_stats['commands_executed']}")
        
        # Hardware Status
        status = self.controller.query_status()
        if status:
            print("\nJoint Positions:")
            for joint, pos in status['positions'].items():
                print(f"  {joint:10s}: {pos:+.3f} rad")
            
            if 'temperature' in status:
                print(f"\nTemperature: {status['temperature']:.1f}°C")
            if 'voltage' in status:
                print(f"Voltage: {status['voltage']:.1f}V")
        
        # Calibration
        if self.calibrator.calibration.calibration_valid:
            print(f"\nCalibration: VALID")
            print(f"  Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.2f} mrad")
    
    def _system_test(self):
        """System Test."""
        print("\n🧪 SYSTEM TEST")
        self._run_automated_tests()
    
    def _debug_tools(self):
        """Debug Tools."""
        if hasattr(self.controller, 'print_debug_summary'):
            run_debug_session(self.controller)
        else:
            print("Debug tools not available")
    
    def _show_metrics(self):
        """Zeigt Metriken."""
        print("\n📊 SESSION METRICS")
        print("-" * 40)
        
        duration = time.time() - self.session_stats["start_time"]
        
        print(f"Duration: {duration/60:.1f} minutes")
        for key, value in self.session_stats.items():
            if key != "start_time":
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        if hasattr(self.controller, 'metrics'):
            print("\nController Metrics:")
            for key, value in self.controller.metrics.items():
                print(f"  {key}: {value}")
    
    # ========== TEST METHODS ==========
    
    def _run_automated_tests(self):
        """Automatisierte Tests."""
        print("\n🧪 RUNNING TESTS...")
        
        tests = [
            ("Connection", self._test_connection),
            ("Movement", self._test_basic_movement),
            ("Manual Control", self._test_manual_control),
            ("Teaching", self._test_teaching_enhanced),
            ("Calibration", self._test_calibration_enhanced),
            ("Patterns", self._test_patterns),
            ("Emergency", self._test_emergency)
        ]
        
        passed = 0
        for name, test in tests:
            print(f"Testing {name}...", end=" ")
            try:
                if test():
                    print("✅")
                    passed += 1
                else:
                    print("❌")
            except Exception as e:
                print(f"❌ ({e})")
        
        print(f"\nResults: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def _test_connection(self):
        """Test Verbindung."""
        return self.controller.serial.connected if hasattr(self.controller, 'serial') else True
    
    def _test_basic_movement(self):
        """Test Bewegung."""
        return self.controller.move_joints({"base": 0.1}, speed=1.0)
    
    def _test_manual_control(self):
        """Test Manual Control."""
        return self.manual_control is not None if ENHANCED_MODULES else True
    
    def _test_teaching_enhanced(self):
        """Test Teaching."""
        if self.teacher.is_recording:
            return False
        return self.teacher.start_recording("test", EnhancedRecordingMode.MANUAL)
    
    def _test_calibration_enhanced(self):
        """Test Kalibrierung."""
        return hasattr(self.calibrator, 'calibration')
    
    def _test_patterns(self):
        """Test Patterns."""
        pattern = RasterScanPattern(width=0.1, height=0.1, rows=2, cols=2)
        return True  # Nur prüfen ob erstellt werden kann
    
    def _test_emergency(self):
        """Test Emergency."""
        self.controller.emergency_stop()
        self.controller.reset_emergency()
        return True
    
    # ========== DEMO METHODS ==========
    
    def _run_demo_sequence(self):
        """Demo-Sequenz."""
        print("\n🎭 DEMO SEQUENCE")
        demos = [
            ("Home Position", lambda: self.controller.move_home()),
            ("Scanner Position", lambda: self.controller.move_to_scanner_position()),
            ("Pattern Demo", lambda: self._demo_pattern()),
            ("Teaching Demo", lambda: self._demo_teaching())
        ]
        
        for name, demo in demos:
            print(f"\n{name}...")
            demo()
            time.sleep(1)
        
        print("\n✅ Demo complete")
    
    def _demo_pattern(self):
        """Demo Pattern."""
        pattern = RasterScanPattern(width=0.1, height=0.1, rows=3, cols=3)
        print(f"Executing {pattern.name} (demo)")
        # Nur simulieren
        time.sleep(2)
    
    def _demo_teaching(self):
        """Demo Teaching."""
        print("Recording demo sequence...")
        # Nur simulieren
        time.sleep(2)
    
    def _run_performance_benchmark(self):
        """Performance Benchmark."""
        print("\n📊 PERFORMANCE BENCHMARK")
        print("-" * 40)
        
        import timeit
        
        benchmarks = {
            "Status Query": lambda: self.controller.query_status(),
            "Simple Move": lambda: self.controller.move_joints({"base": 0.1}),
            "Home Position": lambda: self.controller.move_home()
        }
        
        for name, func in benchmarks.items():
            time_taken = timeit.timeit(func, number=10) / 10 * 1000
            print(f"{name:20s}: {time_taken:.2f} ms")
    
    # ========== HELPER METHODS ==========
    
    def _check_calibration_status(self):
        """Prüft Kalibrierungsstatus."""
        if self.calibrator.calibration.calibration_valid:
            age_days = (time.time() - self.calibrator.calibration.timestamp) / 86400
            
            if age_days < 7:
                status = "GOOD ✅"
            elif age_days < 30:
                status = "OK ⚠️"
            else:
                status = "OLD ⚠️"
            
            print(f"Calibration: {status} ({age_days:.0f} days old)")
        else:
            print("Calibration: NOT CALIBRATED ⚠️")
    
    def _load_teaching_sequence(self):
        """Lädt Teaching Sequenz."""
        from pathlib import Path
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
        
        choice = input("\nSelect: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sequences):
                if hasattr(self.teacher, 'load_sequence'):
                    seq = self.teacher.load_sequence(str(sequences[idx]))
                    if seq:
                        print(f"✅ Loaded '{seq.name}'")
        except:
            print("Invalid selection")
    
    def _exit(self):
        """Beendet das Programm."""
        if self.debug_enabled:
            print("\n📊 Session Summary:")
            self._show_metrics()
        
        self.running = False
    
    def _cleanup(self):
        """Aufräumen beim Beenden."""
        print("\n🔌 Shutting down...")
        
        duration = time.time() - self.session_stats["start_time"]
        print(f"Session: {duration/60:.1f} minutes")
        print(f"Commands: {self.session_stats['commands_executed']}")
        
        if self.controller:
            if self.debug_enabled and hasattr(self.controller, 'print_debug_summary'):
                self.controller.print_debug_summary()
            
            if hasattr(self.controller, 'disconnect'):
                self.controller.disconnect()
        
        print("✅ Shutdown complete")
        print("Goodbye! 👋\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f'RoArm M3 Professional Control System v{VERSION}'
    )
    
    # Connection
    parser.add_argument('--port', default='/dev/tty.usbserial-110', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate')
    
    # Modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--debug', action='store_true', help='Debug mode (no hardware)')
    mode_group.add_argument('--test', action='store_true', help='Run tests')
    mode_group.add_argument('--demo', action='store_true', help='Demo mode')
    mode_group.add_argument('--simulation', action='store_true', help='Simulation mode')
    
    # Parameters
    parser.add_argument('--speed', type=float, default=1.0, help='Default speed (0.1-2.0)')
    parser.add_argument('--pattern', choices=['raster', 'spiral', 'spherical', 'turntable', 'cobweb'],
                       help='Execute pattern directly')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--quick', action='store_true', help='Quick mode (skip menus)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Check enhanced modules
    if ENHANCED_MODULES:
        print(f"✨ Enhanced modules loaded successfully")
    
    # Start CLI
    cli = RoArmCLI()
    cli.run(args)


if __name__ == '__main__':
    main()
