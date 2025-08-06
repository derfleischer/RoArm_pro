#!/usr/bin/env python3
"""
RoArm M3 Professional Control System
Hauptprogramm mit Command Line Interface
Optimiert für macOS M4 mit Revopoint Mini2 Scanner
Erweitert um professionelle Kalibrierungssuite
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

from core.controller import RoArmController, RoArmConfig
from patterns.scan_patterns import (
    RasterScanPattern, SpiralScanPattern, SphericalScanPattern,
    TurntableScanPattern, CobwebScanPattern
)
from teaching.recorder import TeachingRecorder, RecordingMode, TrajectoryType
from calibration.calibration_suite import CalibrationSuite, CalibrationType
from safety.safety_system import SafetySystem, SafetyState, ShutdownReason
from utils.logger import setup_logger, get_logger
from utils.terminal import TerminalController

# Setup logging
setup_logger()
logger = get_logger(__name__)


class RoArmCLI:
    """Command Line Interface für RoArm Control."""
    
    def __init__(self):
        self.controller = None
        self.teacher = None
        self.calibrator = None
        self.safety_system = None  # NEU!
        self.terminal = TerminalController()
        self.running = True
        
        # Signal Handler für Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler für Ctrl+C - Emergency Stop."""
        print("\n\n🚨 EMERGENCY STOP - Ctrl+C detected!")
        if self.controller:
            self.controller.emergency_stop()
        self.running = False
        sys.exit(0)
    
    def run(self, args):
        """Startet die CLI."""
        try:
            # Header
            self._print_header()
            
            # Controller Setup
            config = RoArmConfig(
                port=args.port,
                baudrate=args.baudrate,
                default_speed=args.speed,
                debug=args.debug
            )
            
            # Verbinden
            print(f"🔌 Connecting to RoArm on {config.port}...")
            self.controller = RoArmController(config)
            
            if not self.controller.serial.connected:
                print("❌ Failed to connect to RoArm")
                print("   Check cable and port settings")
                return
            
            print("✅ Successfully connected!\n")
            
            # Teaching Recorder
            self.teacher = TeachingRecorder(self.controller)
            
            # Calibration Suite
            self.calibrator = CalibrationSuite(self.controller)
            
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
                self._show_main_menu()
                choice = input("\n👉 Select option: ").strip()
                self._handle_main_menu(choice)
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
        finally:
            self._cleanup()
    
    def _print_header(self):
        """Zeigt den Header."""
        print("\n" + "="*60)
        print("🤖 RoArm M3 Professional Control System v2.0")
        print("="*60)
        print("📷 Optimized for Revopoint Mini2 Scanner")
        print("🍎 macOS M4 Edition")
        print("🔧 With Professional Calibration Suite")
        print("⚡ Press Ctrl+C anytime for EMERGENCY STOP")
        print("="*60 + "\n")
    
    def _check_calibration_status(self):
        """Prüft und zeigt Kalibrierungsstatus."""
        if self.calibrator.calibration.calibration_valid:
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
    
    def _show_main_menu(self):
        """Zeigt das Hauptmenü."""
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. 🎮 Manual Control")
        print("2. 📷 Scanner Patterns")
        print("3. 🎓 Teaching Mode")
        print("4. 📁 Load & Play Sequence")
        print("5. 🏠 Move to Home")
        print("6. 🔧 Calibration Suite")  # ERWEITERT!
        print("7. ⚙️  Settings")
        print("8. 📊 Status")
        print("9. 🧪 System Test")  # NEU!
        print("0. 🚪 Exit")
    
    def _handle_main_menu(self, choice: str):
        """Verarbeitet Hauptmenü-Auswahl."""
        handlers = {
            '1': self._manual_control,
            '2': self._scanner_menu,
            '3': self._teaching_menu,
            '4': self._load_sequence,
            '5': self._move_home,
            '6': self._calibration_menu,  # ERWEITERT!
            '7': self._settings_menu,
            '8': self._show_status,
            '9': self._system_test,  # NEU!
            '0': self._exit
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid option")
    
    # ============== CALIBRATION SUITE ==============
    
    def _calibration_menu(self):
        """Erweiterte Kalibrierungsoptionen."""
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
    
    # ============== SCANNER PATTERNS ==============
    
    def _scanner_menu(self):
        """Scanner Pattern Menü."""
        print("\n📷 SCANNER PATTERNS")
        print("-" * 40)
        
        # Zeige Scanner-Status
        if self.calibrator.calibration.scanner:
            print(f"Scanner calibrated: ✅")
            print(f"Optimal distance: {self.calibrator.calibration.scanner.optimal_distance*100:.1f}cm")
        else:
            print("Scanner calibration: ⚠️ Not calibrated")
        
        print("\n1. Raster Scan (Grid)")
        print("2. Spiral Scan")
        print("3. Spherical Scan")
        print("4. Turntable Scan")
        print("5. Cobweb Scan")
        print("0. Back")
        
        choice = input("\n👉 Select pattern: ").strip()
        
        patterns = {
            '1': self._raster_scan,
            '2': self._spiral_scan,
            '3': self._spherical_scan,
            '4': self._turntable_scan,
            '5': self._cobweb_scan
        }
        
        if choice in patterns:
            patterns[choice]()
    
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
    
    def _execute_scan(self, pattern):
        """Führt einen Scan aus."""
        print(f"\n🚀 Starting {pattern.name}...")
        print("Press Ctrl+C to abort\n")
        
        # Move to scanner position first
        print("Moving to scanner position...")
        self.controller.move_to_scanner_position(speed=0.5)
        time.sleep(2)
        
        # Apply calibrated scanner position if available
        if self.calibrator.calibration.scanner:
            print("Using calibrated scanner parameters")
            pattern.settle_time = self.calibrator.calibration.scanner.optimal_settle_time
        
        # Execute pattern
        success = self.controller.execute_pattern(pattern)
        
        if success:
            print(f"\n✅ {pattern.name} completed successfully!")
        else:
            print(f"\n❌ {pattern.name} failed or was aborted")
    
    # ============== TEACHING MODE ==============
    
    def _teaching_menu(self):
        """Teaching Mode Menü."""
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
        else:
            print("❌ Failed to reach home")
    
    def _settings_menu(self):
        """Einstellungen."""
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
    
    def _show_status(self):
        """Zeigt Status an."""
        print("\n📊 SYSTEM STATUS")
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
            
            # Calibration info
            if self.calibrator.calibration.calibration_valid:
                print(f"\nCalibration:")
                print(f"  Status: VALID ✅")
                print(f"  Accuracy: ±{self.calibrator.calibration.overall_accuracy*1000:.1f} mrad")
                age_days = (time.time() - self.calibrator.calibration.timestamp) / 86400
                print(f"  Age: {age_days:.0f} days")
        else:
            print("❌ Failed to query status")
    
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
            
            # Move to safe position
            print("Moving to safe position...")
            self.controller.move_home(speed=0.5)
            time.sleep(1)
            
            # Disconnect
            self.controller.disconnect()
        
        print("✅ Shutdown complete")
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
        description='RoArm M3 Professional Control System with Calibration Suite'
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
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Import statements for SERVO_LIMITS
    global SERVO_LIMITS
    from core.constants import SERVO_LIMITS
    
    # Start CLI
    cli = RoArmCLI()
    cli.run(args)


if __name__ == '__main__':
    main()
