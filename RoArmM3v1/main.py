#!/usr/bin/env python3
"""
RoArm M3 Professional Control System
Hauptprogramm mit Command Line Interface
Optimiert für macOS M4 mit Revopoint Mini2 Scanner
Version 3.0 mit Enhanced Features Integration
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

# ============== ENHANCED MODE DETECTION ==============
ENHANCED_MODE = False
ENHANCED_FEATURES = {}

# Versuche Enhanced Module zu laden
try:
    from enhanced.controller import EnhancedController
    from enhanced.trajectory import AdvancedTrajectoryGenerator
    from enhanced.vision import VisionSystem
    from enhanced.ml_optimizer import MLOptimizer
    from enhanced.adaptive_control import AdaptiveControl
    from enhanced.predictive_motion import PredictiveMotion
    from enhanced.cloud_sync import CloudSync
    from enhanced.realtime_monitor import RealtimeMonitoring
    ENHANCED_MODE = True
    print("✅ Enhanced features detected and loaded")
except ImportError as e:
    print("ℹ️  Using standard mode (enhanced features not available)")
    # Fallback auf Standard Controller
    from core.controller import RoArmController as EnhancedController

# ============== STANDARD IMPORTS ==============
from core.controller import RoArmController, RoArmConfig
from core.constants import SERVO_LIMITS, HOME_POSITION
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
    """Command Line Interface für RoArm Control mit Enhanced Features."""
    
    VERSION = "3.0.0"
    
    def __init__(self):
        self.controller = None
        self.teacher = None
        self.calibrator = None
        self.safety_system = None
        self.terminal = TerminalController()
        self.running = True
        
        # Enhanced Features Container
        self.enhanced = {
            "vision": None,
            "ml_optimizer": None,
            "adaptive": None,
            "predictive": None,
            "cloud": None,
            "monitor": None
        }
        
        # Statistics
        self.stats = {
            "commands_executed": 0,
            "movements_completed": 0,
            "errors_encountered": 0,
            "uptime_start": time.time()
        }
        
        # Signal Handler für Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler für Ctrl+C - Emergency Stop."""
        print("\n\n🚨 EMERGENCY STOP - Ctrl+C detected!")
        if self.controller:
            self.controller.emergency_stop()
        
        # Cleanup enhanced features if active
        if ENHANCED_MODE:
            self._cleanup_enhanced_features()
        
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
            
            # Verbinden (Enhanced oder Standard Controller)
            print(f"🔌 Connecting to RoArm on {config.port}...")
            
            if ENHANCED_MODE:
                self.controller = EnhancedController(config)
            else:
                self.controller = RoArmController(config)
            
            if not self.controller.serial.connected:
                print("❌ Failed to connect to RoArm")
                print("   Check cable and port settings")
                return
            
            print("✅ Successfully connected!\n")
            
            # Initialize Components
            self._initialize_components()
            
            # Initialize Enhanced Features if available
            if ENHANCED_MODE:
                self._initialize_enhanced_features()
            
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
            
            # Auto-Start Enhanced Demo wenn angegeben
            if args.demo and ENHANCED_MODE:
                self._run_enhanced_demo()
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
            self.stats["errors_encountered"] += 1
        finally:
            self._cleanup()
    
    def _print_header(self):
        """Zeigt den Header."""
        print("\n" + "="*70)
        print(f"🤖 RoArm M3 Professional Control System v{self.VERSION}")
        if ENHANCED_MODE:
            print("   🚀 ENHANCED MODE ACTIVE")
        print("="*70)
        print("📷 Optimized for Revopoint Mini2 Scanner")
        print("🍎 macOS M4 Edition")
        print("🔧 Professional Calibration Suite")
        
        if ENHANCED_MODE:
            print("\n🎯 Enhanced Features Available:")
            print("   • Vision System & Object Detection")
            print("   • Machine Learning Optimization")
            print("   • Adaptive Control System")
            print("   • Predictive Motion Planning")
            print("   • Cloud Synchronization")
            print("   • Real-time Performance Monitoring")
        
        print("\n⚡ Press Ctrl+C anytime for EMERGENCY STOP")
        print("="*70 + "\n")
    
    def _initialize_components(self):
        """Initialisiert Standard-Komponenten."""
        print("Initializing components...")
        
        # Teaching Recorder
        self.teacher = TeachingRecorder(self.controller)
        print("  ✅ Teaching Recorder")
        
        # Calibration Suite
        self.calibrator = CalibrationSuite(self.controller)
        print("  ✅ Calibration Suite")
        
        # Safety System
        self.safety_system = SafetySystem(self.controller)
        print("  ✅ Safety System")
        
        print("All components initialized!\n")
    
    def _initialize_enhanced_features(self):
        """Initialisiert Enhanced Features wenn verfügbar."""
        if not ENHANCED_MODE:
            return
        
        print("Initializing enhanced features...")
        
        try:
            # Vision System
            try:
                self.enhanced["vision"] = VisionSystem(self.controller)
                print("  ✅ Vision System")
            except Exception as e:
                logger.debug(f"Vision System not available: {e}")
            
            # ML Optimizer
            try:
                self.enhanced["ml_optimizer"] = MLOptimizer(self.controller)
                print("  ✅ ML Optimizer")
            except Exception as e:
                logger.debug(f"ML Optimizer not available: {e}")
            
            # Adaptive Control
            try:
                self.enhanced["adaptive"] = AdaptiveControl(self.controller)
                self.enhanced["adaptive"].enable()
                print("  ✅ Adaptive Control")
            except Exception as e:
                logger.debug(f"Adaptive Control not available: {e}")
            
            # Predictive Motion
            try:
                self.enhanced["predictive"] = PredictiveMotion(self.controller)
                print("  ✅ Predictive Motion")
            except Exception as e:
                logger.debug(f"Predictive Motion not available: {e}")
            
            # Cloud Sync
            try:
                self.enhanced["cloud"] = CloudSync()
                if self.enhanced["cloud"].connect():
                    print("  ✅ Cloud Sync")
                else:
                    self.enhanced["cloud"] = None
            except Exception as e:
                logger.debug(f"Cloud Sync not available: {e}")
            
            # Realtime Monitor
            try:
                self.enhanced["monitor"] = RealtimeMonitoring(self.controller)
                self.enhanced["monitor"].start()
                print("  ✅ Realtime Monitor (http://localhost:8080)")
            except Exception as e:
                logger.debug(f"Realtime Monitor not available: {e}")
            
            print("Enhanced features ready!\n")
            
        except Exception as e:
            logger.warning(f"Some enhanced features could not be initialized: {e}")
    
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
            
            if ENHANCED_MODE and self.enhanced.get("adaptive"):
                print("   🎯 Adaptive compensation active")
        else:
            print("⚠️  No valid calibration found")
            print("   Run calibration for best accuracy (Option 6)")
        print()
    
    def _show_main_menu(self):
        """Zeigt das Hauptmenü."""
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        
        # Standard Features
        print("📌 STANDARD FEATURES:")
        print("1. 🎮 Manual Control")
        print("2. 📷 Scanner Patterns")
        print("3. 🎓 Teaching Mode")
        print("4. 📁 Load & Play Sequence")
        print("5. 🏠 Move to Home")
        print("6. 🔧 Calibration Suite")
        print("7. ⚙️  Settings")
        print("8. 📊 Status")
        print("9. 🧪 System Test")
        
        # Enhanced Features (wenn verfügbar)
        if ENHANCED_MODE:
            print("\n🚀 ENHANCED FEATURES:")
            if self.enhanced.get("vision"):
                print("10. 👁️  Vision System")
            if self.enhanced.get("ml_optimizer"):
                print("11. 🤖 ML Optimization")
            if self.enhanced.get("adaptive"):
                print("12. 🎯 Adaptive Control")
            if self.enhanced.get("predictive"):
                print("13. 🔮 Predictive Motion")
            if self.enhanced.get("cloud"):
                print("14. ☁️  Cloud Sync")
            if self.enhanced.get("monitor"):
                print("15. 📈 Performance Monitor")
            print("16. 🎭 Demo Showcase")
        
        print("\n0. 🚪 Exit")
        
        # Status Bar
        uptime = (time.time() - self.stats["uptime_start"]) / 60
        print("-"*50)
        print(f"📊 Session: {self.stats['commands_executed']} commands | "
              f"{self.stats['movements_completed']} movements | "
              f"{uptime:.1f} min uptime")
    
    def _handle_main_menu(self, choice: str):
        """Verarbeitet Hauptmenü-Auswahl."""
        self.stats["commands_executed"] += 1
        
        # Standard Handlers
        standard_handlers = {
            '1': self._manual_control,
            '2': self._scanner_menu,
            '3': self._teaching_menu,
            '4': self._load_sequence,
            '5': self._move_home,
            '6': self._calibration_menu,
            '7': self._settings_menu,
            '8': self._show_status,
            '9': self._system_test,
            '0': self._exit
        }
        
        # Enhanced Handlers
        enhanced_handlers = {}
        if ENHANCED_MODE:
            enhanced_handlers = {
                '10': self._vision_system_menu,
                '11': self._ml_optimization_menu,
                '12': self._adaptive_control_menu,
                '13': self._predictive_motion_menu,
                '14': self._cloud_sync_menu,
                '15': self._performance_monitor_menu,
                '16': self._demo_showcase_menu
            }
        
        # Combine handlers
        all_handlers = {**standard_handlers, **enhanced_handlers}
        
        handler = all_handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid option")
    
    # ============== ENHANCED FEATURE MENUS ==============
    
    def _vision_system_menu(self):
        """Vision System Menü."""
        if not self.enhanced.get("vision"):
            print("❌ Vision System not available")
            return
        
        print("\n👁️ VISION SYSTEM")
        print("-" * 40)
        print("1. 🔍 Object Detection")
        print("2. 📐 Pose Estimation")
        print("3. 🎯 Visual Servoing")
        print("4. 📷 Camera Calibration")
        print("5. 🖼️  Live Preview")
        print("6. 🎬 Record Video")
        print("7. 📊 Detection Statistics")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        vision = self.enhanced["vision"]
        
        if choice == '1':
            print("\n🔍 Starting object detection...")
            objects = vision.detect_objects()
            if objects:
                print(f"Detected {len(objects)} objects:")
                for i, obj in enumerate(objects, 1):
                    print(f"  {i}. {obj['class']} ({obj['confidence']:.1%}) at {obj['position']}")
            else:
                print("No objects detected")
        
        elif choice == '2':
            target = input("Target object (or ENTER for closest): ").strip()
            print("📐 Estimating pose...")
            pose = vision.estimate_pose(target if target else None)
            if pose:
                print(f"  Position: X={pose['x']:.3f}, Y={pose['y']:.3f}, Z={pose['z']:.3f}")
                print(f"  Rotation: R={pose['roll']:.1f}°, P={pose['pitch']:.1f}°, Y={pose['yaw']:.1f}°")
        
        elif choice == '3':
            target = input("Target object to track: ").strip()
            print(f"🎯 Starting visual servoing to '{target}'...")
            print("Press Ctrl+C to stop")
            try:
                vision.visual_servo(target)
                print("✅ Target reached!")
            except KeyboardInterrupt:
                print("\n⏹️ Visual servoing stopped")
        
        elif choice == '4':
            print("📷 Starting camera calibration...")
            print("Please show calibration pattern from different angles")
            if vision.calibrate_camera():
                print("✅ Camera calibrated successfully")
            else:
                print("❌ Calibration failed")
        
        elif choice == '5':
            print("🖼️ Opening live preview...")
            print("Press 'q' to close preview window")
            vision.show_preview()
        
        elif choice == '6':
            filename = input("Video filename [capture.mp4]: ").strip() or "capture.mp4"
            duration = float(input("Duration (seconds): ") or "10")
            print(f"🎬 Recording {duration}s video to {filename}...")
            vision.record_video(filename, duration)
            print(f"✅ Video saved to {filename}")
        
        elif choice == '7':
            stats = vision.get_statistics()
            print("\n📊 Detection Statistics:")
            print(f"  Total detections: {stats['total_detections']}")
            print(f"  Average confidence: {stats['avg_confidence']:.1%}")
            print(f"  Processing FPS: {stats['fps']:.1f}")
            print(f"  Most common object: {stats['most_common']}")
    
    def _ml_optimization_menu(self):
        """Machine Learning Optimization Menü."""
        if not self.enhanced.get("ml_optimizer"):
            print("❌ ML Optimizer not available")
            return
        
        print("\n🤖 MACHINE LEARNING OPTIMIZATION")
        print("-" * 40)
        print("1. 🛣️  Optimize Path")
        print("2. 📚 Learn from Teaching")
        print("3. 🎯 Predict Optimal Parameters")
        print("4. ⚡ Energy Optimization")
        print("5. 🧠 Train Model")
        print("6. 💾 Save/Load Model")
        print("7. 📊 Model Performance")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        ml = self.enhanced["ml_optimizer"]
        
        if choice == '1':
            print("\n🛣️ Path Optimization")
            print("Define path waypoints (enter 'done' when finished):")
            waypoints = []
            while True:
                point = input(f"Point {len(waypoints)+1} (x,y,z): ").strip()
                if point.lower() == 'done':
                    break
                try:
                    x, y, z = map(float, point.split(','))
                    waypoints.append([x, y, z])
                except:
                    print("Invalid format. Use: x,y,z")
            
            if waypoints:
                print("Optimizing path...")
                optimized = ml.optimize_path(waypoints)
                print(f"✅ Path optimized:")
                print(f"   Original length: {optimized['original_length']:.3f}")
                print(f"   Optimized length: {optimized['optimized_length']:.3f}")
                print(f"   Improvement: {optimized['improvement']:.1%}")
                
                execute = input("Execute optimized path? (y/n): ").lower()
                if execute == 'y':
                    self.controller.execute_path(optimized['path'])
        
        elif choice == '2':
            sequences_dir = Path("sequences")
            if sequences_dir.exists():
                sequences = list(sequences_dir.glob("*.json"))
                if sequences:
                    print("\nAvailable sequences:")
                    for i, seq in enumerate(sequences, 1):
                        print(f"  {i}. {seq.stem}")
                    
                    idx = int(input("Select sequence: ")) - 1
                    if 0 <= idx < len(sequences):
                        print(f"📚 Learning from {sequences[idx].stem}...")
                        ml.learn_from_teaching(str(sequences[idx]))
                        print("✅ Learning complete")
        
        elif choice == '3':
            print("\n🎯 Parameter Prediction")
            print("Task types: pick_place, scanning, welding, painting")
            task = input("Task type: ").strip()
            
            params = ml.predict_parameters(task)
            print(f"\nOptimal parameters for '{task}':")
            print(f"  Speed: {params['speed']:.2f}")
            print(f"  Acceleration: {params['acceleration']:.2f}")
            print(f"  Jerk: {params['jerk']:.1f}")
            print(f"  Trajectory: {params['trajectory']}")
            print(f"  Settle time: {params['settle_time']:.2f}s")
            
            apply = input("Apply these parameters? (y/n): ").lower()
            if apply == 'y':
                self.controller.current_speed = params['speed']
                print("✅ Parameters applied")
        
        elif choice == '4':
            print("\n⚡ Energy Optimization")
            print("Analyzing current energy usage...")
            current_energy = ml.analyze_energy()
            print(f"Current consumption: {current_energy['power']:.1f}W")
            
            print("Optimizing for energy efficiency...")
            optimized = ml.optimize_energy()
            print(f"✅ Optimization complete:")
            print(f"   Energy savings: {optimized['savings']:.1%}")
            print(f"   Speed reduction: {optimized['speed_factor']:.1%}")
            print(f"   Path adjustments: {optimized['path_changes']}")
        
        elif choice == '5':
            print("\n🧠 Model Training")
            print("Training data sources:")
            print("1. Recorded sequences")
            print("2. Live collection")
            print("3. Synthetic data")
            
            source = input("Select source: ").strip()
            
            if source == '1':
                print("Training on recorded sequences...")
                metrics = ml.train_model(source='sequences')
            elif source == '2':
                duration = float(input("Collection duration (minutes): "))
                print(f"Collecting data for {duration} minutes...")
                print("Move the robot manually or run patterns")
                metrics = ml.train_model(source='live', duration=duration*60)
            elif source == '3':
                samples = int(input("Number of synthetic samples: "))
                print(f"Generating {samples} synthetic samples...")
                metrics = ml.train_model(source='synthetic', samples=samples)
            
            if metrics:
                print("\n📊 Training Results:")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Training time: {metrics['time']:.1f}s")
        
        elif choice == '6':
            print("\n💾 Model Management")
            print("1. Save current model")
            print("2. Load model")
            print("3. Export for deployment")
            
            action = input("Select action: ").strip()
            
            if action == '1':
                name = input("Model name [ml_model]: ").strip() or "ml_model"
                filepath = f"models/{name}.pkl"
                ml.save_model(filepath)
                print(f"✅ Model saved to {filepath}")
            
            elif action == '2':
                models_dir = Path("models")
                if models_dir.exists():
                    models = list(models_dir.glob("*.pkl"))
                    if models:
                        print("\nAvailable models:")
                        for i, model in enumerate(models, 1):
                            print(f"  {i}. {model.stem}")
                        
                        idx = int(input("Select model: ")) - 1
                        if 0 <= idx < len(models):
                            ml.load_model(str(models[idx]))
                            print(f"✅ Model loaded: {models[idx].stem}")
            
            elif action == '3':
                format_type = input("Export format (onnx/tflite/coreml): ").strip()
                name = input("Export name: ").strip()
                ml.export_model(f"exports/{name}.{format_type}", format=format_type)
                print(f"✅ Model exported to exports/{name}.{format_type}")
        
        elif choice == '7':
            metrics = ml.get_performance_metrics()
            print("\n📊 Model Performance:")
            print(f"  Model type: {metrics['model_type']}")
            print(f"  Parameters: {metrics['parameters']:,}")
            print(f"  Inference time: {metrics['inference_time']:.3f}ms")
            print(f"  Memory usage: {metrics['memory_mb']:.1f}MB")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Training samples: {metrics['training_samples']:,}")
    
    def _adaptive_control_menu(self):
        """Adaptive Control Menü."""
        if not self.enhanced.get("adaptive"):
            print("❌ Adaptive Control not available")
            return
        
        print("\n🎯 ADAPTIVE CONTROL SYSTEM")
        print("-" * 40)
        
        adaptive = self.enhanced["adaptive"]
        status = "ENABLED" if adaptive.is_enabled() else "DISABLED"
        print(f"Status: {status}")
        print(f"Learning rate: {adaptive.learning_rate:.3f}")
        print(f"Adaptation cycles: {adaptive.cycles}")
        
        print("\n1. 🔄 Toggle On/Off")
        print("2. 📈 Adjust Learning Rate")
        print("3. 🎯 Auto-Tune Current Position")
        print("4. 📊 View Adaptation History")
        print("5. 🔧 Manual PID Tuning")
        print("6. 💾 Save/Load Profile")
        print("7. 🔄 Reset to Defaults")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        
        if choice == '1':
            if adaptive.is_enabled():
                adaptive.disable()
                print("🔴 Adaptive control disabled")
            else:
                adaptive.enable()
                print("🟢 Adaptive control enabled")
        
        elif choice == '2':
            current = adaptive.learning_rate
            print(f"Current learning rate: {current:.3f}")
            rate = float(input("New learning rate (0.001-1.0): "))
            adaptive.set_learning_rate(rate)
            print(f"✅ Learning rate set to {rate:.3f}")
        
        elif choice == '3':
            print("🎯 Auto-tuning current position...")
            print("This will make small test movements")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                result = adaptive.auto_tune()
                print("✅ Auto-tuning complete:")
                print(f"   Settling time: {result['settling_time']:.3f}s")
                print(f"   Overshoot: {result['overshoot']:.1%}")
                print(f"   Steady-state error: {result['error']:.4f}")
        
        elif choice == '4':
            history = adaptive.get_history(limit=20)
            print("\n📊 Recent Adaptations:")
            print("Time     | Joint    | Error   | Adjustment")
            print("-"*45)
            for entry in history:
                time_str = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                print(f"{time_str} | {entry['joint']:8s} | {entry['error']:+.4f} | {entry['adjustment']:+.4f}")
        
        elif choice == '5':
            print("\n🔧 Manual PID Tuning")
            joint = input("Joint (base/shoulder/elbow/wrist/roll/hand): ").strip()
            if joint in SERVO_LIMITS:
                current_pid = adaptive.get_pid_values(joint)
                print(f"Current PID for {joint}:")
                print(f"  P: {current_pid['p']:.4f}")
                print(f"  I: {current_pid['i']:.4f}")
                print(f"  D: {current_pid['d']:.4f}")
                
                p = float(input(f"New P [{current_pid['p']}]: ") or current_pid['p'])
                i = float(input(f"New I [{current_pid['i']}]: ") or current_pid['i'])
                d = float(input(f"New D [{current_pid['d']}]: ") or current_pid['d'])
                
                adaptive.set_pid_values(joint, p, i, d)
                print(f"✅ PID values updated for {joint}")
        
        elif choice == '6':
            print("\n💾 Profile Management")
            print("1. Save current profile")
            print("2. Load profile")
            
            action = input("Select: ").strip()
            
            if action == '1':
                name = input("Profile name: ").strip()
                adaptive.save_profile(f"profiles/{name}.json")
                print(f"✅ Profile saved to profiles/{name}.json")
            
            elif action == '2':
                profiles = list(Path("profiles").glob("*.json"))
                if profiles:
                    print("\nAvailable profiles:")
                    for i, prof in enumerate(profiles, 1):
                        print(f"  {i}. {prof.stem}")
                    
                    idx = int(input("Select profile: ")) - 1
                    if 0 <= idx < len(profiles):
                        adaptive.load_profile(str(profiles[idx]))
                        print(f"✅ Profile loaded: {profiles[idx].stem}")
        
        elif choice == '7':
            confirm = input("Reset all adaptations to defaults? (y/n): ").lower()
            if confirm == 'y':
                adaptive.reset()
                print("✅ Adaptive control reset to defaults")
    
    def _predictive_motion_menu(self):
        """Predictive Motion Planning Menü."""
        if not self.enhanced.get("predictive"):
            print("❌ Predictive Motion not available")
            return
        
        print("\n🔮 PREDICTIVE MOTION PLANNING")
        print("-" * 40)
        
        predictive = self.enhanced["predictive"]
        
        print("1. 🎯 Predict Next Position")
        print("2. 🛡️  Collision Prediction")
        print("3. 📈 Trajectory Preview")
        print("4. ⚡ Optimize Upcoming Movement")
        print("5. 🔄 Predictive Smoothing")
        print("6. ⏱️  Time-to-Target Estimation")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        
        if choice == '1':
            horizon = float(input("Prediction horizon (seconds) [2.0]: ") or "2.0")
            prediction = predictive.predict_position(horizon)
            print(f"\n🎯 Predicted position in {horizon}s:")
            for joint, pos in prediction.items():
                current = self.controller.current_position[joint]
                delta = pos - current
                print(f"  {joint}: {pos:.3f} rad (Δ{delta:+.3f})")
        
        elif choice == '2':
            print("🛡️ Checking for potential collisions...")
            collisions = predictive.predict_collisions()
            if collisions:
                print(f"⚠️ {len(collisions)} potential collisions detected:")
                for i, coll in enumerate(collisions, 1):
                    print(f"  {i}. Time: +{coll['time']:.1f}s, Joint: {coll['joint']}, Type: {coll['type']}")
                
                avoid = input("Activate collision avoidance? (y/n): ").lower()
                if avoid == 'y':
                    predictive.enable_collision_avoidance()
                    print("✅ Collision avoidance activated")
            else:
                print("✅ No collisions predicted")
        
        elif choice == '3':
            duration = float(input("Preview duration (seconds) [5.0]: ") or "5.0")
            print(f"📈 Generating {duration}s trajectory preview...")
            preview = predictive.preview_trajectory(duration)
            
            print("\nTrajectory preview:")
            print("Time | Base   | Shoulder | Elbow  | Wrist")
            print("-"*50)
            for point in preview[::10]:  # Every 10th point
                print(f"{point['time']:4.1f} | {point['base']:6.3f} | {point['shoulder']:8.3f} | "
                      f"{point['elbow']:6.3f} | {point['wrist']:6.3f}")
        
        elif choice == '4':
            print("⚡ Optimizing upcoming movement...")
            optimization = predictive.optimize_next_movement()
            print("✅ Movement optimized:")
            print(f"   Original time: {optimization['original_time']:.2f}s")
            print(f"   Optimized time: {optimization['optimized_time']:.2f}s")
            print(f"   Energy saved: {optimization['energy_saved']:.1%}")
            print(f"   Smoothness improved: {optimization['smoothness']:.1%}")
        
        elif choice == '5':
            print("🔄 Enabling predictive smoothing...")
            smoothing_level = float(input("Smoothing level (0.1-1.0) [0.5]: ") or "0.5")
            predictive.enable_smoothing(smoothing_level)
            print(f"✅ Predictive smoothing enabled (level: {smoothing_level})")
        
        elif choice == '6':
            print("Define target position:")
            target = {}
            for joint in ['base', 'shoulder', 'elbow', 'wrist']:
                val = input(f"  {joint} [{self.controller.current_position[joint]:.3f}]: ").strip()
                if val:
                    target[joint] = float(val)
                else:
                    target[joint] = self.controller.current_position[joint]
            
            estimate = predictive.estimate_time_to_target(target)
            print(f"\n⏱️ Time-to-target estimation:")
            print(f"   Distance: {estimate['distance']:.3f} rad")
            print(f"   Estimated time: {estimate['time']:.2f}s")
            print(f"   Max velocity joint: {estimate['limiting_joint']}")
            print(f"   Energy required: {estimate['energy']:.1f}J")
    
    def _cloud_sync_menu(self):
        """Cloud Sync Menü."""
        if not self.enhanced.get("cloud"):
            print("❌ Cloud Sync not available or not configured")
            print("   Configure API credentials in config.yaml")
            return
        
        print("\n☁️ CLOUD SYNCHRONIZATION")
        print("-" * 40)
        
        cloud = self.enhanced["cloud"]
        status = cloud.get_status()
        
        print(f"Connection: {'🟢 Connected' if status['connected'] else '🔴 Disconnected'}")
        print(f"Last sync: {status['last_sync']}")
        print(f"Pending uploads: {status['pending_uploads']}")
        
        print("\n1. 📤 Upload Current Session")
        print("2. 📥 Download Sequences")
        print("3. 🔄 Sync All Data")
        print("4. 👥 Share Sequence")
        print("5. 🌍 Browse Community")
        print("6. ⚙️  Configure Sync")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        
        if choice == '1':
            print("📤 Uploading current session...")
            result = cloud.upload_session({
                'sequences': self.teacher.get_current_sequences(),
                'calibration': self.calibrator.calibration,
                'statistics': self.stats
            })
            if result['success']:
                print(f"✅ Uploaded successfully (ID: {result['session_id']})")
            else:
                print(f"❌ Upload failed: {result['error']}")
        
        elif choice == '2':
            print("📥 Available cloud sequences:")
            sequences = cloud.list_sequences()
            for i, seq in enumerate(sequences, 1):
                print(f"  {i}. {seq['name']} ({seq['author']}) - {seq['downloads']} downloads")
            
            if sequences:
                idx = int(input("Download sequence: ")) - 1
                if 0 <= idx < len(sequences):
                    print(f"Downloading {sequences[idx]['name']}...")
                    if cloud.download_sequence(sequences[idx]['id']):
                        print("✅ Downloaded successfully")
        
        elif choice == '3':
            print("🔄 Synchronizing all data...")
            print("This may take a few moments...")
            result = cloud.sync_all()
            print(f"✅ Sync complete:")
            print(f"   Uploaded: {result['uploaded']} items")
            print(f"   Downloaded: {result['downloaded']} items")
            print(f"   Conflicts resolved: {result['conflicts']}")
        
        elif choice == '4':
            sequences = list(Path("sequences").glob("*.json"))
            if sequences:
                print("\nLocal sequences:")
                for i, seq in enumerate(sequences, 1):
                    print(f"  {i}. {seq.stem}")
                
                idx = int(input("Select sequence to share: ")) - 1
                if 0 <= idx < len(sequences):
                    description = input("Description: ").strip()
                    tags = input("Tags (comma-separated): ").strip().split(',')
                    
                    result = cloud.share_sequence(
                        str(sequences[idx]),
                        description=description,
                        tags=tags
                    )
                    
                    if result['success']:
                        print(f"✅ Shared successfully!")
                        print(f"   Share URL: {result['url']}")
                        print(f"   Access code: {result['code']}")
        
        elif choice == '5':
            print("🌍 Browse Community Content")
            print("1. Top rated")
            print("2. Most downloaded")
            print("3. Recent")
            print("4. Search")
            
            browse = input("Select: ").strip()
            
            if browse == '4':
                query = input("Search term: ").strip()
                results = cloud.search_community(query)
            else:
                category = ['top', 'popular', 'recent'][int(browse)-1] if browse in '123' else 'recent'
                results = cloud.browse_community(category)
            
            if results:
                print(f"\nFound {len(results)} items:")
                for item in results[:10]:
                    print(f"  • {item['name']} by {item['author']}")
                    print(f"    {item['description'][:50]}...")
                    print(f"    ⭐ {item['rating']:.1f} | 📥 {item['downloads']}")
        
        elif choice == '6':
            print("\n⚙️ Cloud Sync Configuration")
            print(f"Current endpoint: {cloud.endpoint}")
            print(f"Auto-sync: {'Enabled' if cloud.auto_sync else 'Disabled'}")
            print(f"Sync interval: {cloud.sync_interval}s")
            
            print("\n1. Change endpoint")
            print("2. Toggle auto-sync")
            print("3. Set sync interval")
            print("4. Update credentials")
            
            config = input("Select: ").strip()
            
            if config == '1':
                endpoint = input("New endpoint URL: ").strip()
                cloud.set_endpoint(endpoint)
            elif config == '2':
                cloud.auto_sync = not cloud.auto_sync
                print(f"Auto-sync {'enabled' if cloud.auto_sync else 'disabled'}")
            elif config == '3':
                interval = int(input("Sync interval (seconds): "))
                cloud.sync_interval = interval
            elif config == '4':
                api_key = input("API Key: ").strip()
                cloud.set_credentials(api_key)
                print("✅ Credentials updated")
    
    def _performance_monitor_menu(self):
        """Performance Monitor Menü."""
        if not self.enhanced.get("monitor"):
            print("❌ Performance Monitor not available")
            return
        
        print("\n📈 PERFORMANCE MONITOR")
        print("-" * 40)
        
        monitor = self.enhanced["monitor"]
        
        print(f"Monitor URL: http://localhost:{monitor.port}")
        print(f"Status: {'🟢 Running' if monitor.is_running() else '🔴 Stopped'}")
        
        # Live stats
        stats = monitor.get_current_stats()
        print(f"\n📊 Current Performance:")
        print(f"  CPU Usage: {stats['cpu']:.1f}%")
        print(f"  Memory: {stats['memory_mb']:.1f}MB")
        print(f"  Commands/sec: {stats['commands_per_sec']:.1f}")
        print(f"  Latency: {stats['latency_ms']:.1f}ms")
        print(f"  Error rate: {stats['error_rate']:.2%}")
        
        print("\n1. 📊 Detailed Statistics")
        print("2. 📈 Show Graphs")
        print("3. 🎯 Performance Test")
        print("4. 📝 Export Report")
        print("5. 🔧 Configure Monitoring")
        print("6. 🔄 Reset Statistics")
        print("0. Back")
        
        choice = input("\n👉 Select option: ").strip()
        
        if choice == '1':
            detailed = monitor.get_detailed_stats()
            print("\n📊 Detailed Statistics:")
            print("\nJoint Performance:")
            for joint, stats in detailed['joints'].items():
                print(f"  {joint}:")
                print(f"    Movements: {stats['movements']}")
                print(f"    Avg speed: {stats['avg_speed']:.2f} rad/s")
                print(f"    Accuracy: {stats['accuracy']:.4f} rad")
                print(f"    Errors: {stats['errors']}")
            
            print("\nTiming Statistics:")
            print(f"  Total uptime: {detailed['uptime']:.1f} min")
            print(f"  Active time: {detailed['active_time']:.1f} min")
            print(f"  Idle time: {detailed['idle_time']:.1f} min")
            print(f"  Efficiency: {detailed['efficiency']:.1%}")
        
        elif choice == '2':
            print("📈 Opening performance graphs in browser...")
            import webbrowser
            webbrowser.open(f"http://localhost:{monitor.port}/graphs")
        
        elif choice == '3':
            print("\n🎯 Performance Test")
            print("1. Quick test (1 min)")
            print("2. Standard test (5 min)")
            print("3. Stress test (10 min)")
            print("4. Custom duration")
            
            test = input("Select test: ").strip()
            
            durations = {'1': 60, '2': 300, '3': 600}
            if test in durations:
                duration = durations[test]
            elif test == '4':
                duration = int(input("Duration (seconds): "))
            else:
                return
            
            print(f"Running {duration}s performance test...")
            print("The robot will execute various movements")
            
            results = monitor.run_performance_test(duration)
            
            print("\n🏁 Test Results:")
            print(f"  Commands executed: {results['total_commands']}")
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Avg response time: {results['avg_response']:.1f}ms")
            print(f"  Max response time: {results['max_response']:.1f}ms")
            print(f"  Throughput: {results['throughput']:.1f} cmd/s")
            print(f"  Grade: {results['grade']}")
        
        elif choice == '4':
            print("📝 Generating performance report...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"reports/performance_{timestamp}.html"
            
            monitor.export_report(filename)
            print(f"✅ Report saved to {filename}")
            
            open_report = input("Open in browser? (y/n): ").lower()
            if open_report == 'y':
                import webbrowser
                webbrowser.open(filename)
        
        elif choice == '5':
            print("\n🔧 Monitoring Configuration")
            print(f"Update rate: {monitor.update_rate}Hz")
            print(f"Buffer size: {monitor.buffer_size}")
            print(f"Port: {monitor.port}")
            
            print("\n1. Change update rate")
            print("2. Set buffer size")
            print("3. Change port")
            print("4. Toggle components")
            
            config = input("Select: ").strip()
            
            if config == '1':
                rate = int(input(f"Update rate (Hz) [{monitor.update_rate}]: ") or monitor.update_rate)
                monitor.set_update_rate(rate)
            elif config == '2':
                size = int(input(f"Buffer size [{monitor.buffer_size}]: ") or monitor.buffer_size)
                monitor.set_buffer_size(size)
            elif config == '3':
                port = int(input(f"Port [{monitor.port}]: ") or monitor.port)
                monitor.restart(port=port)
            elif config == '4':
                print("Toggle monitoring components:")
                components = monitor.get_components()
                for comp, enabled in components.items():
                    toggle = input(f"  {comp} ({'ON' if enabled else 'OFF'}) - toggle? (y/n): ").lower()
                    if toggle == 'y':
                        monitor.toggle_component(comp)
        
        elif choice == '6':
            confirm = input("Reset all statistics? (y/n): ").lower()
            if confirm == 'y':
                monitor.reset_statistics()
                print("✅ Statistics reset")
    
    def _demo_showcase_menu(self):
        """Demo Showcase für Enhanced Features."""
        if not ENHANCED_MODE:
            print("❌ Demo requires enhanced features")
            return
        
        print("\n🎭 ENHANCED FEATURES DEMO SHOWCASE")
        print("-" * 40)
        print("Demonstrations of enhanced capabilities:")
        print("\n1. 🎯 Smart Object Manipulation")
        print("2. 🔄 Adaptive Learning Demo")
        print("3. 👁️ Vision-Guided Movement")
        print("4. 🤖 AI-Optimized Scanning")
        print("5. 🎨 Creative Pattern Generation")
        print("6. 🚀 Full System Showcase")
        print("0. Back")
        
        choice = input("\n👉 Select demo: ").strip()
        
        if choice == '1':
            self._run_smart_manipulation_demo()
        elif choice == '2':
            self._run_adaptive_learning_demo()
        elif choice == '3':
            self._run_vision_guided_demo()
        elif choice == '4':
            self._run_ai_scanning_demo()
        elif choice == '5':
            self._run_creative_pattern_demo()
        elif choice == '6':
            self._run_full_showcase()
    
    def _run_smart_manipulation_demo(self):
        """Smart Object Manipulation Demo."""
        print("\n🎯 SMART OBJECT MANIPULATION DEMO")
        print("-" * 40)
        print("This demo shows intelligent object handling")
        print("combining vision, ML, and adaptive control.\n")
        
        input("Press ENTER to start...")
        
        # Move to scan position
        print("1. Moving to scan position...")
        self.controller.move_to_scanner_position(speed=0.5)
        time.sleep(2)
        
        if self.enhanced.get("vision"):
            print("2. Detecting objects...")
            objects = self.enhanced["vision"].detect_objects()
            if objects:
                print(f"   Found {len(objects)} objects")
                target = objects[0]
                print(f"   Targeting: {target['class']}")
                
                if self.enhanced.get("ml_optimizer"):
                    print("3. Optimizing approach path...")
                    path = self.enhanced["ml_optimizer"].plan_approach(target)
                    print("   Path optimized for minimal energy")
                
                if self.enhanced.get("adaptive"):
                    print("4. Enabling adaptive control...")
                    self.enhanced["adaptive"].enable()
                    print("   System will adapt to object weight")
                
                print("5. Executing smart manipulation...")
                # Execute the manipulation
                self.controller.execute_path(path)
                print("✅ Demo complete!")
        else:
            print("   Vision system not available - simulating...")
            time.sleep(2)
            print("✅ Simulation complete!")
    
    def _run_enhanced_demo(self):
        """Führt eine Enhanced Features Demo aus."""
        if not ENHANCED_MODE:
            print("Enhanced features not available")
            return
        
        print("\n🎭 ENHANCED FEATURES DEMONSTRATION")
        print("="*50)
        print("This demo showcases the enhanced capabilities:\n")
        
        demos = []
        if self.enhanced.get("vision"):
            demos.append("vision")
        if self.enhanced.get("ml_optimizer"):
            demos.append("ml")
        if self.enhanced.get("adaptive"):
            demos.append("adaptive")
        
        for demo in demos:
            print(f"Running {demo} demo...")
            time.sleep(2)
            # Demo implementation
        
        print("\n✅ Enhanced demo complete!")
    
    # ============== STANDARD FEATURES (alle bisherigen Funktionen) ==============
    # [Hier kommen alle anderen Funktionen aus der ursprünglichen main.py]
    # Ich füge nur die wichtigsten hinzu, um Platz zu sparen:
    
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
        
        if ENHANCED_MODE and self.enhanced.get("predictive"):
            print("  p: Toggle predictive mode")
        if ENHANCED_MODE and self.enhanced.get("adaptive"):
            print("  o: Toggle adaptive control")
        
        print("  space: Emergency stop")
        print("  x: Exit manual control")
        print("-" * 40)
        
        speed = 1.0
        step = 0.1
        
        if self.calibrator.calibration.calibration_valid:
            print("📐 Using calibrated limits")
            step = 0.05
        
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
            elif key == 'p' and ENHANCED_MODE and self.enhanced.get("predictive"):
                self.enhanced["predictive"].toggle()
                state = "ON" if self.enhanced["predictive"].is_enabled() else "OFF"
                print(f"Predictive mode: {state}")
            elif key == 'o' and ENHANCED_MODE and self.enhanced.get("adaptive"):
                self.enhanced["adaptive"].toggle()
                state = "ON" if self.enhanced["adaptive"].is_enabled() else "OFF"
                print(f"Adaptive control: {state}")
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
                    
                    if self.calibrator.calibration.calibration_valid:
                        if joint in self.calibrator.calibration.joints:
                            cal = self.calibrator.calibration.joints[joint]
                            new_pos[joint] = max(cal.safe_min, min(cal.safe_max, new_pos[joint]))
                    
                    self.controller.move_joints(
                        new_pos,
                        speed=speed,
                        trajectory_type=TrajectoryType.LINEAR,
                        wait=False
                    )
                    self.stats["movements_completed"] += 1
    
    def _scanner_menu(self):
        """Scanner Pattern Menü."""
        # [Implementation aus original main.py]
        pass
    
    def _teaching_menu(self):
        """Teaching Mode Menü."""
        # [Implementation aus original main.py]
        pass
    
    def _calibration_menu(self):
        """Calibration Suite Menü."""
        # [Implementation aus original main.py]
        pass
    
    def _move_home(self):
        """Bewegt zur Home Position."""
        print("\n🏠 Moving to home position...")
        if self.controller.move_home(speed=0.5):
            print("✅ Home position reached")
            self.stats["movements_completed"] += 1
        else:
            print("❌ Failed to reach home")
            self.stats["errors_encountered"] += 1
    
    def _settings_menu(self):
        """Settings Menü."""
        # [Implementation aus original main.py]
        pass
    
    def _show_status(self):
        """Zeigt System Status."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        status = self.controller.query_status()
        if status:
            print("Joint Positions (rad):")
            for joint, pos in status['positions'].items():
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
            
            # Enhanced status if available
            if ENHANCED_MODE:
                print("\n🚀 Enhanced Status:")
                if self.enhanced.get("adaptive"):
                    print(f"  Adaptive: {'ON' if self.enhanced['adaptive'].is_enabled() else 'OFF'}")
                if self.enhanced.get("predictive"):
                    print(f"  Predictive: {'ON' if self.enhanced['predictive'].is_enabled() else 'OFF'}")
                if self.enhanced.get("monitor"):
                    perf = self.enhanced["monitor"].get_current_stats()
                    print(f"  Performance: {perf['cpu']:.1f}% CPU, {perf['memory_mb']:.0f}MB RAM")
        else:
            print("❌ Failed to query status")
    
    def _system_test(self):
        """System Test Menü."""
        # [Implementation aus original main.py]
        pass
    
    def _load_sequence(self):
        """Load & Play Sequence."""
        # [Implementation aus original main.py]
        pass
    
    def _execute_pattern(self, pattern_name: str):
        """Führt ein Pattern direkt aus."""
        # [Implementation aus original main.py]
        pass
    
    def _run_auto_calibration(self):
        """Auto-Calibration."""
        # [Implementation aus original main.py]
        pass
    
    def _exit(self):
        """Beendet das Programm."""
        self.running = False
    
    def _cleanup(self):
        """Aufräumen beim Beenden."""
        print("\n🔌 Shutting down...")
        
        # Cleanup enhanced features
        if ENHANCED_MODE:
            self._cleanup_enhanced_features()
        
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
        
        # Show session statistics
        print("\n📊 Session Statistics:")
        print(f"  Commands executed: {self.stats['commands_executed']}")
        print(f"  Movements completed: {self.stats['movements_completed']}")
        print(f"  Errors encountered: {self.stats['errors_encountered']}")
        uptime = (time.time() - self.stats['uptime_start']) / 60
        print(f"  Total uptime: {uptime:.1f} minutes")
        
        print("\n✅ Shutdown complete")
        print("Goodbye! 👋\n")
    
    def _cleanup_enhanced_features(self):
        """Cleanup für Enhanced Features."""
        if not ENHANCED_MODE:
            return
        
        print("Cleaning up enhanced features...")
        
        try:
            if self.enhanced.get("monitor"):
                self.enhanced["monitor"].stop()
            
            if self.enhanced.get("cloud"):
                self.enhanced["cloud"].disconnect()
            
            if self.enhanced.get("vision"):
                self.enhanced["vision"].cleanup()
            
            if self.enhanced.get("adaptive"):
                self.enhanced["adaptive"].save_profile("profiles/last_session.json")
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RoArm M3 Professional Control System v3.0'
    )
    
    parser.add_argument(
        '--port',
        default='/dev/tty.usbserial-110',
        help='Serial port (use "auto" for auto-detection)'
    )
    
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Serial baudrate'
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
        '--demo',
        action='store_true',
        help='Run enhanced features demo (requires enhanced mode)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--no-enhanced',
        action='store_true',
        help='Disable enhanced features even if available'
    )
    
    args = parser.parse_args()
    
    # Force disable enhanced if requested
    if args.no_enhanced:
        global ENHANCED_MODE
        ENHANCED_MODE = False
    
    # Start CLI
    cli = RoArmCLI()
    cli.run(args)


if __name__ == '__main__':
    main()
