#!/usr/bin/env python3
"""
RoArm M3 System Validation Script
Prüft ob das gesamte System lauffähig ist
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class SystemValidator:
    """Comprehensive system validation."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        self.critical_errors = []
        
    def print_colored(self, text: str, color: str = 'white'):
        """Print colored text."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
        print(f"{colors.get(color, '')}{text}{colors['reset']}")
    
    def validate_all(self) -> bool:
        """Run all validations."""
        self.print_colored("\n" + "="*60, 'cyan')
        self.print_colored("🔍 RoArm M3 System Validation", 'cyan')
        self.print_colored("="*60, 'cyan')
        
        # Run all tests
        all_passed = True
        
        # 1. File Structure
        if not self.validate_file_structure():
            all_passed = False
            
        # 2. Python Imports
        if not self.validate_imports():
            all_passed = False
            
        # 3. Config File
        if not self.validate_config():
            all_passed = False
            
        # 4. Class Instantiation
        if not self.validate_class_instantiation():
            all_passed = False
            
        # 5. Dependencies
        if not self.validate_dependencies():
            all_passed = False
            
        # 6. Cross-Module Integration
        if not self.validate_integration():
            all_passed = False
        
        # Print summary
        self.print_summary()
        
        return all_passed and len(self.critical_errors) == 0
    
    def validate_file_structure(self) -> bool:
        """Validate all required files exist."""
        self.print_colored("\n📁 Validating File Structure...", 'blue')
        
        required_files = {
            # Main files
            "main.py": "Main program",
            "config.yaml": "Configuration",
            "requirements.txt": "Dependencies",
            
            # Core module
            "core/__init__.py": "Core package init",
            "core/controller.py": "Main controller",
            "core/serial_comm.py": "Serial communication",
            "core/constants.py": "Hardware constants",
            
            # Motion module
            "motion/__init__.py": "Motion package init",
            "motion/trajectory.py": "Trajectory generation",
            
            # Patterns module
            "patterns/__init__.py": "Patterns package init",
            "patterns/scan_patterns.py": "Scan patterns",
            
            # Teaching module
            "teaching/__init__.py": "Teaching package init",
            "teaching/recorder.py": "Teaching recorder",
            
            # Calibration module
            "calibration/__init__.py": "Calibration package init",
            "calibration/calibration_suite.py": "Calibration suite",
            
            # Safety module
            "safety/__init__.py": "Safety package init",
            "safety/safety_system.py": "Safety system",
            
            # Utils module
            "utils/__init__.py": "Utils package init",
            "utils/logger.py": "Logging system",
            "utils/terminal.py": "Terminal control",
            "utils/safety.py": "Safety monitor",
            "utils/debug_tool.py": "Debug tool"
        }
        
        all_exist = True
        for filepath, description in required_files.items():
            path = Path(filepath)
            if path.exists():
                self.successes.append(f"✅ {filepath}: {description}")
                print(f"  ✅ {filepath}")
            else:
                self.critical_errors.append(f"Missing file: {filepath}")
                self.print_colored(f"  ❌ {filepath} - MISSING!", 'red')
                all_exist = False
                
                # Try to create missing __init__.py
                if filepath.endswith("__init__.py"):
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("")
                        self.print_colored(f"     Created empty {filepath}", 'yellow')
                    except Exception as e:
                        self.print_colored(f"     Could not create: {e}", 'red')
        
        return all_exist
    
    def validate_imports(self) -> bool:
        """Validate all modules can be imported."""
        self.print_colored("\n🔗 Validating Module Imports...", 'blue')
        
        modules_to_test = [
            # Core modules
            ("core.controller", ["RoArmController", "RoArmConfig"]),
            ("core.serial_comm", ["SerialManager"]),
            ("core.constants", ["SERVO_LIMITS", "HOME_POSITION"]),
            
            # Motion modules
            ("motion.trajectory", ["TrajectoryGenerator", "TrajectoryType"]),
            
            # Pattern modules
            ("patterns.scan_patterns", ["RasterScanPattern", "SpiralScanPattern"]),
            
            # Teaching modules
            ("teaching.recorder", ["TeachingRecorder", "RecordingMode"]),
            
            # Calibration modules
            ("calibration.calibration_suite", ["CalibrationSuite", "CalibrationType"]),
            
            # Safety modules
            ("safety.safety_system", ["SafetySystem", "SafetyState"]),
            
            # Utils modules
            ("utils.logger", ["setup_logger", "get_logger"]),
            ("utils.terminal", ["TerminalController"]),
            ("utils.safety", ["SafetyMonitor"]),
            ("utils.debug_tool", ["SystemDebugger"])
        ]
        
        all_imported = True
        for module_name, expected_attrs in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                
                # Check expected attributes
                missing_attrs = []
                for attr in expected_attrs:
                    if not hasattr(module, attr):
                        missing_attrs.append(attr)
                
                if missing_attrs:
                    self.warnings.append(f"Module {module_name} missing attributes: {missing_attrs}")
                    self.print_colored(f"  ⚠️  {module_name} - missing: {missing_attrs}", 'yellow')
                else:
                    self.successes.append(f"Module {module_name} imported successfully")
                    print(f"  ✅ {module_name}")
                    
            except ImportError as e:
                self.critical_errors.append(f"Cannot import {module_name}: {e}")
                self.print_colored(f"  ❌ {module_name} - {e}", 'red')
                all_imported = False
            except Exception as e:
                self.errors.append(f"Error importing {module_name}: {e}")
                self.print_colored(f"  ❌ {module_name} - {e}", 'red')
                all_imported = False
        
        return all_imported
    
    def validate_config(self) -> bool:
        """Validate configuration file."""
        self.print_colored("\n⚙️ Validating Configuration...", 'blue')
        
        try:
            import yaml
            
            config_path = Path("config.yaml")
            if not config_path.exists():
                self.critical_errors.append("config.yaml not found")
                self.print_colored("  ❌ config.yaml not found", 'red')
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = {
                'system': ['port', 'baudrate'],
                'hardware': ['servo_limits', 'home_position'],
                'scanner': ['model', 'weight'],
                'motion': ['default_speed'],
                'safety': ['emergency_deceleration']
            }
            
            all_valid = True
            for section, keys in required_sections.items():
                if section not in config:
                    self.errors.append(f"Config missing section: {section}")
                    self.print_colored(f"  ❌ Missing section: {section}", 'red')
                    all_valid = False
                else:
                    for key in keys:
                        if key not in config[section]:
                            self.warnings.append(f"Config section {section} missing key: {key}")
                            self.print_colored(f"  ⚠️  {section}.{key} missing", 'yellow')
                    else:
                        print(f"  ✅ Section: {section}")
            
            return all_valid
            
        except yaml.YAMLError as e:
            self.critical_errors.append(f"Invalid YAML in config: {e}")
            self.print_colored(f"  ❌ Invalid YAML: {e}", 'red')
            return False
        except Exception as e:
            self.errors.append(f"Config validation error: {e}")
            self.print_colored(f"  ❌ Error: {e}", 'red')
            return False
    
    def validate_class_instantiation(self) -> bool:
        """Test if main classes can be instantiated."""
        self.print_colored("\n🔧 Validating Class Instantiation...", 'blue')
        
        all_valid = True
        
        # Test RoArmConfig
        try:
            from core.controller import RoArmConfig
            config = RoArmConfig()
            print(f"  ✅ RoArmConfig")
        except Exception as e:
            self.errors.append(f"Cannot create RoArmConfig: {e}")
            self.print_colored(f"  ❌ RoArmConfig: {e}", 'red')
            all_valid = False
        
        # Test SerialManager
        try:
            from core.serial_comm import SerialManager
            serial = SerialManager(port="/dev/null")  # Use dummy port
            print(f"  ✅ SerialManager")
        except Exception as e:
            self.errors.append(f"Cannot create SerialManager: {e}")
            self.print_colored(f"  ❌ SerialManager: {e}", 'red')
            all_valid = False
        
        # Test TrajectoryGenerator
        try:
            from motion.trajectory import TrajectoryGenerator
            traj = TrajectoryGenerator()
            print(f"  ✅ TrajectoryGenerator")
        except Exception as e:
            self.errors.append(f"Cannot create TrajectoryGenerator: {e}")
            self.print_colored(f"  ❌ TrajectoryGenerator: {e}", 'red')
            all_valid = False
        
        # Test SafetyMonitor
        try:
            from utils.safety import SafetyMonitor
            from core.constants import SERVO_LIMITS
            safety = SafetyMonitor(SERVO_LIMITS)
            print(f"  ✅ SafetyMonitor")
        except Exception as e:
            self.errors.append(f"Cannot create SafetyMonitor: {e}")
            self.print_colored(f"  ❌ SafetyMonitor: {e}", 'red')
            all_valid = False
        
        # Test Logger
        try:
            from utils.logger import setup_logger, get_logger
            setup_logger(level="INFO", log_to_file=False)
            logger = get_logger(__name__)
            print(f"  ✅ Logger System")
        except Exception as e:
            self.errors.append(f"Cannot setup logger: {e}")
            self.print_colored(f"  ❌ Logger: {e}", 'red')
            all_valid = False
        
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Check if all required packages are installed."""
        self.print_colored("\n📦 Validating Dependencies...", 'blue')
        
        required = {
            'serial': 'pyserial',
            'yaml': 'pyyaml',
            'numpy': 'numpy',
            'colorama': 'colorama'
        }
        
        all_installed = True
        for import_name, package_name in required.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✅ {package_name} ({version})")
            except ImportError:
                self.critical_errors.append(f"Missing package: {package_name}")
                self.print_colored(f"  ❌ {package_name} - NOT INSTALLED", 'red')
                self.print_colored(f"     Install with: pip install {package_name}", 'yellow')
                all_installed = False
        
        return all_installed
    
    def validate_integration(self) -> bool:
        """Test cross-module integration."""
        self.print_colored("\n🔄 Validating Cross-Module Integration...", 'blue')
        
        all_valid = True
        
        # Test 1: Controller can use Serial
        try:
            from core.controller import RoArmController, RoArmConfig
            config = RoArmConfig(port="/dev/null", auto_connect=False)
            controller = RoArmController(config)
            print(f"  ✅ Controller + Serial integration")
        except Exception as e:
            self.errors.append(f"Controller-Serial integration failed: {e}")
            self.print_colored(f"  ❌ Controller-Serial: {e}", 'red')
            all_valid = False
        
        # Test 2: Controller can use Trajectory
        try:
            from core.controller import RoArmController, RoArmConfig
            from motion.trajectory import TrajectoryType
            config = RoArmConfig(port="/dev/null", auto_connect=False)
            controller = RoArmController(config)
            # Test trajectory type is accessible
            _ = TrajectoryType.S_CURVE
            print(f"  ✅ Controller + Trajectory integration")
        except Exception as e:
            self.errors.append(f"Controller-Trajectory integration failed: {e}")
            self.print_colored(f"  ❌ Controller-Trajectory: {e}", 'red')
            all_valid = False
        
        # Test 3: Main can import everything
        try:
            # Simulate main.py imports
            from core.controller import RoArmController, RoArmConfig
            from core.constants import SERVO_LIMITS
            from patterns.scan_patterns import RasterScanPattern
            from teaching.recorder import TeachingRecorder, RecordingMode
            from calibration.calibration_suite import CalibrationSuite
            from safety.safety_system import SafetySystem
            from utils.logger import setup_logger, get_logger
            from utils.terminal import TerminalController
            print(f"  ✅ Main module imports")
        except Exception as e:
            self.critical_errors.append(f"Main module imports failed: {e}")
            self.print_colored(f"  ❌ Main imports: {e}", 'red')
            all_valid = False
        
        return all_valid
    
    def print_summary(self):
        """Print validation summary."""
        self.print_colored("\n" + "="*60, 'cyan')
        self.print_colored("📊 VALIDATION SUMMARY", 'cyan')
        self.print_colored("="*60, 'cyan')
        
        # Count results
        total_tests = len(self.successes) + len(self.errors) + len(self.warnings) + len(self.critical_errors)
        
        print(f"\n  ✅ Passed:    {len(self.successes)}")
        print(f"  ⚠️  Warnings:  {len(self.warnings)}")
        print(f"  ❌ Errors:    {len(self.errors)}")
        print(f"  🚨 Critical:  {len(self.critical_errors)}")
        print(f"  📊 Total:     {total_tests}")
        
        # Overall status
        print("\n  Overall Status: ", end='')
        if len(self.critical_errors) > 0:
            self.print_colored("❌ CRITICAL ISSUES - System will NOT run", 'red')
            print("\n  Critical issues that must be fixed:")
            for error in self.critical_errors[:5]:
                self.print_colored(f"    - {error}", 'red')
        elif len(self.errors) > 0:
            self.print_colored("⚠️ ISSUES DETECTED - System may have problems", 'yellow')
            print("\n  Issues to fix:")
            for error in self.errors[:5]:
                self.print_colored(f"    - {error}", 'yellow')
        else:
            self.print_colored("✅ READY TO RUN - System validated successfully!", 'green')
        
        # Recommendations
        if len(self.critical_errors) > 0 or len(self.errors) > 0:
            print("\n📝 Recommendations:")
            
            # Check for missing dependencies
            missing_deps = [e for e in self.critical_errors if "Missing package" in e]
            if missing_deps:
                print("\n  1. Install missing dependencies:")
                print("     pip install -r requirements.txt")
            
            # Check for missing files
            missing_files = [e for e in self.critical_errors if "Missing file" in e]
            if missing_files:
                print("\n  2. Missing files detected. Check if all files were copied correctly.")
            
            # Check for import errors
            import_errors = [e for e in self.critical_errors if "Cannot import" in e]
            if import_errors:
                print("\n  3. Import errors detected. Check:")
                print("     - Are you in the correct directory? (RoArmM3v1)")
                print("     - Is PYTHONPATH set correctly?")
                print("     - Run: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")
        
        print("\n" + "="*60)
    
    def save_report(self, filename: str = "validation_report.json"):
        """Save validation report."""
        report = {
            "successes": self.successes,
            "warnings": self.warnings,
            "errors": self.errors,
            "critical_errors": self.critical_errors,
            "summary": {
                "passed": len(self.successes),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
                "critical": len(self.critical_errors),
                "ready_to_run": len(self.critical_errors) == 0
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to {filename}")
        except Exception as e:
            print(f"\n❌ Could not save report: {e}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RoArm M3 System Validator')
    parser.add_argument('--report', help='Save report to file')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    
    args = parser.parse_args()
    
    # Run validation
    validator = SystemValidator()
    is_valid = validator.validate_all()
    
    # Save report if requested
    if args.report:
        validator.save_report(args.report)
    
    # Fix issues if requested
    if args.fix and not is_valid:
        print("\n🔧 Attempting to fix issues...")
        # Here you could add auto-fix logic
        print("   Auto-fix not yet implemented")
    
    # Return appropriate exit code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
