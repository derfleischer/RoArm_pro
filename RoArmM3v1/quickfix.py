#!/usr/bin/env python3
"""
RoArm M3 QuickFix Script
Behebt automatisch häufige Probleme
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil

class QuickFix:
    """Automatic problem fixer for RoArm M3 system."""
    
    def __init__(self):
        self.fixes_applied = []
        self.fixes_failed = []
        
    def print_colored(self, text: str, color: str = 'white'):
        """Print colored text."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'reset': '\033[0m'
        }
        print(f"{colors.get(color, '')}{text}{colors['reset']}")
    
    def run_all_fixes(self):
        """Run all automatic fixes."""
        self.print_colored("\n🔧 RoArm M3 QuickFix Tool", 'cyan')
        self.print_colored("="*50, 'cyan')
        
        # 1. Create missing directories
        self.fix_directories()
        
        # 2. Create missing __init__.py files
        self.fix_init_files()
        
        # 3. Install missing dependencies
        self.fix_dependencies()
        
        # 4. Fix Python path
        self.fix_python_path()
        
        # 5. Fix file permissions
        self.fix_permissions()
        
        # 6. Create default config if missing
        self.fix_config()
        
        # Print summary
        self.print_summary()
    
    def fix_directories(self):
        """Create missing directories."""
        self.print_colored("\n📁 Fixing directory structure...", 'blue')
        
        required_dirs = [
            "core",
            "motion",
            "patterns",
            "teaching",
            "calibration",
            "safety",
            "utils",
            "sequences",
            "logs",
            "logs/debug",
            "calibration/data"
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.fixes_applied.append(f"Created directory: {dir_name}")
                    print(f"  ✅ Created: {dir_name}")
                except Exception as e:
                    self.fixes_failed.append(f"Could not create {dir_name}: {e}")
                    self.print_colored(f"  ❌ Failed: {dir_name} - {e}", 'red')
            else:
                print(f"  ✓ Exists: {dir_name}")
    
    def fix_init_files(self):
        """Create missing __init__.py files."""
        self.print_colored("\n📄 Fixing __init__.py files...", 'blue')
        
        init_contents = {
            "core": '''"""RoArm M3 Core Module"""
from .controller import RoArmController, RoArmConfig
from .serial_comm import SerialManager
from .constants import *

__all__ = ['RoArmController', 'RoArmConfig', 'SerialManager']
''',
            "motion": '''"""RoArm M3 Motion Module"""
from .trajectory import TrajectoryGenerator, TrajectoryType, TrajectoryPoint

__all__ = ['TrajectoryGenerator', 'TrajectoryType', 'TrajectoryPoint']
''',
            "patterns": '''"""RoArm M3 Patterns Module"""
from .scan_patterns import *

__all__ = ['ScanPattern', 'RasterScanPattern', 'SpiralScanPattern', 
           'SphericalScanPattern', 'TurntableScanPattern', 'CobwebScanPattern']
''',
            "teaching": '''"""RoArm M3 Teaching Module"""
from .recorder import TeachingRecorder, TeachingWaypoint, TeachingSequence, RecordingMode

__all__ = ['TeachingRecorder', 'TeachingWaypoint', 'TeachingSequence', 'RecordingMode']
''',
            "calibration": '''"""RoArm M3 Calibration Module"""
from .calibration_suite import *

__all__ = ['CalibrationSuite', 'CalibrationType', 'SystemCalibration']
''',
            "safety": '''"""RoArm M3 Safety Module"""
from .safety_system import *

__all__ = ['SafetySystem', 'SafetyState', 'ShutdownReason']
''',
            "utils": '''"""RoArm M3 Utils Module"""
from .logger import setup_logger, get_logger
from .terminal import TerminalController
from .safety import SafetyMonitor

__all__ = ['setup_logger', 'get_logger', 'TerminalController', 'SafetyMonitor']
'''
        }
        
        for module, content in init_contents.items():
            init_file = Path(module) / "__init__.py"
            if not init_file.exists():
                try:
                    init_file.parent.mkdir(parents=True, exist_ok=True)
                    init_file.write_text(content)
                    self.fixes_applied.append(f"Created {module}/__init__.py")
                    print(f"  ✅ Created: {module}/__init__.py")
                except Exception as e:
                    self.fixes_failed.append(f"Could not create {module}/__init__.py: {e}")
                    self.print_colored(f"  ❌ Failed: {module}/__init__.py - {e}", 'red')
            else:
                print(f"  ✓ Exists: {module}/__init__.py")
    
    def fix_dependencies(self):
        """Install missing Python packages."""
        self.print_colored("\n📦 Checking dependencies...", 'blue')
        
        required_packages = ['pyserial', 'pyyaml', 'numpy', 'colorama']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✓ {package} installed")
            except ImportError:
                missing.append(package)
                self.print_colored(f"  ❌ {package} missing", 'yellow')
        
        if missing:
            self.print_colored("\nInstalling missing packages...", 'blue')
            try:
                cmd = [sys.executable, '-m', 'pip', 'install'] + missing
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.fixes_applied.append(f"Installed packages: {', '.join(missing)}")
                    self.print_colored("  ✅ Packages installed successfully", 'green')
                else:
                    self.fixes_failed.append(f"Failed to install packages: {result.stderr}")
                    self.print_colored(f"  ❌ Installation failed: {result.stderr}", 'red')
            except Exception as e:
                self.fixes_failed.append(f"Could not install packages: {e}")
                self.print_colored(f"  ❌ Error: {e}", 'red')
    
    def fix_python_path(self):
        """Create a script to set PYTHONPATH."""
        self.print_colored("\n🐍 Creating Python path setup...", 'blue')
        
        setup_script = Path("setup_env.sh")
        content = '''#!/bin/bash
# Set Python path for RoArm M3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ Python path set for RoArm M3"
echo "   PYTHONPATH=$PYTHONPATH"
'''
        
        try:
            setup_script.write_text(content)
            setup_script.chmod(0o755)
            self.fixes_applied.append("Created setup_env.sh script")
            print("  ✅ Created setup_env.sh")
            print("     Run: source setup_env.sh")
        except Exception as e:
            self.fixes_failed.append(f"Could not create setup script: {e}")
            self.print_colored(f"  ❌ Failed: {e}", 'red')
    
    def fix_permissions(self):
        """Fix file permissions."""
        self.print_colored("\n🔐 Fixing file permissions...", 'blue')
        
        # Make Python files executable
        python_files = list(Path('.').glob('**/*.py'))
        for file in python_files:
            try:
                file.chmod(0o644)
            except:
                pass
        
        # Make main scripts executable
        scripts = ['main.py', 'test_connection.py', 'validate_system.py', 'quickfix.py']
        for script in scripts:
            script_path = Path(script)
            if script_path.exists():
                try:
                    script_path.chmod(0o755)
                    print(f"  ✓ Made executable: {script}")
                except Exception as e:
                    self.print_colored(f"  ❌ Could not chmod {script}: {e}", 'red')
        
        # Make shell scripts executable
        for shell_script in Path('.').glob('*.sh'):
            try:
                shell_script.chmod(0o755)
                print(f"  ✓ Made executable: {shell_script.name}")
            except:
                pass
    
    def fix_config(self):
        """Create default config if missing."""
        self.print_colored("\n⚙️ Checking configuration...", 'blue')
        
        config_file = Path("config.yaml")
        if not config_file.exists():
            self.print_colored("  Creating default config.yaml...", 'yellow')
            
            default_config = '''# RoArm M3 Configuration
system:
  port: "/dev/tty.usbserial-110"  # Update this!
  baudrate: 115200
  timeout: 2.0
  debug: false
  auto_connect: true

hardware:
  servo_limits:
    base: [-3.14, 3.14]
    shoulder: [-1.57, 1.57]
    elbow: [0.0, 3.14]
    wrist: [-1.57, 1.57]
    roll: [-3.14, 3.14]
    hand: [1.08, 3.14]
  
  home_position:
    base: 0.0
    shoulder: 0.0
    elbow: 1.57
    wrist: 0.0
    roll: 0.0
    hand: 3.14
  
  scanner_position:
    base: 0.0
    shoulder: 0.35
    elbow: 1.22
    wrist: -1.57
    roll: 1.57
    hand: 2.5

scanner:
  model: "Revopoint Mini2"
  weight: 0.2
  optimal_distance: 0.15

motion:
  default_speed: 1.0
  default_acceleration: 2.0
  default_jerk: 5.0

safety:
  emergency_deceleration: 10.0
  temperature_warning: 50
  temperature_critical: 60
  voltage_min: 5.5
  voltage_max: 7.0

teaching:
  sample_rate: 50
  save_directory: "sequences"

logging:
  level: "INFO"
  file: "logs/roarm_system.log"
'''
            
            try:
                config_file.write_text(default_config)
                self.fixes_applied.append("Created default config.yaml")
                self.print_colored("  ✅ Created config.yaml", 'green')
                self.print_colored("  ⚠️  Remember to update the serial port!", 'yellow')
            except Exception as e:
                self.fixes_failed.append(f"Could not create config: {e}")
                self.print_colored(f"  ❌ Failed: {e}", 'red')
        else:
            print("  ✓ config.yaml exists")
    
    def print_summary(self):
        """Print fix summary."""
        self.print_colored("\n" + "="*50, 'cyan')
        self.print_colored("📊 QUICKFIX SUMMARY", 'cyan')
        self.print_colored("="*50, 'cyan')
        
        if self.fixes_applied:
            self.print_colored(f"\n✅ Fixes Applied: {len(self.fixes_applied)}", 'green')
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        
        if self.fixes_failed:
            self.print_colored(f"\n❌ Fixes Failed: {len(self.fixes_failed)}", 'red')
            for fail in self.fixes_failed:
                print(f"  • {fail}")
        
        if not self.fixes_applied and not self.fixes_failed:
            self.print_colored("\n✅ No fixes needed - system appears ready!", 'green')
        
        # Next steps
        self.print_colored("\n📝 Next Steps:", 'cyan')
        print("  1. Run validation: python3 validate_system.py")
        print("  2. Update config.yaml with your serial port")
        print("  3. Test connection: python3 test_connection.py")
        print("  4. Run main program: python3 main.py")
        
        print("\n" + "="*50)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RoArm M3 QuickFix Tool')
    parser.add_argument('--check-only', action='store_true', help='Only check, don\'t fix')
    
    args = parser.parse_args()
    
    if args.check_only:
        print("Check-only mode not yet implemented")
        print("Run validate_system.py instead")
    else:
        fixer = QuickFix()
        fixer.run_all_fixes()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
