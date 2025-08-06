#!/usr/bin/env python3
"""
RoArm M3 Debug and Diagnostic Tool
Comprehensive system testing and troubleshooting
"""

import sys
import os
import time
import json
import traceback
import platform
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    test_name: str
    category: str
    status: str  # 'pass', 'fail', 'warning', 'skip'
    message: str
    details: Optional[Dict] = None
    error: Optional[str] = None
    duration: float = 0.0


class SystemDebugger:
    """
    Comprehensive debugging and diagnostic tool for RoArm M3.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize debugger.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results = []
        self.start_time = time.time()
        
        # Color codes for terminal output
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
        
        # Status symbols
        self.symbols = {
            'pass': '✅',
            'fail': '❌',
            'warning': '⚠️',
            'skip': '⏭️',
            'info': 'ℹ️'
        }
    
    def print_colored(self, text: str, color: str = 'white', bold: bool = False):
        """Print colored text."""
        if not self.verbose:
            return
        
        prefix = self.colors['bold'] if bold else ''
        print(f"{prefix}{self.colors.get(color, '')}{text}{self.colors['reset']}")
    
    def print_header(self):
        """Print debug session header."""
        self.print_colored("\n" + "="*60, 'cyan', bold=True)
        self.print_colored("🔍 RoArm M3 System Debugger & Diagnostics", 'cyan', bold=True)
        self.print_colored("="*60, 'cyan', bold=True)
        self.print_colored(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'white')
        self.print_colored(f"Platform: {platform.platform()}", 'white')
        self.print_colored(f"Python: {sys.version.split()[0]}", 'white')
        self.print_colored("="*60 + "\n", 'cyan', bold=True)
    
    def run_all_diagnostics(self) -> bool:
        """
        Run all diagnostic tests.
        
        Returns:
            True if all critical tests pass
        """
        self.print_header()
        
        # Run test categories
        self.test_python_environment()
        self.test_project_structure()
        self.test_dependencies()
        self.test_imports()
        self.test_configuration()
        self.test_serial_ports()
        self.test_permissions()
        
        # Print summary
        self.print_summary()
        
        # Save report
        self.save_report()
        
        # Return overall status
        critical_failures = [r for r in self.results if r.status == 'fail' and r.category in ['environment', 'imports', 'structure']]
        return len(critical_failures) == 0
    
    def test_python_environment(self):
        """Test Python environment."""
        self.print_colored("\n📦 Testing Python Environment...", 'blue', bold=True)
        
        # Python version
        start = time.time()
        try:
            version = sys.version_info
            if version.major == 3 and version.minor >= 7:
                self.add_result(
                    "Python Version",
                    "environment",
                    "pass",
                    f"Python {version.major}.{version.minor}.{version.micro}",
                    {"version": f"{version.major}.{version.minor}.{version.micro}"}
                )
            else:
                self.add_result(
                    "Python Version",
                    "environment",
                    "warning",
                    f"Python {version.major}.{version.minor} (3.7+ recommended)",
                    {"version": f"{version.major}.{version.minor}.{version.micro}"}
                )
        except Exception as e:
            self.add_result("Python Version", "environment", "fail", "Failed to check", error=str(e))
        
        # Virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.add_result("Virtual Environment", "environment", "pass", "Active")
        else:
            self.add_result("Virtual Environment", "environment", "warning", "Not in virtual environment")
        
        # Platform
        self.add_result(
            "Platform",
            "environment",
            "pass",
            platform.system(),
            {"platform": platform.platform(), "machine": platform.machine()}
        )
    
    def test_project_structure(self):
        """Test project directory structure."""
        self.print_colored("\n📁 Testing Project Structure...", 'blue', bold=True)
        
        required_dirs = [
            "core",
            "motion",
            "patterns", 
            "teaching",
            "calibration",
            "safety",
            "utils",
            "sequences",
            "logs"
        ]
        
        required_files = [
            "main.py",
            "config.yaml",
            "core/controller.py",
            "core/serial_comm.py",
            "core/constants.py",
            "motion/trajectory.py",
            "patterns/scan_patterns.py",
            "teaching/recorder.py",
            "calibration/calibration_suite.py",
            "safety/safety_system.py",
            "utils/logger.py",
            "utils/terminal.py",
            "utils/safety.py"
        ]
        
        base_path = Path(__file__).parent.parent
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.add_result(f"Directory: {dir_name}", "structure", "pass", "Found")
            else:
                self.add_result(f"Directory: {dir_name}", "structure", "fail", "Missing")
                # Try to create missing directory
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.add_result(f"Create: {dir_name}", "structure", "pass", "Created missing directory")
                except Exception as e:
                    self.add_result(f"Create: {dir_name}", "structure", "fail", f"Could not create: {e}")
        
        # Check files
        for file_name in required_files:
            file_path = base_path / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                self.add_result(
                    f"File: {file_name}",
                    "structure",
                    "pass",
                    f"Found ({size} bytes)",
                    {"size": size}
                )
            else:
                self.add_result(f"File: {file_name}", "structure", "fail", "Missing")
        
        # Check __init__.py files
        for dir_name in ["core", "motion", "patterns", "teaching", "calibration", "safety", "utils"]:
            init_file = base_path / dir_name / "__init__.py"
            if not init_file.exists():
                # Create empty __init__.py
                try:
                    init_file.write_text("")
                    self.add_result(f"Init: {dir_name}/__init__.py", "structure", "pass", "Created")
                except Exception as e:
                    self.add_result(f"Init: {dir_name}/__init__.py", "structure", "warning", f"Could not create: {e}")
    
    def test_dependencies(self):
        """Test required Python packages."""
        self.print_colored("\n📚 Testing Dependencies...", 'blue', bold=True)
        
        required_packages = {
            'serial': 'pyserial',
            'yaml': 'pyyaml',
            'numpy': 'numpy',
            'colorama': 'colorama'
        }
        
        optional_packages = {
            'matplotlib': 'matplotlib',
            'pandas': 'pandas'
        }
        
        # Test required packages
        for import_name, package_name in required_packages.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                self.add_result(
                    f"Package: {package_name}",
                    "dependencies",
                    "pass",
                    f"Version {version}",
                    {"version": version}
                )
            except ImportError:
                self.add_result(
                    f"Package: {package_name}",
                    "dependencies",
                    "fail",
                    "Not installed - run: pip install " + package_name
                )
        
        # Test optional packages
        for import_name, package_name in optional_packages.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                self.add_result(
                    f"Optional: {package_name}",
                    "dependencies",
                    "pass",
                    f"Version {version}",
                    {"version": version}
                )
            except ImportError:
                self.add_result(
                    f"Optional: {package_name}",
                    "dependencies",
                    "warning",
                    "Not installed (optional)"
                )
    
    def test_imports(self):
        """Test project imports."""
        self.print_colored("\n🔗 Testing Module Imports...", 'blue', bold=True)
        
        test_imports = [
            "core.controller",
            "core.serial_comm",
            "core.constants",
            "motion.trajectory",
            "patterns.scan_patterns",
            "teaching.recorder",
            "calibration.calibration_suite",
            "safety.safety_system",
            "utils.logger",
            "utils.terminal",
            "utils.safety"
        ]
        
        for module_name in test_imports:
            try:
                module = importlib.import_module(module_name)
                self.add_result(f"Import: {module_name}", "imports", "pass", "Success")
            except ImportError as e:
                self.add_result(f"Import: {module_name}", "imports", "fail", f"Import error: {e}")
            except Exception as e:
                self.add_result(f"Import: {module_name}", "imports", "fail", f"Error: {e}")
    
    def test_configuration(self):
        """Test configuration file."""
        self.print_colored("\n⚙️ Testing Configuration...", 'blue', bold=True)
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            self.add_result("Config File", "configuration", "fail", "config.yaml not found")
            return
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.add_result("Config File", "configuration", "pass", "Loaded successfully")
            
            # Check required sections
            required_sections = ['system', 'hardware', 'scanner', 'motion', 'safety']
            for section in required_sections:
                if section in config:
                    self.add_result(f"Config Section: {section}", "configuration", "pass", "Present")
                else:
                    self.add_result(f"Config Section: {section}", "configuration", "warning", "Missing")
            
            # Check port configuration
            if 'system' in config and 'port' in config['system']:
                port = config['system']['port']
                self.add_result("Serial Port", "configuration", "pass", f"Configured: {port}")
            else:
                self.add_result("Serial Port", "configuration", "fail", "Not configured")
                
        except yaml.YAMLError as e:
            self.add_result("Config File", "configuration", "fail", f"Invalid YAML: {e}")
        except Exception as e:
            self.add_result("Config File", "configuration", "fail", f"Error: {e}")
    
    def test_serial_ports(self):
        """Test available serial ports."""
        self.print_colored("\n🔌 Testing Serial Ports...", 'blue', bold=True)
        
        try:
            import serial.tools.list_ports
            
            ports = list(serial.tools.list_ports.comports())
            
            if not ports:
                self.add_result("Serial Ports", "hardware", "warning", "No serial ports found")
            else:
                self.add_result(
                    "Serial Ports",
                    "hardware",
                    "pass",
                    f"Found {len(ports)} port(s)",
                    {"count": len(ports)}
                )
                
                for port in ports:
                    self.add_result(
                        f"Port: {port.device}",
                        "hardware",
                        "pass",
                        port.description or "Unknown device",
                        {
                            "device": port.device,
                            "description": port.description,
                            "hwid": port.hwid
                        }
                    )
                    
                    # Check if it's likely the RoArm
                    if 'usb' in port.device.lower() or 'serial' in port.device.lower():
                        self.add_result(
                            f"Potential RoArm",
                            "hardware",
                            "pass",
                            f"Found at {port.device}"
                        )
                        
        except ImportError:
            self.add_result("Serial Ports", "hardware", "fail", "pyserial not installed")
        except Exception as e:
            self.add_result("Serial Ports", "hardware", "fail", f"Error: {e}")
    
    def test_permissions(self):
        """Test file and device permissions."""
        self.print_colored("\n🔐 Testing Permissions...", 'blue', bold=True)
        
        # Test write permissions in project directory
        try:
            test_file = Path(__file__).parent.parent / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            self.add_result("Write Permission", "permissions", "pass", "Can write to project directory")
        except Exception as e:
            self.add_result("Write Permission", "permissions", "fail", f"Cannot write: {e}")
        
        # Test logs directory
        logs_dir = Path(__file__).parent.parent / "logs"
        try:
            logs_dir.mkdir(exist_ok=True)
            test_log = logs_dir / "test.log"
            test_log.write_text("test")
            test_log.unlink()
            self.add_result("Logs Directory", "permissions", "pass", "Can write to logs")
        except Exception as e:
            self.add_result("Logs Directory", "permissions", "warning", f"Cannot write to logs: {e}")
        
        # Check serial port permissions (macOS specific)
        if platform.system() == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.run(['ls', '-la', '/dev/tty.*'], capture_output=True, text=True)
                if 'usbserial' in result.stdout:
                    self.add_result("Serial Device", "permissions", "pass", "USB serial devices found")
                else:
                    self.add_result("Serial Device", "permissions", "warning", "No USB serial devices found")
            except Exception as e:
                self.add_result("Serial Device", "permissions", "skip", f"Could not check: {e}")
    
    def add_result(self, test_name: str, category: str, status: str, 
                  message: str, details: Optional[Dict] = None, error: Optional[str] = None):
        """Add a diagnostic result."""
        result = DiagnosticResult(
            test_name=test_name,
            category=category,
            status=status,
            message=message,
            details=details,
            error=error,
            duration=time.time() - self.start_time
        )
        self.results.append(result)
        
        # Print result
        if self.verbose:
            symbol = self.symbols.get(status, '?')
            color = {
                'pass': 'green',
                'fail': 'red',
                'warning': 'yellow',
                'skip': 'cyan'
            }.get(status, 'white')
            
            print(f"  {symbol} {test_name}: ", end='')
            self.print_colored(message, color)
            
            if error and self.verbose:
                self.print_colored(f"     Error: {error}", 'red')
    
    def print_summary(self):
        """Print diagnostic summary."""
        self.print_colored("\n" + "="*60, 'cyan', bold=True)
        self.print_colored("📊 Diagnostic Summary", 'cyan', bold=True)
        self.print_colored("="*60, 'cyan', bold=True)
        
        # Count results by status
        counts = {'pass': 0, 'fail': 0, 'warning': 0, 'skip': 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        
        # Print counts
        print(f"\n  {self.symbols['pass']} Passed:  {counts['pass']}")
        print(f"  {self.symbols['fail']} Failed:  {counts['fail']}")
        print(f"  {self.symbols['warning']} Warnings: {counts['warning']}")
        print(f"  {self.symbols['skip']} Skipped: {counts['skip']}")
        
        # Overall status
        print("\n  Overall Status: ", end='')
        if counts['fail'] == 0:
            self.print_colored("✅ READY", 'green', bold=True)
            print("\n  The system appears to be properly configured.")
        elif counts['fail'] < 3:
            self.print_colored("⚠️ MINOR ISSUES", 'yellow', bold=True)
            print("\n  Some issues need attention but system may work.")
        else:
            self.print_colored("❌ CRITICAL ISSUES", 'red', bold=True)
            print("\n  Critical issues must be resolved before running.")
        
        # Print critical failures
        failures = [r for r in self.results if r.status == 'fail']
        if failures:
            self.print_colored("\n❌ Failed Tests:", 'red', bold=True)
            for failure in failures[:5]:  # Show first 5
                print(f"  - {failure.test_name}: {failure.message}")
        
        # Print warnings
        warnings = [r for r in self.results if r.status == 'warning']
        if warnings:
            self.print_colored("\n⚠️ Warnings:", 'yellow', bold=True)
            for warning in warnings[:5]:  # Show first 5
                print(f"  - {warning.test_name}: {warning.message}")
        
        print("\n" + "="*60 + "\n")
    
    def save_report(self, filename: Optional[str] = None):
        """Save diagnostic report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"
        
        report_dir = Path(__file__).parent.parent / "logs"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / filename
        
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.platform(),
                "python_version": sys.version,
                "duration": time.time() - self.start_time,
                "results": [asdict(r) for r in self.results],
                "summary": {
                    "total": len(self.results),
                    "passed": len([r for r in self.results if r.status == 'pass']),
                    "failed": len([r for r in self.results if r.status == 'fail']),
                    "warnings": len([r for r in self.results if r.status == 'warning']),
                    "skipped": len([r for r in self.results if r.status == 'skip'])
                }
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.print_colored(f"📄 Report saved to: {report_path}", 'green')
            
        except Exception as e:
            self.print_colored(f"Failed to save report: {e}", 'red')
    
    def quick_test(self) -> bool:
        """Run quick essential tests only."""
        self.print_colored("\n⚡ Running Quick Diagnostics...", 'cyan', bold=True)
        
        # Only test essentials
        self.test_python_environment()
        self.test_imports()
        self.test_serial_ports()
        
        # Check for critical failures
        failures = [r for r in self.results if r.status == 'fail']
        
        if not failures:
            self.print_colored("\n✅ Quick test PASSED", 'green', bold=True)
            return True
        else:
            self.print_colored(f"\n❌ Quick test FAILED ({len(failures)} issues)", 'red', bold=True)
            return False


def main():
    """Run debugger as standalone tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RoArm M3 System Debugger')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--report', help='Save report to specific file')
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = SystemDebugger(verbose=not args.quiet)
    
    # Run diagnostics
    if args.quick:
        success = debugger.quick_test()
    else:
        success = debugger.run_all_diagnostics()
    
    # Save report if requested
    if args.report:
        debugger.save_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
