#!/usr/bin/env python3
"""
Test script to verify RoArm Pro installation
Run this after installation to check everything is working
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    modules = [
        'roarm_pro',
        'roarm_pro.config',
        'roarm_pro.hardware',
        'roarm_pro.motion',
        'roarm_pro.control',
        'roarm_pro.ui',
    ]
    
    failed = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_dependencies():
    """Test if required dependencies are installed"""
    print("\nTesting dependencies...")
    
    dependencies = {
        'serial': 'pyserial',
        'yaml': 'pyyaml',
        'numpy': 'numpy',
    }
    
    optional = {
        'scipy': 'scipy (optional for advanced trajectories)',
    }
    
    failed = []
    
    # Required dependencies
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Required!")
            failed.append(name)
    
    # Optional dependencies
    for module, name in optional.items():
        try:
            importlib.import_module(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - Not installed")
    
    return len(failed) == 0

def test_serial_ports():
    """Test serial port detection"""
    print("\nTesting serial port detection...")
    
    try:
        from roarm_pro.hardware import get_default_port, list_available_ports
        
        # Get default port
        default_port = get_default_port()
        print(f"  Default port: {default_port}")
        
        # List all ports
        ports = list_available_ports()
        print(f"  Found {len(ports)} serial ports:")
        
        for port in ports:
            print(f"    - {port['device']}: {port['description']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration...")
    
    try:
        from roarm_pro.config import Settings, SERVO_LIMITS, HOME_POSITION
        
        # Test settings
        settings = Settings()
        print(f"  ✅ Settings class loaded")
        print(f"     Speed factor: {settings.speed_factor}")
        print(f"     Scanner mounted: {settings.scanner_mounted}")
        
        # Test constants
        print(f"  ✅ Configuration constants loaded")
        print(f"     Servo limits defined: {len(SERVO_LIMITS)} joints")
        print(f"     Home position defined: {len(HOME_POSITION)} joints")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without hardware"""
    print("\nTesting basic functionality...")
    
    try:
        from roarm_pro.motion import TrajectoryGenerator, TrajectoryType
        from roarm_pro.motion import JointLimits
        
        # Test trajectory generation
        traj_gen = TrajectoryGenerator()
        start = {'base': 0.0, 'shoulder': 0.0}
        end = {'base': 1.0, 'shoulder': 0.5}
        
        trajectory = traj_gen.generate(start, end, 2.0, TrajectoryType.S_CURVE)
        print(f"  ✅ Trajectory generation: {len(trajectory)} points")
        
        # Test joint limits
        limits = JointLimits()
        validated = limits.validate(base=5.0, shoulder=2.0)  # Out of range
        print(f"  ✅ Joint validation working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("RoArm Pro Installation Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Serial Ports", test_serial_ports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! RoArm Pro is ready to use.")
        print("\nNext steps:")
        print("1. Connect your robot")
        print("2. Run: roarm")
        print("3. Or run: roarm --list-ports")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure proper installation: python setup.py develop")
        print("3. Check Python version (3.8+ required)")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
