#!/usr/bin/env python3
"""
RoArm M3 Connection Test Script
Simple test to verify robot connection and basic functions
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported."""
    print("\n1️⃣ Testing imports...")
    
    try:
        from core.controller import RoArmController, RoArmConfig
        print("   ✅ Core modules imported")
    except ImportError as e:
        print(f"   ❌ Failed to import core: {e}")
        return False
    
    try:
        from utils.logger import setup_logger, get_logger
        print("   ✅ Logger imported")
    except ImportError as e:
        print(f"   ❌ Failed to import logger: {e}")
        return False
    
    try:
        from motion.trajectory import TrajectoryGenerator
        print("   ✅ Motion modules imported")
    except ImportError as e:
        print(f"   ❌ Failed to import motion: {e}")
        return False
    
    return True


def test_serial_ports():
    """List available serial ports."""
    print("\n2️⃣ Checking serial ports...")
    
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("   ⚠️  No serial ports found")
            print("   Make sure the robot is connected via USB")
            return None
        
        print(f"   Found {len(ports)} port(s):")
        for port in ports:
            print(f"      📍 {port.device}: {port.description}")
            if 'usb' in port.device.lower():
                print(f"         → Likely robot port")
                return port.device
        
        return ports[0].device if ports else None
        
    except Exception as e:
        print(f"   ❌ Error checking ports: {e}")
        return None


def test_connection(port: str, baudrate: int = 115200):
    """Test connection to robot."""
    print(f"\n3️⃣ Testing connection to {port}...")
    
    try:
        from core.controller import RoArmController, RoArmConfig
        from utils.logger import setup_logger
        
        # Setup logging
        setup_logger(level="INFO")
        
        # Create config
        config = RoArmConfig(
            port=port,
            baudrate=baudrate,
            timeout=2.0,
            auto_connect=False
        )
        
        # Create controller
        print("   Creating controller...")
        controller = RoArmController(config)
        
        # Try to connect
        print("   Attempting connection...")
        if controller.connect():
            print("   ✅ Successfully connected!")
            return controller
        else:
            print("   ❌ Failed to connect")
            print("   Check:")
            print("      - Cable connection")
            print("      - Power to robot")
            print("      - Correct port")
            print("      - Port permissions")
            return None
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return None


def test_basic_functions(controller):
    """Test basic robot functions."""
    print("\n4️⃣ Testing basic functions...")
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Query status
    print("   Testing status query...", end='')
    try:
        status = controller.query_status()
        if status:
            print(" ✅")
            tests_passed += 1
        else:
            print(" ❌")
    except Exception as e:
        print(f" ❌ ({e})")
    
    # Test 2: LED control
    print("   Testing LED control...", end='')
    try:
        controller.led_control(True, brightness=128)
        time.sleep(0.5)
        controller.led_control(False)
        print(" ✅")
        tests_passed += 1
    except Exception as e:
        print(f" ❌ ({e})")
    
    # Test 3: Torque enable
    print("   Testing torque control...", end='')
    try:
        controller.set_torque(True)
        time.sleep(0.5)
        print(" ✅")
        tests_passed += 1
    except Exception as e:
        print(f" ❌ ({e})")
    
    # Test 4: Small movement
    print("   Testing small movement...", end='')
    try:
        # Get current position
        if hasattr(controller, 'current_position'):
            pos = controller.current_position.copy()
            # Small base movement
            pos['base'] = 0.1
            controller.move_joints(pos, speed=0.5, wait=True)
            time.sleep(1)
            # Return
            pos['base'] = 0.0
            controller.move_joints(pos, speed=0.5, wait=True)
            print(" ✅")
            tests_passed += 1
        else:
            print(" ⏭️ (skipped)")
    except Exception as e:
        print(f" ❌ ({e})")
    
    # Test 5: Gripper
    print("   Testing gripper...", end='')
    try:
        controller.gripper_control(0.5)  # Half open
        time.sleep(0.5)
        controller.gripper_control(1.0)  # Close
        print(" ✅")
        tests_passed += 1
    except Exception as e:
        print(f" ❌ ({e})")
    
    print(f"\n   Results: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def interactive_test(controller):
    """Interactive test mode."""
    print("\n5️⃣ Interactive test mode")
    print("   Commands:")
    print("      h - Home position")
    print("      l - LED on/off")
    print("      g - Gripper open/close")
    print("      s - Status")
    print("      q - Quit")
    
    led_state = False
    gripper_state = False
    
    while True:
        cmd = input("\n   Command: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'h':
            print("   Moving to home...")
            controller.move_home(speed=0.5)
            print("   Done")
        elif cmd == 'l':
            led_state = not led_state
            controller.led_control(led_state, brightness=255 if led_state else 0)
            print(f"   LED {'ON' if led_state else 'OFF'}")
        elif cmd == 'g':
            gripper_state = not gripper_state
            controller.gripper_control(0.0 if gripper_state else 1.0)
            print(f"   Gripper {'OPEN' if gripper_state else 'CLOSED'}")
        elif cmd == 's':
            status = controller.query_status()
            if status:
                print("   Status:")
                for key, value in status.items():
                    print(f"      {key}: {value}")
            else:
                print("   Failed to get status")
        else:
            print("   Unknown command")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='RoArm M3 Connection Test')
    parser.add_argument('--port', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🤖 RoArm M3 Connection Test")
    print("="*50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check your installation.")
        return 1
    
    # Find port if not specified
    if not args.port:
        args.port = test_serial_ports()
        if not args.port:
            print("\n❌ No serial port found or specified.")
            print("   Use --port to specify the port manually.")
            return 1
    
    # Test connection
    controller = test_connection(args.port, args.baudrate)
    if not controller:
        print("\n❌ Connection test failed.")
        return 1
    
    # Test basic functions
    if test_basic_functions(controller):
        print("\n✅ All basic tests passed!")
    else:
        print("\n⚠️  Some tests failed, but connection works.")
    
    # Interactive mode
    if args.interactive:
        interactive_test(controller)
    
    # Cleanup
    print("\n🔌 Disconnecting...")
    controller.disconnect()
    
    print("\n✅ Test complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⛔ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
