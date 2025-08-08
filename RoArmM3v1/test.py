#!/usr/bin/env python3
"""
Quick test for fixed RoArm controller
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.controller import RoArmController, RoArmConfig

def test_connection():
    """Test basic connection and commands."""
    print("="*60)
    print("ROARM M3 - TESTING FIXED CONTROLLER")
    print("="*60)
    
    # Create controller
    config = RoArmConfig(
        port="/dev/tty.usbserial-110",
        baudrate=115200,
        debug=True
    )
    
    controller = RoArmController(config)
    
    print("\n1. Testing connection...")
    if not controller.connect():
        print("❌ Connection failed!")
        return False
    
    print("✅ Connected!")
    
    # Test LED
    print("\n2. Testing LED control...")
    controller.led_control(True, 255)
    time.sleep(0.5)
    controller.led_control(False)
    time.sleep(0.5)
    controller.led_control(True, 128)
    time.sleep(0.5)
    controller.led_control(False)
    
    led_worked = input("Did LED blink? (y/n): ").lower() == 'y'
    if led_worked:
        print("✅ LED control works!")
    else:
        print("⚠️ LED control may not work")
    
    # Test movement
    print("\n3. Testing movement...")
    print("⚠️ Robot will move - ensure clear space!")
    input("Press ENTER when ready...")
    
    # Small movement
    print("Moving base +0.3 rad...")
    success = controller.move_joints({"base": 0.3}, speed=0.5)
    
    if success:
        print("✅ Movement command sent")
        time.sleep(2)
        
        print("Moving back to center...")
        controller.move_joints({"base": 0.0}, speed=0.5)
        time.sleep(2)
        
        move_worked = input("Did robot move? (y/n): ").lower() == 'y'
        if move_worked:
            print("✅ Movement works!")
        else:
            print("❌ Movement command sent but robot didn't move")
            print("   Try command IDs 102 instead of 104 in controller.py")
    else:
        print("❌ Movement command failed")
    
    # Test home position
    print("\n4. Testing home position...")
    confirm = input("Move to home position? (y/n): ").lower()
    if confirm == 'y':
        controller.move_home(speed=0.5)
        print("✅ Home position command sent")
    
    # Disconnect
    print("\n5. Disconnecting...")
    controller.disconnect()
    print("✅ Test complete!")
    
    return True

if __name__ == "__main__":
    try:
        test_connection()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
