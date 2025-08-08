#!/usr/bin/env python3
"""
RoArm M3 Debug Tool
Test all functions step by step
"""

import sys
import os
import time
import logging
import math

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import controller
from core.controller import RoArmController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Debug')


class RoArmDebugger:
    """Debug tool for RoArm."""
    
    def __init__(self):
        self.controller = None
        
    def run(self):
        """Run debug tool."""
        print("\n" + "="*60)
        print("RoArm M3 DEBUG TOOL")
        print("="*60)
        
        while True:
            print("\n=== DEBUG MENU ===")
            print("1. Connect to robot")
            print("2. Test LED")
            print("3. Test single joint")
            print("4. Test all joints")
            print("5. Test gripper")
            print("6. Test home position")
            print("7. Test scanner position")
            print("8. Test emergency stop")
            print("9. Query status")
            print("10. Custom movement")
            print("11. Test sequence")
            print("12. Monitor serial data")
            print("0. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.test_connection()
            elif choice == '2':
                self.test_led()
            elif choice == '3':
                self.test_single_joint()
            elif choice == '4':
                self.test_all_joints()
            elif choice == '5':
                self.test_gripper()
            elif choice == '6':
                self.test_home()
            elif choice == '7':
                self.test_scanner_position()
            elif choice == '8':
                self.test_emergency()
            elif choice == '9':
                self.query_status()
            elif choice == '10':
                self.custom_movement()
            elif choice == '11':
                self.test_sequence()
            elif choice == '12':
                self.monitor_serial()
            else:
                print("Invalid option")
    
    def test_connection(self):
        """Test connection."""
        print("\n=== CONNECTION TEST ===")
        
        port = input("Port [/dev/tty.usbserial-110 or 'auto']: ").strip()
        if not port:
            port = "/dev/tty.usbserial-110"
        
        print(f"Connecting to {port}...")
        
        self.controller = RoArmController(port=port)
        
        if self.controller.connect():
            print("✅ Connection successful!")
            
            # Test basic communication
            print("\nTesting communication...")
            
            # LED test
            print("LED ON...")
            self.controller.led_control(True, 200)
            time.sleep(1)
            
            print("LED OFF...")
            self.controller.led_control(False)
            
            print("✅ Communication test passed!")
        else:
            print("❌ Connection failed!")
    
    def test_led(self):
        """Test LED control."""
        if not self.check_connection():
            return
        
        print("\n=== LED TEST ===")
        
        # Test different brightness levels
        for brightness in [50, 100, 150, 200, 255]:
            print(f"Brightness: {brightness}")
            self.controller.led_control(True, brightness)
            time.sleep(0.5)
        
        # Blink test
        print("Blinking...")
        for _ in range(5):
            self.controller.led_control(True, 255)
            time.sleep(0.2)
            self.controller.led_control(False)
            time.sleep(0.2)
        
        print("✅ LED test complete")
    
    def test_single_joint(self):
        """Test single joint movement."""
        if not self.check_connection():
            return
        
        print("\n=== SINGLE JOINT TEST ===")
        print("Joints: base, shoulder, elbow, wrist, roll, hand")
        
        joint = input("Select joint: ").strip()
        if joint not in ["base", "shoulder", "elbow", "wrist", "roll", "hand"]:
            print("Invalid joint")
            return
        
        # Get angle
        angle_deg = float(input(f"Angle in degrees (current limits apply): "))
        angle_rad = math.radians(angle_deg)
        
        print(f"Moving {joint} to {angle_deg}°...")
        
        success = self.controller.move_joints({joint: angle_rad}, speed=0.5)
        
        if success:
            print("✅ Movement successful")
            time.sleep(2)
            
            # Return to center
            print("Returning to center...")
            self.controller.move_joints({joint: 0.0}, speed=0.5)
        else:
            print("❌ Movement failed")
    
    def test_all_joints(self):
        """Test all joints."""
        if not self.check_connection():
            return
        
        print("\n=== ALL JOINTS TEST ===")
        print("Testing each joint with small movements...")
        
        joints = ["base", "shoulder", "elbow", "wrist", "roll"]
        
        for joint in joints:
            print(f"\nTesting {joint}...")
            
            # Small positive movement
            print(f"  Moving {joint} +20°")
            self.controller.move_joints({joint: math.radians(20)}, speed=0.5)
            time.sleep(1)
            
            # Small negative movement
            print(f"  Moving {joint} -20°")
            self.controller.move_joints({joint: math.radians(-20)}, speed=0.5)
            time.sleep(1)
            
            # Return to center
            print(f"  Centering {joint}")
            self.controller.move_joints({joint: 0.0}, speed=0.5)
            time.sleep(1)
        
        print("✅ All joints test complete")
    
    def test_gripper(self):
        """Test gripper."""
        if not self.check_connection():
            return
        
        print("\n=== GRIPPER TEST ===")
        
        print("Opening gripper...")
        self.controller.gripper_control(0.0)
        time.sleep(2)
        
        print("Closing gripper...")
        self.controller.gripper_control(1.0)
        time.sleep(2)
        
        print("Half open...")
        self.controller.gripper_control(0.5)
        time.sleep(2)
        
        print("✅ Gripper test complete")
    
    def test_home(self):
        """Test home position."""
        if not self.check_connection():
            return
        
        print("\n=== HOME POSITION TEST ===")
        print("Moving to home position...")
        
        success = self.controller.move_home(speed=0.5)
        
        if success:
            print("✅ Home position reached")
        else:
            print("❌ Failed to reach home")
    
    def test_scanner_position(self):
        """Test scanner position."""
        if not self.check_connection():
            return
        
        print("\n=== SCANNER POSITION TEST ===")
        print("Moving to scanner position...")
        
        success = self.controller.move_to_scanner_position(speed=0.5)
        
        if success:
            print("✅ Scanner position reached")
            time.sleep(3)
            print("Returning home...")
            self.controller.move_home(speed=0.5)
        else:
            print("❌ Failed to reach scanner position")
    
    def test_emergency(self):
        """Test emergency stop."""
        if not self.check_connection():
            return
        
        print("\n=== EMERGENCY STOP TEST ===")
        print("⚠️ This will trigger emergency stop!")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        print("Starting movement...")
        self.controller.move_joints({"base": 1.0}, speed=0.2, wait=False)
        
        time.sleep(1)
        
        print("EMERGENCY STOP!")
        self.controller.emergency_stop()
        
        time.sleep(2)
        
        print("Resetting emergency...")
        self.controller.reset_emergency()
        
        print("Re-enabling torque...")
        self.controller.set_torque(True)
        
        print("✅ Emergency stop test complete")
    
    def query_status(self):
        """Query robot status."""
        if not self.check_connection():
            return
        
        print("\n=== STATUS QUERY ===")
        
        status = self.controller.query_status()
        
        if status:
            print("Current status:")
            print(f"  Connected: {status['connected']}")
            print(f"  Torque enabled: {status['torque_enabled']}")
            print("  Positions:")
            for joint, angle in status['positions'].items():
                print(f"    {joint}: {math.degrees(angle):.1f}°")
        else:
            print("❌ Failed to query status")
    
    def custom_movement(self):
        """Custom movement input."""
        if not self.check_connection():
            return
        
        print("\n=== CUSTOM MOVEMENT ===")
        print("Enter joint angles in degrees (leave empty to skip)")
        
        positions = {}
        
        for joint in ["base", "shoulder", "elbow", "wrist", "roll", "hand"]:
            value = input(f"{joint}: ").strip()
            if value:
                try:
                    positions[joint] = math.radians(float(value))
                except ValueError:
                    print(f"Invalid value for {joint}")
        
        if positions:
            speed = float(input("Speed (0.1-2.0) [1.0]: ") or "1.0")
            
            print("Executing movement...")
            success = self.controller.move_joints(positions, speed=speed)
            
            if success:
                print("✅ Movement complete")
            else:
                print("❌ Movement failed")
    
    def test_sequence(self):
        """Test movement sequence."""
        if not self.check_connection():
            return
        
        print("\n=== SEQUENCE TEST ===")
        print("Executing test sequence...")
        
        sequence = [
            ({"base": 0.5, "shoulder": 0.2}, 1.0),
            ({"base": -0.5, "shoulder": -0.2}, 1.0),
            ({"elbow": 2.0, "wrist": -0.5}, 0.5),
            ({"base": 0.0, "shoulder": 0.0, "elbow": 1.57, "wrist": 0.0}, 0.5)
        ]
        
        for i, (positions, speed) in enumerate(sequence, 1):
            print(f"Step {i}/{len(sequence)}")
            success = self.controller.move_joints(positions, speed=speed)
            
            if not success:
                print(f"❌ Failed at step {i}")
                break
            
            time.sleep(1)
        
        print("✅ Sequence complete")
    
    def monitor_serial(self):
        """Monitor raw serial data."""
        if not self.check_connection():
            return
        
        print("\n=== SERIAL MONITOR ===")
        print("Press Ctrl+C to stop")
        
        try:
            # Send a test command
            print("Sending test command...")
            self.controller.serial.ser.write(b'{"T":1}\n')
            self.controller.serial.ser.flush()
            
            # Monitor responses
            while True:
                if self.controller.serial.ser.in_waiting > 0:
                    data = self.controller.serial.ser.read(self.controller.serial.ser.in_waiting)
                    print(f"Received: {data}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def check_connection(self):
        """Check if connected."""
        if not self.controller or not self.controller.serial.connected:
            print("❌ Not connected! Use option 1 first.")
            return False
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.controller:
            print("\nDisconnecting...")
            self.controller.disconnect()


def main():
    """Main entry point."""
    debugger = RoArmDebugger()
    
    try:
        debugger.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        debugger.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()
