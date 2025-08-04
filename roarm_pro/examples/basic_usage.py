#!/usr/bin/env python3
"""
Basic usage example for RoArm Pro
Shows how to use the library programmatically
"""

import time
import math
from roarm_pro import RoArmController
from roarm_pro.hardware import get_default_port

def main():
    # Create controller
    port = get_default_port()
    print(f"Using port: {port}")
    
    controller = RoArmController(port)
    
    # Connect
    if not controller.connect():
        print("Failed to connect!")
        return
    
    print("Connected successfully!")
    
    try:
        # 1. Move to home position
        print("\n1. Moving to home position...")
        controller.move_to_home()
        time.sleep(2)
        
        # 2. Get current position
        print("\n2. Current position:")
        position = controller.get_current_position()
        for joint, angle in position.items():
            print(f"   {joint}: {math.degrees(angle):.1f}°")
        
        # 3. Move individual joints
        print("\n3. Moving base 45° right...")
        controller.move_joints(base=math.radians(45), duration=2.0)
        time.sleep(1)
        
        print("   Moving shoulder up 30°...")
        controller.move_joints(shoulder=math.radians(30), duration=2.0)
        time.sleep(1)
        
        # 4. Control gripper
        print("\n4. Opening gripper...")
        controller.set_gripper(80)  # 80% open
        time.sleep(1)
        
        print("   Closing gripper...")
        controller.set_gripper(20)  # 20% open
        time.sleep(1)
        
        # 5. Coordinated movement
        print("\n5. Coordinated movement...")
        controller.move_joints(
            base=math.radians(-45),
            shoulder=math.radians(-15),
            elbow=math.radians(90),
            duration=3.0
        )
        time.sleep(1)
        
        # 6. LED control
        print("\n6. LED test...")
        controller.led_on(255)
        time.sleep(1)
        controller.led_off()
        
        # 7. Return home
        print("\n7. Returning to home...")
        controller.move_to_home()
        
    except KeyboardInterrupt:
        print("\nInterrupted!")
        controller.emergency_stop()
    finally:
        # Safe shutdown
        print("\nShutting down...")
        controller.safe_shutdown()

if __name__ == "__main__":
    main()
