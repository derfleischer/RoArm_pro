#!/usr/bin/env python3
"""
Scanner operation example for RoArm Pro
Shows how to perform various scanning patterns
"""

import time
from roarm_pro import RoArmController
from roarm_pro.control import ScannerControl
from roarm_pro.hardware import get_default_port

def main():
    # Create controller
    port = get_default_port()
    controller = RoArmController(port)
    
    # Connect
    if not controller.connect():
        print("Failed to connect!")
        return
    
    # Create scanner control
    scanner = ScannerControl(controller)
    
    try:
        # 1. Prepare for scanner mounting
        print("1. Moving to scanner mount position...")
        scanner.mount_scanner()
        time.sleep(2)
        
        input("Please mount the scanner and press Enter...")
        
        # Set scanner grip
        controller.set_scanner_grip(2.5)
        time.sleep(1)
        
        # 2. Move to scan start position
        print("\n2. Moving to scan start position...")
        scanner.start_position()
        time.sleep(2)
        
        # 3. Configure scan parameters
        print("\n3. Configuring scan parameters...")
        scanner.set_scan_parameters(
            width=0.15,         # 15cm scan width
            height=0.10,        # 10cm scan height
            step_size=0.01,     # 1cm steps
            settle_time=0.2,    # 200ms settle time
            led_brightness=150  # Medium brightness
        )
        
        # Show menu
        while True:
            print("\n" + "="*50)
            print("SCANNER DEMO MENU")
            print("="*50)
            print("1. Continuous rotation (60s)")
            print("2. Quick raster scan (5x5)")
            print("3. Detailed raster scan (10x10)")
            print("4. Spiral scan (3 turns)")
            print("5. Detail scan (current position)")
            print("0. Exit")
            
            choice = input("\nSelect scan type: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                print("\nStarting continuous rotation scan...")
                print("Press Ctrl+C to stop early")
                scanner.continuous_rotation_scan(duration=60)
            elif choice == '2':
                print("\nStarting quick raster scan...")
                scanner.raster_scan(rows=5, cols=5)
            elif choice == '3':
                print("\nStarting detailed raster scan...")
                scanner.raster_scan(rows=10, cols=10)
            elif choice == '4':
                print("\nStarting spiral scan...")
                scanner.spiral_scan(turns=3, points_per_turn=20)
            elif choice == '5':
                print("\nStarting detail scan...")
                scanner.detail_scan(size=0.03, points=9)
            else:
                print("Invalid choice!")
                continue
            
            # Return to start position after each scan
            print("\nReturning to start position...")
            scanner.start_position()
            time.sleep(2)
        
    except KeyboardInterrupt:
        print("\nInterrupted!")
        controller.emergency_stop()
    finally:
        # Safe shutdown
        print("\nReturning to home position...")
        controller.move_to_home()
        time.sleep(2)
        
        print("Shutting down...")
        controller.safe_shutdown()

if __name__ == "__main__":
    main()
