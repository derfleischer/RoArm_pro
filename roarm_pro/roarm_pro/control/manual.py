"""
Manual control implementation
Keyboard control for direct robot manipulation
"""

import sys
import time
import math
import threading
from typing import Dict, Optional

# Platform-specific imports for keyboard input
try:
    import termios
    import tty
    import select
    MACOS_TERMINAL = True
except ImportError:
    MACOS_TERMINAL = False

class ManualControl:
    """Manual keyboard control for RoArm"""
    
    def __init__(self, controller):
        """
        Initialize manual control
        Args:
            controller: RoArmController instance
        """
        self.controller = controller
        self.active = False
        self.movement_speed = 0.1  # Radians per key press
        
        # Key mappings
        self.key_map = {
            'q': ('base', -1),      # Base left
            'a': ('base', 1),       # Base right
            'w': ('shoulder', 1),   # Shoulder up
            's': ('shoulder', -1),  # Shoulder down
            'e': ('elbow', 1),      # Elbow up
            'd': ('elbow', -1),     # Elbow down
            'r': ('wrist', 1),      # Wrist up
            'f': ('wrist', -1),     # Wrist down
            't': ('roll', 1),       # Roll left
            'g': ('roll', -1),      # Roll right
            'y': ('hand', -0.1),    # Gripper open
            'h': ('hand', 0.1),     # Gripper close
        }
        
    def start(self):
        """Start manual control mode"""
        self.active = True
        
        print("\n" + "="*60)
        print("🎮 MANUAL CONTROL MODE")
        print("="*60)
        print("Controls:")
        print("  Q/A = Base left/right      | W/S = Shoulder up/down")
        print("  E/D = Elbow up/down        | R/F = Wrist up/down")
        print("  T/G = Roll left/right      | Y/H = Gripper open/close")
        print("  +/- = Speed up/down        | SPACE = Emergency stop")
        print("  P = Save position          | L = List positions")
        print("  ESC = Exit")
        print("="*60)
        
        if MACOS_TERMINAL:
            self._run_macos()
        else:
            self._run_generic()
    
    def _run_macos(self):
        """Run manual control for macOS"""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            while self.active:
                # Check for key press
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == '\x1b':  # ESC
                        break
                    elif key == ' ':   # Space - emergency stop
                        self.controller.emergency_stop()
                        print("\n🛑 EMERGENCY STOP!")
                    elif key in self.key_map:
                        self._handle_movement(key)
                    elif key == '+':
                        self.movement_speed = min(self.movement_speed + 0.05, 0.5)
                        print(f"\n⚡ Speed: {self.movement_speed:.2f}")
                    elif key == '-':
                        self.movement_speed = max(self.movement_speed - 0.05, 0.05)
                        print(f"\n⚡ Speed: {self.movement_speed:.2f}")
                    elif key == 'p':
                        self._save_position()
                    elif key == 'l':
                        self._list_positions()
                        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.active = False
            print("\n✅ Manual control ended")
    
    def _run_generic(self):
        """Run manual control for non-macOS systems"""
        print("\n⚠️ Raw keyboard input not available")
        print("Enter commands (e.g., 'q' for base left, 'exit' to quit):")
        
        while self.active:
            try:
                cmd = input("> ").strip().lower()
                
                if cmd == 'exit':
                    break
                elif cmd == 'stop':
                    self.controller.emergency_stop()
                elif cmd in self.key_map:
                    self._handle_movement(cmd)
                elif cmd.startswith('speed '):
                    try:
                        self.movement_speed = float(cmd.split()[1])
                        print(f"Speed set to {self.movement_speed}")
                    except:
                        print("Invalid speed")
                        
            except KeyboardInterrupt:
                break
        
        self.active = False
    
    def _handle_movement(self, key: str):
        """Handle movement key press"""
        joint, direction = self.key_map[key]
        
        # Get current position
        current = self.controller.get_current_position()
        
        # Calculate new position
        delta = self.movement_speed * direction
        new_value = current[joint] + delta
        
        # Move joint
        self.controller.move_joints(**{joint: new_value}, duration=0.2)
    
    def _save_position(self):
        """Save current position"""
        # This would integrate with teaching mode
        pos = self.controller.get_current_position()
        print(f"\n💾 Position saved: {pos}")
    
    def _list_positions(self):
        """List saved positions"""
        # This would integrate with teaching mode
        print("\n📋 Saved positions:")
        print("  1. Home")
        print("  2. Scanner mount")
        # etc.
