"""
Main application class
Coordinates all components and provides user interface
"""

import sys
import signal
import logging
from typing import Optional

from ..control import RoArmController, ManualControl, TeachingMode, ScannerControl
from ..config import Settings
from .menus import MenuSystem

logger = logging.getLogger(__name__)

class RoArmApp:
    """Main application for RoArm control"""
    
    def __init__(self, port: str = None, debug: bool = False):
        """
        Initialize application
        
        Args:
            port: Serial port (auto-detect if None)
            debug: Enable debug logging
        """
        # Load settings
        self.settings = Settings.load()
        
        # Use provided port or from settings
        if port is None:
            from ..hardware import get_default_port
            port = get_default_port()
        
        # Create controller
        self.controller = RoArmController(port)
        
        # Create control modules
        self.manual_control = ManualControl(self.controller)
        self.teaching_mode = TeachingMode(self.controller)
        self.scanner_control = ScannerControl(self.controller)
        
        # Create menu system
        self.menu_system = MenuSystem(self)
        
        # Application state
        self.running = True
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Debug mode
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def run(self):
        """Run the application"""
        self._print_header()
        
        # Auto-connect if enabled
        if self.settings.auto_connect:
            if not self._auto_connect():
                if not self._connection_help():
                    return
        
        # Check calibration
        self._check_calibration()
        
        # Main loop
        try:
            while self.running:
                choice = self.menu_system.show_main_menu()
                
                if choice == '0':
                    break
                else:
                    self.menu_system.handle_choice(choice)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted")
        finally:
            self._shutdown()
    
    def _print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print("🤖 RoArm Pro - Professional Robot Control")
        print("📷 Optimized for Revopoint Mini2 Scanner")
        print("🍎 macOS Focused Edition")
        print("="*60)
    
    def _auto_connect(self) -> bool:
        """Try to auto-connect"""
        print(f"\n🔌 Connecting to {self.controller.serial.port}...")
        
        if self.controller.connect():
            print("✅ Connected successfully")
            
            # Apply settings
            self.controller.speed_factor = self.settings.speed_factor
            self.controller.set_scanner_mounted(self.settings.scanner_mounted)
            
            return True
        else:
            print("❌ Connection failed")
            return False
    
    def _connection_help(self) -> bool:
        """Show connection help"""
        print("\n⚠️ Connection Failed")
        print("="*40)
        print("Please check:")
        print("  1. Robot power is ON")
        print("  2. USB cable is connected")
        print("  3. Correct port selected")
        
        from ..hardware import list_available_ports
        ports = list_available_ports()
        
        if ports:
            print("\nAvailable ports:")
            for p in ports:
                print(f"  - {p['device']}: {p['description']}")
        else:
            print("\n❌ No serial ports found!")
        
        print("\nOptions:")
        print("  1. Retry connection")
        print("  2. Select different port")
        print("  3. Continue without connection")
        print("  0. Exit")
        
        choice = input("\n➤ Choice: ").strip()
        
        if choice == '1':
            return self._auto_connect()
        elif choice == '2':
            port = input("Enter port: ").strip()
            self.controller.serial.port = port
            return self._auto_connect()
        elif choice == '3':
            return True
        else:
            return False
    
    def _check_calibration(self):
        """Check and suggest calibration"""
        if not self.settings.calibrated:
            print("\n🎯 CALIBRATION RECOMMENDED")
            print("="*40)
            print("For optimal performance, run a full")
            print("calibration sequence (Main Menu → 1)")
            print("="*40)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n🛑 EMERGENCY STOP (Ctrl+C)")
        self.controller.emergency_stop()
        print("\nUse menu option to clear emergency stop")
    
    def _shutdown(self):
        """Clean shutdown"""
        print("\n👋 Shutting down...")
        
        # Save settings
        self.settings.save()
        
        # Safe robot shutdown
        if self.controller.connected:
            self.controller.safe_shutdown()
        
        print("✅ Goodbye!")
