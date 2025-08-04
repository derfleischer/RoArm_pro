#!/usr/bin/env python3
"""
RoArm Pro - Main Entry Point
Professional robot arm control system
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from roarm_pro import RoArmApp, __version__
from roarm_pro.hardware import get_default_port, list_available_ports

def setup_logging(level: str = "INFO"):
    """Configure logging system"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def list_ports_command():
    """List available serial ports"""
    print("📋 Available Serial Ports:")
    print("-" * 50)
    
    ports = list_available_ports()
    
    if not ports:
        print("❌ No serial ports found!")
        print("\nPossible issues:")
        print("  - No USB devices connected")
        print("  - Missing USB drivers")
        print("  - Permission issues")
        return
    
    for port in ports:
        print(f"\n📍 {port['device']}")
        print(f"   Description: {port['description']}")
        print(f"   Hardware ID: {port['hwid']}")
        if port['is_usb']:
            print("   ✅ USB Device")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f'RoArm Pro v{__version__} - Professional Robot Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  roarm                    # Start with auto-detected port
  roarm --port COM3        # Use specific port
  roarm --list-ports       # Show available ports
  roarm --debug            # Enable debug logging
  
For more information, visit the documentation.
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'RoArm Pro v{__version__}'
    )
    
    parser.add_argument(
        '--port', '-p',
        default=None,
        help='Serial port (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--list-ports', '-l',
        action='store_true',
        help='List available serial ports and exit'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-auto-connect',
        action='store_true',
        help='Disable automatic connection on startup'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        args.log_level = 'DEBUG'
    setup_logging(args.log_level)
    
    # Handle list ports command
    if args.list_ports:
        list_ports_command()
        return 0
    
    # Create and run application
    try:
        app = RoArmApp(port=args.port, debug=args.debug)
        
        # Override auto-connect if specified
        if args.no_auto_connect:
            app.settings.auto_connect = False
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
