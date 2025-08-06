# ============================================
# utils/terminal.py
# ============================================
#!/usr/bin/env python3
"""
Terminal utilities für macOS.
"""

import sys
import tty
import termios
import select


class TerminalController:
    """Terminal control für macOS."""
    
    def __init__(self):
        self.old_settings = None
        
    def get_key(self, timeout: float = 0.1):
        """Get single keypress (non-blocking)."""
        if sys.platform != 'darwin':
            # Fallback for non-macOS
            return input()
        
        try:
            # Save old settings
            if not self.old_settings:
                self.old_settings = termios.tcgetattr(sys.stdin)
            
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            # Check if input available
            if select.select([sys.stdin], [], [], timeout)[0]:
                key = sys.stdin.read(1)
            else:
                key = None
            
            # Restore settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
            return key
            
        except Exception:
            # Restore on error
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            return None
    
    def __del__(self):
        """Restore terminal settings on cleanup."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
