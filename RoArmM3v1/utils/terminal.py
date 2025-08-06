#!/usr/bin/env python3
"""
RoArm M3 Terminal Controller
Cross-platform keyboard input and terminal control
"""

import sys
import os
import termios
import tty
import select
import threading
from typing import Optional, Callable, Dict, Any
import time

import logging
logger = logging.getLogger(__name__)


class TerminalController:
    """
    Terminal controller for keyboard input and display control.
    Optimized for macOS but works cross-platform.
    """
    
    def __init__(self):
        """Initialize terminal controller."""
        self.running = False
        self.key_handlers = {}
        self.input_thread = None
        
        # Store original terminal settings
        if sys.stdin.isatty():
            self.original_settings = termios.tcgetattr(sys.stdin)
        else:
            self.original_settings = None
        
        # Key buffer
        self.key_buffer = []
        self._lock = threading.Lock()
        
        logger.debug("TerminalController initialized")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.restore_terminal()
    
    def setup_raw_mode(self):
        """Set terminal to raw mode for immediate key input."""
        if not sys.stdin.isatty():
            logger.warning("Not a TTY, raw mode not available")
            return
        
        try:
            # Store original settings
            if self.original_settings is None:
                self.original_settings = termios.tcgetattr(sys.stdin)
            
            # Set raw mode
            tty.setraw(sys.stdin.fileno())
            logger.debug("Terminal set to raw mode")
            
        except Exception as e:
            logger.error(f"Failed to set raw mode: {e}")
    
    def restore_terminal(self):
        """Restore original terminal settings."""
        if self.original_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
                logger.debug("Terminal settings restored")
            except Exception as e:
                logger.error(f"Failed to restore terminal: {e}")
    
    def get_key(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get a single key press (non-blocking).
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Key character or None if no key pressed
        """
        if not sys.stdin.isatty():
            return None
        
        try:
            # Use select for non-blocking read
            if select.select([sys.stdin], [], [], timeout)[0]:
                key = sys.stdin.read(1)
                
                # Handle special keys
                if key == '\x1b':  # ESC sequence
                    # Check for arrow keys
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key += sys.stdin.read(2)
                        if key == '\x1b[A':
                            return 'UP'
                        elif key == '\x1b[B':
                            return 'DOWN'
                        elif key == '\x1b[C':
                            return 'RIGHT'
                        elif key == '\x1b[D':
                            return 'LEFT'
                    return 'ESC'
                elif key == '\r' or key == '\n':
                    return 'ENTER'
                elif key == '\x7f' or key == '\x08':
                    return 'BACKSPACE'
                elif key == '\t':
                    return 'TAB'
                elif key == ' ':
                    return 'SPACE'
                elif key == '\x03':  # Ctrl+C
                    return 'CTRL+C'
                else:
                    return key
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading key: {e}")
            return None
    
    def get_line(self, prompt: str = "> ") -> str:
        """
        Get a line of input with prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Input string
        """
        # Restore terminal for normal input
        self.restore_terminal()
        
        try:
            # Use normal input
            line = input(prompt)
            return line
        finally:
            # Back to raw mode if needed
            pass
    
    def clear_screen(self):
        """Clear the terminal screen."""
        if sys.platform == 'win32':
            os.system('cls')
        else:
            os.system('clear')
    
    def move_cursor(self, row: int, col: int):
        """
        Move cursor to specific position.
        
        Args:
            row: Row position (1-based)
            col: Column position (1-based)
        """
        print(f'\033[{row};{col}H', end='')
    
    def hide_cursor(self):
        """Hide the cursor."""
        print('\033[?25l', end='')
    
    def show_cursor(self):
        """Show the cursor."""
        print('\033[?25h', end='')
    
    def set_color(self, foreground: Optional[str] = None, 
                  background: Optional[str] = None,
                  bold: bool = False):
        """
        Set terminal color.
        
        Args:
            foreground: Foreground color name
            background: Background color name
            bold: Bold text
        """
        colors = {
            'black': 0, 'red': 1, 'green': 2, 'yellow': 3,
            'blue': 4, 'magenta': 5, 'cyan': 6, 'white': 7
        }
        
        codes = []
        
        if bold:
            codes.append('1')
        
        if foreground and foreground.lower() in colors:
            codes.append(f'3{colors[foreground.lower()]}')
        
        if background and background.lower() in colors:
            codes.append(f'4{colors[background.lower()]}')
        
        if codes:
            print(f'\033[{";".join(codes)}m', end='')
    
    def reset_color(self):
        """Reset terminal color to default."""
        print('\033[0m', end='')
    
    def print_colored(self, text: str, foreground: str = None,
                     background: str = None, bold: bool = False):
        """
        Print colored text.
        
        Args:
            text: Text to print
            foreground: Foreground color
            background: Background color
            bold: Bold text
        """
        self.set_color(foreground, background, bold)
        print(text)
        self.reset_color()
    
    def create_progress_bar(self, current: int, total: int, 
                          width: int = 50, title: str = "Progress"):
        """
        Create a progress bar.
        
        Args:
            current: Current value
            total: Total value
            width: Bar width in characters
            title: Progress bar title
            
        Returns:
            Progress bar string
        """
        if total == 0:
            percent = 0
        else:
            percent = min(100, int(100 * current / total))
        
        filled = int(width * percent / 100)
        empty = width - filled
        
        bar = '█' * filled + '░' * empty
        
        return f"{title}: [{bar}] {percent}% ({current}/{total})"
    
    def print_progress(self, current: int, total: int, 
                      width: int = 50, title: str = "Progress"):
        """
        Print progress bar (updates in place).
        
        Args:
            current: Current value
            total: Total value
            width: Bar width
            title: Progress title
        """
        bar = self.create_progress_bar(current, total, width, title)
        print(f'\r{bar}', end='', flush=True)
        
        if current >= total:
            print()  # New line when complete
    
    def print_table(self, headers: list, rows: list, 
                   column_widths: Optional[list] = None):
        """
        Print a formatted table.
        
        Args:
            headers: List of header strings
            rows: List of row lists
            column_widths: Optional column widths
        """
        if not headers:
            return
        
        # Calculate column widths if not provided
        if column_widths is None:
            column_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    if i < len(column_widths):
                        column_widths[i] = max(column_widths[i], len(str(cell)))
        
        # Print header
        header_line = ' | '.join(
            str(h).ljust(w) for h, w in zip(headers, column_widths)
        )
        print(header_line)
        print('-' * len(header_line))
        
        # Print rows
        for row in rows:
            row_line = ' | '.join(
                str(cell).ljust(column_widths[i]) 
                for i, cell in enumerate(row) 
                if i < len(column_widths)
            )
            print(row_line)
    
    def create_menu(self, title: str, options: list, 
                   current_selection: int = 0) -> str:
        """
        Create an interactive menu.
        
        Args:
            title: Menu title
            options: List of option strings
            current_selection: Current selected index
            
        Returns:
            Selected option string
        """
        self.setup_raw_mode()
        
        try:
            selection = current_selection
            
            while True:
                # Clear and print menu
                self.clear_screen()
                print(f"\n{title}")
                print("=" * len(title))
                print("\nUse arrow keys to navigate, Enter to select, ESC to cancel\n")
                
                for i, option in enumerate(options):
                    if i == selection:
                        print(f"  > {option}")
                    else:
                        print(f"    {option}")
                
                # Get key input
                key = self.get_key()
                
                if key == 'UP':
                    selection = (selection - 1) % len(options)
                elif key == 'DOWN':
                    selection = (selection + 1) % len(options)
                elif key == 'ENTER':
                    return options[selection]
                elif key == 'ESC' or key == 'CTRL+C':
                    return None
                
        finally:
            self.restore_terminal()
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask for confirmation.
        
        Args:
            message: Confirmation message
            default: Default value if just Enter pressed
            
        Returns:
            True if confirmed
        """
        default_str = "Y/n" if default else "y/N"
        
        self.restore_terminal()
        
        try:
            response = input(f"{message} [{default_str}]: ").strip().lower()
            
            if not response:
                return default
            
            return response in ['y', 'yes']
            
        finally:
            pass
    
    def wait_for_key(self, message: str = "Press any key to continue..."):
        """
        Wait for any key press.
        
        Args:
            message: Message to display
        """
        print(message)
        self.setup_raw_mode()
        
        try:
            # Wait for any key
            while not self.get_key(timeout=0.1):
                pass
        finally:
            self.restore_terminal()


class KeyboardHandler:
    """
    Advanced keyboard handler with key mapping and callbacks.
    """
    
    def __init__(self):
        """Initialize keyboard handler."""
        self.handlers = {}
        self.running = False
        self.thread = None
        self.terminal = TerminalController()
        
    def register_handler(self, key: str, callback: Callable, 
                        description: str = ""):
        """
        Register a key handler.
        
        Args:
            key: Key to handle
            callback: Callback function
            description: Optional description
        """
        self.handlers[key.lower()] = {
            'callback': callback,
            'description': description
        }
        logger.debug(f"Registered handler for key '{key}': {description}")
    
    def unregister_handler(self, key: str):
        """
        Unregister a key handler.
        
        Args:
            key: Key to unregister
        """
        if key.lower() in self.handlers:
            del self.handlers[key.lower()]
            logger.debug(f"Unregistered handler for key '{key}'")
    
    def start(self):
        """Start keyboard handling in background thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self._handle_keys,
                daemon=True,
                name="KeyboardHandler"
            )
            self.thread.start()
            logger.info("Keyboard handler started")
    
    def stop(self):
        """Stop keyboard handling."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.terminal.restore_terminal()
        logger.info("Keyboard handler stopped")
    
    def _handle_keys(self):
        """Background thread for handling keys."""
        self.terminal.setup_raw_mode()
        
        try:
            while self.running:
                key = self.terminal.get_key(timeout=0.1)
                
                if key:
                    # Check for handler
                    if key.lower() in self.handlers:
                        try:
                            self.handlers[key.lower()]['callback']()
                        except Exception as e:
                            logger.error(f"Handler error for key '{key}': {e}")
                    
                    # Always check for exit keys
                    if key == 'CTRL+C' or key == '\x03':
                        logger.info("Ctrl+C detected, stopping handler")
                        self.running = False
                        break
                        
        finally:
            self.terminal.restore_terminal()
    
    def print_help(self):
        """Print registered key handlers."""
        print("\nKeyboard Controls:")
        print("-" * 40)
        
        for key, info in sorted(self.handlers.items()):
            desc = info['description'] or "No description"
            print(f"  {key:10s} : {desc}")
        
        print("-" * 40)
