#!/usr/bin/env python3
"""
RoArm M3 Logging System
Centralized logging with color output and file rotation
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    
    # Color mapping for log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    # Icons for log levels (if terminal supports)
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def __init__(self, use_colors: bool = True, use_icons: bool = True):
        """
        Initialize colored formatter.
        
        Args:
            use_colors: Enable color output
            use_icons: Enable emoji icons
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_icons = use_icons
        
        # Format with or without timestamp
        if self.use_colors:
            fmt = '%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s'
        else:
            fmt = '%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
        
        super().__init__(fmt, datefmt='%H:%M:%S')
    
    def format(self, record):
        # Save original level name
        levelname = record.levelname
        
        if self.use_colors:
            # Add color to level name
            color = self.COLORS.get(levelname, '')
            record.levelname = f"{color}{levelname}{Style.RESET_ALL}"
            
            # Add icon if enabled
            if self.use_icons and levelname in self.ICONS:
                record.msg = f"{self.ICONS[levelname]} {record.msg}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original level name
        record.levelname = levelname
        
        return result


class RoArmLogger:
    """Centralized logger configuration for RoArm system."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            RoArmLogger._initialized = True
    
    def _setup_logger(self):
        """Setup the logging system."""
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Root logger configuration
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.root_logger.handlers.clear()
        
        # Console handler with colors
        self._setup_console_handler()
        
        # File handler with rotation
        self._setup_file_handler()
        
        # System info handler (for critical events)
        self._setup_system_handler()
        
        # Set levels for specific modules
        self._configure_module_levels()
        
        # Log startup
        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info("RoArm M3 Logging System Initialized")
        logger.info(f"Log directory: {self.log_dir.absolute()}")
        logger.info("="*60)
    
    def _setup_console_handler(self):
        """Setup console output with colors."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use colored formatter
        formatter = ColoredFormatter(use_colors=True, use_icons=True)
        console_handler.setFormatter(formatter)
        
        self.root_logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler."""
        # Main log file
        log_file = self.log_dir / f"roarm_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed format for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.root_logger.addHandler(file_handler)
    
    def _setup_system_handler(self):
        """Setup system event handler for critical events."""
        # System events log (errors and critical only)
        system_log = self.log_dir / "system_events.log"
        
        system_handler = logging.handlers.RotatingFileHandler(
            system_log,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        system_handler.setLevel(logging.ERROR)
        
        # Simple format for system events
        system_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        system_handler.setFormatter(system_formatter)
        
        self.root_logger.addHandler(system_handler)
    
    def _configure_module_levels(self):
        """Configure logging levels for specific modules."""
        # Set specific module levels
        module_levels = {
            'serial': logging.WARNING,           # Less verbose for serial
            'matplotlib': logging.WARNING,       # Suppress matplotlib debug
            'PIL': logging.WARNING,              # Suppress PIL debug
            'urllib3': logging.WARNING,          # Suppress urllib3 debug
            'core.serial_comm': logging.INFO,    # Important serial info
            'safety.safety_system': logging.INFO,  # Safety is important
            'calibration': logging.INFO,         # Calibration info
            'teaching': logging.INFO,            # Teaching info
            'patterns': logging.INFO,            # Pattern execution
            'motion': logging.DEBUG,             # Debug trajectory generation
        }
        
        for module, level in module_levels.items():
            logging.getLogger(module).setLevel(level)
    
    def set_level(self, level: str):
        """
        Set global logging level.
        
        Args:
            level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.root_logger.setLevel(level_map[level.upper()])
            logging.getLogger(__name__).info(f"Logging level set to {level}")
        else:
            logging.getLogger(__name__).warning(f"Invalid level: {level}")
    
    def get_log_files(self):
        """Get list of all log files."""
        return list(self.log_dir.glob("*.log"))
    
    def clear_old_logs(self, days: int = 30):
        """
        Clear log files older than specified days.
        
        Args:
            days: Number of days to keep
        """
        import time
        current_time = time.time()
        
        for log_file in self.get_log_files():
            file_age = current_time - log_file.stat().st_mtime
            if file_age > days * 86400:  # Convert days to seconds
                log_file.unlink()
                logging.getLogger(__name__).info(f"Deleted old log: {log_file.name}")


# Singleton instance
_logger_instance = None


def setup_logger(level: str = "INFO", log_to_file: bool = True):
    """
    Setup the logging system.
    
    Args:
        level: Logging level
        log_to_file: Enable file logging
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = RoArmLogger()
        _logger_instance.set_level(level)
    
    return _logger_instance


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Ensure logger is setup
    setup_logger()
    
    # Return module logger
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exception: Exception, 
                  message: str = "Exception occurred"):
    """
    Log an exception with traceback.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        message: Additional message
    """
    import traceback
    
    logger.error(f"{message}: {str(exception)}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")


def create_debug_logger(name: str, filename: str) -> logging.Logger:
    """
    Create a separate debug logger for a specific module.
    
    Args:
        name: Logger name
        filename: Debug log filename
        
    Returns:
        Debug logger instance
    """
    debug_logger = logging.getLogger(f"debug.{name}")
    debug_logger.setLevel(logging.DEBUG)
    
    # Create debug file handler
    log_dir = Path("logs/debug")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    debug_handler = logging.FileHandler(log_dir / filename, mode='w')
    debug_handler.setLevel(logging.DEBUG)
    
    # Very detailed format for debugging
    debug_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%H:%M:%S'
    )
    debug_handler.setFormatter(debug_formatter)
    
    debug_logger.addHandler(debug_handler)
    
    return debug_logger
