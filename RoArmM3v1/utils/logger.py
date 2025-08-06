# ============================================
# utils/logger.py
# ============================================
#!/usr/bin/env python3
"""
Logging utilities für RoArm M3 System.
"""

import logging
import sys
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(level=logging.INFO):
    """Setup the logging system."""
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('roarm_system.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    ))
    root_logger.addHandler(file_handler)


def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)
