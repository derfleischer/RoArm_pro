"""
Runtime settings management
Handles loading/saving user preferences
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Settings:
    """Runtime settings that can be modified by user"""
    
    # Motion settings
    speed_factor: float = 1.0
    acceleration_limit: float = 2.0
    settle_time: float = 0.1
    
    # Scanner settings
    scanner_mounted: bool = False
    scanner_grip: float = 2.5
    scan_speed: float = 0.5
    
    # Safety settings
    emergency_stop_enabled: bool = True
    soft_limits_enabled: bool = True
    
    # UI settings
    language: str = "de"  # German by default
    debug_mode: bool = False
    auto_connect: bool = True
    
    # Calibration
    calibrated: bool = False
    calibration_version: str = "1.0"
    
    @classmethod
    def load(cls, filepath: str = None) -> "Settings":
        """Load settings from YAML file"""
        if filepath is None:
            filepath = cls._get_default_path()
            
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                    return cls(**data)
            except Exception as e:
                print(f"Warning: Could not load settings: {e}")
                
        return cls()  # Return defaults
    
    def save(self, filepath: str = None):
        """Save settings to YAML file"""
        if filepath is None:
            filepath = self._get_default_path()
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @staticmethod
    def _get_default_path() -> str:
        """Get default settings path"""
        home = Path.home()
        return str(home / ".roarm_pro" / "settings.yaml")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        default = Settings()
        for field in default.__dataclass_fields__:
            setattr(self, field, getattr(default, field))
