#!/usr/bin/env python3
"""
Setup Script für RoArm M3 Enhanced Features
Erstellt die richtige Verzeichnisstruktur und __init__.py Dateien
"""

import os
import sys
from pathlib import Path

def create_init_files():
    """Erstellt alle __init__.py Dateien mit dem richtigen Inhalt."""
    
    print("🔧 Setting up RoArm M3 Enhanced Structure...")
    print("="*50)
    
    # Base directory
    base_dir = Path(__file__).parent
    
    # ============================================
    # core/__init__.py
    # ============================================
    core_init = '''"""
RoArm M3 Core Module
Basis-Controller und Hardware-Kommunikation
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .constants import (
    SERVO_LIMITS,
    HOME_POSITION,
    SCANNER_POSITION,
    PARK_POSITION,
    COMMANDS,
    SPEED_LIMITS,
    SCANNER_SPECS,
    TRAJECTORY_PROFILES,
    DEFAULT_SPEED
)

from .serial_comm import SerialManager

try:
    from enhanced.controller import EnhancedController as RoArmController
    from enhanced.controller import EnhancedConfig as RoArmConfig
    ENHANCED_CONTROLLER = True
except ImportError:
    from .controller import RoArmController, RoArmConfig
    ENHANCED_CONTROLLER = False

__all__ = [
    'RoArmController',
    'RoArmConfig',
    'SerialManager',
    'SERVO_LIMITS',
    'HOME_POSITION',
    'SCANNER_POSITION',
    'PARK_POSITION',
    'COMMANDS',
    'SPEED_LIMITS',
    'SCANNER_SPECS',
    'TRAJECTORY_PROFILES',
    'DEFAULT_SPEED',
    'ENHANCED_CONTROLLER'
]
'''
    
    # ============================================
    # calibration/__init__.py
    # ============================================
    calibration_init = '''"""
RoArm M3 Calibration Suite
Professionelle Kalibrierung für präzise Bewegungen
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .calibration_suite import (
    CalibrationSuite,
    CalibrationType,
    CalibrationPoint,
    JointCalibration,
    ScannerCalibration,
    SystemCalibration
)

ENHANCED_CALIBRATION = False

try:
    from enhanced.calibration import AutoCalibration, VisionCalibration
    ENHANCED_CALIBRATION = True
except ImportError:
    pass

__all__ = [
    'CalibrationSuite',
    'CalibrationType',
    'CalibrationPoint',
    'JointCalibration',
    'ScannerCalibration',
    'SystemCalibration',
    'ENHANCED_CALIBRATION'
]
'''
    
    # ============================================
    # teaching/__init__.py
    # ============================================
    teaching_init = '''"""
RoArm M3 Teaching System
Advanced recording and playback
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .recorder import (
    TeachingRecorder,
    TeachingWaypoint,
    TeachingSequence,
    RecordingMode
)

ENHANCED_TEACHING = False

try:
    from enhanced.teaching import SmartTeaching
    ENHANCED_TEACHING = True
except ImportError:
    pass

__all__ = [
    'TeachingRecorder',
    'TeachingWaypoint',
    'TeachingSequence',
    'RecordingMode',
    'ENHANCED_TEACHING'
]
'''
    
    # ============================================
    # motion/__init__.py
    # ============================================
    motion_init = '''"""
RoArm M3 Motion Control
Trajectory generation and motion planning
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .trajectory import (
    TrajectoryGenerator,
    TrajectoryType,
    TrajectoryPoint
)

ENHANCED_MOTION = False

try:
    from enhanced.motion import AdvancedTrajectoryGenerator
    TrajectoryGenerator = AdvancedTrajectoryGenerator
    ENHANCED_MOTION = True
except ImportError:
    pass

__all__ = [
    'TrajectoryGenerator',
    'TrajectoryType',
    'TrajectoryPoint',
    'ENHANCED_MOTION'
]
'''
    
    # ============================================
    # patterns/__init__.py
    # ============================================
    patterns_init = '''"""
RoArm M3 Scan Patterns
Optimized patterns for 3D scanning
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .scan_patterns import (
    ScanPattern,
    ScanPoint,
    RasterScanPattern,
    SpiralScanPattern,
    SphericalScanPattern,
    TurntableScanPattern,
    CobwebScanPattern
)

ENHANCED_PATTERNS = False

__all__ = [
    'ScanPattern',
    'ScanPoint',
    'RasterScanPattern',
    'SpiralScanPattern',
    'SphericalScanPattern',
    'TurntableScanPattern',
    'CobwebScanPattern',
    'ENHANCED_PATTERNS'
]
'''
    
    # ============================================
    # safety/__init__.py
    # ============================================
    safety_init = '''"""
RoArm M3 Safety System
Emergency stop, graceful shutdown, and monitoring
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .safety_system import (
    SafetySystem,
    SafetyState,
    ShutdownReason,
    SafetyEvent,
    SystemState
)

ENHANCED_SAFETY = False

__all__ = [
    'SafetySystem',
    'SafetyState',
    'ShutdownReason',
    'SafetyEvent',
    'SystemState',
    'ENHANCED_SAFETY'
]
'''
    
    # ============================================
    # utils/__init__.py
    # ============================================
    utils_init = '''"""
RoArm M3 Utilities
Logging, terminal control, and helper functions
"""

__version__ = '3.0.0'
__author__ = 'RoArm Professional Team'

from .logger import setup_logger, get_logger
from .terminal import TerminalController

# Optional utilities
try:
    from .safety import SafetyMonitor
    SAFETY_MONITOR = True
except ImportError:
    SafetyMonitor = None
    SAFETY_MONITOR = False

try:
    from .debug_mode import MockController, DebugMode
    DEBUG_MODE = True
except ImportError:
    MockController = None
    DebugMode = None
    DEBUG_MODE = False

__all__ = [
    'setup_logger',
    'get_logger',
    'TerminalController',
    'SAFETY_MONITOR',
    'DEBUG_MODE'
]

if SAFETY_MONITOR:
    __all__.append('SafetyMonitor')
if DEBUG_MODE:
    __all__.extend(['MockController', 'DebugMode'])
'''
    
    # ============================================
    # enhanced/__init__.py
    # ============================================
    enhanced_init = '''"""
RoArm M3 Enhanced Features
Advanced capabilities with ML, Vision, and Cloud
"""

__version__ = '1.0.0'
__author__ = 'RoArm Professional Team'

FEATURES = {
    'vision': False,
    'ml_optimizer': False,
    'adaptive_control': False,
    'predictive_motion': False,
    'cloud_sync': False,
    'realtime_monitor': False
}

# Try importing enhanced modules
try:
    from .vision import VisionSystem
    FEATURES['vision'] = True
except ImportError:
    VisionSystem = None

try:
    from .ml_optimizer import MLOptimizer
    FEATURES['ml_optimizer'] = True
except ImportError:
    MLOptimizer = None

try:
    from .adaptive_control import AdaptiveControl
    FEATURES['adaptive_control'] = True
except ImportError:
    AdaptiveControl = None

try:
    from .predictive_motion import PredictiveMotion
    FEATURES['predictive_motion'] = True
except ImportError:
    PredictiveMotion = None

try:
    from .cloud_sync import CloudSync
    FEATURES['cloud_sync'] = True
except ImportError:
    CloudSync = None

try:
    from .realtime_monitor import RealtimeMonitoring
    FEATURES['realtime_monitor'] = True
except ImportError:
    RealtimeMonitoring = None

ENHANCED_AVAILABLE = any(FEATURES.values())

__all__ = ['FEATURES', 'ENHANCED_AVAILABLE']

if FEATURES['vision']:
    __all__.append('VisionSystem')
if FEATURES['ml_optimizer']:
    __all__.append('MLOptimizer')
if FEATURES['adaptive_control']:
    __all__.append('AdaptiveControl')
if FEATURES['predictive_motion']:
    __all__.append('PredictiveMotion')
if FEATURES['cloud_sync']:
    __all__.append('CloudSync')
if FEATURES['realtime_monitor']:
    __all__.append('RealtimeMonitoring')

def get_available_features():
    """Return list of available enhanced features."""
    return [name for name, available in FEATURES.items() if available]
'''
    
    # Files to create
    init_files = {
        'core/__init__.py': core_init,
        'calibration/__init__.py': calibration_init,
        'teaching/__init__.py': teaching_init,
        'motion/__init__.py': motion_init,
        'patterns/__init__.py': patterns_init,
        'safety/__init__.py': safety_init,
        'utils/__init__.py': utils_init,
        'enhanced/__init__.py': enhanced_init
    }
    
    # Create files
    created = []
    updated = []
    skipped = []
    
    for filepath, content in init_files.items():
        full_path = base_dir / filepath
        
        # Create directory if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        if full_path.exists():
            # Check if content is different
            with open(full_path, 'r') as f:
                existing = f.read()
            
            if existing.strip() != content.strip():
                # Backup existing
                backup_path = full_path.with_suffix('.py.backup')
                with open(backup_path, 'w') as f:
                    f.write(existing)
                
                # Write new content
                with open(full_path, 'w') as f:
                    f.write(content)
                updated.append(filepath)
                print(f"  ✅ Updated: {filepath} (backup saved)")
            else:
                skipped.append(filepath)
                print(f"  ⏭️  Skipped: {filepath} (already up to date)")
        else:
            # Create new file
            with open(full_path, 'w') as f:
                f.write(content)
            created.append(filepath)
            print(f"  ✅ Created: {filepath}")
    
    # Create additional directories
    additional_dirs = [
        'enhanced',
        'calibration',
        'sequences',
        'models',
        'profiles',
        'reports',
        'logs',
        'exports'
    ]
    
    print("\n📁 Creating directories...")
    for dir_name in additional_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_name}/")
        else:
            print(f"  ⏭️  Exists: {dir_name}/")
    
    # Summary
    print("\n" + "="*50)
    print("📊 SETUP SUMMARY")
    print("="*50)
    print(f"  Files created: {len(created)}")
    print(f"  Files updated: {len(updated)}")
    print(f"  Files skipped: {len(skipped)}")
    
    # Check for enhanced features
    print("\n🔍 Checking for Enhanced Features...")
    enhanced_dir = base_dir / 'enhanced'
    enhanced_modules = list(enhanced_dir.glob('*.py'))
    enhanced_modules = [m for m in enhanced_modules if m.name != '__init__.py']
    
    if enhanced_modules:
        print(f"  ✅ Found {len(enhanced_modules)} enhanced modules:")
        for module in enhanced_modules:
            print(f"     - {module.stem}")
    else:
        print("  ℹ️  No enhanced modules found (standard mode)")
    
    print("\n✅ Setup complete!")
    print("\nYou can now run:")
    print("  python3 main.py           # Standard mode")
    print("  python3 main.py --debug   # Debug mode")
    print("  python3 main.py --help    # Show all options")
    
    return True


def test_imports():
    """Test if all imports work correctly."""
    print("\n🧪 Testing imports...")
    print("-"*40)
    
    test_results = []
    
    # Test core
    try:
        from core import RoArmController, RoArmConfig
        test_results.append(("core", True, "Controller imported"))
    except ImportError as e:
        test_results.append(("core", False, str(e)))
    
    # Test calibration
    try:
        from calibration import CalibrationSuite
        test_results.append(("calibration", True, "CalibrationSuite imported"))
    except ImportError as e:
        test_results.append(("calibration", False, str(e)))
    
    # Test teaching
    try:
        from teaching import TeachingRecorder
        test_results.append(("teaching", True, "TeachingRecorder imported"))
    except ImportError as e:
        test_results.append(("teaching", False, str(e)))
    
    # Test patterns
    try:
        from patterns import RasterScanPattern
        test_results.append(("patterns", True, "Patterns imported"))
    except ImportError as e:
        test_results.append(("patterns", False, str(e)))
    
    # Test safety
    try:
        from safety import SafetySystem
        test_results.append(("safety", True, "SafetySystem imported"))
    except ImportError as e:
        test_results.append(("safety", False, str(e)))
    
    # Test utils
    try:
        from utils import setup_logger, get_logger
        test_results.append(("utils", True, "Logger imported"))
    except ImportError as e:
        test_results.append(("utils", False, str(e)))
    
    # Test enhanced (optional)
    try:
        import enhanced
        if enhanced.ENHANCED_AVAILABLE:
            features = enhanced.get_available_features()
            test_results.append(("enhanced", True, f"{len(features)} features"))
        else:
            test_results.append(("enhanced", True, "No features (stub)"))
    except ImportError as e:
        test_results.append(("enhanced", False, str(e)))
    
    # Print results
    for module, success, message in test_results:
        status = "✅" if success else "❌"
        print(f"  {status} {module:15s}: {message}")
    
    # Summary
    successful = sum(1 for _, s, _ in test_results if s)
    total = len(test_results)
    
    print("-"*40)
    if successful == total:
        print(f"✅ All {total} modules imported successfully!")
        return True
    else:
        print(f"⚠️  {successful}/{total} modules imported")
        print("   Check error messages above")
        return False


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 RoArm M3 Enhanced Setup Script")
    print("="*50)
    
    # Run setup
    if create_init_files():
        # Test imports
        test_imports()
        
        print("\n💡 Next steps:")
        print("1. Copy the new main.py to your project root")
        print("2. Copy system_calibration.json to calibration/")
        print("3. Run: python3 main.py")
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)
