"""
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
