"""
RoArm M3 Scan Patterns Module
Professional 3D Scanning Patterns for RoArm M3
Version 3.1.0 - Complete Implementation
"""

__version__ = '3.1.0'
__author__ = 'RoArm Professional Team'

# Import core pattern classes
try:
    from .scan_patterns import (
        ScanPattern,
        ScanPoint,
        RasterScanPattern,
        SpiralScanPattern,
        SphericalScanPattern,
        TurntableScanPattern,
        HelixScanPattern,
        AdaptiveScanPattern,
        CobwebScanPattern,
        TableScanPattern,
        StatueSpiralPattern,
        create_scan_pattern,
        get_pattern_presets
    )
    
    BASIC_PATTERNS_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Basic patterns import warning: {e}")
    
    # Fallback definitions
    ScanPattern = None
    ScanPoint = None
    RasterScanPattern = None
    SpiralScanPattern = None
    SphericalScanPattern = None
    TurntableScanPattern = None
    HelixScanPattern = None
    AdaptiveScanPattern = None
    CobwebScanPattern = None
    TableScanPattern = None
    StatueSpiralPattern = None
    create_scan_pattern = None
    get_pattern_presets = None
    
    BASIC_PATTERNS_AVAILABLE = False

# Import technical configurator
try:
    from .technical_configurator import TechnicalScanningConfigurator
    TECHNICAL_CONFIGURATOR_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Technical configurator import warning: {e}")
    TechnicalScanningConfigurator = None
    TECHNICAL_CONFIGURATOR_AVAILABLE = False

# Import technical scan patterns (optional)
try:
    from .technical_scan_patterns import (
        TechnicalScanParameters,
        ScanningAlgorithm,
        AdvancedScanningOptions,
        PatternGenerationEngine,
        OptimizationEngine
    )
    
    TECHNICAL_PATTERNS_AVAILABLE = True
    
except ImportError as e:
    print(f"ℹ️ Technical scan patterns not available (optional): {e}")
    
    TechnicalScanParameters = None
    ScanningAlgorithm = None
    AdvancedScanningOptions = None
    PatternGenerationEngine = None
    OptimizationEngine = None
    
    TECHNICAL_PATTERNS_AVAILABLE = False

# Availability flags
ENHANCED_PATTERNS = TECHNICAL_PATTERNS_AVAILABLE

# Export all available classes
__all__ = [
    # Core pattern classes
    'ScanPattern',
    'ScanPoint',
    
    # Basic patterns
    'RasterScanPattern',
    'SpiralScanPattern', 
    'SphericalScanPattern',
    'TurntableScanPattern',
    'HelixScanPattern',
    'AdaptiveScanPattern',
    'CobwebScanPattern',
    'TableScanPattern',
    'StatueSpiralPattern',
    
    # Utility functions
    'create_scan_pattern',
    'get_pattern_presets',
    
    # Technical configurator
    'TechnicalScanningConfigurator',
    
    # Technical pattern classes (if available)
    'TechnicalScanParameters',
    'ScanningAlgorithm',
    'AdvancedScanningOptions',
    'PatternGenerationEngine',
    'OptimizationEngine',
    
    # Availability flags
    'BASIC_PATTERNS_AVAILABLE',
    'TECHNICAL_CONFIGURATOR_AVAILABLE', 
    'TECHNICAL_PATTERNS_AVAILABLE',
    'ENHANCED_PATTERNS'
]

# Clean up __all__ to only include available classes
__all__ = [name for name in __all__ if globals().get(name) is not None]

# Module information
def get_module_info():
    """Returns information about available pattern modules."""
    return {
        'version': __version__,
        'basic_patterns': BASIC_PATTERNS_AVAILABLE,
        'technical_configurator': TECHNICAL_CONFIGURATOR_AVAILABLE,
        'technical_patterns': TECHNICAL_PATTERNS_AVAILABLE,
        'enhanced_patterns': ENHANCED_PATTERNS,
        'available_classes': len(__all__)
    }

# Print status on import (for debugging)
if __name__ != "__main__":
    info = get_module_info()
    print(f"📦 RoArm Patterns Module v{info['version']}")
    print(f"   Basic patterns: {'✅' if info['basic_patterns'] else '❌'}")
    print(f"   Technical configurator: {'✅' if info['technical_configurator'] else '❌'}")
    print(f"   Technical patterns: {'✅' if info['technical_patterns'] else '❌'}")
    print(f"   Available classes: {info['available_classes']}")
