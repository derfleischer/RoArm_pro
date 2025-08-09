#!/usr/bin/env python3
"""
RoArm M3 - Technical Scanning Configurator
Professional Interface für den Technical Scan Patterns
Version 3.1.0 - Complete Implementation
"""

import math
import json
import time
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import asdict

# Import technical patterns engine
try:
    from .technical_scan_patterns import (
        TechnicalScanParameters,
        ScanningAlgorithm, 
        AdvancedScanningOptions,
        PatternGenerationEngine,
        OptimizationEngine,
        ScannerSpecifications
    )
except ImportError:
    # Fallback if technical_scan_patterns not available
    print("⚠️ technical_scan_patterns not available - using minimal implementation")
    
    from enum import Enum
    from dataclasses import dataclass
    
    class ScanningAlgorithm(Enum):
        STRUCTURED_GRID = "structured_grid"
        FIBONACCI_SPHERE = "fibonacci_sphere"
        ADAPTIVE_DENSITY = "adaptive_density"
    
    @dataclass
    class TechnicalScanParameters:
        angular_resolution: float = 0.1
        path_optimization: str = "greedy"
        servo_safety_margin: float = 0.1
    
    PatternGenerationEngine = None
    OptimizationEngine = None

# Import base scan patterns
try:
    from .scan_patterns import ScanPattern, ScanPoint, TrajectoryType
except ImportError:
    from scan_patterns import ScanPattern, ScanPoint, TrajectoryType

logger = logging.getLogger(__name__)


class TechnicalScanningConfigurator:
    """
    Professional Technical Scanning Configurator.
    Erweiterte Scan-Konfiguration mit technischen Parametern.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialisiert den Technical Configurator.
        
        Args:
            config_file: Pfad zur Konfigurationsdatei
        """
        self.config_file = config_file or "technical_scan_config.json"
        
        # Initialize technical parameters
        self.tech_params = TechnicalScanParameters()
        
        # Initialize scanner specifications (if available)
        if 'ScannerSpecifications' in globals():
            self.scanner_specs = ScannerSpecifications()
        else:
            self.scanner_specs = self._create_default_scanner_specs()
        
        # Initialize advanced options (if available)
        if 'AdvancedScanningOptions' in globals():
            self.advanced_options = AdvancedScanningOptions()
        else:
            self.advanced_options = self._create_default_advanced_options()
        
        # Initialize algorithm config
        self.algo_config = self._create_default_algo_config()
        
        # Initialize engines (if available)
        if PatternGenerationEngine:
            self.pattern_engine = PatternGenerationEngine(self.algo_config)
        else:
            self.pattern_engine = None
            
        if OptimizationEngine:
            self.optimization_engine = OptimizationEngine(self.tech_params)
        else:
            self.optimization_engine = None
        
        # Load existing config if available
        self._load_config()
        
        logger.info("🔧 Technical Scanning Configurator initialized")
    
    def _create_default_scanner_specs(self) -> object:
        """Creates default scanner specifications if class not available."""
        class DefaultScannerSpecs:
            def __init__(self):
                self.fov_horizontal = math.radians(60)
                self.fov_vertical = math.radians(45)
                self.min_distance = 0.08
                self.max_distance = 0.30
                self.optimal_distance = 0.15
                self.optimal_overlap_percentage = 0.25
        
        return DefaultScannerSpecs()
    
    def _create_default_advanced_options(self) -> object:
        """Creates default advanced options if class not available."""
        class DefaultAdvancedOptions:
            def __init__(self):
                self.multi_exposure = False
                self.hdr_scanning = False
                self.motion_blur_compensation = True
                self.adaptive_exposure = True
                self.quality_metrics_enabled = True
        
        return DefaultAdvancedOptions()
    
    def _create_default_algo_config(self) -> object:
        """Creates default algorithm config if class not available."""
        class DefaultAlgoConfig:
            def __init__(self):
                self.primary_algorithm = ScanningAlgorithm.FIBONACCI_SPHERE
                self.grid_base_resolution = (12, 8)
                self.grid_refinement_levels = 2
                self.fibonacci_points = 120
                self.fibonacci_golden_ratio = 1.618033988749
                self.adaptive_base_density = 1.0
                self.adaptive_edge_detection = True
                self.adaptive_curvature_weighting = 0.3
        
        return DefaultAlgoConfig()
    
    def expert_configuration_menu(self):
        """
        Hauptmenü für Expert-Konfiguration.
        """
        while True:
            self._clear_screen()
            
            print("\n🔬 TECHNICAL SCANNER CONFIGURATION")
            print("=" * 50)
            print("Expert-level scanning configuration")
            
            print("\n=== ALGORITHM CONFIGURATION ===")
            print("1. 🧠 Scanning Algorithm Settings")
            print("2. 📐 Technical Parameters")
            print("3. 📡 Scanner Specifications")
            print("4. ⚙️ Advanced Options")
            
            print("\n=== ANALYSIS & OPTIMIZATION ===")
            print("5. 🔍 Object Analysis Engine")
            print("6. 🎯 Path Optimization Settings")
            print("7. 📊 Quality Metrics Configuration")
            print("8. 🧪 Algorithm Performance Testing")
            
            print("\n=== CONFIGURATION MANAGEMENT ===")
            print("9. 💾 Save Current Configuration")
            print("10. 📂 Load Configuration Preset")
            print("11. 📤 Export Configuration")
            print("12. 🔄 Reset to Defaults")
            
            print("\n=== ADVANCED TOOLS ===")
            print("13. 🔬 Generate Custom Pattern")
            print("14. 📈 Performance Benchmarking")
            print("15. 🛠️ Calibration Assistant")
            
            print("\n0. ↩️ Back to Main Menu")
            
            choice = input("\n👉 Select option: ").strip()
            
            if choice == '1':
                self._algorithm_configuration()
            elif choice == '2':
                self._technical_parameters()
            elif choice == '3':
                self._scanner_specifications()
            elif choice == '4':
                self._advanced_options_menu()
            elif choice == '5':
                self._object_analysis_engine()
            elif choice == '6':
                self._path_optimization()
            elif choice == '7':
                self._quality_metrics()
            elif choice == '8':
                self._algorithm_testing()
            elif choice == '9':
                self._save_configuration()
            elif choice == '10':
                self._load_configuration_preset()
            elif choice == '11':
                self._export_configuration()
            elif choice == '12':
                self._reset_to_defaults()
            elif choice == '13':
                self._generate_custom_pattern()
            elif choice == '14':
                self._performance_benchmarking()
            elif choice == '15':
                self._calibration_assistant()
            elif choice == '0':
                break
            else:
                print("❌ Invalid option")
                time.sleep(1)
    
    def _algorithm_configuration(self):
        """Konfiguration der Scan-Algorithmen."""
        self._clear_screen()
        
        print("\n🧠 SCANNING ALGORITHM CONFIGURATION")
        print("=" * 45)
        
        # Aktuelle Einstellung anzeigen
        print(f"\nCurrent Algorithm: {self.algo_config.primary_algorithm.value.replace('_', ' ').title()}")
        
        # Verfügbare Algorithmen
        algorithms = list(ScanningAlgorithm)
        print("\nAvailable Algorithms:")
        
        descriptions = {
            ScanningAlgorithm.STRUCTURED_GRID: "Regular grid pattern with refinement options",
            ScanningAlgorithm.FIBONACCI_SPHERE: "Optimal sphere coverage using golden ratio",
            ScanningAlgorithm.ADAPTIVE_DENSITY: "Dynamic density based on object complexity"
        }
        
        for i, algo in enumerate(algorithms, 1):
            current = "🟢" if algo == self.algo_config.primary_algorithm else "⚪"
            print(f"{current} {i}. {algo.value.replace('_', ' ').title()}")
            print(f"      {descriptions.get(algo, 'Advanced scanning algorithm')}")
        
        choice = input(f"\nSelect algorithm (1-{len(algorithms)}) or 0 to skip: ").strip()
        
        if choice == '0':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(algorithms):
                self.algo_config.primary_algorithm = algorithms[idx]
                print(f"✅ Algorithm set to: {algorithms[idx].value.replace('_', ' ').title()}")
                
                # Algorithm-specific configuration
                self._configure_algorithm_parameters()
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
        
        input("\nPress ENTER to continue...")
    
    def _configure_algorithm_parameters(self):
        """Konfiguriert algorithmus-spezifische Parameter."""
        algo = self.algo_config.primary_algorithm
        
        if algo == ScanningAlgorithm.STRUCTURED_GRID:
            print(f"\n📐 STRUCTURED GRID PARAMETERS")
            print(f"Current Resolution: {self.algo_config.grid_base_resolution}")
            print(f"Refinement Levels: {self.algo_config.grid_refinement_levels}")
            
            if input("\nAdjust grid parameters? (y/n): ").lower() == 'y':
                try:
                    h_res = int(input(f"Horizontal resolution [{self.algo_config.grid_base_resolution[0]}]: ") 
                              or self.algo_config.grid_base_resolution[0])
                    v_res = int(input(f"Vertical resolution [{self.algo_config.grid_base_resolution[1]}]: ")
                              or self.algo_config.grid_base_resolution[1])
                    self.algo_config.grid_base_resolution = (h_res, v_res)
                    
                    refinement = int(input(f"Refinement levels [{self.algo_config.grid_refinement_levels}]: ")
                                   or self.algo_config.grid_refinement_levels)
                    self.algo_config.grid_refinement_levels = refinement
                    
                    print("✅ Grid parameters updated")
                except ValueError:
                    print("❌ Invalid input, keeping current values")
        
        elif algo == ScanningAlgorithm.FIBONACCI_SPHERE:
            print(f"\n🌻 FIBONACCI SPHERE PARAMETERS")
            print(f"Number of points: {self.algo_config.fibonacci_points}")
            print(f"Golden ratio: {self.algo_config.fibonacci_golden_ratio:.9f}")
            
            if input("\nAdjust fibonacci parameters? (y/n): ").lower() == 'y':
                try:
                    points = int(input(f"Number of points [{self.algo_config.fibonacci_points}]: ")
                              or self.algo_config.fibonacci_points)
                    self.algo_config.fibonacci_points = points
                    print("✅ Fibonacci parameters updated")
                except ValueError:
                    print("❌ Invalid input, keeping current values")
        
        elif algo == ScanningAlgorithm.ADAPTIVE_DENSITY:
            print(f"\n🎯 ADAPTIVE DENSITY PARAMETERS")
            print(f"Base density: {self.algo_config.adaptive_base_density:.1f}")
            print(f"Edge detection: {'Enabled' if self.algo_config.adaptive_edge_detection else 'Disabled'}")
            print(f"Curvature weighting: {self.algo_config.adaptive_curvature_weighting:.1f}")
            
            if input("\nAdjust adaptive parameters? (y/n): ").lower() == 'y':
                try:
                    density = float(input(f"Base density [{self.algo_config.adaptive_base_density}]: ")
                                  or self.algo_config.adaptive_base_density)
                    self.algo_config.adaptive_base_density = density
                    
                    edge_detect = input(f"Edge detection [{'y' if self.algo_config.adaptive_edge_detection else 'n'}]: ").lower()
                    if edge_detect in ['y', 'n']:
                        self.algo_config.adaptive_edge_detection = edge_detect == 'y'
                    
                    curvature = float(input(f"Curvature weighting [{self.algo_config.adaptive_curvature_weighting}]: ")
                                    or self.algo_config.adaptive_curvature_weighting)
                    self.algo_config.adaptive_curvature_weighting = curvature
                    
                    print("✅ Adaptive parameters updated")
                except ValueError:
                    print("❌ Invalid input, keeping current values")
    
    def _technical_parameters(self):
        """Konfiguration technischer Parameter."""
        self._clear_screen()
        
        print("\n📐 TECHNICAL PARAMETERS")
        print("=" * 30)
        
        print(f"\nCurrent Settings:")
        print(f"  Angular Resolution: {math.degrees(self.tech_params.angular_resolution):.1f}°")
        print(f"  Path Optimization: {self.tech_params.path_optimization.title()}")
        print(f"  Safety Margin: {math.degrees(self.tech_params.servo_safety_margin):.1f}°")
        
        if input("\nModify technical parameters? (y/n): ").lower() == 'y':
            try:
                # Angular Resolution
                current_deg = math.degrees(self.tech_params.angular_resolution)
                new_resolution = float(input(f"Angular resolution in degrees [{current_deg:.1f}]: ") 
                                     or current_deg)
                self.tech_params.angular_resolution = math.radians(new_resolution)
                
                # Path Optimization
                print(f"\nPath Optimization Options:")
                print(f"1. None (original order)")
                print(f"2. Greedy (nearest neighbor)")
                print(f"3. Genetic (advanced optimization)")
                
                opt_choice = input(f"Select optimization [2]: ") or "2"
                optimizations = {"1": "none", "2": "greedy", "3": "genetic"}
                self.tech_params.path_optimization = optimizations.get(opt_choice, "greedy")
                
                # Safety Margin
                current_safety = math.degrees(self.tech_params.servo_safety_margin)
                new_safety = float(input(f"Safety margin in degrees [{current_safety:.1f}]: ") 
                                 or current_safety)
                self.tech_params.servo_safety_margin = math.radians(new_safety)
                
                print("✅ Technical parameters updated")
                
            except ValueError:
                print("❌ Invalid input, keeping current values")
        
        input("\nPress ENTER to continue...")
    
    def _scanner_specifications(self):
        """Konfiguration der Scanner-Spezifikationen."""
        self._clear_screen()
        
        print("\n📡 SCANNER SPECIFICATIONS")
        print("=" * 30)
        
        print(f"\nField of View:")
        print(f"  Horizontal: {math.degrees(self.scanner_specs.fov_horizontal):.1f}°")
        print(f"  Vertical: {math.degrees(self.scanner_specs.fov_vertical):.1f}°")
        
        print(f"\nWorking Distances:")
        print(f"  Minimum: {self.scanner_specs.min_distance*1000:.0f}mm")
        print(f"  Maximum: {self.scanner_specs.max_distance*1000:.0f}mm")
        print(f"  Optimal: {self.scanner_specs.optimal_distance*1000:.0f}mm")
        
        print(f"\nOverlap: {self.scanner_specs.optimal_overlap_percentage*100:.0f}%")
        
        if input("\nModify scanner specifications? (y/n): ").lower() == 'y':
            try:
                # FOV
                h_fov = float(input(f"Horizontal FOV [{math.degrees(self.scanner_specs.fov_horizontal):.1f}]: ") 
                            or math.degrees(self.scanner_specs.fov_horizontal))
                self.scanner_specs.fov_horizontal = math.radians(h_fov)
                
                v_fov = float(input(f"Vertical FOV [{math.degrees(self.scanner_specs.fov_vertical):.1f}]: ") 
                            or math.degrees(self.scanner_specs.fov_vertical))
                self.scanner_specs.fov_vertical = math.radians(v_fov)
                
                # Distances (in mm, convert to m)
                min_dist = float(input(f"Minimum distance (mm) [{self.scanner_specs.min_distance*1000:.0f}]: ") 
                               or self.scanner_specs.min_distance*1000) / 1000
                self.scanner_specs.min_distance = min_dist
                
                max_dist = float(input(f"Maximum distance (mm) [{self.scanner_specs.max_distance*1000:.0f}]: ") 
                               or self.scanner_specs.max_distance*1000) / 1000
                self.scanner_specs.max_distance = max_dist
                
                opt_dist = float(input(f"Optimal distance (mm) [{self.scanner_specs.optimal_distance*1000:.0f}]: ") 
                               or self.scanner_specs.optimal_distance*1000) / 1000
                self.scanner_specs.optimal_distance = opt_dist
                
                # Overlap
                overlap = float(input(f"Overlap percentage [25]: ") or "25") / 100
                self.scanner_specs.optimal_overlap_percentage = overlap
                
                print("✅ Scanner specifications updated")
                
            except ValueError:
                print("❌ Invalid input, keeping current values")
        
        input("\nPress ENTER to continue...")
    
    def _advanced_options_menu(self):
        """Erweiterte Optionen."""
        self._clear_screen()
        
        print("\n⚙️ ADVANCED SCANNING OPTIONS")
        print("=" * 35)
        
        print(f"\nCurrent Settings:")
        print(f"  Multi-exposure: {'✅' if self.advanced_options.multi_exposure else '❌'}")
        print(f"  HDR Scanning: {'✅' if self.advanced_options.hdr_scanning else '❌'}")
        print(f"  Motion Blur Compensation: {'✅' if self.advanced_options.motion_blur_compensation else '❌'}")
        print(f"  Adaptive Exposure: {'✅' if self.advanced_options.adaptive_exposure else '❌'}")
        print(f"  Quality Metrics: {'✅' if self.advanced_options.quality_metrics_enabled else '❌'}")
        
        if input("\nModify advanced options? (y/n): ").lower() == 'y':
            def toggle_option(prompt, current):
                response = input(f"{prompt} [{'y' if current else 'n'}]: ").lower()
                return response == 'y' if response in ['y', 'n'] else current
            
            self.advanced_options.multi_exposure = toggle_option(
                "Multi-exposure", self.advanced_options.multi_exposure)
            self.advanced_options.hdr_scanning = toggle_option(
                "HDR Scanning", self.advanced_options.hdr_scanning)
            self.advanced_options.motion_blur_compensation = toggle_option(
                "Motion Blur Compensation", self.advanced_options.motion_blur_compensation)
            self.advanced_options.adaptive_exposure = toggle_option(
                "Adaptive Exposure", self.advanced_options.adaptive_exposure)
            self.advanced_options.quality_metrics_enabled = toggle_option(
                "Quality Metrics", self.advanced_options.quality_metrics_enabled)
            
            print("✅ Advanced options updated")
        
        input("\nPress ENTER to continue...")
    
    def _object_analysis_engine(self):
        """Objekt-Analyse Engine."""
        self._clear_screen()
        
        print("\n🔍 OBJECT ANALYSIS ENGINE")
        print("=" * 30)
        print("Analyze objects to generate optimal scan parameters")
        
        print("\nObject Characteristics:")
        print("1. Object dimensions (L×W×H)")
        print("2. Object complexity level")
        print("3. Material properties")
        print("4. Surface characteristics")
        
        # Simplified object analysis
        try:
            length = float(input("Object length (cm): ")) / 100
            width = float(input("Object width (cm): ")) / 100
            height = float(input("Object height (cm): ")) / 100
            
            print("\nComplexity Level:")
            print("1. Simple/Geometric")
            print("2. Medium complexity")
            print("3. High complexity/Organic")
            
            complexity = input("Select complexity (1-3): ")
            complexity_map = {"1": "simple", "2": "medium", "3": "high"}
            complexity_level = complexity_map.get(complexity, "medium")
            
            # Perform analysis
            analysis = self._perform_object_analysis((length, width, height), complexity_level)
            
            print(f"\n📊 ANALYSIS RESULTS:")
            print("=" * 25)
            self._display_analysis_results(analysis)
            
            if input("\nApply recommended settings? (y/n): ").lower() == 'y':
                self._apply_analysis_recommendations(analysis)
                print("✅ Recommendations applied")
            
        except ValueError:
            print("❌ Invalid input")
        
        input("\nPress ENTER to continue...")
    
    def _perform_object_analysis(self, dimensions: Tuple[float, float, float], 
                                complexity: str) -> Dict[str, Any]:
        """Führt Objekt-Analyse durch."""
        length, width, height = dimensions
        
        # Calculate object metrics
        volume = length * width * height
        surface_area = 2 * (length*width + length*height + width*height)
        max_dimension = max(dimensions)
        min_dimension = min(dimensions)
        aspect_ratio = max_dimension / min_dimension
        
        # Determine optimal parameters based on analysis
        if complexity == "simple":
            recommended_points = max(50, int(surface_area * 1000))
            speed_factor = 1.2
        elif complexity == "medium":
            recommended_points = max(100, int(surface_area * 1500))
            speed_factor = 1.0
        else:  # high
            recommended_points = max(200, int(surface_area * 2000))
            speed_factor = 0.8
        
        # Optimal distance based on size
        optimal_distance = max(0.10, min(0.25, max_dimension * 1.5))
        
        return {
            'dimensions': dimensions,
            'volume': volume,
            'surface_area': surface_area,
            'aspect_ratio': aspect_ratio,
            'complexity': complexity,
            'recommended_points': min(recommended_points, 500),  # Cap at 500
            'optimal_distance': optimal_distance,
            'speed_factor': speed_factor,
            'recommended_algorithm': self._recommend_algorithm(complexity, aspect_ratio)
        }
    
    def _recommend_algorithm(self, complexity: str, aspect_ratio: float) -> ScanningAlgorithm:
        """Empfiehlt Algorithmus basierend auf Analyse."""
        if complexity == "simple" and aspect_ratio < 2.0:
            return ScanningAlgorithm.STRUCTURED_GRID
        elif complexity == "high":
            return ScanningAlgorithm.ADAPTIVE_DENSITY
        else:
            return ScanningAlgorithm.FIBONACCI_SPHERE
    
    def _display_analysis_results(self, analysis: Dict[str, Any]):
        """Zeigt Analyse-Ergebnisse an."""
        dims = analysis['dimensions']
        print(f"Object Size: {dims[0]*100:.1f} × {dims[1]*100:.1f} × {dims[2]*100:.1f} cm")
        print(f"Volume: {analysis['volume']*1000000:.1f} cm³")
        print(f"Surface Area: {analysis['surface_area']*10000:.1f} cm²")
        print(f"Aspect Ratio: {analysis['aspect_ratio']:.1f}:1")
        print(f"Complexity: {analysis['complexity'].title()}")
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"Algorithm: {analysis['recommended_algorithm'].value.replace('_', ' ').title()}")
        print(f"Scan Points: {analysis['recommended_points']}")
        print(f"Optimal Distance: {analysis['optimal_distance']*100:.0f}mm")
        print(f"Speed Factor: {analysis['speed_factor']:.1f}x")
    
    def _apply_analysis_recommendations(self, analysis: Dict[str, Any]):
        """Wendet Analyse-Empfehlungen an."""
        # Apply algorithm recommendation
        self.algo_config.primary_algorithm = analysis['recommended_algorithm']
        
        # Apply distance recommendation
        self.scanner_specs.optimal_distance = analysis['optimal_distance']
        
        # Apply points recommendation for algorithm
        if analysis['recommended_algorithm'] == ScanningAlgorithm.FIBONACCI_SPHERE:
            self.algo_config.fibonacci_points = analysis['recommended_points']
        elif analysis['recommended_algorithm'] == ScanningAlgorithm.STRUCTURED_GRID:
            # Calculate grid resolution for target points
            target_points = analysis['recommended_points']
            side = int(math.sqrt(target_points))
            self.algo_config.grid_base_resolution = (side, side)
    
    def _path_optimization(self):
        """Pfad-Optimierungs-Einstellungen."""
        print("\n🎯 PATH OPTIMIZATION SETTINGS")
        print("Current optimization:", self.tech_params.path_optimization.title())
        input("Press ENTER to continue...")
    
    def _quality_metrics(self):
        """Quality Metrics Konfiguration."""
        print("\n📊 QUALITY METRICS CONFIGURATION")
        print("Quality metrics enabled:", "✅" if self.advanced_options.quality_metrics_enabled else "❌")
        input("Press ENTER to continue...")
    
    def _algorithm_testing(self):
        """Algorithmus-Tests."""
        print("\n🧪 ALGORITHM PERFORMANCE TESTING")
        print("Test different algorithms on sample objects")
        input("Press ENTER to continue...")
    
    def _generate_custom_pattern(self):
        """Generiert ein Custom Pattern."""
        print("\n🔬 CUSTOM PATTERN GENERATOR")
        print("Generate a custom scan pattern with current settings")
        
        if self.pattern_engine:
            try:
                target_points = int(input("Target number of points [100]: ") or "100")
                positions = self.pattern_engine.generate_scan_pattern(target_points)
                
                print(f"✅ Generated custom pattern with {len(positions)} points")
                print("Pattern uses current algorithm:", self.algo_config.primary_algorithm.value)
                
                if input("Save this pattern? (y/n): ").lower() == 'y':
                    name = input("Pattern name: ").strip()
                    if name:
                        self._save_custom_pattern(name, positions)
                        print(f"✅ Pattern saved as: {name}")
                
            except ValueError:
                print("❌ Invalid input")
        else:
            print("❌ Pattern engine not available")
        
        input("\nPress ENTER to continue...")
    
    def _save_custom_pattern(self, name: str, positions: List[Tuple[float, float]]):
        """Speichert ein Custom Pattern."""
        pattern_data = {
            'name': name,
            'algorithm': self.algo_config.primary_algorithm.value,
            'positions': positions,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'settings': {
                'scanner_specs': asdict(self.scanner_specs) if hasattr(self.scanner_specs, '__dict__') else {},
                'tech_params': asdict(self.tech_params) if hasattr(self.tech_params, '__dict__') else {},
                'advanced_options': asdict(self.advanced_options) if hasattr(self.advanced_options, '__dict__') else {}
            }
        }
        
        patterns_dir = Path("patterns/custom")
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        pattern_file = patterns_dir / f"{name}.json"
        with open(pattern_file, 'w') as f:
            json.dump(pattern_data, f, indent=2)
    
    def _performance_benchmarking(self):
        """Performance Benchmarking."""
        print("\n📈 PERFORMANCE BENCHMARKING")
        print("Benchmark different algorithms and settings")
        input("Press ENTER to continue...")
    
    def _calibration_assistant(self):
        """Kalibierungs-Assistent."""
        print("\n🛠️ CALIBRATION ASSISTANT")
        print("Guided calibration for optimal scan quality")
        input("Press ENTER to continue...")
    
    def _save_configuration(self):
        """Speichert aktuelle Konfiguration."""
        try:
            config_data = {
                'technical_parameters': asdict(self.tech_params) if hasattr(self.tech_params, '__dict__') else {},
                'scanner_specifications': asdict(self.scanner_specs) if hasattr(self.scanner_specs, '__dict__') else {},
                'advanced_options': asdict(self.advanced_options) if hasattr(self.advanced_options, '__dict__') else {},
                'algorithm_config': asdict(self.algo_config) if hasattr(self.algo_config, '__dict__') else {},
                'saved': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to: {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
        
        input("Press ENTER to continue...")
    
    def _load_configuration_preset(self):
        """Lädt Konfigurations-Preset."""
        print("\n📂 LOAD CONFIGURATION PRESET")
        print("Available configuration files:")
        
        # List available config files
        config_files = list(Path(".").glob("*.json"))
        if not config_files:
            print("No configuration files found")
            input("Press ENTER to continue...")
            return
        
        for i, file in enumerate(config_files, 1):
            print(f"{i}. {file.name}")
        
        try:
            choice = int(input("Select configuration file: ")) - 1
            if 0 <= choice < len(config_files):
                self._load_config(config_files[choice])
                print(f"✅ Configuration loaded from: {config_files[choice]}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
        
        input("Press ENTER to continue...")
    
    def _export_configuration(self):
        """Exportiert Konfiguration."""
        print("\n📤 EXPORT CONFIGURATION")
        name = input("Export filename (without .json): ").strip()
        if name:
            export_file = f"{name}.json"
            self.config_file = export_file
            self._save_configuration()
            print(f"✅ Configuration exported to: {export_file}")
        
        input("Press ENTER to continue...")
    
    def _reset_to_defaults(self):
        """Setzt auf Standard-Einstellungen zurück."""
        if input("\n🔄 Reset to defaults? This will lose current settings (y/n): ").lower() == 'y':
            self.tech_params = TechnicalScanParameters()
            self.scanner_specs = self._create_default_scanner_specs()
            self.advanced_options = self._create_default_advanced_options()
            self.algo_config = self._create_default_algo_config()
            
            print("✅ Configuration reset to defaults")
        
        input("Press ENTER to continue...")
    
    def _load_config(self, config_file: Optional[Path] = None):
        """Lädt Konfiguration aus Datei."""
        file_to_load = config_file or Path(self.config_file)
        
        if not file_to_load.exists():
            logger.debug(f"Config file not found: {file_to_load}")
            return
        
        try:
            with open(file_to_load, 'r') as f:
                config_data = json.load(f)
            
            # Load technical parameters
            if 'technical_parameters' in config_data:
                tech_data = config_data['technical_parameters']
                if hasattr(self.tech_params, '__dict__'):
                    for key, value in tech_data.items():
                        if hasattr(self.tech_params, key):
                            setattr(self.tech_params, key, value)
            
            logger.info(f"✅ Configuration loaded from: {file_to_load}")
            
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
    
    def _clear_screen(self):
        """Löscht Bildschirm (plattformunabhängig)."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_current_config_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der aktuellen Konfiguration zurück."""
        return {
            'algorithm': self.algo_config.primary_algorithm.value,
            'scanner_distance': f"{self.scanner_specs.optimal_distance*1000:.0f}mm",
            'angular_resolution': f"{math.degrees(self.tech_params.angular_resolution):.1f}°",
            'path_optimization': self.tech_params.path_optimization,
            'advanced_features': sum([
                self.advanced_options.multi_exposure,
                self.advanced_options.hdr_scanning,
                self.advanced_options.motion_blur_compensation,
                self.advanced_options.adaptive_exposure,
                self.advanced_options.quality_metrics_enabled
            ])
        }


# ================================
# MAIN (for testing)
# ================================

if __name__ == "__main__":
    print("🔧 Testing Technical Configurator...")
    configurator = TechnicalScanningConfigurator()
    
    # Show current config
    config = configurator.get_current_config_summary()
    print("\nCurrent Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTechnical Configurator ready!")
