#!/usr/bin/env python3
"""
RoArm M3 - Technical Scan Patterns Engine
Advanced scanning algorithms and optimization
Version 3.1.0 - Complete Implementation
"""

import math
import random
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ================================
# ENUMS & CONSTANTS
# ================================

class ScanningAlgorithm(Enum):
    """Verfügbare Scanning-Algorithmen."""
    STRUCTURED_GRID = "structured_grid"
    FIBONACCI_SPHERE = "fibonacci_sphere"
    ADAPTIVE_DENSITY = "adaptive_density"
    OPTIMIZED_SPIRAL = "optimized_spiral"
    HIERARCHICAL_SUBDIVISION = "hierarchical_subdivision"
    MONTE_CARLO_SAMPLING = "monte_carlo_sampling"


class PathOptimizationType(Enum):
    """Pfad-Optimierungs-Algorithmen."""
    NONE = "none"
    GREEDY = "greedy"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    TWO_OPT = "two_opt"


class QualityMetric(Enum):
    """Qualitäts-Metriken für Scan-Bewertung."""
    COVERAGE_UNIFORMITY = "coverage_uniformity"
    PATH_EFFICIENCY = "path_efficiency"
    MOTION_SMOOTHNESS = "motion_smoothness"
    SCAN_TIME = "scan_time"
    POSITION_ACCURACY = "position_accuracy"


# ================================
# DATA CLASSES
# ================================

@dataclass
class TechnicalScanParameters:
    """Technische Parameter für erweiterte Scans."""
    # Grundlegende Parameter
    angular_resolution: float = math.radians(5.0)  # 5° in Radiant
    path_optimization: str = "greedy"
    servo_safety_margin: float = math.radians(5.0)  # 5° Safety margin
    
    # Erweiterte Parameter
    adaptive_threshold: float = 0.02
    max_acceleration: float = 2.0  # rad/s²
    jerk_limit: float = 5.0  # rad/s³
    settle_time_factor: float = 1.0
    
    # Qualitäts-Parameter
    quality_target: float = 0.95  # 0.0-1.0
    coverage_redundancy: float = 0.1  # 10% Überlappung
    
    # Performance-Parameter
    max_computation_time: float = 30.0  # Sekunden
    parallel_processing: bool = True


@dataclass
class ScannerSpecifications:
    """Scanner Hardware-Spezifikationen."""
    # Field of View
    fov_horizontal: float = math.radians(60)  # 60° horizontal
    fov_vertical: float = math.radians(45)    # 45° vertikal
    
    # Arbeitsabstände
    min_distance: float = 0.08   # 8cm
    max_distance: float = 0.30   # 30cm
    optimal_distance: float = 0.15  # 15cm
    
    # Auflösung & Qualität
    resolution: Tuple[int, int] = (1920, 1080)
    depth_resolution: float = 0.001  # 1mm Tiefenauflösung
    
    # Überlappung & Coverage
    optimal_overlap_percentage: float = 0.25  # 25%
    min_overlap_percentage: float = 0.15     # 15%
    
    # Performance
    capture_time: float = 0.5  # Sekunden pro Aufnahme
    processing_time: float = 0.2  # Verarbeitungszeit


@dataclass
class AdvancedScanningOptions:
    """Erweiterte Scanning-Optionen."""
    # Multi-Pass Scanning
    multi_pass_enabled: bool = False
    pass_count: int = 2
    pass_offset_angle: float = math.radians(45)
    
    # HDR & Belichtung
    multi_exposure: bool = False
    exposure_count: int = 3
    hdr_scanning: bool = False
    adaptive_exposure: bool = True
    
    # Motion & Stabilization
    motion_blur_compensation: bool = True
    stabilization_time: float = 0.3
    vibration_dampening: bool = True
    
    # Quality Control
    real_time_quality_check: bool = True
    auto_retry_failed_scans: bool = True
    quality_metrics_enabled: bool = True
    
    # Advanced Features
    predictive_positioning: bool = False
    machine_learning_optimization: bool = False


@dataclass
class AlgorithmConfiguration:
    """Konfiguration für Scan-Algorithmen."""
    # Primärer Algorithmus
    primary_algorithm: ScanningAlgorithm = ScanningAlgorithm.FIBONACCI_SPHERE
    
    # Structured Grid Parameter
    grid_base_resolution: Tuple[int, int] = (12, 8)
    grid_refinement_levels: int = 2
    grid_adaptive_subdivision: bool = True
    
    # Fibonacci Sphere Parameter
    fibonacci_points: int = 120
    fibonacci_golden_ratio: float = 1.618033988749895
    fibonacci_spiral_offset: float = 0.0
    
    # Adaptive Density Parameter
    adaptive_base_density: float = 1.0
    adaptive_edge_detection: bool = True
    adaptive_curvature_weighting: float = 0.3
    adaptive_complexity_factor: float = 1.5
    
    # Spiral Parameter
    spiral_turns: float = 4.0
    spiral_tightness: float = 1.0
    spiral_elevation_variance: float = 0.2
    
    # Monte Carlo Parameter
    monte_carlo_samples: int = 1000
    monte_carlo_convergence_threshold: float = 0.01
    monte_carlo_max_iterations: int = 50


# ================================
# CORE PATTERN GENERATION ENGINE
# ================================

class PatternGenerationEngine:
    """
    Kern-Engine für erweiterte Pattern-Generierung.
    Implementiert verschiedene technische Scan-Algorithmen.
    """
    
    def __init__(self, config: AlgorithmConfiguration):
        """
        Initialisiert die Pattern-Engine.
        
        Args:
            config: Algorithmus-Konfiguration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PatternEngine")
        
        # Cache für generierte Patterns
        self._pattern_cache: Dict[str, List[Tuple[float, float]]] = {}
        
        self.logger.info(f"🧠 Pattern Engine initialized with {config.primary_algorithm.value}")
    
    def generate_scan_pattern(self, target_positions: int) -> List[Tuple[float, float]]:
        """
        Generiert Scan-Pattern basierend auf konfiguriertem Algorithmus.
        
        Args:
            target_positions: Gewünschte Anzahl Scan-Positionen
            
        Returns:
            List[Tuple[float, float]]: Liste von (theta, phi) Koordinaten
        """
        # Cache-Key generieren
        cache_key = f"{self.config.primary_algorithm.value}_{target_positions}"
        
        if cache_key in self._pattern_cache:
            self.logger.debug(f"🔄 Using cached pattern: {cache_key}")
            return self._pattern_cache[cache_key]
        
        # Pattern generieren basierend auf Algorithmus
        if self.config.primary_algorithm == ScanningAlgorithm.STRUCTURED_GRID:
            positions = self._generate_structured_grid(target_positions)
        elif self.config.primary_algorithm == ScanningAlgorithm.FIBONACCI_SPHERE:
            positions = self._generate_fibonacci_sphere(target_positions)
        elif self.config.primary_algorithm == ScanningAlgorithm.ADAPTIVE_DENSITY:
            positions = self._generate_adaptive_density(target_positions)
        elif self.config.primary_algorithm == ScanningAlgorithm.OPTIMIZED_SPIRAL:
            positions = self._generate_optimized_spiral(target_positions)
        elif self.config.primary_algorithm == ScanningAlgorithm.HIERARCHICAL_SUBDIVISION:
            positions = self._generate_hierarchical_subdivision(target_positions)
        elif self.config.primary_algorithm == ScanningAlgorithm.MONTE_CARLO_SAMPLING:
            positions = self._generate_monte_carlo_sampling(target_positions)
        else:
            self.logger.warning(f"⚠️ Unknown algorithm: {self.config.primary_algorithm}")
            positions = self._generate_fibonacci_sphere(target_positions)  # Fallback
        
        # Cache speichern
        self._pattern_cache[cache_key] = positions
        
        self.logger.info(f"✅ Generated {len(positions)} positions using {self.config.primary_algorithm.value}")
        return positions
    
    def _generate_structured_grid(self, target_positions: int) -> List[Tuple[float, float]]:
        """Generiert strukturiertes Grid-Pattern."""
        h_res, v_res = self.config.grid_base_resolution
        
        # Anpassung der Auflösung an Ziel-Punktanzahl
        total_base = h_res * v_res
        scale_factor = math.sqrt(target_positions / total_base)
        
        h_res = int(h_res * scale_factor)
        v_res = int(v_res * scale_factor)
        
        positions = []
        
        for v in range(v_res):
            for h in range(h_res):
                # Theta: 0 bis 2π (Azimuth)
                theta = 2 * math.pi * h / h_res
                
                # Phi: -π/3 bis π/3 (Elevation, eingeschränkt für Scanner)
                phi_range = 2 * math.pi / 3  # 120° Gesamtbereich
                phi = -phi_range/2 + phi_range * v / (v_res - 1) if v_res > 1 else 0
                
                positions.append((theta, phi))
        
        # Adaptive Subdivision wenn aktiviert
        if self.config.grid_adaptive_subdivision:
            positions.extend(self._add_subdivision_points(positions))
        
        return positions
    
    def _generate_fibonacci_sphere(self, n_points: int) -> List[Tuple[float, float]]:
        """Generiert Fibonacci-Spirale auf Sphäre für optimale Punktverteilung."""
        golden_ratio = self.config.fibonacci_golden_ratio
        positions = []
        
        for i in range(n_points):
            # Fibonacci-Spirale auf Einheitssphäre
            y = 1 - (i / (n_points - 1)) * 2  # y von 1 bis -1
            radius_at_y = math.sqrt(1 - y * y)
            
            # Goldener Winkel für optimale Verteilung
            theta = 2 * math.pi * i / golden_ratio + self.config.fibonacci_spiral_offset
            theta = theta % (2 * math.pi)  # Normalisieren auf 0-2π
            
            # Phi aus y-Koordinate berechnen
            phi = math.asin(y)
            
            # Einschränkung auf Scanner-FOV (±60°)
            phi = max(-math.pi/3, min(math.pi/3, phi))
            
            positions.append((theta, phi))
        
        return positions
    
    def _generate_adaptive_density(self, target_positions: int) -> List[Tuple[float, float]]:
        """Generiert adaptive Dichte-Verteilung basierend auf Objektkomplexität."""
        # Basis-Grid als Startpunkt
        base_positions = self._generate_structured_grid(target_positions // 2)
        
        # Zusätzliche Punkte in kritischen Bereichen
        edge_positions = []
        
        if self.config.adaptive_edge_detection:
            # Zusätzliche Punkte an Objekt-Kanten (vereinfacht)
            edge_angles = [0, math.pi/2, math.pi, 3*math.pi/2]  # Kardinal-Richtungen
            
            for angle in edge_angles:
                for phi_offset in [-0.2, 0.0, 0.2]:  # Verschiedene Höhen
                    # Krümmungsgewichtung anwenden
                    density_factor = 1.0 + self.config.adaptive_curvature_weighting
                    
                    for _ in range(int(self.config.adaptive_complexity_factor * density_factor)):
                        # Kleine Variation um Edge-Punkte
                        theta_var = angle + random.uniform(-0.1, 0.1)
                        phi_var = phi_offset + random.uniform(-0.05, 0.05)
                        
                        # Auf gültigen Bereich begrenzen
                        phi_var = max(-math.pi/3, min(math.pi/3, phi_var))
                        
                        edge_positions.append((theta_var, phi_var))
        
        return base_positions + edge_positions
    
    def _generate_optimized_spiral(self, target_positions: int) -> List[Tuple[float, float]]:
        """Generiert optimierte Spirale mit variabler Dichte."""
        positions = []
        
        points_per_turn = target_positions / self.config.spiral_turns
        
        for i in range(target_positions):
            t = i / target_positions
            
            # Spirale mit variabler Dichte
            angle = 2 * math.pi * self.config.spiral_turns * t
            
            # Tightness-Parameter für Spiraldichte
            radius_factor = t ** self.config.spiral_tightness
            
            # Elevation mit Variation
            base_elevation = 0.0
            elevation_var = self.config.spiral_elevation_variance * math.sin(4 * math.pi * t)
            phi = base_elevation + elevation_var
            
            # Auf Scanner-FOV begrenzen
            phi = max(-math.pi/3, min(math.pi/3, phi))
            
            positions.append((angle % (2 * math.pi), phi))
        
        return positions
    
    def _generate_hierarchical_subdivision(self, target_positions: int) -> List[Tuple[float, float]]:
        """Generiert hierarchische Unterteilung für adaptive Dichte."""
        # Start mit groben Grid
        initial_resolution = int(math.sqrt(target_positions / 4))
        coarse_positions = self._generate_structured_grid(initial_resolution * initial_resolution)
        
        # Hierarchische Verfeinerung
        refined_positions = list(coarse_positions)
        
        levels = min(3, int(math.log2(target_positions / len(coarse_positions))))
        
        for level in range(levels):
            new_positions = []
            
            for i in range(len(refined_positions) - 1):
                theta1, phi1 = refined_positions[i]
                theta2, phi2 = refined_positions[i + 1]
                
                # Interpolation zwischen benachbarten Punkten
                mid_theta = (theta1 + theta2) / 2
                mid_phi = (phi1 + phi2) / 2
                
                # Auf Scanner-FOV begrenzen
                mid_phi = max(-math.pi/3, min(math.pi/3, mid_phi))
                
                new_positions.append((mid_theta, mid_phi))
            
            refined_positions.extend(new_positions[:target_positions - len(refined_positions)])
            
            if len(refined_positions) >= target_positions:
                break
        
        return refined_positions[:target_positions]
    
    def _generate_monte_carlo_sampling(self, target_positions: int) -> List[Tuple[float, float]]:
        """Generiert Monte-Carlo Sampling für statistische Abdeckung."""
        positions = []
        
        for _ in range(min(target_positions, self.config.monte_carlo_samples)):
            # Zufällige Winkel im gültigen Bereich
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-math.pi/3, math.pi/3)  # Scanner-FOV
            
            positions.append((theta, phi))
        
        # Optimierung durch Konvergenz-Kriterium
        if len(positions) < target_positions:
            # Fülle mit optimierten Punkten auf
            additional_points = self._optimize_monte_carlo_distribution(
                positions, target_positions - len(positions)
            )
            positions.extend(additional_points)
        
        return positions[:target_positions]
    
    def _add_subdivision_points(self, base_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Fügt Subdivision-Punkte für höhere Dichte hinzu."""
        subdivision_points = []
        
        for i in range(min(len(base_positions) - 1, self.config.grid_refinement_levels * 10)):
            theta1, phi1 = base_positions[i]
            theta2, phi2 = base_positions[i + 1] if i + 1 < len(base_positions) else base_positions[0]
            
            # Mittelpunkt berechnen
            mid_theta = (theta1 + theta2) / 2
            mid_phi = (phi1 + phi2) / 2
            
            subdivision_points.append((mid_theta, mid_phi))
        
        return subdivision_points
    
    def _optimize_monte_carlo_distribution(self, existing_positions: List[Tuple[float, float]], 
                                         additional_count: int) -> List[Tuple[float, float]]:
        """Optimiert Monte-Carlo Verteilung für bessere Abdeckung."""
        new_positions = []
        
        for _ in range(additional_count):
            best_theta, best_phi = 0, 0
            max_min_distance = 0
            
            # Mehrere Kandidaten testen
            for _ in range(20):  # Reduziert für Performance
                candidate_theta = random.uniform(0, 2 * math.pi)
                candidate_phi = random.uniform(-math.pi/3, math.pi/3)
                
                # Minimaler Abstand zu existierenden Punkten
                min_distance = float('inf')
                for ex_theta, ex_phi in existing_positions + new_positions:
                    distance = math.sqrt((candidate_theta - ex_theta)**2 + (candidate_phi - ex_phi)**2)
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_theta, best_phi = candidate_theta, candidate_phi
            
            new_positions.append((best_theta, best_phi))
        
        return new_positions


# ================================
# PATH OPTIMIZATION ENGINE
# ================================

class OptimizationEngine:
    """
    Engine für Pfad- und Parameter-Optimierung.
    Implementiert verschiedene Optimierungsalgorithmen.
    """
    
    def __init__(self, tech_params: TechnicalScanParameters):
        """
        Initialisiert die Optimization-Engine.
        
        Args:
            tech_params: Technische Parameter
        """
        self.params = tech_params
        self.logger = logging.getLogger(f"{__name__}.OptimizationEngine")
        
        self.logger.info(f"🎯 Optimization Engine initialized with {tech_params.path_optimization}")
    
    def optimize_scan_path(self, positions: List[Tuple[float, float]]) -> List[int]:
        """
        Optimiert die Reihenfolge der Scan-Positionen.
        
        Args:
            positions: Liste von Scan-Positionen (theta, phi)
            
        Returns:
            List[int]: Optimierte Reihenfolge als Indizes
        """
        if self.params.path_optimization == "greedy":
            return self._greedy_path_optimization(positions)
        elif self.params.path_optimization == "genetic":
            return self._genetic_path_optimization(positions)
        elif self.params.path_optimization == "simulated_annealing":
            return self._simulated_annealing_optimization(positions)
        elif self.params.path_optimization == "two_opt":
            return self._two_opt_optimization(positions)
        else:
            return list(range(len(positions)))  # Ursprüngliche Reihenfolge
    
    def _greedy_path_optimization(self, positions: List[Tuple[float, float]]) -> List[int]:
        """Greedy Nearest-Neighbor Optimierung."""
        if not positions:
            return []
        
        unvisited = set(range(len(positions)))
        path = [0]  # Start bei Position 0
        unvisited.remove(0)
        
        current_pos = 0
        
        while unvisited:
            nearest_idx = min(unvisited, 
                            key=lambda i: self._calculate_movement_cost(positions[current_pos], positions[i]))
            
            path.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current_pos = nearest_idx
        
        self.logger.debug(f"🎯 Greedy optimization: {len(path)} positions")
        return path
    
    def _genetic_path_optimization(self, positions: List[Tuple[float, float]]) -> List[int]:
        """Genetischer Algorithmus für Pfad-Optimierung."""
        if len(positions) < 4:
            return list(range(len(positions)))
        
        population_size = min(50, len(positions) * 2)
        generations = min(100, len(positions) * 5)
        mutation_rate = 0.1
        
        # Initiale Population
        population = []
        for _ in range(population_size):
            individual = list(range(len(positions)))
            random.shuffle(individual)
            population.append(individual)
        
        # Evolution
        for generation in range(generations):
            # Fitness bewerten
            fitness_scores = [1.0 / (1.0 + self._calculate_path_cost(positions, individual)) 
                            for individual in population]
            
            # Selektion (Tournament)
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._order_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate_path(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Besten Pfad auswählen
        final_fitness = [1.0 / (1.0 + self._calculate_path_cost(positions, individual)) 
                        for individual in population]
        best_idx = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
        
        self.logger.debug(f"🧬 Genetic optimization: {len(population[best_idx])} positions, {generations} generations")
        return population[best_idx]
    
    def _simulated_annealing_optimization(self, positions: List[Tuple[float, float]]) -> List[int]:
        """Simulated Annealing für Pfad-Optimierung."""
        if not positions:
            return []
        
        # Parameter
        initial_temp = 100.0
        cooling_rate = 0.95
        min_temp = 0.01
        max_iterations = len(positions) * 20
        
        # Start mit zufälliger Lösung
        current_path = list(range(len(positions)))
        random.shuffle(current_path)
        current_cost = self._calculate_path_cost(positions, current_path)
        
        best_path = current_path.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Neue Lösung durch 2-opt Swap
            new_path = current_path.copy()
            i, j = sorted(random.sample(range(len(positions)), 2))
            new_path[i:j+1] = reversed(new_path[i:j+1])
            
            new_cost = self._calculate_path_cost(positions, new_path)
            cost_diff = new_cost - current_cost
            
            # Akzeptanzkriterium
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_path = new_path
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_path = current_path.copy()
                    best_cost = current_cost
            
            temperature *= cooling_rate
            iteration += 1
        
        self.logger.debug(f"🌡️ Simulated annealing: {len(best_path)} positions, {iteration} iterations")
        return best_path
    
    def _two_opt_optimization(self, positions: List[Tuple[float, float]]) -> List[int]:
        """2-opt Verbesserung für Pfad-Optimierung."""
        path = list(range(len(positions)))
        improved = True
        iterations = 0
        max_iterations = len(positions) * 10
        
        while improved and iterations < max_iterations:
            improved = False
            
            for i in range(len(path) - 1):
                for j in range(i + 2, len(path)):
                    if j == len(path) - 1 and i == 0:
                        continue  # Skip if it would reverse entire path
                    
                    # Berechne Kostendifferenz
                    old_cost = (self._calculate_movement_cost(positions[path[i]], positions[path[i+1]]) +
                              self._calculate_movement_cost(positions[path[j]], positions[path[(j+1) % len(path)]]))
                    
                    new_cost = (self._calculate_movement_cost(positions[path[i]], positions[path[j]]) +
                              self._calculate_movement_cost(positions[path[i+1]], positions[path[(j+1) % len(path)]]))
                    
                    if new_cost < old_cost:
                        # 2-opt Swap durchführen
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
            
            iterations += 1
        
        self.logger.debug(f"🔄 2-opt optimization: {len(path)} positions, {iterations} iterations")
        return path
    
    def _calculate_movement_cost(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Berechnet Bewegungskosten zwischen zwei Positionen."""
        theta1, phi1 = pos1
        theta2, phi2 = pos2
        
        # Euklidische Distanz im Winkelraum
        d_theta = abs(theta2 - theta1)
        d_theta = min(d_theta, 2 * math.pi - d_theta)  # Kürzester Weg über 0/2π
        d_phi = abs(phi2 - phi1)
        
        # Gewichtung: Theta-Bewegung ist meist schneller als Phi
        theta_weight = 1.0
        phi_weight = 1.5  # Phi-Bewegung kostet mehr
        
        cost = math.sqrt((theta_weight * d_theta)**2 + (phi_weight * d_phi)**2)
        
        # Penalty für große Sprünge
        if cost > math.pi / 2:  # 90° Schwelle
            cost *= 1.5
        
        return cost
    
    def _calculate_path_cost(self, positions: List[Tuple[float, float]], path: List[int]) -> float:
        """Berechnet Gesamtkosten eines Pfads."""
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._calculate_movement_cost(positions[path[i]], positions[path[i + 1]])
        
        return total_cost
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Tournament-Selektion für genetischen Algorithmus."""
        tournament_size = min(5, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX) für genetischen Algorithmus."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Kopiere Segment von parent1
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        # Fülle Rest von parent2 in Reihenfolge
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer] = city
                pointer = (pointer + 1) % size
        
        return child
    
    def _mutate_path(self, path: List[int]) -> List[int]:
        """Mutation für genetischen Algorithmus."""
        mutated = path.copy()
        
        # Swap Mutation
        if len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated


# ================================
# QUALITY ASSESSMENT ENGINE
# ================================

class QualityAssessmentEngine:
    """
    Engine für Qualitätsbewertung und Metriken.
    """
    
    def __init__(self, scanner_specs: ScannerSpecifications):
        """
        Initialisiert die Quality Assessment Engine.
        
        Args:
            scanner_specs: Scanner-Spezifikationen
        """
        self.scanner_specs = scanner_specs
        self.logger = logging.getLogger(f"{__name__}.QualityEngine")
    
    def assess_pattern_quality(self, positions: List[Tuple[float, float]], 
                             path: List[int]) -> Dict[str, float]:
        """
        Bewertet die Qualität eines Scan-Patterns.
        
        Args:
            positions: Scan-Positionen
            path: Optimierter Pfad
            
        Returns:
            Dict[str, float]: Qualitäts-Metriken
        """
        metrics = {}
        
        # Coverage Uniformity
        metrics['coverage_uniformity'] = self._assess_coverage_uniformity(positions)
        
        # Path Efficiency
        metrics['path_efficiency'] = self._assess_path_efficiency(positions, path)
        
        # Motion Smoothness
        metrics['motion_smoothness'] = self._assess_motion_smoothness(positions, path)
        
        # Estimated Scan Time
        metrics['estimated_scan_time'] = self._estimate_scan_time(positions, path)
        
        # Overall Quality Score
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        self.logger.debug(f"📊 Quality assessment: {len(positions)} positions, overall={metrics['overall_quality']:.3f}")
        
        return metrics
    
    def _assess_coverage_uniformity(self, positions: List[Tuple[float, float]]) -> float:
        """Bewertet die Gleichmäßigkeit der Abdeckung."""
        if len(positions) < 3:
            return 1.0
        
        # Berechne Abstände zwischen allen Punktpaaren
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                theta1, phi1 = positions[i]
                theta2, phi2 = positions[j]
                
                d_theta = abs(theta2 - theta1)
                d_theta = min(d_theta, 2 * math.pi - d_theta)
                d_phi = abs(phi2 - phi1)
                
                distance = math.sqrt(d_theta**2 + d_phi**2)
                distances.append(distance)
        
        # Standardabweichung der Abstände (niedriger = gleichmäßiger)
        mean_distance = sum(distances) / len(distances)
        variance = sum((d - mean_distance)**2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)
        
        # Normalisiere auf 0-1 (1 = sehr gleichmäßig)
        uniformity = max(0.0, 1.0 - (std_dev / mean_distance))
        
        return uniformity
    
    def _assess_path_efficiency(self, positions: List[Tuple[float, float]], path: List[int]) -> float:
        """Bewertet die Effizienz des Pfads."""
        if len(path) < 2:
            return 1.0
        
        # Berechne tatsächliche Pfadlänge
        actual_cost = 0.0
        for i in range(len(path) - 1):
            pos1 = positions[path[i]]
            pos2 = positions[path[i + 1]]
            
            theta1, phi1 = pos1
            theta2, phi2 = pos2
            
            d_theta = abs(theta2 - theta1)
            d_theta = min(d_theta, 2 * math.pi - d_theta)
            d_phi = abs(phi2 - phi1)
            
            actual_cost += math.sqrt(d_theta**2 + d_phi**2)
        
        # Berechne theoretisch minimale Pfadlänge (MST Approximation)
        total_angular_range = 2 * math.pi + 2 * math.pi / 3  # Theta + Phi Bereiche
        theoretical_min = total_angular_range / len(positions)
        
        # Effizienz als Verhältnis
        efficiency = max(0.0, min(1.0, theoretical_min / actual_cost))
        
        return efficiency
    
    def _assess_motion_smoothness(self, positions: List[Tuple[float, float]], path: List[int]) -> float:
        """Bewertet die Glätte der Bewegung."""
        if len(path) < 3:
            return 1.0
        
        # Berechne Richtungsänderungen
        direction_changes = []
        
        for i in range(1, len(path) - 1):
            pos_prev = positions[path[i - 1]]
            pos_curr = positions[path[i]]
            pos_next = positions[path[i + 1]]
            
            # Vektoren
            vec1 = (pos_curr[0] - pos_prev[0], pos_curr[1] - pos_prev[1])
            vec2 = (pos_next[0] - pos_curr[0], pos_next[1] - pos_curr[1])
            
            # Winkel zwischen Vektoren
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
            mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp
                angle = math.acos(cos_angle)
                direction_changes.append(angle)
        
        if not direction_changes:
            return 1.0
        
        # Glätte basierend auf durchschnittlicher Richtungsänderung
        avg_change = sum(direction_changes) / len(direction_changes)
        smoothness = max(0.0, 1.0 - (avg_change / math.pi))  # Weniger Änderung = glätter
        
        return smoothness
    
    def _estimate_scan_time(self, positions: List[Tuple[float, float]], path: List[int]) -> float:
        """Schätzt die gesamte Scan-Zeit."""
        if not path:
            return 0.0
        
        total_time = 0.0
        
        # Zeit für jede Position
        total_time += len(positions) * self.scanner_specs.capture_time
        total_time += len(positions) * self.scanner_specs.processing_time
        
        # Zeit für Bewegungen
        for i in range(len(path) - 1):
            pos1 = positions[path[i]]
            pos2 = positions[path[i + 1]]
            
            # Vereinfachte Bewegungszeit basierend auf Winkelabstand
            theta1, phi1 = pos1
            theta2, phi2 = pos2
            
            d_theta = abs(theta2 - theta1)
            d_theta = min(d_theta, 2 * math.pi - d_theta)
            d_phi = abs(phi2 - phi1)
            
            angular_distance = math.sqrt(d_theta**2 + d_phi**2)
            
            # Angenommene Winkelgeschwindigkeit: 1 rad/s
            movement_time = angular_distance / 1.0
            total_time += movement_time
        
        return total_time
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Berechnet Gesamt-Qualitätsscore."""
        # Gewichtungen für verschiedene Metriken
        weights = {
            'coverage_uniformity': 0.3,
            'path_efficiency': 0.3,
            'motion_smoothness': 0.2,
            'estimated_scan_time': 0.2  # Invertiert: kürzere Zeit = besser
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                
                # Invertiere Scan-Zeit (kürzere Zeit = höhere Qualität)
                if metric == 'estimated_scan_time':
                    # Normalisiere auf typischen Bereich (60-600 Sekunden)
                    normalized_time = max(0.0, min(1.0, (600 - value) / 540))
                    weighted_sum += weight * normalized_time
                else:
                    weighted_sum += weight * value
                
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


# ================================
# MAIN DEMO FUNCTION
# ================================

def demo_technical_configurator():
    """Demonstriert den technischen Konfigurator."""
    
    # Setup
    algo_config = AlgorithmConfiguration()
    tech_params = TechnicalScanParameters()
    scanner_specs = ScannerSpecifications()
    
    # Engines initialisieren
    pattern_engine = PatternGenerationEngine(algo_config)
    optimization_engine = OptimizationEngine(tech_params)
    quality_engine = QualityAssessmentEngine(scanner_specs)
    
    print("🔧 TECHNISCHER KONFIGURATOR DEMO")
    print("=" * 50)
    
    # Zeige aktuelle Konfiguration
    print("\n📡 Aktuelle Scanner-Spezifikationen:")
    print(f"   FOV: {math.degrees(scanner_specs.fov_horizontal):.1f}° × {math.degrees(scanner_specs.fov_vertical):.1f}°")
    print(f"   Arbeitsabstand: {scanner_specs.min_distance*1000:.0f}-{scanner_specs.max_distance*1000:.0f}mm (optimal: {scanner_specs.optimal_distance*1000:.0f}mm)")
    print(f"   Überlappung: {scanner_specs.optimal_overlap_percentage*100:.0f}%")
    
    print("\n⚙️ Aktuelle technische Parameter:")
    print(f"   Winkel-Auflösung: {math.degrees(tech_params.angular_resolution):.1f}°")
    print(f"   Pfad-Optimierung: {tech_params.path_optimization}")
    print(f"   Sicherheitsmarge: {math.degrees(tech_params.servo_safety_margin):.1f}°")
    
    print("\n🧠 Aktueller Algorithmus:")
    print(f"   Primär: {algo_config.primary_algorithm.value}")
    
    if algo_config.primary_algorithm == ScanningAlgorithm.STRUCTURED_GRID:
        print(f"   Grid-Auflösung: {algo_config.grid_base_resolution}")
    elif algo_config.primary_algorithm == ScanningAlgorithm.FIBONACCI_SPHERE:
        print(f"   Fibonacci-Punkte: {algo_config.fibonacci_points}")
    
    # Beispiel-Pattern generieren
    print(f"\n🔍 Beispiel-Pattern-Generierung (100 Punkte):")
    positions = pattern_engine.generate_scan_pattern(100)
    print(f"   Generierte Positionen: {len(positions)}")
    
    # Pfad optimieren
    print(f"\n🎯 Pfad-Optimierung:")
    optimized_path = optimization_engine.optimize_scan_path(positions)
    print(f"   Optimierter Pfad: {len(optimized_path)} Schritte")
    
    # Qualität bewerten
    print(f"\n📊 Qualitätsbewertung:")
    quality_metrics = quality_engine.assess_pattern_quality(positions, optimized_path)
    
    for metric, value in quality_metrics.items():
        if metric == 'estimated_scan_time':
            print(f"   {metric.replace('_', ' ').title()}: {value:.1f} Sekunden")
        else:
            print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")


if __name__ == "__main__":
    demo_technical_configurator()
