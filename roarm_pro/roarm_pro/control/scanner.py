"""
Scanner-specific control functions
Optimized for Revopoint Mini2 3D scanner
"""

import time
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ScanParameters:
    """Parameters for scanning operations"""
    width: float = 0.2          # Scan width in meters
    height: float = 0.15        # Scan height in meters
    distance: float = 0.15      # Distance from object
    step_size: float = 0.01     # Step size between points
    rotation_speed: float = 0.5 # rad/s for continuous scans
    settle_time: float = 0.1    # Time to settle at each point
    led_brightness: int = 128   # LED brightness during scan

class ScannerControl:
    """Control functions for 3D scanning operations"""
    
    def __init__(self, controller):
        """
        Initialize scanner control
        Args:
            controller: RoArmController instance
        """
        self.controller = controller
        self.params = ScanParameters()
        
        # Ensure scanner mode is set
        self.controller.set_scanner_mounted(True)
    
    def mount_scanner(self) -> bool:
        """Move to scanner mounting position"""
        print("📷 Moving to scanner mount position...")
        
        # First go to safe position
        if not self.controller.move_to_home():
            return False
        
        # Move to mount position
        return self.controller.move_to_position('scanner_mount')
    
    def start_position(self) -> bool:
        """Move to scan start position"""
        print("🎯 Moving to scan start position...")
        return self.controller.move_to_position('scan_start')
    
    # ==================== Scan Patterns ====================
    
    def continuous_rotation_scan(self, duration: float = 60.0) -> bool:
        """
        Continuous 360° rotation scan
        Perfect for objects on turntable
        
        Args:
            duration: Total scan duration in seconds
        """
        print("🌊 Starting continuous rotation scan...")
        print(f"   Duration: {duration}s")
        print("   Press Ctrl+C to stop")
        
        # Get starting position
        start_pos = self.controller.get_current_position()
        start_base = start_pos['base']
        
        # Turn on LED
        self.controller.led_on(self.params.led_brightness)
        
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                if self.controller.abort_flag:
                    break
                
                # Calculate rotation progress
                progress = (time.time() - start_time) / duration
                
                # Smooth rotation
                base_angle = start_base + progress * 2 * math.pi
                
                # Optional: Add slight vertical movement for better coverage
                shoulder_offset = 0.05 * math.sin(progress * 4 * math.pi)
                
                # Move
                self.controller.move_joints(
                    base=base_angle,
                    shoulder=start_pos['shoulder'] + shoulder_offset,
                    duration=0.5
                )
                
                # Small delay
                time.sleep(0.1)
            
            print("✅ Continuous scan complete")
            return True
            
        finally:
            # Turn off LED
            self.controller.led_off()
    
    def raster_scan(self, rows: int = 10, cols: int = 10) -> bool:
        """
        Raster (grid) scan pattern
        Good for flat surfaces or frontal scans
        
        Args:
            rows: Number of horizontal rows
            cols: Number of points per row
        """
        print(f"📐 Starting raster scan ({rows}x{cols})...")
        
        # Generate scan points
        points = self._generate_raster_points(rows, cols)
        
        # Execute scan
        return self._execute_scan_points(points, "Raster scan")
    
    def spiral_scan(self, turns: int = 5, points_per_turn: int = 20) -> bool:
        """
        Spiral scan pattern
        Good for cylindrical objects
        
        Args:
            turns: Number of spiral turns
            points_per_turn: Points per turn
        """
        print(f"🌀 Starting spiral scan ({turns} turns)...")
        
        # Generate spiral points
        points = self._generate_spiral_points(turns, points_per_turn)
        
        # Execute scan
        return self._execute_scan_points(points, "Spiral scan")
    
    def detail_scan(self, center: Dict[str, float] = None, 
                   size: float = 0.05, points: int = 25) -> bool:
        """
        Detailed scan of small area
        
        Args:
            center: Center position (current if None)
            size: Scan area size
            points: Number of scan points
        """
        print(f"🔍 Starting detail scan...")
        
        if center is None:
            center = self.controller.get_current_position()
        
        # Generate detail scan points
        scan_points = self._generate_detail_points(center, size, points)
        
        # Execute scan
        return self._execute_scan_points(scan_points, "Detail scan")
    
    # ==================== Helper Methods ====================
    
    def _generate_raster_points(self, rows: int, cols: int) -> List[Dict[str, float]]:
        """Generate raster scan points"""
        points = []
        
        # Get current position as center
        center = self.controller.get_current_position()
        
        # Calculate offsets
        width = self.params.width
        height = self.params.height
        
        for i in range(rows):
            # Calculate row position
            y_offset = (i / (rows - 1) - 0.5) * height if rows > 1 else 0
            
            # Alternate direction for each row (snake pattern)
            if i % 2 == 0:
                col_range = range(cols)
            else:
                col_range = range(cols - 1, -1, -1)
            
            for j in col_range:
                # Calculate column position
                x_offset = (j / (cols - 1) - 0.5) * width if cols > 1 else 0
                
                # Create point
                point = center.copy()
                
                # Apply offsets (simplified - should use proper kinematics)
                point['base'] = center['base'] + x_offset
                point['shoulder'] = center['shoulder'] + y_offset * 0.5
                point['elbow'] = center['elbow'] - y_offset * 0.5
                
                points.append(point)
        
        return points
    
    def _generate_spiral_points(self, turns: int, points_per_turn: int) -> List[Dict[str, float]]:
        """Generate spiral scan points"""
        points = []
        total_points = turns * points_per_turn
        
        # Get current position as center
        center = self.controller.get_current_position()
        
        for i in range(total_points):
            # Calculate spiral parameters
            angle = (i / points_per_turn) * 2 * math.pi
            radius = (i / total_points) * self.params.width / 2
            
            # Calculate position
            point = center.copy()
            point['base'] = center['base'] + radius * math.cos(angle)
            
            # Vertical movement
            height_offset = (i / total_points) * self.params.height
            point['shoulder'] = center['shoulder'] + height_offset * 0.5
            point['elbow'] = center['elbow'] - height_offset * 0.5
            
            points.append(point)
        
        return points
    
    def _generate_detail_points(self, center: Dict[str, float], 
                               size: float, num_points: int) -> List[Dict[str, float]]:
        """Generate detail scan points in a grid"""
        points = []
        
        # Calculate grid size
        grid_size = int(math.sqrt(num_points))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate offsets
                x_offset = (i / (grid_size - 1) - 0.5) * size if grid_size > 1 else 0
                y_offset = (j / (grid_size - 1) - 0.5) * size if grid_size > 1 else 0
                
                # Create point
                point = center.copy()
                point['base'] = center['base'] + x_offset
                point['shoulder'] = center['shoulder'] + y_offset * 0.3
                
                points.append(point)
        
        return points
    
    def _execute_scan_points(self, points: List[Dict[str, float]], 
                            scan_name: str) -> bool:
        """Execute a list of scan points"""
        total = len(points)
        print(f"   Executing {total} scan points...")
        
        # Turn on LED
        self.controller.led_on(self.params.led_brightness)
        
        try:
            for i, point in enumerate(points):
                if self.controller.abort_flag:
                    print("\n❌ Scan aborted")
                    return False
                
                # Progress
                progress = (i + 1) / total * 100
                print(f"   Progress: {progress:.1f}%", end='\r')
                
                # Move to point
                if not self.controller.move_joints(**point, duration=0.5):
                    print(f"\n❌ Failed at point {i+1}")
                    return False
                
                # Settle time
                time.sleep(self.params.settle_time)
            
            print(f"\n✅ {scan_name} complete")
            return True
            
        finally:
            # Turn off LED
            self.controller.led_off()
    
    # ==================== Configuration ====================
    
    def set_scan_parameters(self, **kwargs):
        """Update scan parameters"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    def get_scan_parameters(self) -> Dict:
        """Get current scan parameters"""
        return {
            'width': self.params.width,
            'height': self.params.height,
            'distance': self.params.distance,
            'step_size': self.params.step_size,
            'rotation_speed': self.params.rotation_speed,
            'settle_time': self.params.settle_time,
            'led_brightness': self.params.led_brightness
        }
