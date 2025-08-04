"""
Kinematics calculations for RoArm
Includes wrist compensation for level scanner
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional

class Kinematics:
    """Forward and inverse kinematics for RoArm"""
    
    # Robot dimensions (meters)
    # These are approximate - adjust based on actual measurements
    LINK_LENGTHS = {
        'base_height': 0.06,     # Height from base to shoulder
        'upper_arm': 0.105,      # Shoulder to elbow
        'forearm': 0.098,        # Elbow to wrist
        'wrist_to_tool': 0.155   # Wrist to end effector (with scanner)
    }
    
    @staticmethod
    def calculate_wrist_compensation(shoulder: float, elbow: float) -> float:
        """
        Calculate wrist angle to keep scanner level
        
        Args:
            shoulder: Shoulder angle in radians
            elbow: Elbow angle in radians
            
        Returns:
            Compensated wrist angle in radians
        """
        # The wrist needs to compensate for shoulder and elbow tilts
        # to keep the scanner horizontal
        
        # Basic compensation: opposite of the sum of tilts
        total_tilt = shoulder + (elbow - math.pi/2)
        compensated_wrist = -total_tilt
        
        return compensated_wrist
    
    @staticmethod
    def forward_kinematics(joint_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate end effector position from joint angles
        
        Args:
            joint_positions: Dict with base, shoulder, elbow, wrist angles
            
        Returns:
            Dict with x, y, z position and orientation
        """
        # Extract joint angles
        base = joint_positions.get('base', 0.0)
        shoulder = joint_positions.get('shoulder', 0.0)
        elbow = joint_positions.get('elbow', math.pi/2)
        wrist = joint_positions.get('wrist', 0.0)
        
        # Get link lengths
        l0 = Kinematics.LINK_LENGTHS['base_height']
        l1 = Kinematics.LINK_LENGTHS['upper_arm']
        l2 = Kinematics.LINK_LENGTHS['forearm']
        l3 = Kinematics.LINK_LENGTHS['wrist_to_tool']
        
        # Calculate positions of each joint
        # Shoulder position
        shoulder_x = 0
        shoulder_y = 0
        shoulder_z = l0
        
        # Elbow position
        elbow_x = l1 * math.cos(shoulder)
        elbow_y = 0
        elbow_z = shoulder_z + l1 * math.sin(shoulder)
        
        # Wrist position
        wrist_angle_global = shoulder + elbow - math.pi/2
        wrist_x = elbow_x + l2 * math.cos(wrist_angle_global)
        wrist_y = 0
        wrist_z = elbow_z + l2 * math.sin(wrist_angle_global)
        
        # End effector position
        tool_angle_global = wrist_angle_global + wrist
        tool_x = wrist_x + l3 * math.cos(tool_angle_global)
        tool_y = 0
        tool_z = wrist_z + l3 * math.sin(tool_angle_global)
        
        # Apply base rotation
        x = tool_x * math.cos(base) - tool_y * math.sin(base)
        y = tool_x * math.sin(base) + tool_y * math.cos(base)
        z = tool_z
        
        # Calculate orientation
        roll = joint_positions.get('roll', 0.0)
        pitch = tool_angle_global
        yaw = base
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
    
    @staticmethod
    def inverse_kinematics(x: float, y: float, z: float,
                          pitch: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        Calculate joint angles for target position
        Simplified 2D IK in the plane, with base rotation
        
        Args:
            x, y, z: Target position in meters
            pitch: Desired end effector pitch (optional)
            
        Returns:
            Dict of joint angles or None if unreachable
        """
        # Calculate base angle
        base = math.atan2(y, x)
        
        # Distance in XY plane
        r = math.sqrt(x**2 + y**2)
        
        # Adjust z for base height
        z_adj = z - Kinematics.LINK_LENGTHS['base_height']
        
        # Get link lengths
        l1 = Kinematics.LINK_LENGTHS['upper_arm']
        l2 = Kinematics.LINK_LENGTHS['forearm']
        l3 = Kinematics.LINK_LENGTHS['wrist_to_tool']
        
        # If pitch is specified, account for tool offset
        if pitch is not None:
            r -= l3 * math.cos(pitch)
            z_adj -= l3 * math.sin(pitch)
        
        # Check reachability
        distance = math.sqrt(r**2 + z_adj**2)
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            return None  # Target unreachable
        
        # Calculate elbow angle using law of cosines
        cos_elbow = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        cos_elbow = max(-1, min(1, cos_elbow))  # Clamp for numerical stability
        elbow = math.acos(cos_elbow)
        
        # Calculate shoulder angle
        angle_to_target = math.atan2(z_adj, r)
        cos_shoulder_offset = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        cos_shoulder_offset = max(-1, min(1, cos_shoulder_offset))
        shoulder_offset = math.acos(cos_shoulder_offset)
        shoulder = angle_to_target - shoulder_offset
        
        # Calculate wrist for level scanner
        wrist = Kinematics.calculate_wrist_compensation(shoulder, elbow)
        
        return {
            'base': base,
            'shoulder': shoulder,
            'elbow': elbow,
            'wrist': wrist
        }
    
    @staticmethod
    def check_workspace(x: float, y: float, z: float) -> bool:
        """
        Check if position is within robot workspace
        
        Args:
            x, y, z: Position to check
            
        Returns:
            True if reachable
        """
        # Simple cylindrical workspace check
        r = math.sqrt(x**2 + y**2)
        
        # Get total reach
        max_reach = (Kinematics.LINK_LENGTHS['upper_arm'] + 
                    Kinematics.LINK_LENGTHS['forearm'] + 
                    Kinematics.LINK_LENGTHS['wrist_to_tool'])
        
        min_z = 0.0
        max_z = 0.4  # Approximate max height
        
        return r <= max_reach and min_z <= z <= max_z
