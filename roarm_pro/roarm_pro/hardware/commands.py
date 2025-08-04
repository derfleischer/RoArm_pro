"""
Command builder for RoArm JSON protocol
Provides a clean interface for building commands
"""

from typing import Dict, Any, Optional
from ..config import COMMANDS

class CommandBuilder:
    """Build RoArm commands with validation"""
    
    @staticmethod
    def emergency_stop() -> Dict[str, int]:
        """Build emergency stop command"""
        return COMMANDS["EMERGENCY_STOP"].copy()
    
    @staticmethod
    def joint_control(**kwargs) -> Dict[str, Any]:
        """
        Build joint control command
        Args: base, shoulder, elbow, wrist, roll, hand (in radians)
        """
        cmd = COMMANDS["JOINT_CONTROL"].copy()
        
        # Map joint names to command parameters
        joint_mapping = {
            'base': 'joint1',
            'shoulder': 'joint2',
            'elbow': 'joint3',
            'wrist': 'joint4',
            'roll': 'joint5',
            'hand': 'joint6'
        }
        
        for joint, value in kwargs.items():
            if joint in joint_mapping and value is not None:
                cmd[joint_mapping[joint]] = float(value)
        
        return cmd
    
    @staticmethod
    def led_control(brightness: int = 255, on: bool = True) -> Dict[str, Any]:
        """
        Build LED control command
        Args:
            brightness: 0-255
            on: True for on, False for off
        """
        if on:
            cmd = COMMANDS["LED_ON"].copy()
            cmd["brightness"] = max(0, min(255, brightness))
        else:
            cmd = COMMANDS["LED_OFF"].copy()
        
        return cmd
    
    @staticmethod
    def torque_control(enable: bool = True) -> Dict[str, Any]:
        """Build torque enable/disable command"""
        cmd = COMMANDS["TORQUE_CONTROL"].copy()
        cmd["enable"] = 1 if enable else 0
        return cmd
    
    @staticmethod
    def status_query() -> Dict[str, int]:
        """Build status query command"""
        return COMMANDS["STATUS_QUERY"].copy()
    
    @staticmethod
    def position_query() -> Dict[str, int]:
        """Build position query command"""
        return COMMANDS["POSITION_QUERY"].copy()
    
    @staticmethod
    def coordinate_control(x: float, y: float, z: float, 
                         roll: Optional[float] = None,
                         pitch: Optional[float] = None,
                         yaw: Optional[float] = None) -> Dict[str, Any]:
        """
        Build coordinate control command (Cartesian)
        Args: x, y, z in meters, angles in radians
        """
        cmd = COMMANDS["COORDINATE_CONTROL"].copy()
        cmd.update({
            'x': float(x),
            'y': float(y),
            'z': float(z)
        })
        
        if roll is not None:
            cmd['roll'] = float(roll)
        if pitch is not None:
            cmd['pitch'] = float(pitch)
        if yaw is not None:
            cmd['yaw'] = float(yaw)
            
        return cmd
    
    @staticmethod
    def gripper_control(percentage: float) -> Dict[str, Any]:
        """
        Build gripper control command
        Args: percentage 0-100 (0=closed, 100=open)
        """
        cmd = COMMANDS["GRIPPER_CONTROL"].copy()
        cmd["percentage"] = max(0.0, min(100.0, float(percentage)))
        return cmd
