"""
Main RoArm controller
Coordinates hardware, motion, and high-level control
"""

import time
import threading
import logging
import math
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from ..config import (HOME_POSITION, CALIBRATION_POSITIONS, 
                     MOTION_DEFAULTS, SPEED_PROFILES)
from ..hardware import SerialConnection, CommandBuilder
from ..motion import TrajectoryGenerator, TrajectoryType, JointLimits, Kinematics

logger = logging.getLogger(__name__)

@dataclass
class RobotState:
    """Current robot state"""
    current_position: Dict[str, float] = field(default_factory=lambda: HOME_POSITION.copy())
    target_position: Dict[str, float] = field(default_factory=dict)
    is_moving: bool = False
    torque_enabled: bool = True
    emergency_stop: bool = False
    scanner_mounted: bool = False

class RoArmController:
    """Main controller for RoArm robot"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        """Initialize controller"""
        # Hardware
        self.serial = SerialConnection(port, baudrate)
        self.cmd_builder = CommandBuilder()
        
        # Motion
        self.trajectory_gen = TrajectoryGenerator()
        self.joint_limits = JointLimits()
        self.kinematics = Kinematics()
        
        # State
        self.state = RobotState()
        self._state_lock = threading.Lock()
        
        # Configuration
        self.speed_factor = 1.0
        self.min_duration = MOTION_DEFAULTS["min_duration"]
        
        # Calibration data
        self.calibration = {}
        
        # Abort flag for emergency stop
        self.abort_flag = False
        
    # ==================== Connection Management ====================
    
    def connect(self) -> bool:
        """Connect to robot"""
        if self.serial.connect():
            # Query initial position
            self.update_position()
            return True
        return False
    
    def disconnect(self):
        """Disconnect from robot"""
        self.serial.disconnect()
    
    @property
    def connected(self) -> bool:
        """Check if connected"""
        return self.serial.is_connected
    
    # ==================== State Management ====================
    
    def update_position(self) -> bool:
        """Update current position from robot"""
        response = self.serial.send_command(
            self.cmd_builder.position_query()
        )
        
        if response:
            with self._state_lock:
                # Update positions from response
                # Response format depends on firmware
                if 'joint1' in response:
                    self.state.current_position['base'] = response.get('joint1', 0)
                    self.state.current_position['shoulder'] = response.get('joint2', 0)
                    self.state.current_position['elbow'] = response.get('joint3', 0)
                    self.state.current_position['wrist'] = response.get('joint4', 0)
                    self.state.current_position['roll'] = response.get('joint5', 0)
                    self.state.current_position['hand'] = response.get('joint6', 0)
                elif 'base' in response:
                    # Alternative format
                    for joint in ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']:
                        if joint in response:
                            self.state.current_position[joint] = response[joint]
            return True
        return False
    
    def get_current_position(self) -> Dict[str, float]:
        """Get current joint positions"""
        with self._state_lock:
            return self.state.current_position.copy()
    
    # ==================== Basic Movement ====================
    
    def move_joints(self, duration: Optional[float] = None, **joint_positions) -> bool:
        """
        Move joints to target positions
        
        Args:
            duration: Movement duration (auto if None)
            **joint_positions: Target positions for joints
            
        Returns:
            True if movement successful
        """
        if self.abort_flag or self.state.emergency_stop:
            logger.warning("Movement blocked - emergency stop active")
            return False
        
        # Validate positions
        validated = self.joint_limits.validate(**joint_positions)
        
        if not validated:
            return True  # No movement needed
        
        # Get current positions
        current = self.get_current_position()
        
        # Create target position dict
        target = current.copy()
        target.update(validated)
        
        # Auto-calculate duration if not specified
        if duration is None:
            duration = self.joint_limits.suggest_duration(current, target)
        
        # Apply speed factor
        duration = duration / self.speed_factor
        
        # Ensure minimum duration
        duration = max(duration, self.min_duration)
        
        # Check velocity limits
        if not self.joint_limits.check_velocity(current, target, duration):
            logger.warning("Velocity limits exceeded - increasing duration")
            duration = self.joint_limits.suggest_duration(current, target)
        
        # Generate trajectory
        trajectory = self.trajectory_gen.generate(
            current, target, duration, 
            TrajectoryType.S_CURVE
        )
        
        # Execute trajectory
        return self._execute_trajectory(trajectory)
    
    def _execute_trajectory(self, trajectory: List) -> bool:
        """Execute a trajectory"""
        with self._state_lock:
            self.state.is_moving = True
            self.state.target_position = trajectory[-1].positions.copy()
        
        try:
            for point in trajectory:
                if self.abort_flag:
                    logger.warning("Trajectory aborted")
                    return False
                
                # Send position command
                cmd = self.cmd_builder.joint_control(**point.positions)
                response = self.serial.send_command(cmd, expect_response=False)
                
                if not response:
                    logger.error("Command failed")
                    return False
                
                # Update current position
                with self._state_lock:
                    self.state.current_position.update(point.positions)
                
                # Wait for next point
                if point != trajectory[-1]:
                    time.sleep(1.0 / self.trajectory_gen.sample_rate)
            
            return True
            
        finally:
            with self._state_lock:
                self.state.is_moving = False
    
    # ==================== Convenience Methods ====================
    
    def move_to_home(self) -> bool:
        """Move to home position"""
        logger.info("🏠 Moving to home position")
        return self.move_joints(duration=2.0, **HOME_POSITION)
    
    def move_to_position(self, position_name: str) -> bool:
        """Move to named position"""
        if position_name in CALIBRATION_POSITIONS:
            logger.info(f"📍 Moving to {position_name}")
            return self.move_joints(
                duration=2.0, 
                **CALIBRATION_POSITIONS[position_name]
            )
        else:
            logger.error(f"Unknown position: {position_name}")
            return False
    
    def emergency_stop(self):
        """Emergency stop"""
        logger.critical("🛑 EMERGENCY STOP")
        self.abort_flag = True
        self.state.emergency_stop = True
        
        # Send emergency stop command
        self.serial.emergency_stop()
    
    def clear_emergency_stop(self):
        """Clear emergency stop"""
        self.abort_flag = False
        self.state.emergency_stop = False
        logger.info("✅ Emergency stop cleared")
    
    # ==================== Gripper Control ====================
    
    def set_gripper(self, percentage: float) -> bool:
        """
        Set gripper opening
        Args:
            percentage: 0-100 (0=closed, 100=open)
        """
        # Convert percentage to radians
        # hand joint: 1.08 (open) to 3.14 (closed)
        hand_range = 3.14 - 1.08
        hand_angle = 3.14 - (percentage / 100.0) * hand_range
        
        return self.move_joints(hand=hand_angle, duration=1.0)
    
    def set_scanner_grip(self, grip_value: float = 2.5) -> bool:
        """Set scanner-safe grip"""
        # Validate grip is in safe range
        min_grip, max_grip = self.joint_limits.get_limits('hand')
        if self.state.scanner_mounted:
            min_grip, max_grip = self.joint_limits._limits['hand']
        
        grip_value = max(min_grip, min(max_grip, grip_value))
        return self.move_joints(hand=grip_value, duration=1.0)
    
    # ==================== LED Control ====================
    
    def led_on(self, brightness: int = 255):
        """Turn LED on"""
        cmd = self.cmd_builder.led_control(brightness, True)
        self.serial.send_command(cmd, expect_response=False)
    
    def led_off(self):
        """Turn LED off"""
        cmd = self.cmd_builder.led_control(0, False)
        self.serial.send_command(cmd, expect_response=False)
    
    # ==================== Torque Control ====================
    
    def enable_torque(self):
        """Enable servo torque"""
        cmd = self.cmd_builder.torque_control(True)
        response = self.serial.send_command(cmd)
        if response:
            self.state.torque_enabled = True
            logger.info("⚡ Torque enabled")
        return bool(response)
    
    def disable_torque(self):
        """Disable servo torque"""
        cmd = self.cmd_builder.torque_control(False)
        response = self.serial.send_command(cmd)
        if response:
            self.state.torque_enabled = False
            logger.warning("⚡ Torque disabled - arm will be limp")
        return bool(response)
    
    # ==================== Scanner Mode ====================
    
    def set_scanner_mounted(self, mounted: bool):
        """Set scanner mounted state"""
        self.state.scanner_mounted = mounted
        self.joint_limits.set_scanner_mounted(mounted)
        
        if mounted:
            logger.info("📷 Scanner mode activated - limits applied")
        else:
            logger.info("🤖 Normal mode - full range available")
    
    # ==================== Calibration ====================
    
    def calibrate(self, full: bool = True) -> bool:
        """Run calibration sequence"""
        logger.info("🎯 Starting calibration...")
        
        # Move to known positions
        positions = ['home', 'rest', 'scanner_mount'] if full else ['home']
        
        for pos in positions:
            if not self.move_to_position(pos):
                logger.error(f"Calibration failed at {pos}")
                return False
            time.sleep(1.0)
        
        # Store calibration data
        self.calibration['calibrated'] = True
        self.calibration['timestamp'] = time.time()
        self.calibration['version'] = '2.0_systematic'
        
        logger.info("✅ Calibration complete")
        return True
    
    # ==================== Safety Shutdown ====================
    
    def safe_shutdown(self):
        """Safe shutdown sequence"""
        logger.info("👋 Safe shutdown sequence...")
        
        # Move to rest position
        self.move_to_position('rest')
        time.sleep(1.0)
        
        # Turn off LED
        self.led_off()
        
        # Disconnect
        self.disconnect()
