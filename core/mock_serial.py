#!/usr/bin/env python3
"""
Mock Serial Manager für RoArm M3
Simuliert die Hardware für Tests ohne physischen Roboter
"""

import json
import time
import random
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MockRobotState:
    """Simulierter Roboter-Zustand."""
    positions: Dict[str, float]
    velocities: Dict[str, float]
    torque_enabled: bool = True
    temperature: float = 25.0
    voltage: float = 6.0
    led_on: bool = False
    led_brightness: int = 0
    
    def __init__(self):
        # Start at home position
        self.positions = {
            "base": 0.0,
            "shoulder": 0.0,
            "elbow": 1.57,
            "wrist": 0.0,
            "roll": 0.0,
            "hand": 3.14
        }
        self.velocities = {joint: 0.0 for joint in self.positions}
        self.torque_enabled = True
        self.temperature = 25.0 + random.uniform(-2, 2)
        self.voltage = 6.0 + random.uniform(-0.2, 0.2)


class MockSerialManager:
    """
    Mock Serial Manager - Simuliert die serielle Kommunikation.
    Perfekt für Tests ohne Hardware!
    """
    
    def __init__(self, port: str = "MOCK", baudrate: int = 115200, timeout: float = 2.0):
        """
        Initialisiert Mock Serial Manager.
        
        Args:
            port: Port-Name (wird ignoriert, nur für Kompatibilität)
            baudrate: Baudrate (wird ignoriert)
            timeout: Timeout (wird simuliert)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False
        
        # Simulierter Robot State
        self.robot_state = MockRobotState()
        
        # Command history für Debugging
        self.command_history = []
        self.response_history = []
        
        # Simulation settings
        self.simulate_delays = True
        self.simulate_errors = False
        self.error_rate = 0.05  # 5% error rate when enabled
        
        logger.info(f"🤖 MockSerialManager initialized (port={port})")
    
    def connect(self) -> bool:
        """
        Simuliert Verbindungsaufbau.
        
        Returns:
            Immer True für Mock
        """
        if self.simulate_delays:
            time.sleep(0.5)  # Simulate connection time
        
        self.connected = True
        logger.info("✅ Mock connection established")
        print("🎮 SIMULATOR MODE - No real hardware connected")
        return True
    
    def disconnect(self):
        """Simuliert Verbindungstrennung."""
        self.connected = False
        logger.info("Mock connection closed")
        
        # Print command statistics
        if self.command_history:
            print(f"\n📊 Simulation Statistics:")
            print(f"  Commands sent: {len(self.command_history)}")
            print(f"  Responses: {len(self.response_history)}")
            
            # Count command types
            cmd_types = {}
            for cmd in self.command_history:
                cmd_type = cmd.get('T', 'unknown')
                cmd_types[cmd_type] = cmd_types.get(cmd_type, 0) + 1
            
            print(f"  Command types:")
            for cmd_type, count in cmd_types.items():
                print(f"    Type {cmd_type}: {count}x")
    
    def send_command(self, command: Dict, wait_response: bool = False) -> Optional[str]:
        """
        Simuliert Command-Versand.
        
        Args:
            command: Command Dictionary
            wait_response: Ob auf Antwort gewartet werden soll
            
        Returns:
            Simulierte Antwort oder None
        """
        if not self.connected:
            logger.error("Mock: Not connected")
            return None
        
        # Log command
        self.command_history.append(command.copy())
        logger.debug(f"Mock received command: {command}")
        
        # Print important commands
        if command.get('T') == 0:  # Emergency stop
            print("🚨 MOCK: Emergency Stop triggered!")
        elif command.get('T') == 102:  # Joint movement
            print(f"🦾 MOCK: Moving joints to {command.get('positions', {})}")
        elif command.get('T') == 51:  # LED
            print(f"💡 MOCK: LED {'ON' if command.get('led') else 'OFF'}")
        
        # Simulate processing time
        if self.simulate_delays:
            time.sleep(0.01)
        
        # Simulate errors
        if self.simulate_errors and random.random() < self.error_rate:
            logger.warning("Mock: Simulated communication error")
            return None
        
        # Process command
        response = self._process_command(command)
        
        if wait_response and response:
            self.response_history.append(response)
            return json.dumps(response)
        
        return None
    
    def _process_command(self, command: Dict) -> Optional[Dict]:
        """
        Verarbeitet Commands und updated simulierten State.
        
        Args:
            command: Command Dictionary
            
        Returns:
            Response Dictionary
        """
        cmd_type = command.get('T')
        
        # Emergency Stop
        if cmd_type == 0:
            self.robot_state.torque_enabled = False
            # Stop all movements
            for joint in self.robot_state.velocities:
                self.robot_state.velocities[joint] = 0.0
            return {"status": "emergency_stop"}
        
        # Status Query
        elif cmd_type == 1 or cmd_type == 2:
            return {
                "positions": self.robot_state.positions.copy(),
                "velocities": self.robot_state.velocities.copy(),
                "torque_enabled": self.robot_state.torque_enabled,
                "temperature": self.robot_state.temperature + random.uniform(-0.5, 0.5),
                "voltage": self.robot_state.voltage + random.uniform(-0.1, 0.1)
            }
        
        # LED Control
        elif cmd_type == 51:
            self.robot_state.led_on = bool(command.get('led', 0))
            self.robot_state.led_brightness = command.get('brightness', 0)
            return {"led": self.robot_state.led_on}
        
        # Joint Control
        elif cmd_type == 102:
            target_positions = command.get('positions', {})
            
            # Simulate movement (instant for mock)
            for joint, pos in target_positions.items():
                if joint in self.robot_state.positions:
                    old_pos = self.robot_state.positions[joint]
                    self.robot_state.positions[joint] = pos
                    
                    # Calculate simulated velocity
                    self.robot_state.velocities[joint] = (pos - old_pos) * 10
            
            # Clear velocities after "movement"
            if self.simulate_delays:
                time.sleep(0.1)  # Simulate movement time
            
            for joint in self.robot_state.velocities:
                self.robot_state.velocities[joint] = 0.0
            
            return {"status": "movement_complete"}
        
        # Torque Control
        elif cmd_type == 210:
            self.robot_state.torque_enabled = bool(command.get('enabled', 0))
            return {"torque": self.robot_state.torque_enabled}
        
        # Unknown command
        else:
            logger.warning(f"Mock: Unknown command type {cmd_type}")
            return {"error": f"unknown_command_{cmd_type}"}
    
    def read_response(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Simuliert Response-Lesen.
        
        Args:
            timeout: Read timeout
            
        Returns:
            Simulierte Response
        """
        if self.simulate_delays:
            time.sleep(0.01)
        
        # Return a mock response
        return json.dumps({"status": "ok"})
    
    def query_status(self) -> Optional[Dict]:
        """
        Gibt simulierten Status zurück.
        
        Returns:
            Status Dictionary
        """
        return {
            "positions": self.robot_state.positions.copy(),
            "velocities": self.robot_state.velocities.copy(),
            "torque_enabled": self.robot_state.torque_enabled,
            "temperature": self.robot_state.temperature,
            "voltage": self.robot_state.voltage,
            "led": self.robot_state.led_on,
            "simulator": True  # Flag to indicate this is simulated
        }
    
    def get_debug_info(self) -> Dict:
        """
        Gibt Debug-Informationen zurück.
        
        Returns:
            Debug info dictionary
        """
        return {
            "mode": "SIMULATOR",
            "connected": self.connected,
            "commands_sent": len(self.command_history),
            "responses_received": len(self.response_history),
            "current_state": {
                "positions": self.robot_state.positions,
                "torque": self.robot_state.torque_enabled,
                "led": self.robot_state.led_on
            },
            "settings": {
                "simulate_delays": self.simulate_delays,
                "simulate_errors": self.simulate_errors,
                "error_rate": self.error_rate
            }
        }
    
    def enable_debug_output(self, enabled: bool = True):
        """
        Aktiviert/Deaktiviert Debug-Output.
        
        Args:
            enabled: Debug output ein/aus
        """
        if enabled:
            print("\n📝 Mock Debug Output ENABLED")
            print("  All commands will be printed")
        else:
            print("\n🔇 Mock Debug Output DISABLED")
    
    def get_command_log(self) -> List[Dict]:
        """
        Gibt Command-History zurück.
        
        Returns:
            Liste aller gesendeten Commands
        """
        return self.command_history.copy()
    
    def clear_history(self):
        """Löscht Command/Response History."""
        self.command_history.clear()
        self.response_history.clear()
        logger.info("Mock history cleared")
    
    def set_simulation_mode(self, delays: bool = True, errors: bool = False, error_rate: float = 0.05):
        """
        Konfiguriert Simulations-Modus.
        
        Args:
            delays: Simuliere Verzögerungen
            errors: Simuliere Fehler
            error_rate: Fehlerrate (0.0-1.0)
        """
        self.simulate_delays = delays
        self.simulate_errors = errors
        self.error_rate = max(0.0, min(1.0, error_rate))
        
        print(f"\n⚙️ Simulation Mode Updated:")
        print(f"  Delays: {'ON' if delays else 'OFF'}")
        print(f"  Errors: {'ON' if errors else 'OFF'} (rate: {error_rate*100:.0f}%)")


def create_mock_serial(*args, **kwargs):
    """
    Factory function to create MockSerialManager.
    Kann als Drop-in Replacement für SerialManager verwendet werden.
    """
    return MockSerialManager(*args, **kwargs)
