#!/usr/bin/env python3
"""
RoArm M3 Debug & Simulation Mode
Ermöglicht vollständige Tests ohne Hardware
"""

import json
import time
import random
import threading
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from core.constants import (
    SERVO_LIMITS, HOME_POSITION, SCANNER_POSITION,
    COMMANDS, DEFAULT_SPEED
)
from utils.logger import get_logger

logger = get_logger(__name__)


class SimulationMode(Enum):
    """Verschiedene Simulationsmodi."""
    PERFECT = "perfect"          # Perfekte Ausführung
    REALISTIC = "realistic"      # Mit realistischen Fehlern
    FAILURE = "failure"          # Simuliert Fehler
    RANDOM = "random"           # Zufällige Ereignisse


@dataclass
class SimulatedState:
    """Simulierter Roboterzustand."""
    positions: Dict[str, float]
    velocities: Dict[str, float]
    accelerations: Dict[str, float]
    torque_enabled: bool = True
    temperature: float = 25.0
    voltage: float = 6.0
    led_state: bool = False
    led_brightness: int = 0
    gripper_force: float = 0.5
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MockSerialManager:
    """Mock Serial Manager für Debug-Modus."""
    
    def __init__(self, port: str = "MOCK", baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False
        self.command_history = []
        self.response_delay = 0.01  # Simulierte Latenz
        
        # Simulierter Zustand
        self.state = SimulatedState(
            positions=HOME_POSITION.copy(),
            velocities={joint: 0.0 for joint in HOME_POSITION},
            accelerations={joint: 0.0 for joint in HOME_POSITION}
        )
        
        logger.info(f"🔧 MockSerialManager initialized (port={port})")
    
    def connect(self) -> bool:
        """Simuliert Verbindung."""
        time.sleep(0.5)  # Simuliere Verbindungsaufbau
        self.connected = True
        logger.info("✅ [MOCK] Connected to virtual RoArm")
        return True
    
    def disconnect(self):
        """Simuliert Trennung."""
        self.connected = False
        logger.info("🔌 [MOCK] Disconnected from virtual RoArm")
    
    def send_command(self, command: Dict, wait_response: bool = False) -> Optional[str]:
        """
        Simuliert Command-Verarbeitung.
        
        Args:
            command: JSON Command
            wait_response: Auf Antwort warten
            
        Returns:
            Simulierte Antwort oder None
        """
        # Log command
        self.command_history.append({
            "timestamp": time.time(),
            "command": command,
            "type": self._get_command_type(command)
        })
        
        # Simuliere Latenz
        time.sleep(self.response_delay)
        
        # Verarbeite Command
        cmd_type = command.get("T", 0)
        
        if cmd_type == COMMANDS["EMERGENCY_STOP"]:
            logger.warning("🚨 [MOCK] Emergency Stop received")
            return '{"status": "emergency_stopped"}'
            
        elif cmd_type == COMMANDS["STATUS_QUERY"]:
            return self._generate_status_response()
            
        elif cmd_type == COMMANDS["JOINT_CONTROL"]:
            self._process_joint_command(command)
            return '{"status": "moving"}'
            
        elif cmd_type == COMMANDS["LED_CONTROL"]:
            self.state.led_state = bool(command.get("led", 0))
            self.state.led_brightness = command.get("brightness", 0)
            logger.debug(f"💡 [MOCK] LED: {self.state.led_state} ({self.state.led_brightness})")
            return '{"status": "ok"}'
            
        elif cmd_type == COMMANDS["TORQUE_CONTROL"]:
            self.state.torque_enabled = bool(command.get("enabled", 0))
            logger.debug(f"⚡ [MOCK] Torque: {self.state.torque_enabled}")
            return '{"status": "ok"}'
        
        return '{"status": "unknown_command"}'
    
    def _get_command_type(self, command: Dict) -> str:
        """Identifiziert Command-Typ."""
        cmd_id = command.get("T", -1)
        for name, value in COMMANDS.items():
            if value == cmd_id:
                return name
        return "UNKNOWN"
    
    def _generate_status_response(self) -> str:
        """Generiert Status-Antwort."""
        # Füge leichte Noise hinzu für Realismus
        positions_with_noise = {}
        for joint, pos in self.state.positions.items():
            noise = random.gauss(0, 0.001)  # ±1 mrad noise
            positions_with_noise[joint] = pos + noise
        
        response = {
            "positions": positions_with_noise,
            "velocities": self.state.velocities,
            "torque": self.state.torque_enabled,
            "temperature": self.state.temperature + random.gauss(0, 0.5),
            "voltage": self.state.voltage + random.gauss(0, 0.05)
        }
        
        return json.dumps(response)
    
    def _process_joint_command(self, command: Dict):
        """Verarbeitet Joint-Bewegung."""
        target_positions = command.get("positions", {})
        
        # Simuliere schrittweise Bewegung
        for joint, target in target_positions.items():
            if joint in self.state.positions:
                # Validiere gegen Limits
                if joint in SERVO_LIMITS:
                    min_val, max_val = SERVO_LIMITS[joint]
                    target = max(min_val, min(max_val, target))
                
                # Update Position (instant für Mock)
                self.state.positions[joint] = target
                logger.debug(f"🔄 [MOCK] {joint}: {target:.3f} rad")


class MockController:
    """Mock Controller für Debug-Modus."""
    
    def __init__(self, config, simulation_mode: SimulationMode = SimulationMode.REALISTIC):
        """
        Initialisiert Mock Controller.
        
        Args:
            config: RoArmConfig
            simulation_mode: Simulationsmodus
        """
        self.config = config
        self.simulation_mode = simulation_mode
        
        # Mock Serial
        self.serial = MockSerialManager(
            port="MOCK_PORT",
            baudrate=config.baudrate
        )
        
        # State tracking
        self.current_position = HOME_POSITION.copy()
        self.current_speed = config.default_speed
        self.torque_enabled = True
        self.emergency_stop_flag = False
        self.scanner_mounted = False
        
        # Command Queue (Mock)
        self.command_queue = MockCommandQueue()
        
        # Bewegungs-Historie
        self.movement_history = []
        self.pattern_history = []
        
        # Performance Metriken
        self.metrics = {
            "total_commands": 0,
            "total_movements": 0,
            "total_distance": 0.0,
            "emergency_stops": 0,
            "patterns_executed": 0
        }
        
        # Automatisch verbinden
        self.serial.connect()
        
        logger.info(f"🎮 MockController initialized (mode={simulation_mode.value})")
    
    def move_joints(self, positions: Dict[str, float], 
                   speed: Optional[float] = None,
                   trajectory_type: Any = None,
                   wait: bool = True) -> bool:
        """Simuliert Joint-Bewegung."""
        
        # Emergency check
        if self.emergency_stop_flag:
            logger.warning("[MOCK] Movement blocked - emergency stop active")
            return False
        
        # Validate positions
        for joint, pos in positions.items():
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                if pos < min_val or pos > max_val:
                    logger.error(f"[MOCK] Position out of limits: {joint}={pos:.3f}")
                    if self.simulation_mode == SimulationMode.FAILURE:
                        return False
        
        # Log movement
        self.movement_history.append({
            "timestamp": time.time(),
            "from": self.current_position.copy(),
            "to": positions.copy(),
            "speed": speed or self.current_speed,
            "trajectory": str(trajectory_type) if trajectory_type else "linear"
        })
        
        # Calculate distance
        distance = self._calculate_distance(self.current_position, positions)
        self.metrics["total_distance"] += distance
        self.metrics["total_movements"] += 1
        
        # Simuliere Bewegungszeit
        move_time = distance / (speed or self.current_speed) * 0.5  # Vereinfacht
        
        if wait and self.simulation_mode != SimulationMode.PERFECT:
            # Simuliere realistische Bewegungszeit
            logger.debug(f"[MOCK] Moving... (simulated {move_time:.2f}s)")
            time.sleep(min(move_time, 0.5))  # Max 0.5s für Tests
        
        # Update position
        self.current_position.update(positions)
        self.serial.state.positions.update(positions)
        
        # Zufällige Fehler im RANDOM Mode
        if self.simulation_mode == SimulationMode.RANDOM:
            if random.random() < 0.05:  # 5% Fehlerchance
                logger.warning("[MOCK] Random failure simulated")
                return False
        
        logger.info(f"✅ [MOCK] Moved to position (distance={distance:.3f} rad)")
        return True
    
    def move_home(self, speed: float = 1.0) -> bool:
        """Simuliert Home-Bewegung."""
        logger.info("[MOCK] Moving to home position...")
        return self.move_joints(HOME_POSITION, speed=speed)
    
    def move_to_scanner_position(self, speed: float = 0.5) -> bool:
        """Simuliert Scanner-Position."""
        logger.info("[MOCK] Moving to scanner position...")
        self.scanner_mounted = True
        return self.move_joints(SCANNER_POSITION, speed=speed)
    
    def gripper_control(self, position: float) -> bool:
        """Simuliert Greifer."""
        logger.debug(f"[MOCK] Gripper position: {position:.2f}")
        min_pos = SERVO_LIMITS["hand"][0]
        max_pos = SERVO_LIMITS["hand"][1]
        servo_pos = min_pos + (max_pos - min_pos) * position
        return self.move_joints({"hand": servo_pos}, speed=1.0)
    
    def set_torque(self, enabled: bool) -> bool:
        """Simuliert Torque-Control."""
        self.torque_enabled = enabled
        self.serial.state.torque_enabled = enabled
        logger.info(f"[MOCK] Torque {'enabled' if enabled else 'disabled'}")
        return True
    
    def led_control(self, on: bool, brightness: int = 255) -> bool:
        """Simuliert LED."""
        self.serial.state.led_state = on
        self.serial.state.led_brightness = brightness if on else 0
        logger.debug(f"[MOCK] LED: {'ON' if on else 'OFF'} ({brightness})")
        return True
    
    def emergency_stop(self):
        """Simuliert Emergency Stop."""
        logger.warning("🚨 [MOCK] EMERGENCY STOP")
        self.emergency_stop_flag = True
        self.metrics["emergency_stops"] += 1
        self.command_queue.clear()
    
    def reset_emergency(self):
        """Reset Emergency."""
        self.emergency_stop_flag = False
        logger.info("[MOCK] Emergency stop reset")
    
    def query_status(self) -> Optional[Dict]:
        """Simuliert Status-Abfrage."""
        # Simuliere gelegentliche Timeouts
        if self.simulation_mode == SimulationMode.FAILURE:
            if random.random() < 0.1:
                logger.warning("[MOCK] Status query timeout simulated")
                return None
        
        status = {
            "positions": self.current_position.copy(),
            "velocities": {joint: 0.0 for joint in self.current_position},
            "torque_enabled": self.torque_enabled,
            "temperature": 25 + random.gauss(0, 2),
            "voltage": 6.0 + random.gauss(0, 0.1)
        }
        
        return status
    
    def execute_pattern(self, pattern) -> bool:
        """Simuliert Pattern-Ausführung."""
        try:
            points = pattern.generate_points()
            logger.info(f"[MOCK] Executing pattern: {pattern.name} ({len(points)} points)")
            
            self.pattern_history.append({
                "name": pattern.name,
                "timestamp": time.time(),
                "points": len(points)
            })
            
            # Simuliere nur ersten und letzten Punkt für Geschwindigkeit
            if len(points) > 2:
                # Erster Punkt
                self.move_joints(points[0].positions, speed=points[0].speed, wait=False)
                
                # Progress simulation
                for i in range(1, len(points)-1):
                    if i % 10 == 0:
                        logger.info(f"[MOCK] Pattern progress: {i/len(points)*100:.1f}%")
                    time.sleep(0.01)  # Kurze Pause
                    
                    if self.emergency_stop_flag:
                        logger.warning("[MOCK] Pattern aborted")
                        return False
                
                # Letzter Punkt
                self.move_joints(points[-1].positions, speed=points[-1].speed, wait=False)
            
            self.metrics["patterns_executed"] += 1
            logger.info(f"✅ [MOCK] Pattern completed: {pattern.name}")
            return True
            
        except Exception as e:
            logger.error(f"[MOCK] Pattern execution error: {e}")
            return False
    
    def disconnect(self):
        """Trennt Mock-Verbindung."""
        logger.info("[MOCK] Disconnecting...")
        self.print_debug_summary()
        self.serial.disconnect()
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Berechnet Distanz zwischen Positionen."""
        distance = 0.0
        for joint in pos1:
            if joint in pos2:
                distance += abs(pos2[joint] - pos1[joint])
        return distance
    
    def print_debug_summary(self):
        """Gibt Debug-Zusammenfassung aus."""
        print("\n" + "="*60)
        print("🔍 DEBUG SESSION SUMMARY")
        print("="*60)
        print(f"Mode: {self.simulation_mode.value}")
        print(f"Total Commands: {self.metrics['total_commands']}")
        print(f"Total Movements: {self.metrics['total_movements']}")
        print(f"Total Distance: {self.metrics['total_distance']:.2f} rad")
        print(f"Patterns Executed: {self.metrics['patterns_executed']}")
        print(f"Emergency Stops: {self.metrics['emergency_stops']}")
        
        if self.movement_history:
            print(f"\nLast 5 movements:")
            for move in self.movement_history[-5:]:
                print(f"  - Distance: {self._calculate_distance(move['from'], move['to']):.3f} rad, "
                      f"Speed: {move['speed']:.1f}")
        
        if self.pattern_history:
            print(f"\nExecuted patterns:")
            for pattern in self.pattern_history:
                print(f"  - {pattern['name']}: {pattern['points']} points")
        
        print("="*60 + "\n")


class MockCommandQueue:
    """Mock Command Queue."""
    
    def __init__(self):
        self.commands = []
    
    def put(self, command):
        self.commands.append(command)
    
    def get(self, timeout=None):
        if self.commands:
            return self.commands.pop(0)
        raise Exception("Empty queue")
    
    def empty(self):
        return len(self.commands) == 0
    
    def clear(self):
        self.commands.clear()
    
    @property
    def queue(self):
        """Compatibility property."""
        return self


class DebugMenu:
    """Interaktives Debug-Menü."""
    
    def __init__(self, controller):
        self.controller = controller
        self.running = True
    
    def show(self):
        """Zeigt Debug-Menü."""
        while self.running:
            print("\n" + "="*50)
            print("🔧 DEBUG MENU")
            print("="*50)
            print("1. Show current state")
            print("2. Test movement")
            print("3. Test pattern")
            print("4. Simulate error")
            print("5. Show metrics")
            print("6. Change simulation mode")
            print("7. Command history")
            print("8. Run test sequence")
            print("0. Exit debug menu")
            
            choice = input("\n👉 Select: ").strip()
            
            if choice == '1':
                self.show_state()
            elif choice == '2':
                self.test_movement()
            elif choice == '3':
                self.test_pattern()
            elif choice == '4':
                self.simulate_error()
            elif choice == '5':
                self.show_metrics()
            elif choice == '6':
                self.change_mode()
            elif choice == '7':
                self.show_command_history()
            elif choice == '8':
                self.run_test_sequence()
            elif choice == '0':
                self.running = False
    
    def show_state(self):
        """Zeigt aktuellen Zustand."""
        print("\n📊 Current State:")
        print("-"*40)
        for joint, pos in self.controller.current_position.items():
            print(f"{joint:10s}: {pos:+.3f} rad")
        print(f"\nTorque: {self.controller.torque_enabled}")
        print(f"LED: {self.controller.serial.state.led_state}")
        print(f"Scanner: {self.controller.scanner_mounted}")
    
    def test_movement(self):
        """Testet Bewegung."""
        print("\nTest movement to random position...")
        test_pos = {}
        for joint, (min_val, max_val) in SERVO_LIMITS.items():
            test_pos[joint] = random.uniform(min_val*0.5, max_val*0.5)
        
        if self.controller.move_joints(test_pos, speed=0.5):
            print("✅ Movement successful")
        else:
            print("❌ Movement failed")
    
    def test_pattern(self):
        """Testet Pattern."""
        from patterns.scan_patterns import RasterScanPattern
        pattern = RasterScanPattern(width=0.1, height=0.1, rows=3, cols=3)
        self.controller.execute_pattern(pattern)
    
    def simulate_error(self):
        """Simuliert Fehler."""
        print("\n1. Emergency Stop")
        print("2. Communication timeout")
        print("3. Position limit error")
        
        error_type = input("Select error: ").strip()
        
        if error_type == '1':
            self.controller.emergency_stop()
            print("🚨 Emergency stop triggered")
        elif error_type == '2':
            self.controller.serial.connected = False
            print("❌ Connection lost")
        elif error_type == '3':
            invalid_pos = {"base": 10.0}  # Way out of limits
            self.controller.move_joints(invalid_pos)
    
    def show_metrics(self):
        """Zeigt Metriken."""
        print("\n📈 Session Metrics:")
        print("-"*40)
        for key, value in self.controller.metrics.items():
            print(f"{key}: {value}")
    
    def change_mode(self):
        """Ändert Simulationsmodus."""
        print("\nSelect simulation mode:")
        for mode in SimulationMode:
            print(f"  {mode.value}")
        
        new_mode = input("Mode: ").strip()
        try:
            self.controller.simulation_mode = SimulationMode(new_mode)
            print(f"✅ Mode changed to: {new_mode}")
        except:
            print("❌ Invalid mode")
    
    def show_command_history(self):
        """Zeigt Command-Historie."""
        print("\n📜 Last 10 commands:")
        for cmd in self.controller.serial.command_history[-10:]:
            print(f"  [{cmd['type']}] {cmd['command']}")
    
    def run_test_sequence(self):
        """Führt Test-Sequenz aus."""
        print("\n🧪 Running test sequence...")
        
        # Test sequence
        tests = [
            ("Home position", lambda: self.controller.move_home()),
            ("LED on", lambda: self.controller.led_control(True)),
            ("Gripper open", lambda: self.controller.gripper_control(0.0)),
            ("Gripper close", lambda: self.controller.gripper_control(1.0)),
            ("Scanner position", lambda: self.controller.move_to_scanner_position()),
            ("Status query", lambda: self.controller.query_status() is not None),
            ("LED off", lambda: self.controller.led_control(False)),
            ("Home return", lambda: self.controller.move_home())
        ]
        
        passed = 0
        for name, test in tests:
            print(f"  Testing {name}... ", end='')
            if test():
                print("✅")
                passed += 1
            else:
                print("❌")
        
        print(f"\nResult: {passed}/{len(tests)} tests passed")


def run_debug_session(controller):
    """Startet interaktive Debug-Session."""
    print("\n" + "="*60)
    print("🔧 DEBUG MODE ACTIVE")
    print("="*60)
    print("Running in simulation mode - no hardware required")
    print("All movements and commands are simulated")
    print("-"*60)
    
    menu = DebugMenu(controller)
    menu.show()
    
    # Zusammenfassung am Ende
    controller.print_debug_summary()
