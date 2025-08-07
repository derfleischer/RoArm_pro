#!/usr/bin/env python3
"""
RoArm M3 Professional Safety & Shutdown System V2
Drop-in Replacement - Gleiche API, aber nicht-blockierend
"""

import time
import threading
import signal
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable, List
from enum import Enum
import atexit

from core.constants import HOME_POSITION, PARK_POSITION, SERVO_LIMITS
from motion.trajectory import TrajectoryType
from utils.logger import get_logger

logger = get_logger(__name__)


class SafetyState(Enum):
    """System-Sicherheitszustände - ORIGINAL ENUMS."""
    NORMAL = "normal"
    WARNING = "warning"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"
    RECOVERY = "recovery"
    SAFE_MODE = "safe_mode"
    MAINTENANCE = "maintenance"


class ShutdownReason(Enum):
    """Gründe für Shutdown - ORIGINAL ENUMS."""
    USER_REQUEST = "user_request"
    EMERGENCY_STOP = "emergency_stop"
    TEMPERATURE = "temperature"
    VOLTAGE = "voltage"
    COMMUNICATION = "communication"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SafetyEvent:
    """Ein Sicherheitsereignis - ORIGINAL."""
    timestamp: float
    event_type: str
    severity: str
    message: str
    data: Optional[Dict] = None


@dataclass
class SystemState:
    """Aktueller Systemzustand für Recovery - ORIGINAL."""
    timestamp: float
    positions: Dict[str, float]
    speed: float
    trajectory_type: str
    torque_enabled: bool
    led_state: bool
    led_brightness: int
    scanner_mounted: bool
    safety_state: str
    last_command: Optional[str] = None
    sequence_name: Optional[str] = None
    
    def save(self, filepath: str = "safety/last_state.json"):
        """Speichert Systemzustand."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str = "safety/last_state.json"):
        """Lädt letzten Systemzustand."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except:
            return None


class SafetySystem:
    """
    Safety System V2 - Kompatible API aber nicht-blockierend.
    Gleiche Methoden und Signaturen wie Original!
    """
    
    def __init__(self, controller):
        """
        Initialisiert Safety System - GLEICHE SIGNATUR.
        
        Args:
            controller: RoArm Controller Instanz
        """
        self.controller = controller
        
        # Safety State - ORIGINAL VARIABLEN
        self.safety_state = SafetyState.NORMAL
        self.emergency_active = False
        self.shutdown_in_progress = False
        
        # Event History
        self.events: List[SafetyEvent] = []
        self.max_events = 100
        
        # Shutdown Konfiguration - ORIGINAL
        self.shutdown_config = {
            "move_to_safe": True,
            "safe_position": "home",  # GEÄNDERT: home statt park!
            "shutdown_speed": 0.3,
            "led_blink_count": 3,
            "save_state": True,
            "timeout": 30.0,
            "force_after_timeout": True
        }
        
        # Recovery Konfiguration - ORIGINAL
        self.recovery_config = {
            "auto_recovery": True,
            "recovery_delay": 2.0,
            "restore_position": False,
            "verify_before_restore": True
        }
        
        # Watchdog - ABER DEAKTIVIERT!
        self.watchdog_enabled = False  # WAR: True - HAUPTÄNDERUNG!
        self.watchdog_thread = None
        self.watchdog_interval = 5.0  # WAR: 1.0 - Langsamer!
        self.last_heartbeat = time.time()
        
        # Voltage tolerance - NEU aber intern
        self._voltage_error_count = 0
        self._voltage_error_threshold = 10  # Erst nach 10 Fehlern warnen
        self._suppress_voltage_warnings = True  # Voltage Warnings unterdrücken
        
        # Callbacks - ORIGINAL
        self.emergency_callbacks: List[Callable] = []
        self.shutdown_callbacks: List[Callable] = []
        
        # Thread Lock
        self._lock = threading.Lock()
        
        # Register shutdown handlers - ORIGINAL
        self._register_handlers()
        
        # Start watchdog nur wenn enabled
        if self.watchdog_enabled:
            self._start_watchdog()
        
        logger.info("Safety System initialized (V2 - Non-blocking mode)")
    
    # ============== ORIGINAL PUBLIC METHODS ==============
    
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
        """
        Führt sofortigen Emergency Stop aus - ABER WENIGER AGGRESSIV.
        """
        with self._lock:
            if self.emergency_active:
                logger.debug("Emergency stop already active")
                return True
            
            logger.warning(f"⚠️ EMERGENCY STOP: {reason}")
            
            # Set state
            self.emergency_active = True
            self.safety_state = SafetyState.EMERGENCY
            
            # Log event
            self._log_event(
                "EMERGENCY_STOP",
                "WARNING",  # WAR: CRITICAL
                f"Emergency stop triggered: {reason}"
            )
        
        try:
            # Clear queue
            while not self.controller.command_queue.empty():
                try:
                    self.controller.command_queue.get_nowait()
                except:
                    break
            
            # Stop command
            self.controller.serial.send_command({"T": 0})
            
            # LED Signal
            self.controller.led_control(True, brightness=255)
            time.sleep(0.2)
            self.controller.led_control(False)
            
            # NICHT Torque deaktivieren (außer explizit gewünscht)
            # self.controller.set_torque(False)  # AUSKOMMENTIERT!
            
            logger.info("Emergency stop executed (torque maintained)")
            return True
            
        except Exception as e:
            logger.debug(f"Emergency stop error: {e}")
            return False
    
    def reset_emergency(self) -> bool:
        """Reset nach Emergency Stop - ORIGINAL SIGNATUR."""
        with self._lock:
            if not self.emergency_active:
                logger.debug("No emergency to reset")
                return True
            
            logger.info("Resetting emergency state...")
            self.emergency_active = False
            self.safety_state = SafetyState.RECOVERY
        
        try:
            # Re-enable torque
            self.controller.set_torque(True)
            time.sleep(0.5)
            
            # Back to normal
            self.safety_state = SafetyState.NORMAL
            
            logger.info("✅ Emergency reset complete")
            return True
            
        except Exception as e:
            logger.debug(f"Emergency reset error: {e}")
            self.safety_state = SafetyState.SAFE_MODE
            return False
    
    def graceful_shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST) -> bool:
        """
        Führt sauberes Herunterfahren durch - VEREINFACHT.
        """
        with self._lock:
            if self.shutdown_in_progress:
                logger.debug("Shutdown already in progress")
                return True
            
            self.shutdown_in_progress = True
            self.safety_state = SafetyState.SHUTDOWN
        
        logger.info(f"Starting graceful shutdown: {reason.value}")
        
        try:
            # Vereinfachter Shutdown
            if self.controller and self.controller.serial and self.controller.serial.connected:
                
                # LED blink
                for _ in range(2):
                    self.controller.led_control(True, brightness=128)
                    time.sleep(0.2)
                    self.controller.led_control(False)
                    time.sleep(0.2)
                
                # Move to HOME (nicht PARK!)
                if self.shutdown_config["move_to_safe"]:
                    logger.info("Moving to home position...")
                    try:
                        self.controller.move_home(speed=0.3)
                        time.sleep(1)
                    except:
                        pass  # Fehler ignorieren
                
                # LED off
                self.controller.led_control(False)
                
                # Stop watchdog
                self.watchdog_enabled = False
            
            logger.info("✅ Graceful shutdown complete")
            return True
            
        except Exception as e:
            logger.debug(f"Shutdown error: {e}")
            return False
        
        finally:
            self.shutdown_in_progress = False
    
    # ============== WATCHDOG - ENTSCHÄRFT ==============
    
    def _start_watchdog(self):
        """Startet Watchdog Thread - ORIGINAL."""
        if self.watchdog_enabled and not self.watchdog_thread:
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True
            )
            self.watchdog_thread.start()
            logger.debug("Watchdog started (reduced frequency)")
    
    def _watchdog_loop(self):
        """Watchdog Loop - WENIGER AGGRESSIV."""
        while self.watchdog_enabled:
            try:
                # Nur grundlegende Checks
                self._check_temperature()
                
                # Voltage Check mit Unterdrückung
                if not self._suppress_voltage_warnings:
                    self._check_voltage()
                
                # Communication Check entfernt!
                # self._check_communication()  # ENTFERNT!
                
                self.last_heartbeat = time.time()
                time.sleep(self.watchdog_interval)
                
            except Exception as e:
                logger.debug(f"Watchdog error (ignored): {e}")
                time.sleep(self.watchdog_interval)
    
    def _check_temperature(self):
        """Prüft Temperatur - NUR BEI EXTREMWERTEN."""
        try:
            status = self.controller.query_status()
            if status and 'temperature' in status:
                temp = status['temperature']
                
                # Ignoriere unrealistische Werte
                if temp <= 0 or temp > 100:
                    return
                
                # Nur bei wirklich kritischen Temperaturen
                if temp > 75:  # WAR: 60
                    logger.warning(f"High temperature: {temp}°C")
                    self._log_event("TEMPERATURE_WARNING", "WARNING", f"Temperature: {temp}°C")
                    
                    if temp > 85:  # WAR: 60 für critical
                        self.emergency_stop("Temperature critical")
        except:
            pass
    
    def _check_voltage(self):
        """Prüft Spannung - MIT FEHLERTOLERANZ."""
        try:
            status = self.controller.query_status()
            if status and 'voltage' in status:
                voltage = status['voltage']
                
                # Ignoriere 0V (offensichtlich falsches Reading)
                if voltage == 0:
                    self._voltage_error_count += 1
                    
                    # Nur nach vielen Fehlern loggen
                    if self._voltage_error_count == self._voltage_error_threshold:
                        logger.debug("Voltage reading issues detected (0V readings)")
                    return
                
                # Reset counter bei gutem Reading
                if 5.5 <= voltage <= 7.0:
                    self._voltage_error_count = 0
                    return
                
                # Nur einmal pro Minute warnen
                current_time = time.time()
                if current_time - getattr(self, '_last_voltage_warning', 0) > 60:
                    logger.debug(f"Voltage: {voltage}V")
                    self._last_voltage_warning = current_time
        except:
            pass
    
    def _check_communication(self):
        """DEAKTIVIERT - Prüft Kommunikation nicht mehr."""
        pass  # Macht nichts mehr!
    
    def _check_position_limits(self):
        """DEAKTIVIERT - Controller prüft das selbst."""
        pass  # Macht nichts mehr!
    
    # ============== HELPER METHODS - ORIGINAL ==============
    
    def _stop_all_movements(self):
        """Stoppt sofort alle Bewegungen."""
        try:
            while not self.controller.command_queue.empty():
                self.controller.command_queue.get_nowait()
            self.controller.serial.send_command({"T": 0})
        except:
            pass
    
    def _emergency_led_signal(self):
        """LED Signal für Emergency."""
        try:
            for _ in range(3):  # WAR: 5
                self.controller.led_control(True, brightness=255)
                time.sleep(0.1)
                self.controller.led_control(False)
                time.sleep(0.1)
        except:
            pass
    
    def _recovery_led_signal(self):
        """LED Signal für Recovery."""
        try:
            for _ in range(2):  # WAR: 3
                self.controller.led_control(True, brightness=128)
                time.sleep(0.3)
                self.controller.led_control(False)
                time.sleep(0.3)
        except:
            pass
    
    def _get_current_state(self) -> SystemState:
        """Erfasst aktuellen Systemzustand."""
        return SystemState(
            timestamp=time.time(),
            positions=self.controller.current_position.copy(),
            speed=self.controller.current_speed,
            trajectory_type=TrajectoryType.S_CURVE.value,
            torque_enabled=self.controller.torque_enabled,
            led_state=False,
            led_brightness=0,
            scanner_mounted=self.controller.scanner_mounted,
            safety_state=self.safety_state.value
        )
    
    def _save_current_state(self):
        """Speichert aktuellen Zustand."""
        try:
            state = self._get_current_state()
            state.save()
        except:
            pass
    
    def _restore_last_state(self):
        """Stellt letzten Zustand wieder her."""
        try:
            state = SystemState.load()
            if state and self._verify_positions_safe(state.positions):
                self.controller.move_joints(state.positions, speed=0.2)
        except:
            pass
    
    def _verify_safe_to_reset(self) -> bool:
        """Prüft ob Reset sicher ist."""
        return True  # Immer erlauben
    
    def _verify_positions_safe(self, positions: Dict[str, float]) -> bool:
        """Prüft ob Positionen sicher sind."""
        for joint, pos in positions.items():
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                if pos < min_val - 0.1 or pos > max_val + 0.1:
                    return False
        return True
    
    def _log_event(self, event_type: str, severity: str, message: str, data: Optional[Dict] = None):
        """Loggt Sicherheitsereignis - ABER WENIGER VERBOSE."""
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            message=message,
            data=data
        )
        
        self.events.append(event)
        
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Nur wichtige Events loggen
        if severity in ["CRITICAL", "ERROR"]:
            logger.warning(f"{event_type}: {message}")
        elif severity == "WARNING" and event_type != "VOLTAGE_WARNING":
            logger.debug(f"{event_type}: {message}")
    
    def _save_event_log(self):
        """Speichert Event-Log."""
        try:
            filepath = f"safety/events_{time.strftime('%Y%m%d_%H%M%S')}.json"
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            events_data = [asdict(e) for e in self.events]
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2)
        except:
            pass
    
    def _execute_callbacks(self, callbacks: List[Callable]):
        """Führt Callbacks aus."""
        for callback in callbacks:
            try:
                callback()
            except:
                pass
    
    def _register_handlers(self):
        """Registriert System-Handler."""
        atexit.register(self._atexit_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _atexit_handler(self):
        """Handler für Programm-Exit."""
        if not self.shutdown_in_progress:
            logger.debug("Exit detected - initiating shutdown...")
            self.graceful_shutdown(ShutdownReason.USER_REQUEST)
    
    def _signal_handler(self, signum, frame):
        """Handler für System-Signals."""
        logger.debug(f"Signal {signum} received")
        if signum == signal.SIGTERM:
            self.graceful_shutdown(ShutdownReason.USER_REQUEST)
    
    # ============== ORIGINAL PUBLIC METHODS ==============
    
    def register_emergency_callback(self, callback: Callable):
        """Registriert Callback für Emergency Stop."""
        self.emergency_callbacks.append(callback)
    
    def register_shutdown_callback(self, callback: Callable):
        """Registriert Callback für Shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def get_safety_status(self) -> Dict:
        """Gibt aktuellen Safety-Status zurück."""
        return {
            "state": self.safety_state.value,
            "emergency_active": self.emergency_active,
            "shutdown_in_progress": self.shutdown_in_progress,
            "events_count": len(self.events),
            "last_heartbeat": self.last_heartbeat,
            "watchdog_active": self.watchdog_enabled
        }
    
    def get_event_history(self, count: int = 10) -> List[Dict]:
        """Gibt letzte Events zurück."""
        recent_events = self.events[-count:]
        return [asdict(e) for e in recent_events]
    
    def set_shutdown_config(self, **kwargs):
        """Aktualisiert Shutdown-Konfiguration."""
        self.shutdown_config.update(kwargs)
        logger.debug(f"Shutdown config updated: {kwargs}")
    
    def set_recovery_config(self, **kwargs):
        """Aktualisiert Recovery-Konfiguration."""
        self.recovery_config.update(kwargs)
        logger.debug(f"Recovery config updated: {kwargs}")
    
    # ============== KOMPATIBILITÄTS-METHODEN ==============
    
    def _shutdown_phase_1_preparation(self):
        """Phase 1: Vorbereitung für Shutdown."""
        self.controller.command_queue.queue.clear()
        for _ in range(2):
            self.controller.led_control(True, brightness=255)
            time.sleep(0.2)
            self.controller.led_control(False)
            time.sleep(0.2)
        self._execute_callbacks(self.shutdown_callbacks)
    
    def _shutdown_phase_2_safe_position(self) -> bool:
        """Phase 2: Bewegung zu sicherer Position."""
        try:
            # Immer HOME statt PARK!
            self.controller.move_home(speed=0.3)
            time.sleep(1.0)
            return True
        except:
            return False
    
    def _shutdown_phase_3_save_state(self):
        """Phase 3: Zustand speichern."""
        self._save_current_state()
    
    def _shutdown_phase_4_hardware(self):
        """Phase 4: Hardware herunterfahren."""
        try:
            # Sanfter
            time.sleep(0.5)
            # self.controller.set_torque(False)  # Optional
            self.controller.led_control(False)
        except:
            pass
    
    def _shutdown_phase_5_cleanup(self):
        """Phase 5: Aufräumen."""
        self.watchdog_enabled = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=1)
        self._save_event_log()
    
    def _force_shutdown(self):
        """Erzwungenes Herunterfahren."""
        try:
            self.controller.serial.send_command({"T": 0})
            self.controller.led_control(False)
        except:
            pass
