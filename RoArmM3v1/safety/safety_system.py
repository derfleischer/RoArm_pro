#!/usr/bin/env python3
"""
RoArm M3 Professional Safety & Shutdown System
Sicheres Herunterfahren und Emergency Stop Management
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

# FIXED: Relative imports changed to absolute imports
from core.constants import HOME_POSITION, PARK_POSITION, SERVO_LIMITS
from motion.trajectory import TrajectoryType
from utils.logger import get_logger

logger = get_logger(__name__)


class SafetyState(Enum):
    """System-Sicherheitszustände."""
    NORMAL = "normal"                    # Normalbetrieb
    WARNING = "warning"                  # Warnung (z.B. Temperatur)
    EMERGENCY = "emergency"              # Emergency Stop aktiv
    SHUTDOWN = "shutdown"                # Herunterfahren
    RECOVERY = "recovery"                # Wiederherstellung nach Emergency
    SAFE_MODE = "safe_mode"             # Eingeschränkter Betrieb
    MAINTENANCE = "maintenance"          # Wartungsmodus


class ShutdownReason(Enum):
    """Gründe für Shutdown."""
    USER_REQUEST = "user_request"        # Normal beendet
    EMERGENCY_STOP = "emergency_stop"    # Notaus
    TEMPERATURE = "temperature"          # Überhitzung
    VOLTAGE = "voltage"                  # Spannungsproblem
    COMMUNICATION = "communication"      # Verbindungsverlust
    COLLISION = "collision"              # Kollision erkannt
    TIMEOUT = "timeout"                  # Timeout
    ERROR = "error"                      # Allgemeiner Fehler


@dataclass
class SafetyEvent:
    """Ein Sicherheitsereignis."""
    timestamp: float
    event_type: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    data: Optional[Dict] = None


@dataclass
class SystemState:
    """Aktueller Systemzustand für Recovery."""
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
    Umfassendes Safety & Shutdown Management System.
    Koordiniert sicheres Herunterfahren und Emergency Stops.
    """
    
    def __init__(self, controller):
        """
        Initialisiert Safety System.
        
        Args:
            controller: RoArm Controller Instanz
        """
        self.controller = controller
        
        # Safety State
        self.safety_state = SafetyState.NORMAL
        self.emergency_active = False
        self.shutdown_in_progress = False
        
        # Event History
        self.events: List[SafetyEvent] = []
        self.max_events = 100
        
        # Shutdown Konfiguration
        self.shutdown_config = {
            "move_to_safe": True,           # Zur sicheren Position fahren
            "safe_position": "park",         # park oder home
            "shutdown_speed": 0.3,           # Langsame Bewegung
            "led_blink_count": 3,            # LED-Blinks beim Shutdown
            "save_state": True,              # Zustand speichern
            "timeout": 30.0,                 # Max Zeit für Shutdown
            "force_after_timeout": True      # Erzwinge nach Timeout
        }
        
        # Recovery Konfiguration
        self.recovery_config = {
            "auto_recovery": True,           # Auto-Recovery nach Emergency
            "recovery_delay": 2.0,           # Wartezeit vor Recovery
            "restore_position": False,       # Position wiederherstellen
            "verify_before_restore": True    # Prüfung vor Wiederherstellung
        }
        
        # Watchdog
        self.watchdog_enabled = True
        self.watchdog_thread = None
        self.watchdog_interval = 1.0        # Prüfintervall
        self.last_heartbeat = time.time()
        
        # Callbacks
        self.emergency_callbacks: List[Callable] = []
        self.shutdown_callbacks: List[Callable] = []
        
        # Thread Lock
        self._lock = threading.Lock()
        
        # Register shutdown handlers
        self._register_handlers()
        
        # Start watchdog
        self._start_watchdog()
        
        logger.info("Safety System initialized")
    
    # ============== EMERGENCY STOP ==============
    
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
        """
        Führt sofortigen Emergency Stop aus.
        
        Args:
            reason: Grund für Emergency Stop
            
        Returns:
            True wenn erfolgreich
        """
        with self._lock:
            if self.emergency_active:
                logger.warning("Emergency stop already active")
                return True
            
            logger.critical(f"🚨 EMERGENCY STOP: {reason}")
            
            # Set state
            self.emergency_active = True
            self.safety_state = SafetyState.EMERGENCY
            
            # Log event
            self._log_event(
                "EMERGENCY_STOP",
                "CRITICAL",
                f"Emergency stop triggered: {reason}"
            )
        
        try:
            # 1. Sofort alle Bewegungen stoppen
            logger.info("Stopping all movements...")
            self._stop_all_movements()
            
            # 2. LED Alarm
            self._emergency_led_signal()
            
            # 3. Optional: Bremse aktivieren (wenn vorhanden)
            # self._engage_brakes()
            
            # 4. Speichere aktuellen Zustand
            if self.shutdown_config["save_state"]:
                self._save_current_state()
            
            # 5. Callbacks aufrufen
            self._execute_callbacks(self.emergency_callbacks)
            
            # 6. Warte kurz
            time.sleep(0.5)
            
            # 7. Optional: Torque beibehalten für Stabilität
            if not self.shutdown_config.get("disable_torque_on_emergency", False):
                logger.info("Keeping torque enabled for stability")
            else:
                logger.info("Disabling torque...")
                self.controller.set_torque(False)
            
            logger.info("✅ Emergency stop complete")
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
            # Force stop
            try:
                self.controller.serial.send_command({"T": 0})  # Direct emergency command
            except:
                pass
            return False
    
    def reset_emergency(self) -> bool:
        """
        Reset nach Emergency Stop.
        
        Returns:
            True wenn erfolgreich
        """
        with self._lock:
            if not self.emergency_active:
                logger.warning("No emergency to reset")
                return True
            
            logger.info("Resetting emergency state...")
            
            # Check if safe to reset
            if not self._verify_safe_to_reset():
                logger.error("Not safe to reset - manual intervention required")
                return False
            
            self.emergency_active = False
            self.safety_state = SafetyState.RECOVERY
        
        try:
            # 1. LED Signal
            self._recovery_led_signal()
            
            # 2. Re-enable torque
            logger.info("Re-enabling torque...")
            self.controller.set_torque(True)
            time.sleep(0.5)
            
            # 3. Move to safe position
            if self.recovery_config["auto_recovery"]:
                logger.info("Moving to safe position...")
                self.controller.move_home(speed=0.2)
                time.sleep(2)
            
            # 4. Restore state if configured
            if self.recovery_config["restore_position"]:
                self._restore_last_state()
            
            # 5. Normal state
            self.safety_state = SafetyState.NORMAL
            
            logger.info("✅ Emergency reset complete")
            return True
            
        except Exception as e:
            logger.error(f"Emergency reset error: {e}")
            self.safety_state = SafetyState.SAFE_MODE
            return False
    
    # ============== GRACEFUL SHUTDOWN ==============
    
    def graceful_shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST) -> bool:
        """
        Führt sauberes, schrittweises Herunterfahren durch.
        
        Args:
            reason: Grund für Shutdown
            
        Returns:
            True wenn erfolgreich
        """
        with self._lock:
            if self.shutdown_in_progress:
                logger.warning("Shutdown already in progress")
                return True
            
            self.shutdown_in_progress = True
            self.safety_state = SafetyState.SHUTDOWN
        
        logger.info(f"🔌 Starting graceful shutdown: {reason.value}")
        
        # Log event
        self._log_event(
            "SHUTDOWN",
            "INFO",
            f"Graceful shutdown initiated: {reason.value}"
        )
        
        shutdown_successful = False
        start_time = time.time()
        
        try:
            # Phase 1: Vorbereitung
            logger.info("Phase 1/5: Preparation...")
            self._shutdown_phase_1_preparation()
            
            # Phase 2: Bewegung zu sicherer Position
            if self.shutdown_config["move_to_safe"]:
                logger.info("Phase 2/5: Moving to safe position...")
                if not self._shutdown_phase_2_safe_position():
                    logger.warning("Failed to reach safe position")
            
            # Phase 3: Zustand speichern
            if self.shutdown_config["save_state"]:
                logger.info("Phase 3/5: Saving state...")
                self._shutdown_phase_3_save_state()
            
            # Phase 4: Hardware herunterfahren
            logger.info("Phase 4/5: Hardware shutdown...")
            self._shutdown_phase_4_hardware()
            
            # Phase 5: Aufräumen
            logger.info("Phase 5/5: Cleanup...")
            self._shutdown_phase_5_cleanup()
            
            shutdown_successful = True
            logger.info("✅ Graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            
            # Force shutdown nach Timeout
            if (time.time() - start_time) > self.shutdown_config["timeout"]:
                logger.warning("Shutdown timeout - forcing...")
                self._force_shutdown()
        
        finally:
            self.shutdown_in_progress = False
            
            # Final LED off
            try:
                self.controller.led_control(False)
            except:
                pass
        
        return shutdown_successful
    
    def _shutdown_phase_1_preparation(self):
        """Phase 1: Vorbereitung für Shutdown."""
        # Stop alle laufenden Operationen
        self.controller.command_queue.queue.clear()
        
        # LED Shutdown Signal
        for _ in range(self.shutdown_config["led_blink_count"]):
            self.controller.led_control(True, brightness=255)
            time.sleep(0.2)
            self.controller.led_control(False)
            time.sleep(0.2)
        
        # Callbacks
        self._execute_callbacks(self.shutdown_callbacks)
    
    def _shutdown_phase_2_safe_position(self) -> bool:
        """Phase 2: Bewegung zu sicherer Position."""
        try:
            position = self.shutdown_config["safe_position"]
            speed = self.shutdown_config["shutdown_speed"]
            
            if position == "park":
                target = PARK_POSITION
                logger.info("Moving to PARK position...")
            elif position == "home":
                target = HOME_POSITION
                logger.info("Moving to HOME position...")
            elif position == "current":
                logger.info("Staying at current position")
                return True
            else:
                target = HOME_POSITION
            
            # Langsame, sanfte Bewegung
            self.controller.move_joints(
                target,
                speed=speed,
                trajectory_type=TrajectoryType.S_CURVE
            )
            
            # Extra Wartezeit für Stabilität
            time.sleep(1.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to move to safe position: {e}")
            return False
    
    def _shutdown_phase_3_save_state(self):
        """Phase 3: Zustand speichern."""
        try:
            state = self._get_current_state()
            state.save()
            logger.info("System state saved")
            
            # Speichere auch Kalibrierung wenn vorhanden
            if hasattr(self.controller, 'calibrator'):
                if self.controller.calibrator.calibration.calibration_valid:
                    self.controller.calibrator.save_calibration()
                    logger.info("Calibration saved")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _shutdown_phase_4_hardware(self):
        """Phase 4: Hardware herunterfahren."""
        try:
            # Reduziere Torque schrittweise
            logger.info("Reducing torque gradually...")
            
            # Torque in Stufen reduzieren
            torque_steps = [0.8, 0.6, 0.4, 0.2, 0.0]
            for torque in torque_steps:
                if torque > 0:
                    # Pseudo-Code: Torque reduzieren
                    time.sleep(0.2)
            
            # Torque komplett aus
            logger.info("Disabling torque...")
            self.controller.set_torque(False)
            time.sleep(0.5)
            
            # LED aus
            self.controller.led_control(False)
            
        except Exception as e:
            logger.error(f"Hardware shutdown error: {e}")
    
    def _shutdown_phase_5_cleanup(self):
        """Phase 5: Aufräumen."""
        try:
            # Stop watchdog
            self.watchdog_enabled = False
            if self.watchdog_thread:
                self.watchdog_thread.join(timeout=2)
            
            # Close serial connection
            if self.controller.serial:
                logger.info("Closing serial connection...")
                self.controller.serial.disconnect()
            
            # Save event log
            self._save_event_log()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _force_shutdown(self):
        """Erzwungenes Herunterfahren bei Timeout."""
        logger.warning("Forcing shutdown...")
        
        try:
            # Direct emergency stop command
            self.controller.serial.send_command({"T": 0})
            time.sleep(0.1)
            
            # Torque off
            self.controller.set_torque(False)
            
            # LED off
            self.controller.led_control(False)
            
            # Close serial
            self.controller.serial.disconnect()
            
        except:
            pass
    
    # ============== WATCHDOG ==============
    
    def _start_watchdog(self):
        """Startet Watchdog Thread."""
        if self.watchdog_enabled and not self.watchdog_thread:
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True
            )
            self.watchdog_thread.start()
            logger.debug("Watchdog started")
    
    def _watchdog_loop(self):
        """Watchdog Loop - überwacht System."""
        while self.watchdog_enabled:
            try:
                # Check various conditions
                self._check_temperature()
                self._check_voltage()
                self._check_communication()
                self._check_position_limits()
                
                # Update heartbeat
                self.last_heartbeat = time.time()
                
                time.sleep(self.watchdog_interval)
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(1)
    
    def _check_temperature(self):
        """Prüft Temperatur."""
        try:
            status = self.controller.query_status()
            if status and 'temperature' in status:
                temp = status['temperature']
                
                if temp > 60:  # Critical
                    logger.critical(f"CRITICAL TEMPERATURE: {temp}°C")
                    self.emergency_stop("Temperature critical")
                elif temp > 50:  # Warning
                    logger.warning(f"High temperature: {temp}°C")
                    self.safety_state = SafetyState.WARNING
                    self._log_event("TEMPERATURE_WARNING", "WARNING", f"Temperature: {temp}°C")
        except:
            pass
    
    def _check_voltage(self):
        """Prüft Spannung."""
        try:
            status = self.controller.query_status()
            if status and 'voltage' in status:
                voltage = status['voltage']
                
                if voltage < 5.5 or voltage > 7.0:
                    logger.warning(f"Voltage out of range: {voltage}V")
                    self._log_event("VOLTAGE_WARNING", "WARNING", f"Voltage: {voltage}V")
        except:
            pass
    
    def _check_communication(self):
        """Prüft Kommunikation."""
        if not self.controller.serial.connected:
            logger.error("Communication lost")
            self.safety_state = SafetyState.EMERGENCY
            self._log_event("COMMUNICATION_LOST", "ERROR", "Serial connection lost")
    
    def _check_position_limits(self):
        """Prüft ob Positionen in sicheren Grenzen."""
        try:
            positions = self.controller.current_position
            for joint, pos in positions.items():
                if joint in SERVO_LIMITS:
                    min_val, max_val = SERVO_LIMITS[joint]
                    if pos < min_val - 0.1 or pos > max_val + 0.1:
                        logger.warning(f"Joint {joint} near limit: {pos:.3f}")
                        self._log_event(
                            "POSITION_LIMIT",
                            "WARNING",
                            f"Joint {joint} at {pos:.3f} rad"
                        )
        except:
            pass
    
    # ============== HELPER METHODS ==============
    
    def _stop_all_movements(self):
        """Stoppt sofort alle Bewegungen."""
        try:
            # Clear command queue
            while not self.controller.command_queue.empty():
                self.controller.command_queue.get()
            
            # Send emergency stop command
            self.controller.serial.send_command({"T": 0})
            
        except Exception as e:
            logger.error(f"Failed to stop movements: {e}")
    
    def _emergency_led_signal(self):
        """LED Signal für Emergency."""
        try:
            # Schnelles Blinken
            for _ in range(5):
                self.controller.led_control(True, brightness=255)
                time.sleep(0.1)
                self.controller.led_control(False)
                time.sleep(0.1)
            
            # Dauerhaft rot (wenn RGB LED)
            self.controller.led_control(True, brightness=255)
            
        except:
            pass
    
    def _recovery_led_signal(self):
        """LED Signal für Recovery."""
        try:
            # Langsames Blinken
            for _ in range(3):
                self.controller.led_control(True, brightness=128)
                time.sleep(0.5)
                self.controller.led_control(False)
                time.sleep(0.5)
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
            logger.debug("Current state saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _restore_last_state(self):
        """Stellt letzten Zustand wieder her."""
        try:
            state = SystemState.load()
            if state:
                logger.info("Restoring last state...")
                
                # Verify positions are safe
                if self._verify_positions_safe(state.positions):
                    self.controller.move_joints(
                        state.positions,
                        speed=0.2,
                        trajectory_type=TrajectoryType.S_CURVE
                    )
                    logger.info("State restored")
                else:
                    logger.warning("Last positions not safe - moving to home")
                    self.controller.move_home(speed=0.2)
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
    
    def _verify_safe_to_reset(self) -> bool:
        """Prüft ob Reset sicher ist."""
        try:
            # Check communication
            if not self.controller.serial.connected:
                return False
            
            # Check if can query status
            status = self.controller.query_status()
            if not status:
                return False
            
            # Check temperature if available
            if 'temperature' in status:
                if status['temperature'] > 60:
                    return False
            
            return True
            
        except:
            return False
    
    def _verify_positions_safe(self, positions: Dict[str, float]) -> bool:
        """Prüft ob Positionen sicher sind."""
        for joint, pos in positions.items():
            if joint in SERVO_LIMITS:
                min_val, max_val = SERVO_LIMITS[joint]
                if pos < min_val or pos > max_val:
                    return False
        return True
    
    def _log_event(self, event_type: str, severity: str, message: str, data: Optional[Dict] = None):
        """Loggt Sicherheitsereignis."""
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            message=message,
            data=data
        )
        
        self.events.append(event)
        
        # Limit event history
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def _save_event_log(self):
        """Speichert Event-Log."""
        try:
            filepath = f"safety/events_{time.strftime('%Y%m%d_%H%M%S')}.json"
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            events_data = [asdict(e) for e in self.events]
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2)
            
            logger.info(f"Event log saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save event log: {e}")
    
    def _execute_callbacks(self, callbacks: List[Callable]):
        """Führt Callbacks aus."""
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _register_handlers(self):
        """Registriert System-Handler."""
        # Register atexit handler
        atexit.register(self._atexit_handler)
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        # SIGINT wird bereits vom Main Handler behandelt
    
    def _atexit_handler(self):
        """Handler für Programm-Exit."""
        if not self.shutdown_in_progress:
            logger.info("Exit detected - initiating shutdown...")
            self.graceful_shutdown(ShutdownReason.USER_REQUEST)
    
    def _signal_handler(self, signum, frame):
        """Handler für System-Signals."""
        logger.info(f"Signal {signum} received")
        if signum == signal.SIGTERM:
            self.graceful_shutdown(ShutdownReason.USER_REQUEST)
    
    # ============== PUBLIC METHODS ==============
    
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
        logger.info(f"Shutdown config updated: {kwargs}")
    
    def set_recovery_config(self, **kwargs):
        """Aktualisiert Recovery-Konfiguration."""
        self.recovery_config.update(kwargs)
        logger.info(f"Recovery config updated: {kwargs}")
