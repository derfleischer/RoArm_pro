# 🛡️ RoArm M3 Safety System Documentation

## Übersicht

Das Safety System ist ein kritischer Bestandteil der RoArm M3 Professional Software und bietet umfassenden Schutz für Hardware und Benutzer durch:

- **Emergency Stop** mit automatischer Recovery
- **Graceful Shutdown** mit schrittweisem Herunterfahren
- **Watchdog Monitoring** für Temperatur, Spannung und Positionen
- **State Persistence** für Recovery nach Stromausfall
- **Event Logging** für Fehleranalyse

## 🚨 Emergency Stop

### Auslösung

Der Emergency Stop kann auf verschiedene Arten ausgelöst werden:

1. **Ctrl+C** - Jederzeit während des Betriebs
2. **Spacebar** - Im Manual Control Mode
3. **Automatisch** - Bei kritischen Bedingungen (Temperatur, Kollision)
4. **Programmatisch** - `safety_system.emergency_stop("Reason")`

### Emergency Stop Sequenz

```
1. SOFORTIGE BEWEGUNGSSTOP
   ↓
2. Command Queue leeren
   ↓
3. LED Alarm Signal (5x schnelles Blinken)
   ↓
4. Zustand speichern
   ↓
5. Optional: Torque beibehalten für Stabilität
   ↓
6. Warten auf Reset oder Shutdown
```

### Recovery nach Emergency Stop

Nach einem Emergency Stop haben Sie zwei Optionen:

#### Option 1: Reset und Fortfahren

```python
# Automatisch nach Ctrl+C
> Options:
> 1. Reset and continue
> 2. Shutdown system
> Select: 1

# Oder manuell
safety_system.reset_emergency()
```

**Recovery-Sequenz:**
1. Sicherheitsprüfung (Temperatur, Spannung, Kommunikation)
2. Torque wieder aktivieren
3. Zur Home-Position fahren (optional)
4. Letzten Zustand wiederherstellen (optional)
5. Normalbetrieb fortsetzen

#### Option 2: Sicheres Herunterfahren

Führt einen kontrollierten Shutdown durch (siehe unten).

## 🔌 Graceful Shutdown

### Shutdown-Phasen

Das System fährt in 5 kontrollierten Phasen herunter:

#### Phase 1: Vorbereitung
- Command Queue stoppen
- LED-Signal (3x blinken)
- Shutdown-Callbacks ausführen
- Aktuelle Sequenzen speichern

#### Phase 2: Sichere Position
- Langsame Bewegung zur Park- oder Home-Position
- S-Kurven Trajectory für sanfte Bewegung
- Extra Wartezeit für Stabilität

#### Phase 3: Zustand speichern
- Aktuelle Positionen speichern
- Kalibrierung sichern
- Teaching-Sequenzen autosave
- Event-Log exportieren

#### Phase 4: Hardware herunterfahren
- Torque schrittweise reduzieren (100% → 80% → 60% → 40% → 20% → 0%)
- Servos sanft entspannen
- LED ausschalten

#### Phase 5: Aufräumen
- Watchdog stoppen
- Serial-Verbindung schließen
- Ressourcen freigeben

### Shutdown-Konfiguration

```yaml
shutdown:
  move_to_safe: true        # Zur sicheren Position fahren
  safe_position: "park"     # park, home oder current
  shutdown_speed: 0.3       # Langsame, sichere Bewegung
  led_blink_count: 3        # Visuelle Warnung
  save_state: true          # Für Recovery speichern
  timeout: 30.0             # Max Zeit bevor Force-Shutdown
  gradual_torque: true      # Sanftes Torque-Reduzieren
```

## 🔍 Watchdog Monitoring

Der Watchdog überwacht kontinuierlich:

### Temperatur-Überwachung
- **Warning**: >50°C - System in Warning-State
- **Critical**: >60°C - Automatischer Emergency Stop

### Spannungs-Überwachung
- **Normal**: 5.5V - 7.0V
- **Außerhalb**: Warning Event + Optional Emergency

### Positions-Überwachung
- Prüft ob Joints sich den Limits nähern
- Warning bei <0.1 rad zu Limits
- Verhindert Hardware-Schäden

### Kommunikations-Überwachung
- Prüft Serial-Verbindung
- Emergency bei Verbindungsverlust

## 💾 State Persistence

### Gespeicherte Informationen

Bei Emergency Stop oder Shutdown wird gespeichert:

```json
{
  "timestamp": 1234567890.123,
  "positions": {
    "base": 0.0,
    "shoulder": 0.35,
    "elbow": 1.57,
    "wrist": -1.57,
    "roll": 1.57,
    "hand": 2.5
  },
  "speed": 1.0,
  "trajectory_type": "s_curve",
  "torque_enabled": true,
  "scanner_mounted": true,
  "safety_state": "emergency",
  "last_command": "scan_pattern",
  "sequence_name": "turntable_scan"
}
```

### Recovery nach Stromausfall

Beim nächsten Start:

1. System erkennt unerwarteten Shutdown
2. Lädt letzten Zustand
3. Fragt ob Recovery gewünscht
4. Stellt Position/Sequenz wieder her

## 📊 Event Logging

### Event-Typen

- **INFO**: Normale Operationen
- **WARNING**: Nicht-kritische Probleme
- **ERROR**: Fehler die Recovery erlauben
- **CRITICAL**: Emergency-Situationen

### Event-Historie

```python
# Letzte Events abrufen
events = safety_system.get_event_history(10)

# Event-Struktur
{
  "timestamp": 1234567890.123,
  "event_type": "TEMPERATURE_WARNING",
  "severity": "WARNING",
  "message": "Temperature: 52°C",
  "data": {"temperature": 52, "joint": "shoulder"}
}
```

## 🎮 Verwendung im Code

### Basic Setup

```python
from safety.safety_system import SafetySystem, ShutdownReason

# Initialize
safety = SafetySystem(controller)

# Configure
safety.set_shutdown_config(
    move_to_safe=True,
    safe_position="park",
    shutdown_speed=0.3
)

# Register callbacks
safety.register_emergency_callback(on_emergency)
safety.register_shutdown_callback(on_shutdown)
```

### Emergency Handling

```python
# Trigger emergency
safety.emergency_stop("Collision detected")

# Check state
if safety.emergency_active:
    print("System in emergency state")
    
# Reset
if safety.reset_emergency():
    print("System recovered")
```

### Graceful Shutdown

```python
# Normal shutdown
safety.graceful_shutdown(ShutdownReason.USER_REQUEST)

# Error shutdown
safety.graceful_shutdown(ShutdownReason.ERROR)

# Temperature shutdown
safety.graceful_shutdown(ShutdownReason.TEMPERATURE)
```

## 🔧 CLI Menü-Integration

### Safety System Menü (Taste 'S')

```
🛡️ SAFETY SYSTEM
----------------------------------------
Current State: normal
Emergency Active: False
Watchdog Active: True

Recent Events:
  - TEMPERATURE_WARNING: Temperature: 52°C
  - POSITION_LIMIT: Joint base at 3.10 rad

1. Test Emergency Stop
2. Reset Emergency State
3. Test Graceful Shutdown
4. Configure Shutdown Behavior
5. View Event History
6. Test Recovery Sequence
7. Configure Watchdog
0. Back
```

## ⚙️ Best Practices

### 1. Immer Safety System initialisieren

```python
# In main.py
self.safety_system = SafetySystem(self.controller)
```

### 2. Kritische Operationen absichern

```python
try:
    # Kritische Operation
    controller.execute_pattern(pattern)
except Exception as e:
    safety.emergency_stop(f"Pattern failed: {e}")
```

### 3. Regelmäßige Status-Checks

```python
status = safety.get_safety_status()
if status['state'] != 'normal':
    # Handle abnormal state
    pass
```

### 4. Sauberes Herunterfahren

```python
# Immer über Safety System
def cleanup():
    safety.graceful_shutdown()
    
atexit.register(cleanup)
```

## 🚦 Status-Indikatoren

### LED-Signale

| Muster | Bedeutung |
|--------|-----------|
| 5x schnell blinken | Emergency Stop |
| 3x langsam blinken | Shutdown in Progress |
| 2x blinken | Recovery |
| Dauerhaft an | Emergency Active |
| Aus | Normal/Shutdown Complete |

### System States

| State | Icon | Bedeutung |
|-------|------|-----------|
| NORMAL | ✅ | Alles OK |
| WARNING | ⚠️ | Warnung (z.B. Temperatur) |
| EMERGENCY | 🚨 | Emergency Stop aktiv |
| SHUTDOWN | 🔌 | Herunterfahren |
| RECOVERY | 🔄 | Wiederherstellung |
| SAFE_MODE | 🛡️ | Eingeschränkter Betrieb |

## 📈 Performance Impact

- **Watchdog**: ~1% CPU (1Hz Prüfung)
- **Event Logging**: <100KB RAM (100 Events)
- **State Save**: ~10ms
- **Emergency Stop**: <50ms Reaktionszeit
- **Graceful Shutdown**: 10-30s (abhängig von Position)

## 🐛 Troubleshooting

### Problem: Emergency Reset fehlgeschlagen

```python
# Manueller Reset
safety.safety_state = SafetyState.NORMAL
safety.emergency_active = False
controller.set_torque(True)
```

### Problem: Shutdown hängt

```python
# Force Shutdown nach Timeout
safety._force_shutdown()
```

### Problem: Watchdog zu sensitiv

```yaml
# In config.yaml anpassen
watchdog:
  temperature_warning: 55  # Erhöhen
  position_margin: 0.2     # Mehr Spielraum
```

## 📝 Changelog

### Version 2.0.0
- Initiale Safety System Implementation
- Emergency Stop mit Recovery
- 5-Phasen Graceful Shutdown
- Watchdog Monitoring
- State Persistence
- Event Logging System

## 🔗 Siehe auch

- [Hauptdokumentation](README.md)
- [Kalibrierungssystem](CALIBRATION.md)
- [Teaching Mode](TEACHING.md)
- [Waveshare Wiki](https://www.waveshare.com/wiki/RoArm-M3)
