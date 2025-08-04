# 🚀 RoArm Pro - Quick Start Guide

## 1️⃣ Installation (5 Minuten)

```bash
# 1. Ordnerstruktur erstellen
mkdir roarm_pro
cd roarm_pro

# 2. Unterordner erstellen
mkdir -p roarm_pro/{config,hardware,motion,control,ui}
mkdir -p examples tests docs

# 3. Alle Dateien in die richtigen Ordner kopieren
# (siehe Dateiliste unten)

# 4. Dependencies installieren
pip install -r requirements.txt

# 5. Paket installieren
python setup.py develop

# 6. Installation testen
python test_installation.py
```

## 2️⃣ Datei-Struktur

Kopieren Sie die Dateien in diese Struktur:

```
roarm_pro/
├── requirements.txt
├── setup.py
├── test_installation.py
├── Makefile
├── README.md
├── QUICKSTART.md (diese Datei)
├── .gitignore
├── roarm_pro/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── defaults.py
│   │   ├── settings.py
│   │   └── default_settings.yaml
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── serial_comm.py
│   │   ├── port_utils.py
│   │   └── commands.py
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── trajectory.py
│   │   ├── limits.py
│   │   └── kinematics.py
│   ├── control/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── manual.py
│   │   ├── teaching.py
│   │   └── scanner.py
│   └── ui/
│       ├── __init__.py
│       ├── app.py
│       └── menus.py
└── examples/
    ├── basic_usage.py
    └── scanner_example.py
```

## 3️⃣ Erste Schritte

### Robot verbinden und starten:

```bash
# Ports anzeigen
roarm --list-ports
# oder
python -m roarm_pro.main --list-ports

# Programm starten
roarm
# oder
python -m roarm_pro.main
```

### Erste Bewegungen:

1. **Verbindung prüfen**: System → 93
2. **Kalibrierung**: Hauptmenü → 1 → 1 (Vollständig)
3. **Home-Position**: Hauptmenü → 2
4. **Manuelle Kontrolle**: Hauptmenü → 4

## 4️⃣ Scanner-Betrieb

1. **Scanner montieren**:
   - Hauptmenü → 6 → 1
   - Scanner physisch montieren
   - Enter drücken

2. **Scan starten**:
   - Kontinuierlich: 6 → 2
   - Raster: 6 → 3
   - Spiral: 6 → 4

## 5️⃣ Tastatur-Steuerung

In der manuellen Kontrolle (Menü → 4):

```
Q/A = Base links/rechts
W/S = Schulter hoch/runter
E/D = Ellbogen hoch/runter
R/F = Handgelenk hoch/runter
T/G = Roll links/rechts
Y/H = Greifer auf/zu

+/- = Geschwindigkeit
SPACE = NOTAUS
ESC = Beenden
```

## 6️⃣ Häufige Probleme

### "Keine Verbindung"
```bash
# 1. USB-Kabel prüfen
# 2. Robot einschalten
# 3. Anderen Port versuchen:
roarm --port /dev/cu.usbserial-110
```

### "Permission denied"
```bash
# macOS: Terminal-Berechtigung geben
# Linux: User zu dialout Gruppe hinzufügen
sudo usermod -a -G dialout $USER
# Dann neu einloggen
```

### "Module not found"
```bash
# Installation wiederholen
pip install -r requirements.txt
python setup.py develop
```

## 7️⃣ Programmatische Nutzung

```python
from roarm_pro import RoArmController

# Controller erstellen
controller = RoArmController("/dev/cu.usbserial-110")

# Verbinden
controller.connect()

# Nach Hause fahren
controller.move_to_home()

# Einzelne Gelenke bewegen (in Radians)
controller.move_joints(base=0.5, shoulder=0.3)

# Greifer steuern (0-100%)
controller.set_gripper(50)

# Sicher beenden
controller.safe_shutdown()
```

## 8️⃣ Nächste Schritte

- **Teaching Mode** ausprobieren (Menü → 5)
- **Eigene Scan-Patterns** programmieren
- **Settings** anpassen (~/.roarm_pro/settings.yaml)
- **Examples** durcharbeiten (examples/ Ordner)

## 💡 Tipps

- Immer mit **Kalibrierung** starten
- **Langsame Geschwindigkeit** für erste Tests
- **Emergency Stop**: Ctrl+C funktioniert überall
- **Scanner-Modus** aktivieren wenn Scanner montiert

---

Bei Fragen: GitHub Issues oder Dokumentation lesen!
