# 🚀 RoArm M3 Professional - Setup Guide

## 📋 Prerequisites

- **Python 3.7+** installed
- **macOS** (optimized for M-series chips) or Linux
- **USB Serial Driver** for your system
- **Waveshare RoArm M3** robot arm
- **Optional:** Revopoint Mini2 Scanner

## 🔧 Installation Steps

### 1. Clone or Download the Project

```bash
git clone https://github.com/derfleischer/RoArm_pro.git
cd RoArm_pro/RoArmM3v1
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install as package (recommended)
pip install -e .
```

### 4. Run System Diagnostics

Before first run, check your system:

```bash
# Full diagnostic check
python3 utils/debug_tool.py

# Quick check only
python3 utils/debug_tool.py --quick
```

### 5. Configure Serial Port

Edit `config.yaml` and set your serial port:

```yaml
system:
  port: "/dev/tty.usbserial-110"  # Update this!
  baudrate: 115200
```

To find your port:
```bash
# On macOS
ls /dev/tty.usb*

# On Linux
ls /dev/ttyUSB*

# Or use Python
python3 -c "import serial.tools.list_ports; [print(p.device) for p in serial.tools.list_ports.comports()]"
```

## 🏃 Running the System

### Method 1: Using the Startup Script (Recommended)

```bash
# Make script executable
chmod +x run.sh

# Run with auto-diagnostics
./run.sh

# With custom port
./run.sh --port /dev/tty.usbserial-110 --speed 0.5
```

### Method 2: Direct Python

```bash
# Basic run
python3 main.py

# With parameters
python3 main.py --port /dev/tty.usbserial-110 --baudrate 115200 --speed 1.0

# Debug mode
python3 main.py --debug

# Auto calibration
python3 main.py --calibrate
```

## 🐛 Troubleshooting

### Issue: "No module named 'core'"

**Solution:** Make sure you're in the correct directory and Python path is set:
```bash
cd RoArmM3v1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Serial port not found"

**Solution:** 
1. Check cable connection
2. Install USB drivers if needed
3. Check permissions:
```bash
# On macOS/Linux
sudo chmod 666 /dev/tty.usbserial*
```

### Issue: "Permission denied on serial port"

**Solution for macOS:**
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Logout and login again
```

### Issue: Missing dependencies

**Solution:**
```bash
pip install pyserial pyyaml numpy colorama
```

### Issue: Import errors

**Solution:** Run diagnostics to identify missing files:
```bash
python3 utils/debug_tool.py
```

## 📁 Project Structure

```
RoArmM3v1/
├── main.py                 # Main program
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
├── run.sh                # Startup script
│
├── core/                 # Core modules
│   ├── __init__.py
│   ├── controller.py     # Main controller
│   ├── serial_comm.py    # Serial communication
│   └── constants.py      # Hardware constants
│
├── motion/               # Motion control
│   ├── __init__.py
│   └── trajectory.py     # Trajectory generation
│
├── patterns/             # Scan patterns
│   ├── __init__.py
│   └── scan_patterns.py
│
├── teaching/             # Teaching mode
│   ├── __init__.py
│   └── recorder.py
│
├── calibration/          # Calibration
│   ├── __init__.py
│   └── calibration_suite.py
│
├── safety/               # Safety system
│   ├── __init__.py
│   └── safety_system.py
│
├── utils/                # Utilities
│   ├── __init__.py
│   ├── logger.py        # Logging
│   ├── terminal.py      # Terminal control
│   ├── safety.py        # Safety monitor
│   └── debug_tool.py    # Diagnostics
│
├── sequences/            # Saved sequences
└── logs/                # Log files
```

## ✅ Verification Checklist

After setup, verify:

- [ ] Python 3.7+ installed
- [ ] All dependencies installed
- [ ] Serial port configured correctly
- [ ] Robot connected and powered
- [ ] Diagnostics pass (`python3 utils/debug_tool.py`)
- [ ] Can connect to robot (`python3 main.py`)
- [ ] Emergency stop works (Ctrl+C)

## 🆘 Getting Help

1. **Run full diagnostics:**
   ```bash
   python3 utils/debug_tool.py
   ```

2. **Check logs:**
   ```bash
   tail -f logs/roarm_system.log
   ```

3. **Enable debug mode:**
   ```bash
   python3 main.py --debug
   ```

4. **Test serial connection:**
   ```python
   from core.serial_comm import SerialManager
   serial = SerialManager(port="/dev/tty.usbserial-110")
   serial.connect()
   ```

## 🎯 Quick Start Commands

```bash
# 1. Setup
git clone <repo>
cd RoArmM3v1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Diagnose
python3 utils/debug_tool.py

# 3. Configure
nano config.yaml  # Set your port

# 4. Run
./run.sh
# or
python3 main.py

# 5. Calibrate (first time)
python3 main.py --calibrate
```

## 📝 Notes

- The system creates `logs/` and `sequences/` directories automatically
- Calibration data is saved in `calibration/`
- Use Ctrl+C for emergency stop at any time
- The LED on the robot indicates system status

## 🔄 Updates

To update the system:
```bash
git pull
pip install -r requirements.txt --upgrade
python3 utils/debug_tool.py  # Verify everything still works
```

---

**Ready to go!** 🎉 Run `./run.sh` or `python3 main.py` to start the RoArm Control System.
