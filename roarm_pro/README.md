# RoArm Pro - Professional Robot Arm Controller

🤖 **Modular, professional control system for Waveshare RoArm M3**  
📷 **Optimized for Revopoint Mini2 3D Scanner**  
🍎 **macOS focused with cross-platform support**

## 🚀 Features

- **Modular Architecture**: Clean separation of hardware, motion, control, and UI
- **Thread-Safe**: Proper concurrent operation handling
- **Scanner Optimized**: Special modes and limits for 3D scanner operation
- **S-Curve Trajectories**: Smooth motion profiles for precise movements
- **Teaching Mode**: Record and replay position sequences
- **Auto-Calibration**: Systematic calibration routines
- **Emergency Stop**: Hardware and software safety features
- **Persistent Settings**: Save and restore configurations

## 📋 Requirements

- Python 3.8+
- macOS (optimized) / Linux / Windows
- Waveshare RoArm M3
- USB Serial adapter

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/roarm-pro.git
cd roarm-pro
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install package**:
```bash
python setup.py develop
```

## 🎯 Quick Start

1. **Basic usage**:
```bash
# Auto-detect port and start
python -m roarm_pro.main

# Or use the installed command
roarm
```

2. **Specify port**:
```bash
roarm --port /dev/cu.usbserial-110
```

3. **List available ports**:
```bash
roarm --list-ports
```

4. **Debug mode**:
```bash
roarm --debug
```

## 📁 Project Structure

```
roarm_pro/
├── config/          # Configuration and constants
│   ├── defaults.py  # Hardware limits, positions
│   └── settings.py  # Runtime settings management
├── hardware/        # Low-level hardware control
│   ├── serial_comm.py   # Serial communication
│   ├── port_utils.py    # Port detection
│   └── commands.py      # Command building
├── motion/          # Motion control
│   ├── trajectory.py    # Trajectory generation
│   ├── limits.py        # Joint validation
│   └── kinematics.py    # Kinematics calculations
├── control/         # High-level control
│   ├── controller.py    # Main controller
│   ├── manual.py        # Manual control
│   ├── teaching.py      # Teaching mode
│   └── scanner.py       # Scanner operations
└── ui/              # User interface
    ├── app.py          # Main application
    └── menus.py        # Menu system
```

## 🎮 Usage Guide

### First Run
1. Connect your RoArm M3 via USB
2. Run `roarm`
3. The system will auto-detect the port
4. Run calibration (Menu → 1)

### Basic Operations
- **Home Position**: Menu → 2
- **Manual Control**: Menu → 4 (keyboard control)
- **Teaching Mode**: Menu → 5 (record positions)
- **Scanner Functions**: Menu → 6

### Scanner Operations
1. Mount scanner: Menu → 6 → 1
2. Continuous scan: Menu → 6 → 2
3. Raster scan: Menu → 6 → 3
4. Spiral scan: Menu → 6 → 4

### Keyboard Controls (Manual Mode)
```
Q/A = Base left/right      | W/S = Shoulder up/down
E/D = Elbow up/down        | R/F = Wrist up/down
T/G = Roll left/right      | Y/H = Gripper open/close
+/- = Speed up/down        | SPACE = Emergency stop
P = Save position          | ESC = Exit
```

## ⚙️ Configuration

Settings are saved in `~/.roarm_pro/settings.yaml`:

```yaml
speed_factor: 1.0
scanner_mounted: true
auto_connect: true
calibrated: true
language: de
```

## 🔌 Hardware Setup

### Servo Limits (radians)
- Base: -3.14 to 3.14 (360°)
- Shoulder: -1.57 to 1.57 (180°)
- Elbow: 0.0 to 3.14 (180°)
- Wrist: -1.57 to 1.57 (180°)
- Roll: -3.14 to 3.14 (360°)
- Hand: 1.08 to 3.14 (62° to 180°)

### Scanner Mode
When scanner is mounted, restricted limits are applied for safety:
- Hand: 2.2 to 2.8 (safe grip range)
- Reduced shoulder/wrist ranges

## 🛡️ Safety Features

- **Emergency Stop**: Ctrl+C anywhere stops all motion
- **Soft Limits**: All movements validated before execution
- **Velocity Limits**: Maximum 2.0 rad/s
- **Scanner Protection**: Special limits when scanner mounted
- **Graceful Shutdown**: Safe position sequence on exit

## 🐛 Troubleshooting

### Connection Issues
1. Check USB cable connection
2. Verify robot power is ON
3. Try different USB port
4. Install USB serial drivers if needed

### macOS Specific
- Grant terminal permissions for keyboard control
- Use Terminal.app (not VS Code terminal) for best results

### Movement Issues
- Run calibration sequence
- Check emergency stop status (Menu → 92)
- Verify torque is enabled
- Reduce speed factor if movements are jerky

## 📝 Development

### Adding New Features
1. Create module in appropriate package
2. Follow existing patterns for consistency
3. Add menu entry in `ui/menus.py`
4. Update documentation

### Testing
```bash
# Run with debug logging
roarm --debug

# Test specific module
python -m pytest tests/test_trajectory.py
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Waveshare for RoArm M3 documentation
- Revopoint for Mini2 scanner
- Contributors and testers

---

For issues and contributions, please visit the GitHub repository.
