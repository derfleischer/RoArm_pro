roarm_professional/
├── main.py                 # Hauptprogramm mit CLI
├── config.yaml            # Konfiguration
├── requirements.txt       # Python Dependencies
├── README.md             # Diese Datei
│
├── core/                 # Kern-Funktionalität
│   ├── controller.py     # Haupt-Controller
│   ├── serial_comm.py    # Serial Communication
│   └── constants.py      # Hardware-Konstanten
│
├── motion/               # Bewegungssteuerung
│   └── trajectory.py     # Trajectory Generation
│
├── patterns/             # Bewegungsmuster
│   └── scan_patterns.py  # Scanner-Patterns
│
├── teaching/             # Teaching Mode
│   └── recorder.py       # Erweiterte Aufzeichnung
│
├── utils/                # Hilfsfunktionen
│   ├── logger.py        # Logging System
│   ├── safety.py        # Safety Monitor
│   └── terminal.py      # macOS Terminal Control
│
└── sequences/            # Gespeicherte Sequenzen
    └── *.json           # Teaching-Aufzeichnungen
