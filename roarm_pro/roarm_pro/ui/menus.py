"""
Menu system for RoArm application
Handles all menu display and navigation
"""

import math
from typing import Dict, Callable

class MenuSystem:
    """Hierarchical menu system"""
    
    def __init__(self, app):
        """
        Initialize menu system
        Args:
            app: RoArmApp instance
        """
        self.app = app
        
        # Menu handlers
        self.handlers = {
            # Main menu
            '1': self._calibration_menu,
            '2': self._move_home,
            '3': self._show_position,
            '4': self._manual_control,
            '5': self._teaching_mode,
            '6': self._scanner_menu,
            '7': self._speed_menu,
            '8': self._settings_menu,
            '9': self._system_menu,
            
            # Quick actions
            '91': self._reconnect,
            '92': self._clear_emergency,
            '93': self._test_connection,
        }
    
    def show_main_menu(self) -> str:
        """Show main menu and get choice"""
        print("\n" + "="*60)
        print("HAUPTMENÜ")
        print("="*60)
        
        # Status line
        self._show_status()
        
        print("\n=== KALIBRIERUNG ===")
        print("  1. 🎯 Kalibrierung")
        
        print("\n=== GRUNDFUNKTIONEN ===")
        print("  2. 🏠 Home-Position")
        print("  3. 📍 Position anzeigen")
        print("  4. 🎮 Manuelle Kontrolle")
        print("  5. 🎓 Teaching Mode")
        
        print("\n=== SCANNER ===")
        print("  6. 📷 Scanner-Funktionen")
        
        print("\n=== EINSTELLUNGEN ===")
        print("  7. ⚡ Geschwindigkeit")
        print("  8. 🔧 Einstellungen")
        print("  9. 💻 System")
        
        print("\n  0. ❌ Beenden")
        
        return input("\n➤ Wahl: ").strip()
    
    def handle_choice(self, choice: str):
        """Handle menu choice"""
        if choice in self.handlers:
            self.handlers[choice]()
        else:
            print("❌ Ungültige Auswahl")
    
    def _show_status(self):
        """Show connection and robot status"""
        ctrl = self.app.controller
        
        # Connection status
        if ctrl.connected:
            print(f"🟢 Verbunden | Port: {ctrl.serial.port}")
        else:
            print("🔴 Nicht verbunden")
        
        # Speed and mode
        print(f"⚡ Speed: {ctrl.speed_factor:.1f}x", end="")
        
        if ctrl.state.scanner_mounted:
            print(" | 📷 Scanner-Modus")
        else:
            print(" | 🤖 Normal-Modus")
        
        # Emergency stop
        if ctrl.state.emergency_stop:
            print("🚨 EMERGENCY STOP AKTIV")
    
    # ==================== Menu Handlers ====================
    
    def _calibration_menu(self):
        """Calibration menu"""
        print("\n🎯 KALIBRIERUNG")
        print("="*30)
        
        print("1. Vollständige Kalibrierung")
        print("2. Schnell-Kalibrierung")
        print("3. Positions-Test")
        print("0. Zurück")
        
        choice = input("\n➤ Wahl: ").strip()
        
        if choice == '1':
            if self.app.controller.calibrate(full=True):
                self.app.settings.calibrated = True
                self.app.settings.calibration_version = "2.0_systematic"
        elif choice == '2':
            self.app.controller.calibrate(full=False)
        elif choice == '3':
            self._test_positions()
    
    def _move_home(self):
        """Move to home position"""
        if not self.app.controller.connected:
            print("❌ Nicht verbunden")
            return
        
        print("🏠 Fahre zur Home-Position...")
        if self.app.controller.move_to_home():
            print("✅ Home-Position erreicht")
        else:
            print("❌ Bewegung fehlgeschlagen")
    
    def _show_position(self):
        """Show current position"""
        if not self.app.controller.connected:
            print("❌ Nicht verbunden")
            return
        
        pos = self.app.controller.get_current_position()
        
        print("\n📍 AKTUELLE POSITION:")
        print("-"*30)
        
        for joint, angle in pos.items():
            deg = math.degrees(angle)
            print(f"  {joint:8} : {angle:7.3f} rad ({deg:7.1f}°)")
    
    def _manual_control(self):
        """Start manual control"""
        if not self.app.controller.connected:
            print("❌ Nicht verbunden")
            return
        
        self.app.manual_control.start()
    
    def _teaching_mode(self):
        """Start teaching mode"""
        if not self.app.controller.connected:
            print("❌ Nicht verbunden")
            return
        
        self.app.teaching_mode.start()
    
    def _scanner_menu(self):
        """Scanner functions menu"""
        print("\n📷 SCANNER-FUNKTIONEN")
        print("="*30)
        
        print("1. 🔧 Scanner montieren")
        print("2. 🌊 Kontinuierlicher Scan")
        print("3. 📐 Raster-Scan")
        print("4. 🌀 Spiral-Scan")
        print("5. 🔍 Detail-Scan")
        print("6. ⚙️  Scan-Parameter")
        print("0. Zurück")
        
        choice = input("\n➤ Wahl: ").strip()
        
        if choice == '1':
            self.app.scanner_control.mount_scanner()
        elif choice == '2':
            duration = float(input("Dauer (Sekunden): ") or "60")
            self.app.scanner_control.continuous_rotation_scan(duration)
        elif choice == '3':
            rows = int(input("Zeilen (default 10): ") or "10")
            cols = int(input("Spalten (default 10): ") or "10")
            self.app.scanner_control.raster_scan(rows, cols)
        elif choice == '4':
            turns = int(input("Umdrehungen (default 5): ") or "5")
            self.app.scanner_control.spiral_scan(turns)
        elif choice == '5':
            self.app.scanner_control.detail_scan()
        elif choice == '6':
            self._scan_parameters()
    
    def _speed_menu(self):
        """Speed adjustment menu"""
        ctrl = self.app.controller
        
        print(f"\n⚡ GESCHWINDIGKEIT (aktuell: {ctrl.speed_factor:.1f}x)")
        print("="*30)
        
        print("1. 🐌 Langsam (0.5x)")
        print("2. 🚶 Normal (1.0x)")
        print("3. 🏃 Schnell (1.5x)")
        print("4. 🚀 Sehr schnell (2.0x)")
        print("5. ✏️  Benutzerdefiniert")
        print("0. Zurück")
        
        choice = input("\n➤ Wahl: ").strip()
        
        speeds = {
            '1': 0.5,
            '2': 1.0,
            '3': 1.5,
            '4': 2.0
        }
        
        if choice in speeds:
            ctrl.speed_factor = speeds[choice]
            self.app.settings.speed_factor = speeds[choice]
            print(f"✅ Geschwindigkeit: {ctrl.speed_factor:.1f}x")
        elif choice == '5':
            try:
                speed = float(input("Faktor (0.1-3.0): "))
                if 0.1 <= speed <= 3.0:
                    ctrl.speed_factor = speed
                    self.app.settings.speed_factor = speed
                    print(f"✅ Geschwindigkeit: {speed:.1f}x")
                else:
                    print("❌ Ungültiger Bereich")
            except ValueError:
                print("❌ Ungültige Eingabe")
    
    def _settings_menu(self):
        """Settings menu"""
        print("\n🔧 EINSTELLUNGEN")
        print("="*30)
        
        print("1. 📷 Scanner-Modus ein/aus")
        print("2. 🔄 Auto-Connect ein/aus")
        print("3. 💾 Einstellungen speichern")
        print("4. 📂 Einstellungen laden")
        print("5. 🔄 Auf Standard zurücksetzen")
        print("0. Zurück")
        
        choice = input("\n➤ Wahl: ").strip()
        
        if choice == '1':
            self.app.settings.scanner_mounted = not self.app.settings.scanner_mounted
            self.app.controller.set_scanner_mounted(self.app.settings.scanner_mounted)
            state = "EIN" if self.app.settings.scanner_mounted else "AUS"
            print(f"✅ Scanner-Modus: {state}")
        elif choice == '2':
            self.app.settings.auto_connect = not self.app.settings.auto_connect
            state = "EIN" if self.app.settings.auto_connect else "AUS"
            print(f"✅ Auto-Connect: {state}")
        elif choice == '3':
            self.app.settings.save()
            print("✅ Einstellungen gespeichert")
        elif choice == '4':
            self.app.settings = Settings.load()
            print("✅ Einstellungen geladen")
        elif choice == '5':
            self.app.settings.reset_to_defaults()
            print("✅ Standard-Einstellungen wiederhergestellt")
    
    def _system_menu(self):
        """System menu"""
        print("\n💻 SYSTEM")
        print("="*30)
        
        print("91. 🔌 Neu verbinden")
        print("92. 🚨 Emergency Stop aufheben")
        print("93. 🔍 Verbindung testen")
        print("94. 📊 Debug-Info")
        print("0. Zurück")
        
        choice = input("\n➤ Wahl: ").strip()
        
        if choice in ['91', '92', '93', '94']:
            self.handlers[choice]()
    
    # ==================== Quick Actions ====================
    
    def _reconnect(self):
        """Reconnect to robot"""
        self.app.controller.disconnect()
        if self.app._auto_connect():
            print("✅ Neu verbunden")
        else:
            print("❌ Verbindung fehlgeschlagen")
    
    def _clear_emergency(self):
        """Clear emergency stop"""
        self.app.controller.clear_emergency_stop()
        print("✅ Emergency Stop aufgehoben")
    
    def _test_connection(self):
        """Test connection"""
        if not self.app.controller.connected:
            print("❌ Nicht verbunden")
            return
        
        print("🔍 Teste Verbindung...")
        
        # Query status
        if self.app.controller.update_position():
            print("✅ Verbindung OK")
        else:
            print("❌ Keine Antwort")
        
        # LED test
        print("💡 LED Test...")
        self.app.controller.led_on(100)
        import time
        time.sleep(0.5)
        self.app.controller.led_off()
        print("✅ LED Test abgeschlossen")
    
    # ==================== Helper Methods ====================
    
    def _test_positions(self):
        """Test calibration positions"""
        positions = ['home', 'rest', 'scanner_mount', 'scan_start']
        
        for pos in positions:
            print(f"\n📍 Teste Position: {pos}")
            if input("Fortfahren? (y/n): ").lower() != 'y':
                break
            
            if not self.app.controller.move_to_position(pos):
                print(f"❌ Fehler bei {pos}")
                break
            
            import time
            time.sleep(1.0)
        
        print("\n✅ Positionstest abgeschlossen")
    
    def _scan_parameters(self):
        """Edit scan parameters"""
        params = self.app.scanner_control.get_scan_parameters()
        
        print("\n⚙️ SCAN-PARAMETER")
        print("-"*30)
        
        for key, value in params.items():
            print(f"{key:15} : {value}")
        
        print("\nParameter ändern (z.B. 'width 0.25'):")
        print("Enter zum Beenden")
        
        while True:
            cmd = input("> ").strip()
            if not cmd:
                break
            
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    key = parts[0]
                    value = float(parts[1])
                    self.app.scanner_control.set_scan_parameters(**{key: value})
                    print(f"✅ {key} = {value}")
                except:
                    print("❌ Ungültige Eingabe")
