"""
Teaching mode for recording and replaying positions
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class TeachingPosition:
    """Single teaching position"""
    name: str
    positions: Dict[str, float]
    timestamp: str
    description: str = ""

class TeachingMode:
    """Teaching mode for position recording and playback"""
    
    def __init__(self, controller):
        """
        Initialize teaching mode
        Args:
            controller: RoArmController instance
        """
        self.controller = controller
        self.positions: List[TeachingPosition] = []
        self.recording = False
        
    def start(self):
        """Start teaching mode interface"""
        print("\n" + "="*60)
        print("🎓 TEACHING MODE")
        print("="*60)
        
        while True:
            self._show_menu()
            choice = input("\n➤ Choice: ").strip()
            
            if choice == '0':
                break
            else:
                self._handle_choice(choice)
    
    def _show_menu(self):
        """Show teaching menu"""
        print(f"\nRecorded positions: {len(self.positions)}")
        print("\n1. 💾 Save current position")
        print("2. ▶️  Play sequence")
        print("3. 📋 List positions")
        print("4. 🗑️  Clear position")
        print("5. 💿 Save to file")
        print("6. 📂 Load from file")
        print("7. ✏️  Edit position")
        print("8. 🔄 Reverse sequence")
        print("0. ↩️  Exit")
    
    def _handle_choice(self, choice: str):
        """Handle menu choice"""
        if choice == '1':
            self.save_position()
        elif choice == '2':
            self.play_sequence()
        elif choice == '3':
            self.list_positions()
        elif choice == '4':
            self.clear_position()
        elif choice == '5':
            self.save_to_file()
        elif choice == '6':
            self.load_from_file()
        elif choice == '7':
            self.edit_position()
        elif choice == '8':
            self.reverse_sequence()
    
    def save_position(self):
        """Save current position"""
        # Get current position
        positions = self.controller.get_current_position()
        
        # Get name and description
        name = input("Position name: ").strip() or f"Position {len(self.positions) + 1}"
        description = input("Description (optional): ").strip()
        
        # Create position
        pos = TeachingPosition(
            name=name,
            positions=positions,
            timestamp=datetime.now().isoformat(),
            description=description
        )
        
        self.positions.append(pos)
        print(f"✅ Position '{name}' saved")
    
    def play_sequence(self):
        """Play recorded sequence"""
        if not self.positions:
            print("❌ No positions recorded")
            return
        
        print(f"\n▶️  Playing {len(self.positions)} positions...")
        
        # Options
        duration = float(input("Duration per move (seconds, default 2.0): ") or "2.0")
        pause = float(input("Pause between moves (seconds, default 1.0): ") or "1.0")
        loops = int(input("Number of loops (default 1): ") or "1")
        
        for loop in range(loops):
            if loops > 1:
                print(f"\n🔄 Loop {loop + 1}/{loops}")
                
            for i, pos in enumerate(self.positions):
                if self.controller.abort_flag:
                    print("\n❌ Playback aborted")
                    return
                
                print(f"📍 Moving to: {pos.name}")
                
                # Move to position
                success = self.controller.move_joints(
                    duration=duration,
                    **pos.positions
                )
                
                if not success:
                    print(f"❌ Failed to reach {pos.name}")
                    return
                
                # Pause between moves
                if i < len(self.positions) - 1 or loop < loops - 1:
                    time.sleep(pause)
        
        print("✅ Sequence complete")
    
    def list_positions(self):
        """List all positions"""
        if not self.positions:
            print("📋 No positions recorded")
            return
        
        print(f"\n📋 {len(self.positions)} positions:")
        print("-" * 60)
        
        for i, pos in enumerate(self.positions):
            # Format joint positions
            joint_str = ", ".join([
                f"{j}={math.degrees(v):.1f}°" 
                for j, v in pos.positions.items()
                if j in ['base', 'shoulder', 'elbow']
            ])
            
            print(f"{i+1:2d}. {pos.name:20s} | {joint_str}")
            if pos.description:
                print(f"    {pos.description}")
    
    def clear_position(self):
        """Clear a position"""
        if not self.positions:
            print("❌ No positions to clear")
            return
        
        self.list_positions()
        
        try:
            idx = int(input("\nPosition number to clear (0 to cancel): ")) - 1
            
            if 0 <= idx < len(self.positions):
                removed = self.positions.pop(idx)
                print(f"✅ Removed '{removed.name}'")
            elif idx != -1:
                print("❌ Invalid position number")
                
        except ValueError:
            print("❌ Invalid input")
    
    def save_to_file(self):
        """Save positions to file"""
        if not self.positions:
            print("❌ No positions to save")
            return
        
        filename = input("Filename (without .json): ").strip()
        if not filename:
            filename = f"teaching_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = f"{filename}.json"
        
        try:
            data = {
                'positions': [asdict(pos) for pos in self.positions],
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'count': len(self.positions)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Saved {len(self.positions)} positions to {filepath}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def load_from_file(self):
        """Load positions from file"""
        # List available files
        files = [f for f in os.listdir('.') if f.endswith('.json') and 'teaching' in f]
        
        if not files:
            print("❌ No teaching files found")
            return
        
        print("\nAvailable files:")
        for i, f in enumerate(files):
            print(f"  {i+1}. {f}")
        
        try:
            idx = int(input("Select file (0 to cancel): ")) - 1
            
            if 0 <= idx < len(files):
                filepath = files[idx]
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Clear existing positions
                if self.positions:
                    if input("Clear existing positions? (y/n): ").lower() == 'y':
                        self.positions.clear()
                
                # Load positions
                for pos_data in data['positions']:
                    pos = TeachingPosition(**pos_data)
                    self.positions.append(pos)
                
                print(f"✅ Loaded {len(data['positions'])} positions from {filepath}")
                
        except Exception as e:
            print(f"❌ Load failed: {e}")
    
    def edit_position(self):
        """Edit a position"""
        if not self.positions:
            print("❌ No positions to edit")
            return
        
        self.list_positions()
        
        try:
            idx = int(input("\nPosition number to edit (0 to cancel): ")) - 1
            
            if 0 <= idx < len(self.positions):
                pos = self.positions[idx]
                
                print(f"\nEditing: {pos.name}")
                
                # Edit name
                new_name = input(f"Name ({pos.name}): ").strip()
                if new_name:
                    pos.name = new_name
                
                # Edit description
                new_desc = input(f"Description ({pos.description}): ").strip()
                if new_desc:
                    pos.description = new_desc
                
                # Update position?
                if input("Update to current position? (y/n): ").lower() == 'y':
                    pos.positions = self.controller.get_current_position()
                    pos.timestamp = datetime.now().isoformat()
                
                print("✅ Position updated")
                
        except ValueError:
            print("❌ Invalid input")
    
    def reverse_sequence(self):
        """Reverse the sequence order"""
        if len(self.positions) < 2:
            print("❌ Need at least 2 positions to reverse")
            return
        
        self.positions.reverse()
        print("✅ Sequence reversed")
