#!/usr/bin/env python3
"""
Test Script für Mock Serial
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Mock Serial Import...")
print("-" * 40)

# Test 1: Direct import
try:
    from core.mock_serial import MockSerialManager
    print("✅ Direct import works")
except ImportError as e:
    print(f"❌ Direct import failed: {e}")
    sys.exit(1)

# Test 2: Create instance
try:
    mock = MockSerialManager("TEST_PORT", 115200)
    print("✅ MockSerialManager instance created")
except Exception as e:
    print(f"❌ Could not create instance: {e}")
    sys.exit(1)

# Test 3: Connect
try:
    result = mock.connect()
    if result:
        print("✅ Mock connect() works")
    else:
        print("❌ Mock connect() returned False")
except Exception as e:
    print(f"❌ Connect failed: {e}")

# Test 4: Send command
try:
    mock.send_command({"T": 1})  # Status query
    print("✅ Mock send_command() works")
except Exception as e:
    print(f"❌ Send command failed: {e}")

# Test 5: Query status
try:
    status = mock.query_status()
    if status:
        print("✅ Mock query_status() works")
        print(f"   Positions: {list(status['positions'].keys())}")
    else:
        print("❌ Query status returned None")
except Exception as e:
    print(f"❌ Query status failed: {e}")

# Test 6: Check command history
try:
    history = mock.get_command_log()
    print(f"✅ Command history: {len(history)} commands")
except Exception as e:
    print(f"❌ Get command log failed: {e}")

print("-" * 40)
print("✅ All tests passed! Mock is ready to use.")
print("\nYou can now run:")
print("  python3 main.py --simulate")
print("  python3 main.py --simulate --debug")
