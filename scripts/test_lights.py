#!/usr/bin/env python3
"""Test script to verify basic light control works."""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hue_visualizer.core.config import settings
from hue_visualizer.bridge.connection import HueBridge
from hue_visualizer.core.exceptions import BridgeConnectionError, ConfigurationError


def main():
    """Run basic light control test."""
    print("=" * 60)
    print("Hue Visualizer - Light Control Test")
    print("=" * 60)
    print()

    # Load configuration
    try:
        bridge_ip = settings.bridge_ip
        username = settings.hue_username
        print(f"Bridge IP: {bridge_ip}")
        print(f"Username: {username[:8]}...")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure you have a .env file with:")
        print("  BRIDGE_IP=...")
        print("  HUE_USERNAME=...")
        print("\nRun scripts/setup_bridge.py first!")
        sys.exit(1)

    print()

    # Connect to bridge
    print("[1/4] Connecting to bridge...")
    try:
        bridge = HueBridge(bridge_ip, username)
        bridge.test_connection()
        print("✓ Connected successfully!")
    except BridgeConnectionError as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)

    print()

    # Get lights
    print("[2/4] Discovering lights...")
    try:
        lights = bridge.get_lights()
        light_ids = list(lights.keys())
        print(f"✓ Found {len(light_ids)} light(s):")
        for light_id in light_ids[:5]:  # Show first 5
            light = lights[light_id]
            name = light.get("name", "Unknown")
            is_on = light.get("state", {}).get("on", False)
            status = "ON" if is_on else "OFF"
            print(f"  • {name} (ID: {light_id}) - {status}")
        if len(light_ids) > 5:
            print(f"  ... and {len(light_ids) - 5} more")
    except BridgeConnectionError as e:
        print(f"✗ Failed to get lights: {e}")
        sys.exit(1)

    if not light_ids:
        print("\n✗ No lights found! Make sure lights are paired with the bridge.")
        sys.exit(1)

    print()

    # Test light control
    print(f"[3/4] Testing light control on all {len(light_ids)} lights...")
    print("      (All lights will blink 3 times simultaneously)")

    try:
        # Store original states for all lights
        original_states = {}
        for light_id in light_ids:
            original_states[light_id] = bridge.get_light(light_id)["state"].get("on", False)

        # Blink all lights 3 times simultaneously
        for i in range(3):
            print(f"      Blink {i+1}/3...")

            # Turn all lights on (bright white)
            for light_id in light_ids:
                bridge.set_light_state(light_id, {"on": True, "bri": 254, "sat": 0})
            time.sleep(0.3)

            # Turn all lights off
            for light_id in light_ids:
                bridge.set_light_state(light_id, {"on": False})
            time.sleep(0.3)

        # Restore original states for all lights
        print("      Restoring original states...")
        for light_id in light_ids:
            bridge.set_light_state(light_id, {"on": original_states[light_id]})

        print(f"✓ Light control test successful! ({len(light_ids)} lights tested)")

    except BridgeConnectionError as e:
        print(f"✗ Light control failed: {e}")
        sys.exit(1)

    print()
    print("[4/4] Testing entertainment areas...")
    try:
        areas = bridge.get_entertainment_areas()
        if areas:
            print(f"✓ Found {len(areas)} entertainment area(s):")
            for area in areas:
                print(
                    f"  • {area['name']} (ID: {area['id']}) - {len(area['lights'])} lights"
                )
        else:
            print("⚠️  No entertainment areas configured")
            print("    Create one in the Hue app for Entertainment API")
    except BridgeConnectionError as e:
        print(f"✗ Failed to get entertainment areas: {e}")

    print()
    print("=" * 60)
    print("Basic Test Complete!")
    print("=" * 60)
    print("\nNext step: Implement Entertainment API for real-time control")
    print()


if __name__ == "__main__":
    main()
