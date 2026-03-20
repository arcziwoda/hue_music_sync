#!/usr/bin/env python3
"""Interactive setup script for Hue Bridge pairing and configuration."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hue_visualizer.bridge.discovery import discover_bridge, create_user, verify_connection
from hue_visualizer.bridge.connection import HueBridge
from hue_visualizer.core.exceptions import BridgeDiscoveryError, BridgeConnectionError


def main():
    """Run interactive setup."""
    print("=" * 60)
    print("Hue Visualizer - Bridge Setup")
    print("=" * 60)
    print()

    # Step 1: Discover bridge
    print("[1/4] Discovering Hue Bridge on network...")
    try:
        bridge_ip = discover_bridge()
        print(f"✓ Found bridge at: {bridge_ip}")
    except BridgeDiscoveryError as e:
        print(f"✗ Discovery failed: {e}")
        print("\nPlease enter bridge IP manually:")
        bridge_ip = input("Bridge IP: ").strip()
        if not bridge_ip:
            print("Error: Bridge IP is required")
            sys.exit(1)

    print()

    # Step 2: Create user
    print("[2/4] Creating API user...")
    print("⚠️  Please press the LINK BUTTON on your Hue Bridge now!")
    print("    You have 30 seconds after pressing the button.")
    input("\nPress ENTER after you've pressed the bridge button...")

    try:
        username = create_user(bridge_ip)
        print(f"✓ User created successfully!")
        print(f"  Username: {username}")
    except BridgeConnectionError as e:
        print(f"✗ Failed to create user: {e}")
        sys.exit(1)

    print()

    # Step 3: Verify connection
    print("[3/4] Verifying connection...")
    try:
        verify_connection(bridge_ip, username)
        print("✓ Connection verified!")
    except BridgeConnectionError as e:
        print(f"✗ Verification failed: {e}")
        sys.exit(1)

    print()

    # Step 4: List entertainment areas
    print("[4/4] Checking entertainment areas...")
    try:
        bridge = HueBridge(bridge_ip, username)
        areas = bridge.get_entertainment_areas()

        if not areas:
            print("⚠️  No entertainment areas found!")
            print("\nYou need to create an Entertainment Area in the Hue app:")
            print("  1. Open Hue app → Settings → Entertainment Areas")
            print("  2. Create a new area and add your lights")
            print("  3. Note the area name")
            print("\nAfter creating an area, run this script again.")
        else:
            print(f"✓ Found {len(areas)} entertainment area(s):")
            for area in areas:
                print(f"\n  • {area['name']} (ID: {area['id']})")
                print(f"    Lights: {len(area['lights'])} light(s)")
                print(f"    Class: {area.get('class', 'N/A')}")

    except BridgeConnectionError as e:
        print(f"✗ Failed to query bridge: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nAdd these values to your .env file:")
    print()
    print(f"BRIDGE_IP={bridge_ip}")
    print(f"HUE_USERNAME={username}")
    if areas:
        print(f"ENTERTAINMENT_AREA_ID={areas[0]['id']}  # {areas[0]['name']}")
    else:
        print("ENTERTAINMENT_AREA_ID=1  # Update after creating area")
    print()


if __name__ == "__main__":
    main()
