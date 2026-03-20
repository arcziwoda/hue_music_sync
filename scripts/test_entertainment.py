#!/usr/bin/env python3
"""
Test script for Hue Entertainment API connection using hue-entertainment-pykit.
This script will:
1. Discover the Hue bridge
2. Get bridge configuration details
3. Get entertainment area information
4. Test basic Entertainment API streaming
"""

import os
import time
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import requests
from hue_entertainment_pykit import Discovery, create_bridge, Entertainment, Streaming

# Load environment variables
load_dotenv()

BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_USERNAME = os.getenv("HUE_USERNAME")
HUE_CLIENTKEY = os.getenv("HUE_CLIENTKEY")
ENTERTAINMENT_AREA_ID = os.getenv("ENTERTAINMENT_AREA_ID", "1")


def get_bridge_config():
    """Get bridge configuration from REST API."""
    print(f"🔍 Getting bridge configuration from {BRIDGE_IP}...")

    url = f"http://{BRIDGE_IP}/api/{HUE_USERNAME}/config"
    response = requests.get(url)
    response.raise_for_status()

    config = response.json()
    print(f"  ✓ Bridge ID: {config.get('bridgeid')}")
    print(f"  ✓ API Version: {config.get('apiversion')}")
    print(f"  ✓ Software Version: {config.get('swversion')}")
    print(f"  ✓ Name: {config.get('name')}")

    return config


def get_entertainment_areas():
    """Get entertainment areas/groups from REST API."""
    print(f"\n🎭 Getting entertainment areas...")

    # Try v2 API first (CLIP API)
    url = f"https://{BRIDGE_IP}/clip/v2/resource/entertainment_configuration"
    headers = {"hue-application-key": HUE_USERNAME}

    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Found {len(data.get('data', []))} entertainment configurations")
            return data.get('data', [])
    except Exception as e:
        print(f"  ! V2 API failed: {e}")

    # Fallback to v1 API
    url = f"http://{BRIDGE_IP}/api/{HUE_USERNAME}/groups"
    response = requests.get(url)
    response.raise_for_status()

    groups = response.json()
    entertainment_areas = {
        gid: group for gid, group in groups.items()
        if group.get('type') == 'Entertainment'
    }

    print(f"  ✓ Found {len(entertainment_areas)} entertainment areas")
    for gid, area in entertainment_areas.items():
        print(f"    - Area {gid}: {area.get('name')} ({len(area.get('lights', []))} lights)")

    return entertainment_areas


def test_discovery():
    """Test bridge discovery."""
    print("\n🔍 Testing bridge discovery...")

    try:
        discovery = Discovery()
        bridges = discovery.discover_bridges()
        print(f"  ✓ Discovered {len(bridges)} bridge(s)")

        for bridge_id, bridge in bridges.items():
            print(f"    - Bridge: {bridge.name} at {bridge.ip_address}")

        return bridges
    except Exception as e:
        print(f"  ✗ Discovery failed: {e}")
        return {}


def test_entertainment_streaming():
    """Test basic Entertainment API streaming."""
    print("\n🎨 Testing Entertainment API streaming...")

    try:
        # Get bridge info from REST API
        config = get_bridge_config()

        # We need to manually create the bridge instance with our credentials
        # The Discovery class tries to create a new user, but we already have one
        print("\n🔧 Creating bridge instance from config...")

        # Create bridge using existing credentials
        bridge = create_bridge(
            identification=config.get('bridgeid'),
            rid=config.get('bridgeid'),  # Using bridgeid as rid for now
            ip_address=BRIDGE_IP,
            swversion=int(config.get('swversion', 0)),
            username=HUE_USERNAME,
            hue_app_id="hue-visualizer-app",  # Our app ID
            clientkey=HUE_CLIENTKEY,
            name=config.get('name', 'Hue Bridge')
        )
        print(f"  ✓ Created bridge instance for {BRIDGE_IP}")

        # Initialize Entertainment service
        print("\n🎬 Initializing Entertainment service...")
        entertainment_service = Entertainment(bridge)

        # Get entertainment configurations
        print("  Getting entertainment configurations...")
        ent_configs = entertainment_service.get_entertainment_configs()

        if not ent_configs:
            print("  ✗ No entertainment configurations found!")
            print("  Make sure you've created an Entertainment Area in the Hue app")
            return False

        print(f"  ✓ Found {len(ent_configs)} entertainment configuration(s)")

        # Use first entertainment config
        ent_config = list(ent_configs.values())[0]
        print(f"  Using entertainment config: {ent_config}")

        # Start streaming
        print("\n🚀 Starting Entertainment stream...")
        streaming = Streaming(
            bridge,
            ent_config,
            entertainment_service.get_ent_conf_repo()
        )

        streaming.start_stream()
        print("  ✓ Stream started!")

        # Set color space
        streaming.set_color_space("xyb")

        # Test: Cycle through some colors
        print("\n🌈 Testing color changes (5 seconds)...")

        colors = [
            (0.675, 0.322, 254),  # Red
            (0.167, 0.04, 254),   # Blue
            (0.408, 0.517, 254),  # Green
            (0.0, 0.0, 254),      # White
        ]

        for i in range(4):
            x, y, bri = colors[i]
            print(f"  Color {i+1}: x={x}, y={y}, bri={bri}")

            # Set color for all lights (light_id 0 might set all)
            streaming.set_input((x, y, bri / 254, 0))
            time.sleep(1.25)

        # Stop streaming
        print("\n⏹️  Stopping stream...")
        streaming.stop_stream()
        print("  ✓ Stream stopped!")

        return True

    except Exception as e:
        print(f"  ✗ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Hue Entertainment API Test")
    print("=" * 60)

    # Verify environment variables
    if not BRIDGE_IP or not HUE_USERNAME or not HUE_CLIENTKEY:
        print("❌ Missing required environment variables!")
        print("   Please set BRIDGE_IP, HUE_USERNAME, and HUE_CLIENTKEY in .env file")
        print("   Run 'uv run python scripts/get_clientkey.py' to get the clientkey")
        return 1

    print(f"\n📋 Configuration:")
    print(f"   Bridge IP: {BRIDGE_IP}")
    print(f"   Username: {HUE_USERNAME[:10]}...")
    print(f"   ClientKey: {HUE_CLIENTKEY[:10]}...")
    print(f"   Entertainment Area ID: {ENTERTAINMENT_AREA_ID}")

    # Run tests
    try:
        # Step 1: Get bridge config
        get_bridge_config()

        # Step 2: Get entertainment areas
        get_entertainment_areas()

        # Step 3: Test discovery
        test_discovery()

        # Step 4: Test entertainment streaming
        success = test_entertainment_streaming()

        if success:
            print("\n" + "=" * 60)
            print("✅ All tests passed!")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("⚠️  Some tests failed - see above for details")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
