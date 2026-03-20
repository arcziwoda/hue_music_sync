#!/usr/bin/env python3
"""
Script to obtain a clientkey for Hue Entertainment API.

The clientkey is required for DTLS authentication when using Entertainment mode.
This script will:
1. Create a new application user with Entertainment API access
2. Display the clientkey that should be added to .env

IMPORTANT: You must press the link button on your Hue Bridge before running this script.
"""

import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BRIDGE_IP = os.getenv("BRIDGE_IP")


def create_entertainment_user():
    """Create a new user with Entertainment API access."""
    print("=" * 60)
    print("Hue Entertainment API - Get ClientKey")
    print("=" * 60)

    if not BRIDGE_IP:
        print("❌ BRIDGE_IP not found in .env file!")
        return 1

    print(f"\n📋 Bridge IP: {BRIDGE_IP}")
    print("\n⚠️  IMPORTANT: Press the link button on your Hue Bridge NOW!")
    print("   The script will try to connect in 3 seconds...")

    # If running interactively, wait for user input, otherwise auto-continue
    try:
        import select
        # Give user 3 seconds to press button
        print("\n⏳ Waiting 3 seconds...")
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1

    # Create user with clientkey
    url = f"http://{BRIDGE_IP}/api"
    payload = {
        "devicetype": "hue_visualizer#python",
        "generateclientkey": True
    }

    print(f"\n🔄 Creating user with Entertainment API access...")
    print(f"   Sending request to {url}")

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        result = response.json()
        print(f"\n📥 Response: {result}")

        if isinstance(result, list) and len(result) > 0:
            if "success" in result[0]:
                success_data = result[0]["success"]
                username = success_data.get("username")
                clientkey = success_data.get("clientkey")

                print("\n" + "=" * 60)
                print("✅ SUCCESS!")
                print("=" * 60)
                print(f"\nUsername:  {username}")
                print(f"ClientKey: {clientkey}")
                print("\n📝 Add/Update these in your .env file:")
                print("=" * 60)
                print(f"HUE_USERNAME={username}")
                print(f"HUE_CLIENTKEY={clientkey}")
                print("=" * 60)
                print("\n💡 Save these values - you cannot retrieve the clientkey later!")

                return 0

            elif "error" in result[0]:
                error_data = result[0]["error"]
                error_type = error_data.get("type")
                description = error_data.get("description")

                print(f"\n❌ Error {error_type}: {description}")

                if "link button not pressed" in description:
                    print("\n💡 The link button was not pressed. Please try again:")
                    print("   1. Press the large button on top of your Hue Bridge")
                    print("   2. Run this script again within 30 seconds")

                return 1

        print("\n❌ Unexpected response format")
        return 1

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out. Is the bridge IP correct?")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to bridge at {BRIDGE_IP}")
        print("   Please verify the IP address in your .env file")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(create_entertainment_user())