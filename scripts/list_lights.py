#!/usr/bin/env python3
"""
List all lights in entertainment area.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import requests

load_dotenv()

BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_USERNAME = os.getenv("HUE_USERNAME")

def list_entertainment_areas():
    """List all entertainment areas and their lights."""
    print("=" * 60)
    print("Entertainment Areas & Lights")
    print("=" * 60)

    # Get all groups
    url = f"http://{BRIDGE_IP}/api/{HUE_USERNAME}/groups"
    response = requests.get(url)
    groups = response.json()

    print(f"\nFound {len(groups)} groups:")

    for gid, group in groups.items():
        if group.get('type') == 'Entertainment':
            print(f"\n🎭 Entertainment Area #{gid}: {group.get('name')}")
            print(f"   Type: {group.get('class')}")
            print(f"   Lights ({len(group.get('lights', []))}):")

            for light_id in group.get('lights', []):
                # Get light info
                light_url = f"http://{BRIDGE_IP}/api/{HUE_USERNAME}/lights/{light_id}"
                light_resp = requests.get(light_url)
                light_data = light_resp.json()

                print(f"      - Light {light_id}: {light_data.get('name')} ({light_data.get('type')})")

            # Get entertainment config if available
            if 'stream' in group:
                stream = group['stream']
                print(f"\n   Stream config:")
                print(f"      Active: {stream.get('active')}")
                print(f"      Owner: {stream.get('owner', 'none')}")

if __name__ == "__main__":
    list_entertainment_areas()
