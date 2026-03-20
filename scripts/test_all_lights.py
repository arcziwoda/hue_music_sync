#!/usr/bin/env python3
"""
Test setting all lights in entertainment area.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.hue_visualizer.bridge import EntertainmentController

load_dotenv()

BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_USERNAME = os.getenv("HUE_USERNAME")
HUE_CLIENTKEY = os.getenv("HUE_CLIENTKEY")


def test_light_control():
    """Test different ways to control lights."""
    print("=" * 60)
    print("Testing All Lights Control")
    print("=" * 60)

    with EntertainmentController(BRIDGE_IP, HUE_USERNAME, HUE_CLIENTKEY) as controller:

        # Test 1: light_id = 0 (current behavior)
        print("\n1️⃣  Test: light_id=0 (Red)")
        controller.set_color_rgb(255, 0, 0, brightness=1.0, light_id=0)
        time.sleep(2)

        # Test 2: Try each light individually
        print("\n2️⃣  Test: Individual lights (cycling through 1-6)")
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        for light_id in range(1, 7):
            r, g, b = colors[light_id - 1]
            print(f"   Setting light {light_id} to {colors[light_id - 1]}")
            controller.set_color_rgb(r, g, b, brightness=1.0, light_id=light_id)
            time.sleep(0.5)

        time.sleep(2)

        # Test 3: Set all lights to same color by looping
        print("\n3️⃣  Test: Loop through all lights - Purple")
        for light_id in range(1, 7):
            controller.set_color_rgb(128, 0, 128, brightness=1.0, light_id=light_id)
        time.sleep(2)

        # Test 4: Back to light_id=0
        print("\n4️⃣  Test: Back to light_id=0 - White")
        controller.set_color_rgb(255, 255, 255, brightness=0.5, light_id=0)
        time.sleep(2)

        print("\n✅ Test complete!")


if __name__ == "__main__":
    test_light_control()
