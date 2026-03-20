#!/usr/bin/env python3
"""
Demo script showing how to use EntertainmentController and LightEffects.

This demonstrates the simple, Pythonic API for controlling Hue lights.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.hue_visualizer.bridge import EntertainmentController, LightEffects

# Load configuration
load_dotenv()

BRIDGE_IP = os.getenv("BRIDGE_IP")
HUE_USERNAME = os.getenv("HUE_USERNAME")
HUE_CLIENTKEY = os.getenv("HUE_CLIENTKEY")


def main():
    """Run effects demo."""
    print("=" * 60)
    print("Hue Entertainment API - Effects Demo")
    print("=" * 60)

    if not all([BRIDGE_IP, HUE_USERNAME, HUE_CLIENTKEY]):
        print("\n❌ Missing configuration!")
        print("Please ensure BRIDGE_IP, HUE_USERNAME, and HUE_CLIENTKEY")
        print("are set in your .env file.")
        return 1

    print(f"\n📋 Bridge: {BRIDGE_IP}")
    print("\n🎬 Starting demo...\n")

    # Use context manager for automatic connection/disconnection
    with EntertainmentController(BRIDGE_IP, HUE_USERNAME, HUE_CLIENTKEY) as controller:
        effects = LightEffects(controller)

        # Demo 1: Solid colors
        print("1️⃣  Solid Colors (2 seconds each)")
        print("   → Red")
        controller.set_color_rgb(255, 0, 0, brightness=1.0)
        import time

        time.sleep(2)

        print("   → Green")
        controller.set_color_rgb(0, 255, 0, brightness=1.0)
        time.sleep(2)

        print("   → Blue")
        controller.set_color_rgb(0, 0, 255, brightness=1.0)
        time.sleep(2)

        # Demo 2: HSV colors
        print("\n2️⃣  HSV Colors - Color wheel")
        print("   → Cycling through hues...")
        for hue in range(0, 360, 30):
            controller.set_color_hsv(hue, 1.0, 1.0)
            time.sleep(0.3)

        # Demo 3: Pulse effect
        print("\n3️⃣  Pulse Effect")
        print("   → Purple pulse (3 cycles)")
        effects.pulse(128, 0, 128, duration=1.0, cycles=3)

        # Demo 4: Color cycle
        print("\n4️⃣  Rainbow Cycle")
        print("   → Smooth rainbow (5 seconds)")
        effects.color_cycle(duration=5.0)

        # Demo 5: Breathe
        print("\n5️⃣  Breathing Effect")
        print("   → Cyan breathe (2 cycles)")
        effects.breathe(0, 255, 255, duration=2.5, cycles=2)

        # Demo 6: Strobe
        print("\n6️⃣  Strobe Effect")
        print("   → White strobe (2 seconds)")
        effects.strobe(255, 255, 255, duration=2.0, frequency=8.0)

        # Demo 7: Fade
        print("\n7️⃣  Fade Effect")
        print("   → Fade up to orange")
        effects.fade_to_color(
            255, 128, 0, duration=2.0, from_brightness=0.0, to_brightness=1.0
        )

        print("\n   → Fade down")
        effects.fade_to_color(
            255, 128, 0, duration=2.0, from_brightness=1.0, to_brightness=0.0
        )

        # Demo 8: Rainbow wave
        print("\n8️⃣  Rainbow Wave")
        print("   → Fast rainbow wave (5 seconds)")
        effects.rainbow_wave(duration=5.0, speed=2.0)

        # Finish with white
        print("\n✨ Demo complete - setting to warm white")
        controller.set_color_rgb(255, 200, 150, brightness=0.5)

    print("\n" + "=" * 60)
    print("✅ Demo finished!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    main()
