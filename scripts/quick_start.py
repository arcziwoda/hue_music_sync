#!/usr/bin/env python3
"""
Quick start example - simplest possible usage of EntertainmentController.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.hue_visualizer.bridge import EntertainmentController

# Load configuration
load_dotenv()

# Simple example: Set lights to red for 3 seconds
with EntertainmentController(
    bridge_ip=os.getenv("BRIDGE_IP"),
    username=os.getenv("HUE_USERNAME"),
    clientkey=os.getenv("HUE_CLIENTKEY")
) as controller:
    print("Setting lights to RED...")
    controller.set_color_rgb(255, 0, 0)

    import time
    time.sleep(3)

    print("Done!")
