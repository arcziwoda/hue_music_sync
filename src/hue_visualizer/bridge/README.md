# Bridge Module - Hue Entertainment API Controller

High-level, Pythonic interface for controlling Philips Hue lights via the Entertainment API.

## Features

- **Simple API**: Easy-to-use methods for color control
- **Multiple color spaces**: RGB, HSV, and XY (CIE) support
- **Context manager**: Automatic connection/disconnection
- **Built-in effects**: Pulse, fade, strobe, rainbow, and more
- **Low latency**: Uses DTLS Entertainment API for real-time control

## Quick Start

```python
from hue_visualizer.bridge import EntertainmentController

# Connect and set color
with EntertainmentController(bridge_ip, username, clientkey) as controller:
    controller.set_color_rgb(255, 0, 0)  # Red
```

## EntertainmentController

### Initialization

```python
controller = EntertainmentController(
    bridge_ip="192.168.1.24",
    username="your-username",
    clientkey="your-clientkey",
    entertainment_area_id=None  # Optional, uses first area if None
)
```

### Connection Management

#### Context Manager (Recommended)
```python
with EntertainmentController(bridge_ip, username, clientkey) as controller:
    # Your code here
    controller.set_color_rgb(0, 255, 0)
# Automatically disconnects when exiting
```

#### Manual Connection
```python
controller = EntertainmentController(bridge_ip, username, clientkey)
controller.connect()
try:
    controller.set_color_rgb(0, 255, 0)
finally:
    controller.disconnect()
```

### Setting Colors

#### RGB Colors (0-255)
```python
controller.set_color_rgb(255, 0, 0, brightness=1.0)  # Red
controller.set_color_rgb(0, 255, 0, brightness=0.5)  # Green at 50%
controller.set_color_rgb(0, 0, 255, brightness=0.2)  # Dim blue
```

#### HSV Colors
```python
# Hue: 0-360 degrees, Saturation: 0-1, Value: 0-1
controller.set_color_hsv(0, 1.0, 1.0)    # Red
controller.set_color_hsv(120, 1.0, 1.0)  # Green
controller.set_color_hsv(240, 1.0, 1.0)  # Blue
controller.set_color_hsv(60, 0.5, 0.8)   # Muted yellow
```

#### XY Colors (CIE Color Space)
```python
# Direct control using Hue's native color space
controller.set_color_xy(0.675, 0.322, brightness=1.0)  # Red
controller.set_color_xy(0.167, 0.04, brightness=1.0)   # Blue
```

#### Using Color Presets
```python
from hue_visualizer.utils.color_conversion import ColorPresets

x, y = ColorPresets.RED
controller.set_color_xy(x, y, brightness=1.0)

# Available presets:
# RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA
# WHITE, WARM_WHITE, COOL_WHITE
# ORANGE, AMBER, PURPLE, VIOLET, PINK, HOT_PINK
```

### Brightness Control

```python
controller.set_brightness(1.0)   # Full brightness (white)
controller.set_brightness(0.5)   # 50% brightness
controller.turn_off()            # Turn off (brightness = 0)
```

### Per-Light Control

By default, methods affect all lights. Use `light_id` parameter for individual control:

```python
controller.set_color_rgb(255, 0, 0, light_id=0)  # All lights
controller.set_color_rgb(0, 255, 0, light_id=1)  # Light 1 only
controller.set_color_rgb(0, 0, 255, light_id=2)  # Light 2 only
```

## LightEffects

Pre-built effects for common lighting patterns.

```python
from hue_visualizer.bridge import LightEffects

with EntertainmentController(bridge_ip, username, clientkey) as controller:
    effects = LightEffects(controller)

    # Run effects...
```

### Available Effects

#### Pulse
Smooth brightness fade in/out:
```python
effects.pulse(
    r=255, g=0, b=128,      # Color (RGB)
    duration=1.0,            # Cycle duration (seconds)
    min_brightness=0.1,      # Minimum brightness
    max_brightness=1.0,      # Maximum brightness
    cycles=3,                # Number of pulses
    fps=30                   # Smoothness (frames per second)
)
```

#### Breathe
Natural, slower fade (like pulse but with easing):
```python
effects.breathe(
    r=0, g=255, b=255,      # Cyan
    duration=3.0,            # Breath duration
    cycles=2,                # Number of breaths
    min_brightness=0.0,
    max_brightness=1.0
)
```

#### Color Cycle
Smooth rotation through color wheel:
```python
effects.color_cycle(
    duration=10.0,           # Full cycle duration
    saturation=1.0,          # Color saturation
    brightness=1.0
)
```

#### Rainbow Wave
Fast-moving rainbow effect:
```python
effects.rainbow_wave(
    duration=5.0,
    speed=2.0,               # Speed multiplier
    saturation=1.0,
    brightness=1.0
)
```

#### Strobe
Flash effect:
```python
effects.strobe(
    r=255, g=255, b=255,    # White
    duration=2.0,            # Total duration
    frequency=10.0,          # Flashes per second
    brightness=1.0
)
```

#### Fade
Smooth brightness transition:
```python
effects.fade_to_color(
    r=255, g=128, b=0,      # Orange
    duration=2.0,
    from_brightness=0.0,     # Start: off
    to_brightness=1.0        # End: full
)
```

#### Solid Color
Simple color hold:
```python
effects.solid_color(
    r=255, g=0, b=0,
    brightness=1.0,
    duration=5.0             # Hold for 5 seconds
)
```

## Color Conversion Utilities

```python
from hue_visualizer.utils.color_conversion import (
    rgb_to_xy, hsv_to_xy, rgb_to_hsv, hsv_to_rgb, ColorPresets
)

# Convert RGB to XY
x, y = rgb_to_xy(255, 0, 0)  # Red

# Convert HSV to XY
x, y = hsv_to_xy(120, 1.0, 1.0)  # Green

# RGB <-> HSV
h, s, v = rgb_to_hsv(255, 128, 0)  # RGB to HSV
r, g, b = hsv_to_rgb(30, 1.0, 1.0)  # HSV to RGB

# Color presets
x, y = ColorPresets.WARM_WHITE
x, y = ColorPresets.get_by_name("purple")
```

## Advanced: LightState

For direct XY control with full control:

```python
from hue_visualizer.bridge import LightState

state = LightState(
    x=0.675,
    y=0.322,
    brightness=0.8,
    light_id=1
)

controller.set_light_state(state)
```

## Performance Considerations

- **Frame rate**: Most effects default to 30 FPS, which is smooth enough while maintaining low CPU usage
- **Network latency**: Entertainment API has <50ms latency (much better than REST API's 300-500ms)
- **Effect blocking**: All effects are blocking (synchronous). For music sync, you'll want to send individual frames in your audio processing loop

## Example: Music Sync Pattern

```python
# In your audio processing loop:
with EntertainmentController(bridge_ip, username, clientkey) as controller:
    while True:
        # Get audio data
        bass_energy = analyze_bass()

        # Map to brightness
        brightness = min(1.0, bass_energy / 100.0)

        # Update lights (this is fast - ~20-50ms)
        controller.set_color_hsv(
            h=current_hue,
            s=1.0,
            v=brightness
        )

        # Process next frame
        time.sleep(1.0 / 43)  # 43 FPS for smooth updates
```

## Troubleshooting

### "Not connected" Error
Make sure to call `connect()` or use context manager:
```python
# Wrong:
controller = EntertainmentController(...)
controller.set_color_rgb(255, 0, 0)  # Error!

# Right:
with EntertainmentController(...) as controller:
    controller.set_color_rgb(255, 0, 0)  # Works!
```

### Colors Look Wrong
- Use RGB values 0-255 (not 0-1)
- HSV hue is 0-360 degrees (not 0-1)
- XY coordinates are 0-1 range

### No Entertainment Area Found
Create an Entertainment Area in the Hue app:
1. Open Hue app → Settings → Entertainment Areas
2. Create new area
3. Add your lights to the area
