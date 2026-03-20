"""
Bridge module - Hue Bridge connection and Entertainment API control.
"""

from .entertainment_controller import EntertainmentController, LightState
from .effects import (
    BaseEffect,
    BreatheEffect,
    ColorCycleEffect,
    EffectState,
    FlashDecayEffect,
    PulseEffect,
    StrobeEffect,
)

__all__ = [
    "EntertainmentController",
    "LightState",
    "BaseEffect",
    "BreatheEffect",
    "ColorCycleEffect",
    "EffectState",
    "FlashDecayEffect",
    "PulseEffect",
    "StrobeEffect",
]
