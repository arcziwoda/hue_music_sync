"""Tick-based light effects — non-blocking, for use standalone or as building blocks.

All effects expose a tick(dt) method that advances state by dt seconds
and returns the current EffectState. No time.sleep() — caller controls timing.
"""

import math
from dataclasses import dataclass


@dataclass
class EffectState:
    """Output of a single tick of an effect."""

    hue: float = 0.0  # 0-360
    saturation: float = 1.0  # 0-1
    brightness: float = 0.0  # 0-1
    active: bool = True


class BaseEffect:
    """Base class for tick-based effects."""

    def tick(self, dt: float) -> EffectState:
        """Advance effect by dt seconds. Returns current state."""
        raise NotImplementedError

    def reset(self) -> None:
        pass


class PulseEffect(BaseEffect):
    """Sine-wave brightness pulse."""

    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        period: float = 1.0,
        min_brightness: float = 0.1,
        max_brightness: float = 1.0,
    ):
        self.hue = hue
        self.saturation = saturation
        self.period = period
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self._phase = 0.0

    def tick(self, dt: float) -> EffectState:
        self._phase = (self._phase + dt / self.period) % 1.0
        b = self.min_brightness + (self.max_brightness - self.min_brightness) * (
            0.5 + 0.5 * math.sin(2 * math.pi * self._phase - math.pi / 2)
        )
        return EffectState(
            hue=self.hue, saturation=self.saturation, brightness=b
        )

    def reset(self) -> None:
        self._phase = 0.0


class BreatheEffect(BaseEffect):
    """Cubic-eased breathing — natural inhale/exhale pattern."""

    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        period: float = 3.0,
        min_brightness: float = 0.0,
        max_brightness: float = 1.0,
    ):
        self.hue = hue
        self.saturation = saturation
        self.period = period
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self._phase = 0.0

    def tick(self, dt: float) -> EffectState:
        self._phase = (self._phase + dt / self.period) % 1.0
        if self._phase < 0.5:
            t = self._phase * 2
            eased = t * t * t
        else:
            t = (self._phase - 0.5) * 2
            eased = 1 - t * t * t
        b = self.min_brightness + (self.max_brightness - self.min_brightness) * eased
        return EffectState(
            hue=self.hue, saturation=self.saturation, brightness=b
        )

    def reset(self) -> None:
        self._phase = 0.0


class ColorCycleEffect(BaseEffect):
    """Continuous hue rotation."""

    def __init__(
        self,
        period: float = 10.0,
        saturation: float = 1.0,
        brightness: float = 1.0,
    ):
        self.period = period
        self.saturation = saturation
        self.brightness = brightness
        self._hue = 0.0

    def tick(self, dt: float) -> EffectState:
        self._hue = (self._hue + 360 * dt / self.period) % 360
        return EffectState(
            hue=self._hue, saturation=self.saturation, brightness=self.brightness
        )

    def reset(self) -> None:
        self._hue = 0.0


class StrobeEffect(BaseEffect):
    """Flash on/off with safety-clamped frequency.

    Max 3 Hz. No saturated red strobe (clamped to sat 0.7).
    """

    MAX_SAFE_HZ = 3.0

    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        frequency: float = 2.0,
        max_brightness: float = 1.0,
    ):
        self.frequency = min(frequency, self.MAX_SAFE_HZ)
        self.hue = hue
        self.max_brightness = max_brightness
        self._phase = 0.0

        # Safety: no saturated red strobe
        if (hue < 15 or hue > 345) and saturation > 0.7:
            self.saturation = 0.7
        else:
            self.saturation = saturation

    def tick(self, dt: float) -> EffectState:
        self._phase = (self._phase + dt * self.frequency) % 1.0
        b = self.max_brightness if self._phase < 0.5 else 0.0
        return EffectState(
            hue=self.hue, saturation=self.saturation, brightness=b
        )

    def reset(self) -> None:
        self._phase = 0.0


class FlashDecayEffect(BaseEffect):
    """Single flash with exponential decay — designed for beat triggers.

    Call trigger() to fire, then tick() advances the decay.
    """

    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        decay_ms: float = 250,
    ):
        self.hue = hue
        self.saturation = saturation
        self.decay_tau = decay_ms / 1000.0
        self._strength = 0.0
        self._time = 0.0
        self._active = False

    def trigger(self, strength: float = 1.0, hue: float | None = None) -> None:
        """Fire the flash."""
        self._strength = strength
        if hue is not None:
            self.hue = hue
        self._time = 0.0
        self._active = True

    def tick(self, dt: float) -> EffectState:
        if not self._active:
            return EffectState(brightness=0.0, active=False)

        self._time += dt
        b = self._strength * math.exp(-self._time / self.decay_tau)

        if b < 0.01:
            self._active = False
            b = 0.0

        return EffectState(
            hue=self.hue, saturation=self.saturation, brightness=b
        )

    def reset(self) -> None:
        self._time = 0.0
        self._active = False
        self._strength = 0.0
