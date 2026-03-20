"""Effect engine — orchestrates audio-to-light mapping.

Hybrid reactive-generative pipeline (Task 1.1 + 1.2):
1. **Generative layer** (always active): slow hue rotation + breathing brightness
   + spatial waves. Produces beautiful ambient patterns even during silence.
2. **Reactive layer**: palette-driven color mapping from audio features + beat flash.
3. **Blend**: energy-driven mix — quiet passages ~80% generative, loud ~80% reactive.
4. Per-light asymmetric EMA smoothing on blended result.
5. Safety limiter: max 3Hz flash, no strobe red.

Palette-driven reactive layer:
- Genre palette provides 3-4 curated hues distributed across lights
- Colors rotate/chase across lights, synced to BPM (1 full rotation = 16 beats)
- Music drives brightness (RMS) and saturation (flatness), not hue
- Centroid adds subtle +/-20 deg timbral shift to palette hues
- Beat = brightness flash (exponential decay), not color change

Per-band effects (Task 1.5 + 1.9 + 1.10):
- Bass pulse: kick onset -> red/orange hue (0-30 deg), brightness proportional
  to bass energy, blended into reactive layer on bass-dominant lights.
- Treble sparkle: hi-hat onset -> brief blue/violet flicker (240-280 deg) on
  1-2 randomly selected lights, very short decay (~75ms).
- Snare onset: triggers the existing white/bright flash (handled by main beat flash).

Predictive beat triggering (Task 0.4):
When PLL confidence is high enough, fires light commands ~80ms BEFORE the
predicted beat arrives to compensate for end-to-end system latency
(audio -> processing -> DTLS -> bridge -> bulb ~ 80-120ms).

Intensity selector (Task 1.12):
3-level intensity (intense/normal/chill) applies multipliers on top of base
parameter values (which come from genre presets). Multipliers affect flash_tau,
attack_alpha, max brightness, and beat threshold.

Effects size (Task 1.13):
Controls how many lights receive reactive effects per beat. At <100%, only a
rotating subset gets full reactive treatment; others stay on generative base.

Light group splitting (Task 1.14):
Auto-splits lights into 2-3 subgroups with different palette phase offsets,
creating color variety across the light array.
"""

import math
import random
import time
from dataclasses import dataclass, field

from ..audio.analyzer import AudioFeatures
from ..audio.beat_detector import BeatInfo
from ..audio.section_detector import Section, SectionInfo
from ..bridge.entertainment_controller import LightState
from ..utils.color_conversion import hsv_to_xy
from .color_mapper import ColorMapper, _ema, _smooth_hue, COLOR_MODE_CENTROID, COLOR_MODES
from .spatial import SpatialMapper


# ---------------------------------------------------------------------------
# Intensity levels (Task 1.12)
# ---------------------------------------------------------------------------

INTENSITY_INTENSE = "intense"
INTENSITY_NORMAL = "normal"
INTENSITY_CHILL = "chill"
INTENSITY_LEVELS = [INTENSITY_INTENSE, INTENSITY_NORMAL, INTENSITY_CHILL]

# Multipliers applied on top of base parameter values.
# Keys: flash_tau, attack_alpha, max_brightness, beat_threshold
INTENSITY_MULTIPLIERS: dict[str, dict[str, float]] = {
    INTENSITY_INTENSE: {
        "flash_tau": 0.7,
        "attack_alpha": 1.3,
        "max_brightness": 1.0,
        "beat_threshold": 0.85,
    },
    INTENSITY_NORMAL: {
        "flash_tau": 1.0,
        "attack_alpha": 1.0,
        "max_brightness": 0.85,
        "beat_threshold": 1.0,
    },
    INTENSITY_CHILL: {
        "flash_tau": 1.5,
        "attack_alpha": 0.6,
        "max_brightness": 0.6,
        "beat_threshold": 1.2,
    },
}


@dataclass
class _LightSmoothed:
    """Per-light smoothed state."""

    hue: float = 180.0
    saturation: float = 0.5
    brightness: float = 0.0
    flash_brightness: float = 0.0

    # Bass pulse state (Task 1.10)
    bass_pulse_brightness: float = 0.0  # Decaying pulse brightness
    bass_pulse_hue: float = 15.0  # Red-orange hue for bass pulse

    # Treble sparkle state (Task 1.9)
    sparkle_brightness: float = 0.0  # Decaying sparkle brightness
    sparkle_hue: float = 260.0  # Blue-violet hue for sparkle


def _palette_hue(palette: tuple[float, ...], phase: float) -> float:
    """Get interpolated hue from palette at a given phase (0-1)."""
    n = len(palette)
    if n == 0:
        return 180.0
    if n == 1:
        return palette[0]

    pos = phase * n
    idx = int(pos) % n
    frac = pos - int(pos)
    next_idx = (idx + 1) % n

    h1 = palette[idx]
    h2 = palette[next_idx]
    diff = h2 - h1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (h1 + frac * diff) % 360


# ---------------------------------------------------------------------------
# Generative layer — always-active ambient patterns
# ---------------------------------------------------------------------------


class GenerativeLayer:
    """Produces ambient light patterns independently of audio input.

    Combines three effects:
    - Slow hue rotation through the palette (30-60s full cycle)
    - Sinusoidal breathing brightness oscillation (0.25 Hz, between min/max)
    - Spatial wave that creates gentle movement across lights

    The result is a beautiful, always-moving base pattern that ensures lights
    are never dark or static, even during complete silence.
    """

    def __init__(
        self,
        num_lights: int,
        hue_cycle_period: float = 45.0,
        breathing_rate_hz: float = 0.25,
        breathing_min: float = 0.20,
        breathing_max: float = 0.80,
        wave_speed: float = 0.15,
        base_saturation: float = 0.7,
    ):
        self.num_lights = max(num_lights, 1)
        self.hue_cycle_period = max(hue_cycle_period, 1.0)
        self.breathing_rate_hz = max(breathing_rate_hz, 0.01)
        self.breathing_min = max(0.0, min(1.0, breathing_min))
        self.breathing_max = max(self.breathing_min, min(1.0, breathing_max))
        self.wave_speed = wave_speed
        self.base_saturation = max(0.0, min(1.0, base_saturation))

        # Internal phase accumulators
        self._hue_phase: float = 0.0  # 0-1, position in palette
        self._breathing_phase: float = 0.0  # 0-1, position in breathing cycle
        self._wave_phase: float = 0.0  # 0-1, spatial wave position

        # Palette reference (set by engine)
        self._palette: tuple[float, ...] = (280.0, 240.0, 0.0, 320.0)

        # Light positions (linear 0-1)
        self._positions = [
            i / max(self.num_lights - 1, 1) for i in range(self.num_lights)
        ]

    def tick(self, dt: float) -> list[tuple[float, float, float]]:
        """Advance generative patterns and return per-light (H, S, V).

        Args:
            dt: Time since last tick in seconds.

        Returns:
            List of (hue, saturation, brightness) tuples, one per light.
        """
        # Advance phases
        self._hue_phase = (self._hue_phase + dt / self.hue_cycle_period) % 1.0
        self._breathing_phase = (
            self._breathing_phase + dt * self.breathing_rate_hz
        ) % 1.0
        self._wave_phase = (self._wave_phase + dt * self.wave_speed) % 1.0

        # Global breathing brightness: sinusoidal oscillation
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * self._breathing_phase)
        breath_brightness = (
            self.breathing_min
            + (self.breathing_max - self.breathing_min) * breath
        )

        result = []
        for i in range(self.num_lights):
            pos = self._positions[i]

            # Per-light hue: offset from palette by light position
            # Creates a rainbow-like spread across lights that slowly rotates
            light_hue_phase = (self._hue_phase + pos * 0.3) % 1.0
            hue = _palette_hue(self._palette, light_hue_phase)

            # Spatial wave: gentle brightness modulation across lights
            wave = 0.5 + 0.5 * math.sin(
                2 * math.pi * (self._wave_phase - pos)
            )
            # Wave modulates brightness by +/-25% around breathing base
            brightness = breath_brightness * (0.75 + 0.25 * wave)
            brightness = max(0.0, min(1.0, brightness))

            result.append((hue, self.base_saturation, brightness))

        return result

    def set_palette(self, palette: tuple[float, ...]) -> None:
        """Update the palette used for hue rotation."""
        if palette and len(palette) >= 1:
            self._palette = palette

    def reset(self) -> None:
        """Reset all phase accumulators."""
        self._hue_phase = 0.0
        self._breathing_phase = 0.0
        self._wave_phase = 0.0


# ---------------------------------------------------------------------------
# Blend utilities
# ---------------------------------------------------------------------------


def _blend_maximum(
    gen: list[tuple[float, float, float]],
    react: list[tuple[float, float, float]],
    reactive_weight: float,
) -> list[tuple[float, float, float]]:
    """Blend two layers using energy-weighted maximum (screen) blend.

    For brightness: weighted max of both layers, so beat flashes always
    punch through the generative base.
    For hue/saturation: crossfade weighted by reactive_weight so the
    reactive palette dominates during loud passages.

    Args:
        gen: Generative layer per-light (H, S, V).
        react: Reactive layer per-light (H, S, V).
        reactive_weight: 0.0 = all generative, 1.0 = all reactive.

    Returns:
        Blended per-light (H, S, V) list.
    """
    gen_weight = 1.0 - reactive_weight
    result = []

    for (gh, gs, gb), (rh, rs, rb) in zip(gen, react):
        # Hue: crossfade using shortest-path interpolation
        # When reactive is dominant, use reactive hue; when gen is dominant, use gen hue
        if reactive_weight > 0.5:
            blended_h = _blend_hue(rh, gh, gen_weight)
        else:
            blended_h = _blend_hue(gh, rh, reactive_weight)

        # Saturation: weighted average
        blended_s = gen_weight * gs + reactive_weight * rs

        # Brightness: maximum blend — the brighter layer wins, weighted.
        # This ensures beat flashes (reactive) always punch through the
        # generative base, while the generative base is always visible
        # during quiet passages.
        blended_b = max(gen_weight * gb, reactive_weight * rb)
        # Add a small contribution from the dimmer layer to soften transitions
        blended_b = min(1.0, blended_b + 0.15 * min(gen_weight * gb, reactive_weight * rb))

        result.append((blended_h, blended_s, blended_b))

    return result


def _blend_hue(primary_h: float, secondary_h: float, secondary_weight: float) -> float:
    """Blend two hues using shortest-path interpolation.

    Args:
        primary_h: Dominant hue (0-360).
        secondary_h: Secondary hue (0-360).
        secondary_weight: Weight of secondary hue (0-1).

    Returns:
        Blended hue (0-360).
    """
    diff = secondary_h - primary_h
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (primary_h + secondary_weight * diff) % 360


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class EffectEngine:
    """Main effect engine: AudioFeatures + BeatInfo -> list[LightState].

    Hybrid reactive-generative model:
    - Generative layer: always active, produces ambient patterns
    - Reactive layer: audio-driven color mapping + beat flash
    - Blend: energy-driven crossfade between layers

    The blend ratio is driven by smoothed RMS energy:
    - Silence (RMS ~0): ~80% generative, ~20% reactive
    - Loud (RMS ~1): ~80% reactive, ~20% generative
    """

    def __init__(
        self,
        num_lights: int,
        gamma: float = 2.2,
        attack_alpha: float = 0.7,
        release_alpha: float = 0.1,
        max_flash_hz: float = 3.0,
        spatial_mode: str = "frequency_zones",
        latency_compensation_ms: float = 80.0,
        predictive_confidence_threshold: float = 0.6,
        generative_hue_cycle_period: float = 45.0,
        generative_breathing_rate_hz: float = 0.25,
        generative_breathing_min: float = 0.20,
        generative_breathing_max: float = 0.80,
    ):
        self.num_lights = max(num_lights, 1)
        self.attack_alpha = attack_alpha
        self.release_alpha = release_alpha

        self.color_mapper = ColorMapper(gamma=gamma)
        self.spatial_mapper = SpatialMapper(
            num_lights=self.num_lights, mode=spatial_mode
        )

        # Per-light smoothed state
        self._lights = [_LightSmoothed() for _ in range(self.num_lights)]

        # Safety: max flash rate (epilepsy limit)
        self._min_flash_interval = 1.0 / max(max_flash_hz, 0.1)
        self._last_flash_time = 0.0

        # Beat flash: exponential decay tau (time constant in seconds)
        # Research: tau = 200-500ms. At tau=0.25, flash decays to 37% in 250ms.
        self._flash_tau = 0.25

        # Generative drift fallback (when no BPM)
        self._base_hue_offset = 0.0
        self._base_hue_speed = 6.0

        # --- Palette rotation ---
        self._palette: tuple[float, ...] = (280.0, 240.0, 0.0, 320.0)
        self._rotation_phase: float = 0.0  # 0-1, rotates across lights
        self._beats_per_rotation: float = 16.0  # full rotation every 16 beats
        self._beat_count: int = 0

        # --- Predictive beat triggering (Task 0.4) ---
        self._latency_compensation_sec = latency_compensation_ms / 1000.0
        self._predictive_confidence_threshold = predictive_confidence_threshold
        self._last_predictive_beat_target: float = 0.0
        self._predictive_beat_fired: bool = False

        # --- Generative layer (Task 1.1) ---
        self._generative = GenerativeLayer(
            num_lights=self.num_lights,
            hue_cycle_period=generative_hue_cycle_period,
            breathing_rate_hz=generative_breathing_rate_hz,
            breathing_min=generative_breathing_min,
            breathing_max=generative_breathing_max,
        )

        # --- Blend state (Task 1.1/1.2) ---
        # Smoothed energy level drives the reactive/generative blend ratio.
        # Range: 0.0 (silence) to 1.0 (loud).
        self._energy_level: float = 0.0
        # EMA alpha for energy level smoothing — slow to prevent jarring transitions
        self._energy_smooth_alpha: float = 0.08
        # Minimum reactive weight (even in silence, a little reactive shows through)
        self._min_reactive_weight: float = 0.15
        # Maximum reactive weight (even at max energy, a little generative shows through)
        self._max_reactive_weight: float = 0.85

        # --- Bass pulse (Task 1.10) ---
        self._bass_pulse_tau: float = 0.20  # Fast decay (200ms time constant)
        self._bass_pulse_intensity: float = 0.5  # Max intensity (subtle, not overwhelming)
        self._bass_pulse_hue_min: float = 0.0  # Red
        self._bass_pulse_hue_max: float = 30.0  # Orange

        # --- Treble sparkle (Task 1.9) ---
        self._sparkle_tau: float = 0.075  # Very fast decay (~75ms)
        self._sparkle_intensity: float = 0.4  # Max intensity (subtle)
        self._sparkle_hue_min: float = 240.0  # Blue
        self._sparkle_hue_max: float = 280.0  # Violet
        self._sparkle_num_lights: int = min(2, self.num_lights)  # 1-2 lights
        self._sparkle_last_lights: list[int] = []  # Which lights are sparkling

        # --- Chase spatial mode (Task 1.8) ---
        # Exponential decay time constant for chase tail (seconds).
        # At 128 BPM with 6 bulbs, one beat = ~468ms, ~78ms per bulb.
        # tau = 0.15 gives a visible tail across 2-3 bulbs.
        self._chase_decay_tau: float = 0.15

        # --- Intensity selector (Task 1.12) ---
        self._intensity_level: str = INTENSITY_NORMAL
        # Base values are the "raw" values set by genre presets.
        # Intensity multipliers are applied on top of these.
        self._base_flash_tau: float = self._flash_tau
        self._base_attack_alpha: float = self.attack_alpha
        self._max_brightness: float = INTENSITY_MULTIPLIERS[INTENSITY_NORMAL]["max_brightness"]

        # --- Effects size (Task 1.13) ---
        # Float 0.0 to 1.0: fraction of lights that get reactive effects per beat.
        # 1.0 = all lights (default). Lower values = rotating subset.
        self._effects_size: float = 1.0
        # Offset that advances each beat to rotate which lights are active.
        self._active_light_offset: int = 0
        # Set of light indices currently receiving reactive effects.
        # When effects_size < 1.0, inactive lights get reduced reactive treatment.
        self._active_lights: set[int] = set(range(self.num_lights))

        # --- Light group splitting (Task 1.14) ---
        # Each light is assigned to a group for palette phase offset diversity.
        # Number of groups: 2 for <=4 lights, 3 for >4 lights.
        self._light_groups: list[int] = []
        self._num_groups: int = 0
        self._group_phase_offsets: list[float] = []
        self._compute_light_groups()

        # --- Section detection state (Task 1.3) ---
        self._current_section = Section.NORMAL
        self._section_intensity: float = 0.0
        # DROP: extra flash brightness boost applied once at transition
        self._drop_flash_pending: bool = False
        # BUILDUP: tracks ramp progress for gradual parameter shifts
        self._buildup_progress: float = 0.0
        # BREAKDOWN: cool palette shift amount (hue offset in degrees)
        self._breakdown_hue_shift: float = 0.0

    def tick(
        self,
        features: AudioFeatures,
        beat_info: BeatInfo,
        dt: float = 0.033,
        now: float | None = None,
        section_info: SectionInfo | None = None,
    ) -> list[LightState]:
        """Process one tick. Call at ~25-30 Hz.

        Pipeline:
        1. Update section state (Task 1.3)
        2. Update energy-based blend ratio (section-modulated)
        3. Generative layer tick -> per-light HSV
        4. Reactive layer (color mapper + spatial + beat flash) -> per-light HSV
        5. Blend generative + reactive based on energy level
        6. Section modulation of blended output
        7. Per-light asymmetric EMA smoothing
        8. Safety limiter
        """
        if now is None:
            now = time.monotonic()

        # --- Predictive beat triggering (Task 0.4) ---
        trigger_beat, beat_strength = self._resolve_beat_trigger(
            beat_info, now
        )

        # --- 1. Update section state (Task 1.3) ---
        section = section_info if section_info is not None else SectionInfo()
        prev_section = self._current_section
        self._current_section = section.section
        self._section_intensity = section.intensity

        # Detect section transitions for one-shot effects
        if section.section == Section.DROP and prev_section != Section.DROP:
            self._drop_flash_pending = True

        # --- 2. Update energy-based blend ratio ---
        energy_raw = min(1.0, features.rms * 2.0)
        self._energy_level = _ema(
            self._energy_level, energy_raw, self._energy_smooth_alpha
        )

        # Map energy to reactive weight: silence -> min_reactive, loud -> max_reactive
        reactive_weight = (
            self._min_reactive_weight
            + (self._max_reactive_weight - self._min_reactive_weight)
            * self._energy_level
        )

        # Section modulation of reactive weight:
        # - DROP: force full reactive (lights should slam to audio)
        # - BUILDUP: gradually increase reactive weight toward max
        # - BREAKDOWN: pull toward generative (cool ambient dominates)
        reactive_weight = self._section_modulate_reactive_weight(
            reactive_weight, section
        )

        # --- 3. Generative layer ---
        gen_hsv = self._generative.tick(dt)

        # --- 4. Reactive layer ---
        react_hsv = self._reactive_layer(features, beat_info, trigger_beat, dt, now)

        # --- 5. Blend generative + reactive ---
        blended_hsv = _blend_maximum(gen_hsv, react_hsv, reactive_weight)

        # --- 5b. Section color modulation (Task 1.3) ---
        blended_hsv = self._section_modulate_colors(blended_hsv, section)

        # --- 5c. Beat flash overlay (applied AFTER blend, so flash always
        #     punches through regardless of reactive/generative balance) ---
        # DROP: fire a massive flash on transition (all lights full brightness)
        if self._drop_flash_pending:
            self._drop_flash_pending = False
            if (now - self._last_flash_time) >= self._min_flash_interval:
                self._last_flash_time = now
                for light in self._lights:
                    light.flash_brightness = 1.0  # Max flash for drop

        # Regular beat flash (also triggered by snare onsets for bright white flash)
        if trigger_beat or beat_info.snare_onset:
            flash_strength = beat_strength
            # Snare onset without main beat: use snare energy as flash strength
            if beat_info.snare_onset and not trigger_beat:
                flash_strength = min(1.0, beat_info.snare_energy * 0.8)
            # BUILDUP: amplify beat flash as buildup progresses
            if section.section == Section.BUILDUP:
                flash_strength = min(1.0, flash_strength * (1.0 + 0.5 * section.intensity))
            if (now - self._last_flash_time) >= self._min_flash_interval:
                self._last_flash_time = now
                # Task 1.13: Only flash active lights when effects_size < 1.0
                for idx, light in enumerate(self._lights):
                    if idx in self._active_lights:
                        light.flash_brightness = flash_strength
                    # Inactive lights keep their current (decaying) flash

            # Task 1.13: Advance active light rotation on beat
            if trigger_beat:
                self._advance_active_lights()

        # --- 5c-chase. Chase mode: advance chase position on beat ---
        if trigger_beat and self.spatial_mapper.mode == SpatialMapper.CHASE:
            sm = self.spatial_mapper
            n = self.num_lights
            # Advance chase to next bulb
            next_pos = sm._chase_position + sm._chase_direction
            # Wrap or reverse direction
            if sm._chase_alternating:
                if next_pos >= n or next_pos < 0:
                    sm._chase_direction *= -1
                    next_pos = sm._chase_position + sm._chase_direction
            else:
                next_pos = next_pos % n
            sm._chase_position = next_pos
            # Mark the newly activated bulb
            active_idx = int(round(sm._chase_position)) % n
            sm._chase_last_activated[active_idx] = now

        # --- 5d. Bass pulse overlay (Task 1.10) ---
        # Kick onset triggers red/orange pulse on bass-dominant lights
        if beat_info.kick_onset:
            bass_strength = min(1.0, beat_info.kick_energy * self._bass_pulse_intensity * 2.0)
            # Map bass energy to hue within red-orange range (0-30 deg)
            bass_hue = self._bass_pulse_hue_min + (
                self._bass_pulse_hue_max - self._bass_pulse_hue_min
            ) * (1.0 - min(1.0, beat_info.kick_energy))  # Lower energy -> more orange

            mode = self.spatial_mapper.mode
            for i, light in enumerate(self._lights):
                # In frequency_zones/mirror mode, apply more to bass-dominant lights
                if mode in ("frequency_zones", "mirror"):
                    pos = self.spatial_mapper._positions[i]
                    if mode == "mirror":
                        center = (self.num_lights - 1) / 2.0
                        bass_weight = abs(i - center) / max(center, 1)
                        bass_weight = 1.0 - bass_weight  # Center lights get less
                    else:
                        bass_weight = 1.0 - pos  # Low positions = bass
                    # Apply with spatial weighting
                    light.bass_pulse_brightness = bass_strength * (0.3 + 0.7 * bass_weight)
                else:
                    # Uniform/wave: apply to all lights equally
                    light.bass_pulse_brightness = bass_strength
                light.bass_pulse_hue = bass_hue

        # --- 5e. Treble sparkle overlay (Task 1.9) ---
        # Hi-hat onset triggers brief blue/violet flicker on 1-2 random lights
        if beat_info.hihat_onset:
            sparkle_strength = min(1.0, beat_info.hihat_energy * self._sparkle_intensity * 2.0)
            # Random hue within blue-violet range
            sparkle_hue = random.uniform(self._sparkle_hue_min, self._sparkle_hue_max)
            # Pick 1-2 random lights (different from last time if possible)
            n_sparkle = min(self._sparkle_num_lights, self.num_lights)
            available = list(range(self.num_lights))
            # Try to pick different lights than last time
            if len(available) > n_sparkle:
                preferred = [l for l in available if l not in self._sparkle_last_lights]
                if len(preferred) >= n_sparkle:
                    available = preferred
            sparkle_targets = random.sample(available, n_sparkle)
            self._sparkle_last_lights = sparkle_targets
            for idx in sparkle_targets:
                self._lights[idx].sparkle_brightness = sparkle_strength
                self._lights[idx].sparkle_hue = sparkle_hue

        # --- 6. Per-light: flash decay + EMA smoothing + safety ---
        # Section-modulated flash decay (faster during DROP for punchy feel)
        active_flash_tau = self._flash_tau
        if section.section == Section.DROP:
            active_flash_tau = max(0.08, self._flash_tau * 0.6)

        light_states = []
        for i, (target_h, target_s, target_b) in enumerate(blended_hsv):
            light = self._lights[i]

            # Exponential flash decay: f(t) = f0 * e^(-dt/tau)
            if light.flash_brightness > 0.01:
                light.flash_brightness *= math.exp(-dt / active_flash_tau)
                if light.flash_brightness < 0.01:
                    light.flash_brightness = 0.0

            # Bass pulse decay (Task 1.10): faster decay than main flash
            if light.bass_pulse_brightness > 0.01:
                light.bass_pulse_brightness *= math.exp(-dt / self._bass_pulse_tau)
                if light.bass_pulse_brightness < 0.01:
                    light.bass_pulse_brightness = 0.0

            # Treble sparkle decay (Task 1.9): very fast decay
            if light.sparkle_brightness > 0.01:
                light.sparkle_brightness *= math.exp(-dt / self._sparkle_tau)
                if light.sparkle_brightness < 0.01:
                    light.sparkle_brightness = 0.0

            # --- Compose target hue/sat from overlays ---
            # Start with blended values
            final_h = target_h
            final_s = target_s

            # Bass pulse blends hue toward red/orange when active
            if light.bass_pulse_brightness > 0.02:
                pulse_weight = light.bass_pulse_brightness * 0.6  # Don't fully override
                final_h = _blend_hue(final_h, light.bass_pulse_hue, pulse_weight)
                final_s = final_s + (1.0 - final_s) * pulse_weight * 0.5  # Boost saturation

            # Sparkle blends hue toward blue/violet when active
            if light.sparkle_brightness > 0.02:
                sparkle_weight = light.sparkle_brightness * 0.7
                final_h = _blend_hue(final_h, light.sparkle_hue, sparkle_weight)
                final_s = min(1.0, final_s + sparkle_weight * 0.3)

            # Task 1.13: Reduce reactive overlays for inactive lights.
            # Active lights get full flash/pulse/sparkle; inactive get reduced.
            # We use a reactive_scale factor rather than modifying the stored
            # brightness values (which would corrupt decay state).
            is_active = i in self._active_lights
            reactive_scale = 1.0 if (is_active or self._effects_size >= 1.0) else 0.15

            # Flash overlays on blended brightness — all effects additive
            combined_b = min(
                1.0,
                target_b
                + light.flash_brightness * reactive_scale
                + light.bass_pulse_brightness * 0.7 * reactive_scale
                + light.sparkle_brightness * 0.5 * reactive_scale
            )

            # Task 1.12: Apply intensity max brightness cap
            combined_b = min(combined_b, self._max_brightness)

            # Asymmetric EMA smoothing — fast attack, slower release
            b_alpha = (
                self.attack_alpha
                if combined_b > light.brightness
                else self.release_alpha
            )
            light.hue = _smooth_hue(light.hue, final_h, 0.2)
            light.saturation = _ema(light.saturation, min(1.0, final_s), 0.2)
            light.brightness = _ema(light.brightness, combined_b, b_alpha)

            # Safety: no strobe saturated red
            any_flash_active = (
                light.flash_brightness > 0.3
                or light.bass_pulse_brightness > 0.3
            )
            if (light.hue < 15 or light.hue > 345) and any_flash_active:
                light.saturation = min(light.saturation, 0.7)

            x, y = hsv_to_xy(light.hue, light.saturation, 1.0)
            light_states.append(
                LightState(x=x, y=y, brightness=light.brightness, light_id=i)
            )

        return light_states

    # --- Section modulation helpers (Task 1.3) ---

    def _section_modulate_reactive_weight(
        self, base_weight: float, section: SectionInfo
    ) -> float:
        """Modulate reactive/generative blend based on current section.

        - DROP: force reactive weight to max (lights slam to audio)
        - BUILDUP: ramp reactive weight up toward max as intensity grows
        - BREAKDOWN: pull reactive weight down (generative ambient dominates)
        - NORMAL: no change
        """
        if section.section == Section.DROP:
            # During a drop, reactive weight is forced high.
            # Blend toward 1.0 based on intensity (smooth, not instant).
            return base_weight + (1.0 - base_weight) * section.intensity

        if section.section == Section.BUILDUP:
            # Gradually ramp up reactive weight as buildup progresses.
            # At full intensity, push weight to ~0.9
            target = min(1.0, self._max_reactive_weight + 0.1)
            return base_weight + (target - base_weight) * section.intensity * 0.7

        if section.section == Section.BREAKDOWN:
            # Pull reactive weight down — generative ambient dominates.
            # At full intensity, reactive weight drops to ~0.1
            target = max(0.0, self._min_reactive_weight - 0.05)
            return base_weight + (target - base_weight) * section.intensity * 0.8

        return base_weight

    def _section_modulate_colors(
        self,
        hsv_list: list[tuple[float, float, float]],
        section: SectionInfo,
    ) -> list[tuple[float, float, float]]:
        """Apply section-specific color modulation to the blended HSV output.

        - DROP: boost brightness, increase saturation (vivid, punchy)
        - BUILDUP: shift hue toward warm, gradually increase brightness
        - BREAKDOWN: dim to 20-40%, shift toward cool colors, reduce saturation (pastels)
        - NORMAL: no change
        """
        if section.section == Section.NORMAL or section.intensity < 0.01:
            return hsv_list

        result = []
        intensity = section.intensity

        for h, s, b in hsv_list:
            if section.section == Section.DROP:
                # Boost brightness: min 70% during drop
                b = b + (1.0 - b) * intensity * 0.5
                # Boost saturation for vivid colors
                s = min(1.0, s + intensity * 0.2)

            elif section.section == Section.BUILDUP:
                # Shift hue toward warm (add 0-30 degrees toward red/orange)
                warm_shift = intensity * 30.0
                h = (h + warm_shift) % 360
                # Gradually ramp brightness
                b = b + (1.0 - b) * intensity * 0.3

            elif section.section == Section.BREAKDOWN:
                # Dim brightness to 20-40% range
                max_b = 0.2 + 0.2 * (1.0 - intensity)  # 0.2 at full, 0.4 at low
                b = b * (1.0 - intensity * 0.6)  # Reduce by up to 60%
                b = max(0.05, min(max_b, b))  # Clamp to dim range
                # Shift toward cool colors (add 180 degrees = blue shift)
                cool_shift = intensity * 40.0
                h = (h + cool_shift) % 360
                # Reduce saturation for pastel effect
                s = s * (1.0 - intensity * 0.3)

            b = max(0.0, min(1.0, b))
            s = max(0.0, min(1.0, s))
            result.append((h, s, b))

        return result

    def _resolve_beat_trigger(
        self, beat_info: BeatInfo, now: float
    ) -> tuple[bool, float]:
        """Determine if a beat should fire this tick (predictive or reactive).

        Returns:
            (trigger_beat, beat_strength) tuple.
        """
        trigger_beat = False
        beat_strength = beat_info.beat_strength

        # Check for predictive trigger
        predictive_available = (
            self._latency_compensation_sec > 0
            and beat_info.predicted_next_beat > 0
            and beat_info.bpm_confidence >= self._predictive_confidence_threshold
        )

        if predictive_available:
            predicted = beat_info.predicted_next_beat
            fire_at = predicted - self._latency_compensation_sec
            is_new_prediction = (
                abs(predicted - self._last_predictive_beat_target) > 0.05
            )

            if is_new_prediction and now >= fire_at:
                trigger_beat = True
                self._last_predictive_beat_target = predicted
                self._predictive_beat_fired = True
                if beat_strength < 0.1:
                    beat_strength = 0.7

        # Reactive beat: only if we didn't already fire predictively
        if beat_info.is_beat and not trigger_beat:
            if self._predictive_beat_fired:
                time_since_prediction = abs(
                    now - self._last_predictive_beat_target
                )
                if time_since_prediction < self._latency_compensation_sec + 0.1:
                    pass  # Skip — corresponds to our predictive trigger
                else:
                    trigger_beat = True
                    beat_strength = beat_info.beat_strength
                    self._predictive_beat_fired = False
            else:
                trigger_beat = True
                beat_strength = beat_info.beat_strength

        # Reset predictive state when a new beat period starts
        if (
            beat_info.predicted_next_beat > 0
            and beat_info.predicted_next_beat
            > self._last_predictive_beat_target + 0.05
            and not trigger_beat
        ):
            self._predictive_beat_fired = False

        return trigger_beat, beat_strength

    def _reactive_layer(
        self,
        features: AudioFeatures,
        beat_info: BeatInfo,
        trigger_beat: bool,
        dt: float,
        now: float = 0.0,
    ) -> list[tuple[float, float, float]]:
        """Produce per-light HSV from audio features (reactive behavior).

        This encapsulates the original monolithic engine logic:
        color mapper -> palette rotation -> spatial distribution.
        Beat flash is handled separately in tick() so it can be blended.

        In centroid color mode, the color mapper returns a direct hue (0-300)
        instead of a palette offset, and palette rotation is bypassed.
        """
        # Audio -> brightness, saturation, hue offset (or direct hue in centroid mode)
        hue_value, base_sat, base_brightness = self.color_mapper.map(features)

        is_centroid_mode = self.color_mapper.color_mode == COLOR_MODE_CENTROID

        # Advance rotation phase (colors chase across lights) -- palette mode only
        if not is_centroid_mode:
            if trigger_beat:
                self._beat_count += 1
                self._rotation_phase = (
                    self._beat_count / self._beats_per_rotation
                ) % 1.0
            elif beat_info.bpm > 0:
                beats_per_sec = beat_info.bpm / 60.0
                self._rotation_phase = (
                    self._rotation_phase
                    + dt * beats_per_sec / self._beats_per_rotation
                ) % 1.0
            else:
                self._base_hue_offset = (
                    self._base_hue_offset + self._base_hue_speed * dt
                ) % 360
                self._rotation_phase = (self._base_hue_offset / 360.0) % 1.0

        # Per-light distribution + brightness from bands
        return self._distribute(
            features, hue_value, base_sat, base_brightness, dt,
            centroid_mode=is_centroid_mode,
            now=now,
        )

    def _distribute(
        self,
        features: AudioFeatures,
        hue_value: float,
        base_sat: float,
        base_brightness: float,
        dt: float,
        centroid_mode: bool = False,
        now: float = 0.0,
    ) -> list[tuple[float, float, float]]:
        """Distribute colors across lights with spatial variation.

        In palette mode: each light samples a different point in the palette,
        offset by rotation_phase (which advances with BPM). hue_value is
        a +-20 deg offset applied on top.

        In centroid mode: hue_value is the direct centroid-derived hue (0-300).
        All lights share the same base hue (no palette rotation), but spatial
        brightness distribution still applies from band energies.
        """
        mode = self.spatial_mapper.mode
        palette = self._palette
        n_lights = self.num_lights

        if mode == SpatialMapper.WAVE:
            self.spatial_mapper._wave_phase = (
                self.spatial_mapper._wave_phase + dt * 2.0
            ) % 1.0

        # Chase mode: compute per-bulb brightness multipliers from decay
        chase_multipliers: list[float] | None = None
        if mode == SpatialMapper.CHASE:
            chase_multipliers = []
            for i in range(n_lights):
                time_since = now - self.spatial_mapper._chase_last_activated[i]
                if time_since < 0:
                    time_since = 0
                # Exponential decay from last activation
                decay = math.exp(-time_since / self._chase_decay_tau)
                chase_multipliers.append(decay)

        if centroid_mode:
            # Centroid mode: direct hue for all lights, no palette rotation
            shared_hue = hue_value
        else:
            # Palette mode: shared palette hue + centroid offset
            shared_hue = _palette_hue(palette, self._rotation_phase)
            shared_hue = (shared_hue + hue_value) % 360

        result = []

        for i in range(n_lights):
            pos = self.spatial_mapper._positions[i]
            # Task 1.14: Per-light group phase offset for color diversity
            group_offset = self.get_group_phase_offset(i)

            if mode == SpatialMapper.UNIFORM:
                if centroid_mode or self._num_groups <= 1:
                    hue = shared_hue
                else:
                    # Task 1.14: Even in uniform mode, apply group offset
                    group_phase = (self._rotation_phase + group_offset) % 1.0
                    hue = _palette_hue(palette, group_phase)
                    hue = (hue + hue_value) % 360
                result.append((hue, base_sat, base_brightness))

            elif mode == SpatialMapper.WAVE:
                wave = 0.5 + 0.5 * math.sin(
                    2 * math.pi * (self.spatial_mapper._wave_phase - pos)
                )
                brightness = min(1.0, base_brightness * (0.15 + 0.85 * wave))
                if centroid_mode or self._num_groups <= 1:
                    hue = shared_hue
                else:
                    group_phase = (self._rotation_phase + group_offset) % 1.0
                    hue = _palette_hue(palette, group_phase)
                    hue = (hue + hue_value) % 360
                result.append((hue, base_sat, brightness))

            elif mode == SpatialMapper.CHASE:
                # Chase mode: sequential per-bulb activation with exponential decay.
                # The "lit" bulb gets full brightness, others decay from their last
                # activation time. Per-light palette hue gives each bulb a distinct
                # color that chases across the array.
                multiplier = chase_multipliers[i] if chase_multipliers else 0.0
                # Minimum floor so inactive bulbs are not completely dark
                brightness = min(
                    1.0,
                    base_brightness * (0.05 + 0.95 * multiplier),
                )

                if centroid_mode:
                    hue = shared_hue
                else:
                    # Each light samples a different palette point for color variety
                    # Task 1.14: Add group offset
                    light_phase = (
                        self._rotation_phase + group_offset + i / max(n_lights - 1, 1)
                    ) % 1.0
                    hue = _palette_hue(palette, light_phase)
                    hue = (hue + hue_value) % 360

                result.append((hue, base_sat, brightness))

            else:
                # FREQ / MIRROR
                if mode == "mirror":
                    center = (n_lights - 1) / 2.0
                    light_spread = 1.0 - abs(i - center) / max(center, 1)
                else:
                    light_spread = pos

                if centroid_mode:
                    # Centroid mode: same hue for all lights
                    hue = shared_hue
                else:
                    # Palette mode: per-light palette sampling + offset
                    # Task 1.14: Add group offset
                    light_phase = (
                        self._rotation_phase + group_offset + light_spread / max(n_lights - 1, 1)
                    ) % 1.0
                    hue = _palette_hue(palette, light_phase)
                    hue = (hue + hue_value) % 360

                bass_weight = 1.0 - light_spread
                mid_weight = 1.0 - abs(light_spread - 0.5) * 2
                treble_weight = light_spread

                energy = (
                    bass_weight * features.bass_energy
                    + mid_weight * features.mid_energy
                    + treble_weight * features.high_energy
                )
                brightness = min(1.0, base_brightness * (0.3 + 0.7 * energy))

                result.append((hue, base_sat, brightness))

        return result

    # --- Public API ---

    @property
    def reactive_weight(self) -> float:
        """Current reactive/generative blend ratio (0=all gen, 1=all reactive)."""
        return (
            self._min_reactive_weight
            + (self._max_reactive_weight - self._min_reactive_weight)
            * self._energy_level
        )

    @property
    def energy_level(self) -> float:
        """Current smoothed energy level (0-1)."""
        return self._energy_level

    def set_spatial_mode(self, mode: str) -> None:
        if mode in SpatialMapper.MODES:
            self.spatial_mapper.mode = mode

    def set_color_mode(self, mode: str) -> None:
        """Switch between 'palette' and 'centroid' color modes."""
        if mode in COLOR_MODES:
            self.color_mapper.set_color_mode(mode)

    def set_palette(self, palette: tuple[float, ...]) -> None:
        if palette and len(palette) >= 1:
            self._palette = palette
            self._generative.set_palette(palette)

    def set_flash_tau(self, tau: float) -> None:
        """Set flash exponential decay time constant in seconds.

        Stores as base value and reapplies intensity multiplier.
        """
        self._base_flash_tau = tau
        mult = INTENSITY_MULTIPLIERS[self._intensity_level]
        self._flash_tau = tau * mult["flash_tau"]

    def set_hue_drift_speed(self, speed: float) -> None:
        """Set generative hue drift speed in degrees/second."""
        self._base_hue_speed = speed

    def set_latency_compensation(self, ms: float) -> None:
        """Set latency compensation in milliseconds."""
        self._latency_compensation_sec = max(0.0, ms) / 1000.0

    def set_predictive_confidence_threshold(self, threshold: float) -> None:
        """Set minimum PLL confidence for predictive triggering (0-1)."""
        self._predictive_confidence_threshold = max(0.0, min(1.0, threshold))

    def set_generative_breathing(
        self,
        rate_hz: float | None = None,
        min_brightness: float | None = None,
        max_brightness: float | None = None,
    ) -> None:
        """Update generative layer breathing parameters."""
        if rate_hz is not None:
            self._generative.breathing_rate_hz = max(0.01, rate_hz)
        if min_brightness is not None:
            self._generative.breathing_min = max(0.0, min(1.0, min_brightness))
        if max_brightness is not None:
            self._generative.breathing_max = max(
                self._generative.breathing_min, min(1.0, max_brightness)
            )

    def set_generative_hue_cycle_period(self, period: float) -> None:
        """Set generative layer hue rotation period in seconds."""
        self._generative.hue_cycle_period = max(1.0, period)

    @property
    def current_section(self) -> Section:
        """Current detected musical section."""
        return self._current_section

    @property
    def section_intensity(self) -> float:
        """Current section effect intensity (0-1)."""
        return self._section_intensity

    def set_light_positions(self, positions: list[float]) -> None:
        """Set light positions from bridge entertainment area data.

        Delegates to SpatialMapper and also updates GenerativeLayer positions.

        Args:
            positions: Normalized 0-1 positions, one per light.
        """
        self.spatial_mapper.set_positions(positions)
        # Also update generative layer positions for consistent spatial wave
        if len(positions) == self.num_lights:
            self._generative._positions = list(positions)

    # --- Intensity selector (Task 1.12) ---

    @property
    def intensity_level(self) -> str:
        """Current intensity level ('intense', 'normal', 'chill')."""
        return self._intensity_level

    def set_intensity(self, level: str) -> None:
        """Set the intensity level, applying multipliers on top of base values.

        The base values come from genre presets. Intensity modifies:
        - flash_tau: shorter = snappier flashes
        - attack_alpha: higher = faster attack response
        - max_brightness: caps the maximum light output
        - beat_threshold: returned for pipeline to apply externally

        Args:
            level: One of 'intense', 'normal', 'chill'.
        """
        if level not in INTENSITY_LEVELS:
            return
        self._intensity_level = level
        self._apply_intensity_multipliers()

    def _apply_intensity_multipliers(self) -> None:
        """Reapply intensity multipliers to current base values."""
        mult = INTENSITY_MULTIPLIERS[self._intensity_level]
        self._flash_tau = self._base_flash_tau * mult["flash_tau"]
        self.attack_alpha = self._base_attack_alpha * mult["attack_alpha"]
        # Clamp attack_alpha to valid range
        self.attack_alpha = max(0.01, min(1.0, self.attack_alpha))
        self._max_brightness = mult["max_brightness"]

    def get_intensity_beat_threshold_multiplier(self) -> float:
        """Return the beat threshold multiplier for the current intensity.

        The server applies this externally to the beat detector.
        """
        return INTENSITY_MULTIPLIERS[self._intensity_level]["beat_threshold"]

    def set_base_attack_alpha(self, alpha: float) -> None:
        """Set the base attack alpha (from genre preset) and reapply intensity.

        This should be called instead of setting attack_alpha directly when
        changing genre presets, so the intensity multiplier stays correct.
        """
        self._base_attack_alpha = alpha
        self._apply_intensity_multipliers()

    # --- Effects size (Task 1.13) ---

    @property
    def effects_size(self) -> float:
        """Current effects size (0.0 to 1.0)."""
        return self._effects_size

    def set_effects_size(self, size: float) -> None:
        """Set the effects size: fraction of lights receiving reactive effects.

        Args:
            size: 0.0 to 1.0. Special values:
                  - 1.0 = all lights (default)
                  - 0.5 = half the lights
                  - 0.25 = quarter of the lights
                  - Values <= 1/num_lights = single light
        """
        self._effects_size = max(0.0, min(1.0, size))
        # Recompute active lights immediately
        self._update_active_lights()

    def _update_active_lights(self) -> None:
        """Recompute which lights are currently active based on effects_size.

        Active lights rotate by _active_light_offset to cycle through all lights.
        """
        n = self.num_lights
        if self._effects_size >= 1.0:
            self._active_lights = set(range(n))
            return

        n_active = max(1, math.ceil(n * self._effects_size))
        self._active_lights = set()
        for k in range(n_active):
            idx = (self._active_light_offset + k) % n
            self._active_lights.add(idx)

    def _advance_active_lights(self) -> None:
        """Advance the active light rotation on each beat."""
        if self._effects_size >= 1.0:
            return
        n_active = max(1, math.ceil(self.num_lights * self._effects_size))
        self._active_light_offset = (
            self._active_light_offset + n_active
        ) % self.num_lights
        self._update_active_lights()

    # --- Light group splitting (Task 1.14) ---

    @property
    def light_groups(self) -> list[int]:
        """Per-light group assignment (0, 1, or 2)."""
        return list(self._light_groups)

    @property
    def num_groups(self) -> int:
        """Number of light groups (2 or 3)."""
        return self._num_groups

    def _compute_light_groups(self) -> None:
        """Assign each light to a group based on position.

        2 groups for <= 4 lights, 3 groups for > 4 lights.
        Groups are assigned by dividing the light array evenly.
        Phase offsets give each group a different starting point in the palette.
        """
        n = self.num_lights
        if n <= 1:
            self._light_groups = [0]
            self._num_groups = 1
            self._group_phase_offsets = [0.0]
            return

        self._num_groups = 2 if n <= 4 else 3
        self._light_groups = []
        for i in range(n):
            group = int(i * self._num_groups / n)
            self._light_groups.append(min(group, self._num_groups - 1))

        # Phase offsets: evenly distributed across 0-1
        # For 2 groups: 0.0 and 0.5 (0 deg and 180 deg)
        # For 3 groups: 0.0, 0.333, 0.667 (0 deg, 120 deg, 240 deg)
        self._group_phase_offsets = [
            g / self._num_groups for g in range(self._num_groups)
        ]

    def get_group_phase_offset(self, light_index: int) -> float:
        """Return the palette phase offset for a given light's group.

        Used in _distribute to shift the palette sampling point per group.

        Args:
            light_index: Index of the light (0 to num_lights-1).

        Returns:
            Phase offset in range [0, 1).
        """
        if not self._light_groups or light_index >= len(self._light_groups):
            return 0.0
        group = self._light_groups[light_index]
        return self._group_phase_offsets[group]

    def reset(self) -> None:
        self.color_mapper.reset()
        self.spatial_mapper.reset()
        self._generative.reset()
        self._lights = [_LightSmoothed() for _ in range(self.num_lights)]
        self._last_flash_time = 0.0
        self._base_hue_offset = 0.0
        self._rotation_phase = 0.0
        self._beat_count = 0
        self._last_predictive_beat_target = 0.0
        self._predictive_beat_fired = False
        self._energy_level = 0.0
        self._current_section = Section.NORMAL
        self._section_intensity = 0.0
        self._drop_flash_pending = False
        self._buildup_progress = 0.0
        self._breakdown_hue_shift = 0.0
        self._sparkle_last_lights = []
        # Task 1.13: Reset effects size rotation
        self._active_light_offset = 0
        self._update_active_lights()
