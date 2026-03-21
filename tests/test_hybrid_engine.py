"""Tests for the hybrid reactive-generative model (Task 1.1 + 1.2).

Verifies:
- GenerativeLayer produces meaningful output independently
- Breathing oscillation works correctly
- Hue rotation progresses through palette
- Spatial wave creates per-light variation
- Blend ratio is driven by smoothed energy (RMS)
- Generative dominates in silence, reactive dominates during loud audio
- Beat flashes punch through generative base
- Safety limiter still works on blended output
- Reset clears all state including generative layer
"""

import math

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.engine import (
    EffectEngine,
    GenerativeLayer,
    _blend_hue,
    _blend_maximum,
)


# --- Test helpers ---


def _silence_features() -> AudioFeatures:
    """AudioFeatures representing complete silence."""
    return AudioFeatures(
        band_energies=np.zeros(7),
        spectral_centroid=0.0,
        spectral_flux=0.0,
        spectral_rolloff=0.0,
        spectral_flatness=0.0,
        rms=0.0,
        peak=0.0,
        spectrum=np.zeros(1024),
    )


def _loud_features(rms: float = 0.7) -> AudioFeatures:
    """AudioFeatures representing loud music."""
    bands = np.array([0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=3000.0,
        spectral_flux=20.0,
        spectral_rolloff=5000.0,
        spectral_flatness=0.2,
        rms=rms,
        peak=rms * 1.5,
        spectrum=np.zeros(1024),
    )


def _no_beat() -> BeatInfo:
    """BeatInfo with no beat detected."""
    return BeatInfo(
        is_beat=False,
        bpm=0.0,
        bpm_confidence=0.0,
        beat_strength=0.0,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


def _beat(strength: float = 0.8, bpm: float = 128.0) -> BeatInfo:
    """BeatInfo with a beat detected."""
    return BeatInfo(
        is_beat=True,
        bpm=bpm,
        bpm_confidence=0.5,
        beat_strength=strength,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


# ============================================================================
# GenerativeLayer unit tests
# ============================================================================


class TestGenerativeLayerBasic:
    """Test that GenerativeLayer produces valid output."""

    def test_produces_per_light_hsv(self):
        """Should return one (H, S, V) tuple per light."""
        gen = GenerativeLayer(num_lights=6)
        result = gen.tick(0.033)
        assert len(result) == 6
        for h, s, v in result:
            assert 0 <= h <= 360
            assert 0 <= s <= 1
            assert 0 <= v <= 1

    def test_single_light(self):
        gen = GenerativeLayer(num_lights=1)
        result = gen.tick(0.033)
        assert len(result) == 1

    def test_produces_nonzero_brightness(self):
        """Even at time 0, generative layer should have visible brightness."""
        gen = GenerativeLayer(num_lights=4)
        result = gen.tick(0.033)
        brightnesses = [v for _, _, v in result]
        assert max(brightnesses) > 0.1, \
            f"Generative layer should produce visible output, got max={max(brightnesses)}"

    def test_reset_zeroes_phases(self):
        gen = GenerativeLayer(num_lights=4)
        # Advance some phases
        for _ in range(100):
            gen.tick(0.033)
        assert gen._hue_phase > 0
        assert gen._breathing_phase > 0

        gen.reset()
        assert gen._hue_phase == 0.0
        assert gen._breathing_phase == 0.0
        assert gen._wave_phase == 0.0


class TestGenerativeBreathing:
    """Test the breathing brightness oscillation."""

    def test_breathing_oscillates_between_min_and_max(self):
        """Over a full breathing cycle, brightness should span min to max range."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=1.0,  # 1 full cycle per second
            breathing_min=0.2,
            breathing_max=0.8,
            wave_speed=0.0,  # Disable wave to isolate breathing
        )

        # Sample brightness over one full cycle (1 second at 1 Hz)
        brightnesses = []
        for _ in range(100):
            result = gen.tick(0.01)  # 100 ticks * 0.01s = 1s
            _, _, b = result[0]
            brightnesses.append(b)

        min_b = min(brightnesses)
        max_b = max(brightnesses)

        # Breathing should reach close to configured min/max
        # (wave is disabled, so brightness is purely breathing)
        assert min_b < 0.35, f"Min brightness {min_b} should be near breathing_min=0.2"
        assert max_b > 0.55, f"Max brightness {max_b} should be near breathing_max=0.8"

    def test_breathing_at_correct_frequency(self):
        """Breathing at 0.25 Hz should complete one cycle in 4 seconds."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=0.25,
            breathing_min=0.2,
            breathing_max=0.8,
            wave_speed=0.0,
        )

        # After exactly half a cycle (2 seconds at 0.25 Hz), breathing phase = 0.5
        for _ in range(200):
            gen.tick(0.01)

        assert abs(gen._breathing_phase - 0.5) < 0.01, \
            f"After 2s at 0.25Hz, phase should be ~0.5, got {gen._breathing_phase}"

    def test_breathing_min_max_clamped(self):
        """Brightness values should always stay within [0, 1]."""
        gen = GenerativeLayer(
            num_lights=4,
            breathing_min=0.0,
            breathing_max=1.0,
        )

        for _ in range(500):
            result = gen.tick(0.033)
            for _, _, b in result:
                assert 0.0 <= b <= 1.0


class TestGenerativeHueRotation:
    """Test slow hue rotation through the palette."""

    def test_hue_changes_over_time(self):
        """Hue should change as hue_phase advances."""
        gen = GenerativeLayer(
            num_lights=1,
            hue_cycle_period=10.0,  # Full cycle in 10s
        )

        result_start = gen.tick(0.001)
        h_start = result_start[0][0]

        # Advance 5 seconds (half cycle)
        for _ in range(500):
            gen.tick(0.01)

        result_mid = gen.tick(0.001)
        h_mid = result_mid[0][0]

        # Hue should have changed significantly
        diff = abs(h_mid - h_start)
        if diff > 180:
            diff = 360 - diff
        assert diff > 20, f"Hue should change over time, diff was only {diff}"

    def test_hue_phase_wraps(self):
        """Hue phase should wrap around after full cycle."""
        gen = GenerativeLayer(
            num_lights=1,
            hue_cycle_period=1.0,  # Very fast cycle for testing
        )

        # Run for exactly 1 cycle
        for _ in range(100):
            gen.tick(0.01)

        assert gen._hue_phase == pytest.approx(0.0, abs=0.02), \
            f"Phase should wrap to ~0 after full cycle, got {gen._hue_phase}"


class TestGenerativeSpatialWave:
    """Test that spatial wave creates per-light brightness variation."""

    def test_wave_creates_brightness_variation(self):
        """Different lights should have different brightness due to spatial wave."""
        gen = GenerativeLayer(
            num_lights=6,
            wave_speed=0.5,
            breathing_rate_hz=0.001,  # Very slow breathing to isolate wave
        )

        # Advance to let wave phase develop
        result = None
        for _ in range(50):
            result = gen.tick(0.033)

        assert result is not None
        brightnesses = [b for _, _, b in result]
        spread = max(brightnesses) - min(brightnesses)
        assert spread > 0.01, \
            f"Spatial wave should create brightness variation, spread={spread}"

    def test_single_light_no_wave_variation(self):
        """With 1 light, spatial wave has no variation to produce."""
        gen = GenerativeLayer(num_lights=1, wave_speed=1.0)

        # Just verify it doesn't crash
        for _ in range(50):
            result = gen.tick(0.033)
            assert len(result) == 1


class TestGenerativePaletteSync:
    """Test that generative layer uses the engine's palette."""

    def test_set_palette_propagates(self):
        gen = GenerativeLayer(num_lights=4)
        gen.set_palette((120.0, 240.0))
        assert gen._palette == (120.0, 240.0)

    def test_engine_set_palette_syncs_generative(self):
        engine = EffectEngine(num_lights=4)
        new_palette = (100.0, 200.0, 300.0)
        engine.set_palette(new_palette)
        assert engine._generative._palette == new_palette


# ============================================================================
# Blend function tests
# ============================================================================


class TestBlendMaximum:
    """Test the _blend_maximum function."""

    def test_pure_reactive(self):
        """With reactive_weight=1.0, output should be dominated by reactive layer."""
        gen = [(180.0, 0.7, 0.5)]
        react = [(90.0, 0.9, 0.8)]
        result = _blend_maximum(gen, react, reactive_weight=1.0)
        h, s, b = result[0]
        # Brightness: max(0.0 * 0.5, 1.0 * 0.8) = 0.8, plus small additive term
        assert b > 0.7
        # Saturation should be near reactive value
        assert abs(s - 0.9) < 0.05

    def test_pure_generative(self):
        """With reactive_weight=0.0, output should be dominated by generative layer."""
        gen = [(180.0, 0.7, 0.5)]
        react = [(90.0, 0.9, 0.8)]
        result = _blend_maximum(gen, react, reactive_weight=0.0)
        h, s, b = result[0]
        # Brightness: max(1.0 * 0.5, 0.0 * 0.8) = 0.5, plus small additive
        assert b > 0.45
        assert b < 0.6
        # Saturation should be near generative value
        assert abs(s - 0.7) < 0.05

    def test_beat_flash_punches_through(self):
        """When reactive brightness is high (beat flash), it should dominate
        even at moderate reactive weight."""
        gen = [(180.0, 0.7, 0.3)]
        react = [(90.0, 0.9, 1.0)]  # Beat flash at full brightness
        result = _blend_maximum(gen, react, reactive_weight=0.6)
        _, _, b = result[0]
        # max(0.4 * 0.3, 0.6 * 1.0) = 0.6
        assert b > 0.5, f"Beat flash should punch through, got brightness={b}"

    def test_multiple_lights(self):
        gen = [(180.0, 0.7, 0.5), (200.0, 0.6, 0.4)]
        react = [(90.0, 0.9, 0.8), (100.0, 0.8, 0.7)]
        result = _blend_maximum(gen, react, reactive_weight=0.5)
        assert len(result) == 2
        for h, s, b in result:
            assert 0 <= h <= 360
            assert 0 <= s <= 1
            assert 0 <= b <= 1


class TestBlendHue:
    """Test the _blend_hue function."""

    def test_no_secondary_weight(self):
        result = _blend_hue(90.0, 180.0, 0.0)
        assert result == pytest.approx(90.0)

    def test_full_secondary_weight(self):
        result = _blend_hue(90.0, 180.0, 1.0)
        assert result == pytest.approx(180.0)

    def test_halfway(self):
        result = _blend_hue(90.0, 180.0, 0.5)
        assert result == pytest.approx(135.0)

    def test_wraps_around_360(self):
        """Should take shortest path: 350 -> 10 = +20, not -340."""
        result = _blend_hue(350.0, 10.0, 0.5)
        assert result == pytest.approx(0.0, abs=0.1) or result == pytest.approx(360.0, abs=0.1)


# ============================================================================
# Engine-level hybrid model tests
# ============================================================================


class TestHybridSilence:
    """Test that the engine produces beautiful output during silence."""

    def test_silence_produces_nonzero_brightness(self):
        """In complete silence, generative layer should keep lights visible."""
        engine = EffectEngine(num_lights=6)
        features = _silence_features()
        beat = _no_beat()

        # Run for several ticks to let smoothing stabilize
        states = None
        for i in range(60):
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert states is not None
        brightnesses = [s.brightness for s in states]
        assert max(brightnesses) > 0.05, \
            f"In silence, lights should still be visible (generative layer), max={max(brightnesses)}"

    def test_silence_energy_level_near_zero(self):
        """In silence, energy level should drop to near zero."""
        engine = EffectEngine(num_lights=4)
        features = _silence_features()
        beat = _no_beat()

        for i in range(100):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert engine.energy_level < 0.05, \
            f"Energy level should be near 0 in silence, got {engine.energy_level}"

    def test_silence_reactive_weight_near_minimum(self):
        """In silence, reactive weight should be near minimum."""
        engine = EffectEngine(num_lights=4)
        features = _silence_features()
        beat = _no_beat()

        for i in range(100):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        rw = engine.reactive_weight
        assert rw < 0.25, \
            f"Reactive weight in silence should be near min (~0.15), got {rw}"

    def test_silence_lights_change_over_time(self):
        """In silence, lights should slowly change (generative animation)."""
        engine = EffectEngine(num_lights=4)
        features = _silence_features()
        beat = _no_beat()

        # Sample at two different times
        for i in range(30):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)
        states_early = engine.tick(features, beat, dt=0.033, now=1001.0)
        early_brightness = [s.brightness for s in states_early]

        # Advance 5 more seconds
        for i in range(150):
            engine.tick(features, beat, dt=0.033, now=1001.0 + i * 0.033)
        states_late = engine.tick(features, beat, dt=0.033, now=1006.0)
        late_brightness = [s.brightness for s in states_late]

        # At least some lights should have changed brightness
        diffs = [abs(a - b) for a, b in zip(early_brightness, late_brightness)]
        assert max(diffs) > 0.01, \
            "Generative layer should create visible changes over time in silence"


class TestHybridLoud:
    """Test that reactive layer dominates during loud audio."""

    def test_loud_reactive_weight_near_maximum(self):
        """During loud audio, reactive weight should approach maximum."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features(rms=0.7)
        beat = _no_beat()

        for i in range(200):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        rw = engine.reactive_weight
        assert rw > 0.65, \
            f"Reactive weight with loud audio should be near max (~0.85), got {rw}"

    def test_loud_high_brightness(self):
        """During loud audio, lights should be bright (reactive layer drives brightness)."""
        engine = EffectEngine(num_lights=6)
        features = _loud_features(rms=0.7)
        beat = _no_beat()

        states = None
        for i in range(100):
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert states is not None
        brightnesses = [s.brightness for s in states]
        assert max(brightnesses) > 0.3, \
            f"Loud audio should produce bright lights, max={max(brightnesses)}"


class TestHybridBlendTransition:
    """Test smooth transitions between silence and loud passages."""

    def test_transition_from_silence_to_loud(self):
        """Reactive weight should increase smoothly when audio gets loud."""
        engine = EffectEngine(num_lights=4)
        beat = _no_beat()

        # Start in silence
        for i in range(50):
            engine.tick(_silence_features(), beat, dt=0.033, now=1000.0 + i * 0.033)
        rw_silence = engine.reactive_weight

        # Switch to loud
        weights = [rw_silence]
        for i in range(100):
            engine.tick(
                _loud_features(rms=0.8), beat,
                dt=0.033, now=1002.0 + i * 0.033,
            )
            weights.append(engine.reactive_weight)

        # Weight should monotonically increase (with EMA, each step increases)
        # Check that the trend is upward
        assert weights[-1] > weights[0], \
            "Reactive weight should increase as audio gets loud"

        # Should not jump abruptly — check that no single step is too large
        max_step = max(abs(weights[i+1] - weights[i]) for i in range(len(weights)-1))
        assert max_step < 0.15, \
            f"Blend transition should be smooth, max step={max_step}"

    def test_transition_from_loud_to_silence(self):
        """Reactive weight should decrease when audio goes silent."""
        engine = EffectEngine(num_lights=4)
        beat = _no_beat()

        # Start loud
        for i in range(200):
            engine.tick(_loud_features(rms=0.8), beat, dt=0.033, now=1000.0 + i * 0.033)
        rw_loud = engine.reactive_weight

        # Switch to silence
        for i in range(200):
            engine.tick(
                _silence_features(), beat,
                dt=0.033, now=1007.0 + i * 0.033,
            )
        rw_after = engine.reactive_weight

        assert rw_after < rw_loud, \
            f"Reactive weight should decrease in silence: {rw_after} < {rw_loud}"


class TestHybridBeatFlash:
    """Test that beat flashes work correctly in the hybrid model."""

    def test_beat_flash_visible_during_loud(self):
        """Beat flash should produce visible brightness increase."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Stabilize the engine first
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record brightness before beat
        states_before = engine.tick(features, _no_beat(), dt=0.033, now=1002.0)
        max_before = max(s.brightness for s in states_before)

        # Fire a beat
        states_after = engine.tick(features, _beat(strength=0.9), dt=0.033, now=1002.033)
        max_after = max(s.brightness for s in states_after)

        assert max_after > max_before, \
            f"Beat flash should increase brightness: {max_after} > {max_before}"

    def test_beat_flash_visible_during_quiet(self):
        """Beat flash should punch through even when generative dominates."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        features = _silence_features()

        # Run in silence to let generative dominate
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record brightness before beat
        states_before = engine.tick(features, _no_beat(), dt=0.033, now=1002.0)
        max_before = max(s.brightness for s in states_before)

        # Fire a beat with loud features (beat with energy)
        loud = _loud_features(rms=0.6)
        states_after = engine.tick(loud, _beat(strength=0.9), dt=0.033, now=1002.033)
        max_after = max(s.brightness for s in states_after)

        assert max_after > max_before, \
            f"Beat flash should be visible even during quiet: {max_after} > {max_before}"


class TestHybridSafety:
    """Test that safety limiter still works on blended output."""

    def test_flash_rate_limit_preserved(self):
        """Max flash rate should still be enforced after blending."""
        engine = EffectEngine(num_lights=3, max_flash_hz=3.0)
        features = _loud_features()

        # First beat fires
        now = 1000.0
        engine.tick(features, _beat(strength=0.9), dt=0.033, now=now)
        assert any(light.flash_brightness > 0 for light in engine._lights)

        # Clear flash
        for light in engine._lights:
            light.flash_brightness = 0.0

        # Second beat too soon (33ms later, rate limit = 333ms)
        now_2 = now + 0.033
        engine.tick(features, _beat(strength=0.9), dt=0.033, now=now_2)
        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Flash rate limiter should prevent the second flash"

    def test_no_strobe_red_preserved(self):
        """Safety: no strobe saturated red should still work (safe mode)."""
        engine = EffectEngine(num_lights=3, max_flash_hz=10.0)
        engine.set_safe_mode(True)  # Red protection only in safe mode
        # Set palette to pure red
        engine.set_palette((0.0, 5.0, 355.0))

        features = _loud_features()

        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire a strong beat
        states = engine.tick(features, _beat(strength=0.9), dt=0.033, now=1002.0)

        # All lights near red hue with flash should have clamped saturation
        for light in engine._lights:
            if (light.hue < 15 or light.hue > 345) and light.flash_brightness > 0.3:
                assert light.saturation <= 0.71, \
                    f"Red strobe safety failed: sat={light.saturation} with flash={light.flash_brightness}"

    def test_brightness_never_exceeds_one(self):
        """Blended brightness should always be clamped to [0, 1]."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        features = _loud_features(rms=0.9)

        for i in range(100):
            states = engine.tick(
                features,
                _beat(strength=1.0) if i % 10 == 0 else _no_beat(),
                dt=0.033,
                now=1000.0 + i * 0.033,
            )
            for s in states:
                assert 0 <= s.brightness <= 1.0, \
                    f"Brightness out of range: {s.brightness}"


class TestHybridReset:
    """Test that reset clears all state including generative layer."""

    def test_reset_clears_energy_level(self):
        engine = EffectEngine(num_lights=4)
        features = _loud_features()

        for i in range(100):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        assert engine.energy_level > 0.1

        engine.reset()

        assert engine.energy_level == 0.0

    def test_reset_clears_generative_state(self):
        engine = EffectEngine(num_lights=4)

        for i in range(100):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        assert engine._generative._hue_phase > 0

        engine.reset()

        assert engine._generative._hue_phase == 0.0
        assert engine._generative._breathing_phase == 0.0
        assert engine._generative._wave_phase == 0.0

    def test_reset_clears_flash_state(self):
        engine = EffectEngine(num_lights=4, max_flash_hz=10.0)
        features = _loud_features()

        engine.tick(features, _beat(), dt=0.033, now=1000.0)
        assert any(light.flash_brightness > 0 for light in engine._lights)

        engine.reset()

        assert all(light.flash_brightness == 0 for light in engine._lights)


class TestHybridProperties:
    """Test public properties for reactive_weight and energy_level."""

    def test_reactive_weight_range(self):
        """Reactive weight should always be within [min, max] bounds."""
        engine = EffectEngine(num_lights=4)

        # In silence
        for i in range(100):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=1000.0 + i * 0.033)
        rw_silence = engine.reactive_weight
        assert rw_silence >= engine._min_reactive_weight - 0.001
        assert rw_silence <= engine._max_reactive_weight + 0.001

        # Loud
        for i in range(200):
            engine.tick(_loud_features(rms=0.9), _no_beat(), dt=0.033, now=1004.0 + i * 0.033)
        rw_loud = engine.reactive_weight
        assert rw_loud >= engine._min_reactive_weight - 0.001
        assert rw_loud <= engine._max_reactive_weight + 0.001

    def test_energy_level_range(self):
        """Energy level should always be in [0, 1]."""
        engine = EffectEngine(num_lights=4)

        for rms in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            features = _loud_features(rms=rms)
            for i in range(50):
                engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)
            assert 0.0 <= engine.energy_level <= 1.0


class TestHybridGenerativeConfig:
    """Test engine-level generative configuration setters."""

    def test_set_generative_breathing(self):
        engine = EffectEngine(num_lights=4)
        engine.set_generative_breathing(rate_hz=0.5, min_brightness=0.1, max_brightness=0.9)
        assert engine._generative.breathing_rate_hz == 0.5
        assert engine._generative.breathing_min == 0.1
        assert engine._generative.breathing_max == 0.9

    def test_set_generative_breathing_partial(self):
        """Should only update specified parameters."""
        engine = EffectEngine(num_lights=4)
        original_rate = engine._generative.breathing_rate_hz
        engine.set_generative_breathing(min_brightness=0.3)
        assert engine._generative.breathing_min == 0.3
        assert engine._generative.breathing_rate_hz == original_rate

    def test_set_generative_hue_cycle_period(self):
        engine = EffectEngine(num_lights=4)
        engine.set_generative_hue_cycle_period(30.0)
        assert engine._generative.hue_cycle_period == 30.0

    def test_set_generative_hue_cycle_period_clamped(self):
        engine = EffectEngine(num_lights=4)
        engine.set_generative_hue_cycle_period(0.1)  # Too low
        assert engine._generative.hue_cycle_period == 1.0

    def test_engine_constructor_generative_params(self):
        """Constructor should pass generative params through."""
        engine = EffectEngine(
            num_lights=4,
            generative_hue_cycle_period=30.0,
            generative_breathing_rate_hz=0.3,
            generative_breathing_min=0.1,
            generative_breathing_max=0.9,
        )
        assert engine._generative.hue_cycle_period == 30.0
        assert engine._generative.breathing_rate_hz == 0.3
        assert engine._generative.breathing_min == 0.1
        assert engine._generative.breathing_max == 0.9


class TestHybridEndToEnd:
    """Integration-style tests simulating realistic scenarios."""

    def test_five_seconds_of_silence_then_music(self):
        """Simulate silence -> loud music transition. Verify smooth behavior."""
        engine = EffectEngine(num_lights=6)
        dt = 0.033
        now = 1000.0

        # 5 seconds of silence
        for i in range(150):
            states = engine.tick(_silence_features(), _no_beat(), dt=dt, now=now)
            now += dt
            for s in states:
                assert 0 <= s.brightness <= 1.0

        rw_before_music = engine.reactive_weight
        assert rw_before_music < 0.3, \
            "Before music, reactive weight should be low"

        # 5 seconds of loud music with beats
        beat_interval = 60.0 / 128.0
        last_beat = now
        for i in range(150):
            is_beat = (now - last_beat) >= beat_interval
            if is_beat:
                last_beat = now
            beat = _beat(strength=0.8) if is_beat else _no_beat()
            states = engine.tick(_loud_features(rms=0.6), beat, dt=dt, now=now)
            now += dt
            for s in states:
                assert 0 <= s.brightness <= 1.0

        rw_after_music = engine.reactive_weight
        assert rw_after_music > rw_before_music, \
            "After music starts, reactive weight should increase"

    def test_continuous_operation_no_crashes(self):
        """Run the engine for 30 simulated seconds with varying input."""
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        dt = 0.033
        now = 1000.0

        for i in range(900):  # ~30 seconds
            # Varying RMS: silence -> loud -> silence
            phase = (i / 900.0) * 2 * math.pi
            rms = max(0.0, 0.5 + 0.5 * math.sin(phase))
            features = _loud_features(rms=rms) if rms > 0.1 else _silence_features()

            is_beat = (i % 15 == 0) and rms > 0.2
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0


# ============================================================================
# Centroid color mode engine tests (Task 1.4)
# ============================================================================


class TestEngineCentroidMode:
    """Tests for centroid-driven color mode in the EffectEngine."""

    def _bass_features(self, rms: float = 0.5) -> AudioFeatures:
        """Audio features with low spectral centroid (bass-heavy)."""
        bands = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.05, 0.02])
        return AudioFeatures(
            band_energies=bands,
            spectral_centroid=200.0,  # Low frequency -> warm hue
            spectral_flux=10.0,
            spectral_rolloff=1000.0,
            spectral_flatness=0.15,
            rms=rms,
            peak=rms * 1.4,
            spectrum=np.zeros(1024),
        )

    def _treble_features(self, rms: float = 0.5) -> AudioFeatures:
        """Audio features with high spectral centroid (treble-heavy)."""
        bands = np.array([0.1, 0.15, 0.3, 0.5, 0.7, 0.8, 0.6])
        return AudioFeatures(
            band_energies=bands,
            spectral_centroid=6000.0,  # High frequency -> cool hue
            spectral_flux=15.0,
            spectral_rolloff=8000.0,
            spectral_flatness=0.3,
            rms=rms,
            peak=rms * 1.4,
            spectrum=np.zeros(1024),
        )

    def test_set_color_mode_on_engine(self):
        """Engine should expose set_color_mode that delegates to ColorMapper."""
        engine = EffectEngine(num_lights=4)
        assert engine.color_mapper.color_mode == "palette"
        engine.set_color_mode("centroid")
        assert engine.color_mapper.color_mode == "centroid"

    def test_centroid_mode_invalid_mode_ignored(self):
        engine = EffectEngine(num_lights=4)
        engine.set_color_mode("nonsense")
        assert engine.color_mapper.color_mode == "palette"

    def test_centroid_mode_produces_valid_output(self):
        """Engine in centroid mode should produce valid LightStates."""
        engine = EffectEngine(num_lights=6)
        engine.set_color_mode("centroid")

        features = _loud_features(rms=0.6)
        beat = _no_beat()

        for i in range(60):
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert len(states) == 6
        for s in states:
            assert 0 <= s.brightness <= 1.0
            assert 0 <= s.x <= 1.0
            assert 0 <= s.y <= 1.0

    def test_centroid_mode_bass_produces_warm_hues(self):
        """Bass-heavy audio in centroid mode should produce warm light hues."""
        engine = EffectEngine(num_lights=4)
        engine.set_color_mode("centroid")

        bass = self._bass_features(rms=0.6)
        beat = _no_beat()

        for i in range(80):
            engine.tick(bass, beat, dt=0.033, now=1000.0 + i * 0.033)

        # Check per-light hues are in warm range (red/orange/yellow < 90 deg)
        warm_count = sum(
            1 for light in engine._lights
            if light.hue < 90 or light.hue > 330
        )
        assert warm_count >= 2, \
            f"Bass audio in centroid mode should produce warm hues, " \
            f"got hues: {[round(l.hue, 1) for l in engine._lights]}"

    def test_centroid_mode_treble_produces_cool_hues(self):
        """Treble-heavy audio in centroid mode should produce cool light hues."""
        engine = EffectEngine(num_lights=4)
        engine.set_color_mode("centroid")

        treble = self._treble_features(rms=0.6)
        beat = _no_beat()

        for i in range(80):
            engine.tick(treble, beat, dt=0.033, now=1000.0 + i * 0.033)

        # Check per-light hues are in cool range (blue/violet 180-300)
        cool_count = sum(
            1 for light in engine._lights
            if 150 < light.hue < 310
        )
        assert cool_count >= 2, \
            f"Treble audio in centroid mode should produce cool hues, " \
            f"got hues: {[round(l.hue, 1) for l in engine._lights]}"

    def test_centroid_mode_beats_still_flash(self):
        """Beat flash should still work in centroid mode."""
        engine = EffectEngine(num_lights=4, max_flash_hz=10.0)
        engine.set_color_mode("centroid")

        features = _loud_features(rms=0.5)

        # Stabilize
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        before = max(s.brightness for s in engine.tick(features, _no_beat(), dt=0.033, now=1002.0))
        after = max(s.brightness for s in engine.tick(features, _beat(strength=0.9), dt=0.033, now=1002.033))

        assert after > before, "Beat flash should work in centroid mode"

    def test_centroid_mode_continuous_operation(self):
        """Run centroid mode for 15 seconds with varying centroid. No crashes."""
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        engine.set_color_mode("centroid")

        dt = 0.033
        now = 1000.0

        for i in range(450):  # ~15 seconds
            # Vary centroid between bass and treble
            phase = (i / 450.0) * 2 * math.pi
            centroid = 100 + (10000 - 100) * (0.5 + 0.5 * math.sin(phase))
            rms = 0.3 + 0.3 * abs(math.sin(phase * 0.5))

            bands = np.array([0.5, 0.4, 0.3, 0.3, 0.2, 0.15, 0.1])
            features = AudioFeatures(
                band_energies=bands,
                spectral_centroid=centroid,
                spectral_flux=10.0,
                spectral_rolloff=centroid * 1.5,
                spectral_flatness=0.2,
                rms=rms,
                peak=rms * 1.3,
                spectrum=np.zeros(1024),
            )

            is_beat = (i % 15 == 0) and rms > 0.3
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0
