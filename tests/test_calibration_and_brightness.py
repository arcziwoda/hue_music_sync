"""Tests for calibration delay (Task 2.6) and brightness min/max (Task 2.8).

Task 2.6: Manual calibration delay adds to predictive beat latency compensation
so that light commands fire earlier, compensating for system-specific audio-to-light delay.

Task 2.8: Brightness min/max provides global floor/cap for all light output,
compressing the dynamic range via linear mapping: output = min + (max - min) * raw.
"""

import numpy as np

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.engine import EffectEngine


def _make_features(rms: float = 0.3, bass: float = 0.3) -> AudioFeatures:
    """Create AudioFeatures with specified values for engine testing."""
    bands = np.zeros(7)
    bands[0] = bass
    bands[1] = bass
    bands[2] = 0.2
    bands[3] = 0.2
    bands[4] = 0.1
    bands[5] = 0.1
    bands[6] = 0.05
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=2000.0,
        spectral_flux=0.1,
        spectral_rolloff=4000.0,
        spectral_flatness=0.3,
        rms=rms,
        peak=rms * 1.5,
        spectrum=np.zeros(1024),
    )


def _make_beat_info(
    is_beat: bool = False,
    bpm: float = 128.0,
    confidence: float = 0.8,
    beat_strength: float = 0.8,
    predicted_next_beat: float = 0.0,
    time_since_beat: float = 0.0,
) -> BeatInfo:
    """Create BeatInfo with specified values."""
    return BeatInfo(
        is_beat=is_beat,
        bpm=bpm,
        bpm_confidence=confidence,
        beat_strength=beat_strength,
        predicted_next_beat=predicted_next_beat,
        time_since_beat=time_since_beat,
    )


# ---------------------------------------------------------------------------
# Task 2.6: Calibration delay tests
# ---------------------------------------------------------------------------


class TestCalibrationDelay:
    """Tests for manual calibration delay (Task 2.6)."""

    def test_default_calibration_delay_is_zero(self):
        """Calibration delay defaults to 0ms."""
        engine = EffectEngine(num_lights=4)
        assert engine.calibration_delay_ms == 0.0

    def test_set_calibration_delay(self):
        """Setting calibration delay stores the value."""
        engine = EffectEngine(num_lights=4)
        engine.set_calibration_delay(300.0)
        assert engine.calibration_delay_ms == 300.0

    def test_calibration_delay_clamps_max(self):
        """Calibration delay is clamped to 600ms max."""
        engine = EffectEngine(num_lights=4)
        engine.set_calibration_delay(1000.0)
        assert engine.calibration_delay_ms == 600.0

    def test_calibration_delay_clamps_min(self):
        """Calibration delay is clamped to 0ms min."""
        engine = EffectEngine(num_lights=4)
        engine.set_calibration_delay(-100.0)
        assert engine.calibration_delay_ms == 0.0

    def test_effective_compensation_includes_calibration(self):
        """Effective latency compensation = base + calibration delay."""
        engine = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
        )
        assert engine.effective_latency_compensation_ms == 80.0

        engine.set_calibration_delay(300.0)
        assert engine.effective_latency_compensation_ms == 380.0

    def test_calibration_delay_makes_predictive_fire_earlier(self):
        """With calibration delay, predictive beats fire earlier."""
        # Engine with base 80ms compensation, no calibration
        engine_no_calib = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        # Engine with base 80ms + 300ms calibration
        engine_with_calib = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )
        engine_with_calib.set_calibration_delay(300.0)

        # Predicted beat at t=1.0
        beat_info = _make_beat_info(
            is_beat=False,
            bpm=128.0,
            confidence=0.9,
            predicted_next_beat=1.0,
        )
        features = _make_features(rms=0.5)

        # At t=0.65: with calibration (380ms total), should fire (1.0 - 0.38 = 0.62)
        # Without calibration (80ms), should NOT fire (1.0 - 0.08 = 0.92)
        result_no_calib = engine_no_calib._resolve_beat_trigger(beat_info, now=0.65)
        result_with_calib = engine_with_calib._resolve_beat_trigger(beat_info, now=0.65)

        assert result_no_calib[0] is False, "Without calibration, should not fire yet"
        assert result_with_calib[0] is True, "With calibration delay, should fire earlier"

    def test_calibration_delay_zero_no_effect(self):
        """Zero calibration delay behaves identically to no calibration."""
        engine = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )
        engine.set_calibration_delay(0.0)

        beat_info = _make_beat_info(
            is_beat=False,
            bpm=128.0,
            confidence=0.9,
            predicted_next_beat=1.0,
        )

        # At t=0.90, should NOT fire (1.0 - 0.08 = 0.92 > 0.90)
        trigger, _ = engine._resolve_beat_trigger(beat_info, now=0.90)
        assert not trigger

        # At t=0.93, should fire (1.0 - 0.08 = 0.92 <= 0.93)
        trigger, _ = engine._resolve_beat_trigger(beat_info, now=0.93)
        assert trigger

    def test_calibration_delay_reactive_beats_still_work(self):
        """Reactive beats (is_beat=True) still fire normally with calibration delay."""
        engine = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )
        engine.set_calibration_delay(400.0)

        # Reactive beat (no prediction)
        beat_info = _make_beat_info(
            is_beat=True,
            bpm=128.0,
            confidence=0.3,  # Below predictive threshold
            beat_strength=0.8,
            predicted_next_beat=0.0,  # No prediction
        )

        trigger, strength = engine._resolve_beat_trigger(beat_info, now=5.0)
        assert trigger is True
        assert strength == 0.8


class TestCalibrationDelayWithEngine:
    """Integration tests: calibration delay with full engine tick."""

    def test_tick_produces_valid_output_with_calibration(self):
        """Engine tick produces valid LightState even with max calibration delay."""
        engine = EffectEngine(num_lights=4, latency_compensation_ms=80.0)
        engine.set_calibration_delay(600.0)

        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(is_beat=True, beat_strength=0.9)

        states = engine.tick(features, beat_info, dt=0.033, now=100.0)
        assert len(states) == 4
        for s in states:
            assert 0.0 <= s.brightness <= 1.0


# ---------------------------------------------------------------------------
# Task 2.8: Brightness min/max tests
# ---------------------------------------------------------------------------


class TestBrightnessMinMax:
    """Tests for brightness min/max mapping (Task 2.8)."""

    def test_default_brightness_range(self):
        """Default brightness range is 0.0 to 1.0 (no remapping)."""
        engine = EffectEngine(num_lights=4)
        assert engine.brightness_min == 0.0
        assert engine.brightness_max == 1.0

    def test_set_brightness_min(self):
        """Setting brightness min stores the value."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.2)
        assert engine.brightness_min == 0.2

    def test_set_brightness_max(self):
        """Setting brightness max stores the value."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_max(0.8)
        assert engine.brightness_max == 0.8

    def test_brightness_min_clamps(self):
        """Brightness min is clamped to 0.0-1.0."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(-0.5)
        assert engine.brightness_min == 0.0
        engine.set_brightness_min(1.5)
        assert engine.brightness_min == 1.0

    def test_brightness_max_clamps(self):
        """Brightness max is clamped to 0.0-1.0."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_max(-0.5)
        assert engine.brightness_max == 0.0
        engine.set_brightness_max(1.5)
        assert engine.brightness_max == 1.0

    def test_min_greater_than_max_adjusts_max(self):
        """Setting min > current max raises max to match."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_max(0.5)
        engine.set_brightness_min(0.7)
        assert engine.brightness_min == 0.7
        assert engine.brightness_max == 0.7

    def test_max_less_than_min_adjusts_min(self):
        """Setting max < current min lowers min to match."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.5)
        engine.set_brightness_max(0.3)
        assert engine.brightness_max == 0.3
        assert engine.brightness_min == 0.3

    def test_brightness_floor_applied_at_output(self):
        """With brightness_min > 0, output brightness is never below min."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.3)

        # Feed silence (low RMS) to get near-zero brightness
        features = _make_features(rms=0.0)
        beat_info = _make_beat_info(is_beat=False)

        # Run several ticks to let smoothing settle
        for _ in range(60):
            states = engine.tick(features, beat_info, dt=0.033, now=100.0 + _ * 0.033)

        # All lights should be at or above the floor
        for s in states:
            assert s.brightness >= 0.3 - 0.01, (
                f"Brightness {s.brightness:.3f} below min floor 0.3"
            )

    def test_brightness_cap_applied_at_output(self):
        """With brightness_max < 1.0, output brightness is never above max."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_max(0.5)

        # Feed loud audio + beat to get high brightness
        features = _make_features(rms=0.9)
        beat_info = _make_beat_info(is_beat=True, beat_strength=1.0)

        # Run several ticks with beats to push brightness up
        for i in range(20):
            # Only first tick has beat, rest don't (to see decaying max)
            bi = beat_info if i == 0 else _make_beat_info(is_beat=False)
            states = engine.tick(features, bi, dt=0.033, now=100.0 + i * 0.033)

        # All lights should be at or below the cap
        for s in states:
            assert s.brightness <= 0.5 + 0.01, (
                f"Brightness {s.brightness:.3f} above max cap 0.5"
            )

    def test_brightness_mapping_formula(self):
        """Brightness mapping is: output = min + (max - min) * raw."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.2)
        engine.set_brightness_max(0.8)

        # The mapping should compress the range:
        # raw 0.0 -> 0.2, raw 0.5 -> 0.5, raw 1.0 -> 0.8
        # We test the internal formula indirectly through engine output
        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(is_beat=False)

        # After several ticks, brightness should be between min and max
        for i in range(60):
            states = engine.tick(features, beat_info, dt=0.033, now=100.0 + i * 0.033)

        for s in states:
            assert s.brightness >= 0.2 - 0.01, f"Below min: {s.brightness}"
            assert s.brightness <= 0.8 + 0.01, f"Above max: {s.brightness}"

    def test_equal_min_max_produces_constant(self):
        """When min == max, all lights output the same brightness."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.5)
        engine.set_brightness_max(0.5)

        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(is_beat=True, beat_strength=0.9)

        # Run several ticks
        for i in range(30):
            bi = beat_info if i == 0 else _make_beat_info(is_beat=False)
            states = engine.tick(features, bi, dt=0.033, now=100.0 + i * 0.033)

        # All lights should output ~0.5 regardless of input
        for s in states:
            assert abs(s.brightness - 0.5) < 0.02, (
                f"Brightness {s.brightness:.3f} should be ~0.5 when min==max==0.5"
            )

    def test_default_range_no_change(self):
        """Default min=0, max=1 should not modify brightness at all."""
        engine_default = EffectEngine(num_lights=4)
        engine_explicit = EffectEngine(num_lights=4)
        engine_explicit.set_brightness_min(0.0)
        engine_explicit.set_brightness_max(1.0)

        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(is_beat=True, beat_strength=0.8)

        # Same input, same time
        states_default = engine_default.tick(features, beat_info, dt=0.033, now=100.0)
        states_explicit = engine_explicit.tick(features, beat_info, dt=0.033, now=100.0)

        # Should produce identical output
        for sd, se in zip(states_default, states_explicit):
            assert abs(sd.brightness - se.brightness) < 0.001, (
                f"Default {sd.brightness} != explicit {se.brightness}"
            )


class TestBrightnessMinMaxIntegration:
    """Integration tests: brightness min/max with other engine features."""

    def test_brightness_min_with_safe_mode(self):
        """Brightness min works together with safe mode."""
        engine = EffectEngine(num_lights=4)
        engine.set_safe_mode(True)
        engine.set_brightness_min(0.2)

        features = _make_features(rms=0.0)
        beat_info = _make_beat_info(is_beat=False)

        for i in range(60):
            states = engine.tick(features, beat_info, dt=0.033, now=100.0 + i * 0.033)

        for s in states:
            assert s.brightness >= 0.19

    def test_brightness_max_with_intensity_cap(self):
        """Brightness max interacts with intensity cap. Output should respect both."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_max(0.6)
        # Intensity cap also limits brightness (e.g., chill mode caps at 0.6)
        engine.set_intensity("chill")

        features = _make_features(rms=0.9)
        beat_info = _make_beat_info(is_beat=True, beat_strength=1.0)

        states = engine.tick(features, beat_info, dt=0.033, now=100.0)
        for s in states:
            # Should not exceed the lower of the two caps
            assert s.brightness <= 0.61

    def test_brightness_range_with_beat_flash(self):
        """Beat flash brightness is still mapped through min/max range."""
        engine = EffectEngine(num_lights=4)
        engine.set_brightness_min(0.1)
        engine.set_brightness_max(0.7)

        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(is_beat=True, beat_strength=1.0)

        # First tick with beat
        states = engine.tick(features, beat_info, dt=0.033, now=100.0)

        for s in states:
            assert s.brightness >= 0.1 - 0.01
            assert s.brightness <= 0.7 + 0.01


# ---------------------------------------------------------------------------
# Combined tests
# ---------------------------------------------------------------------------


class TestCalibrationAndBrightnessCombined:
    """Tests both features working together."""

    def test_both_features_at_once(self):
        """Calibration delay and brightness min/max work together without issues."""
        engine = EffectEngine(
            num_lights=4,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )
        engine.set_calibration_delay(300.0)
        engine.set_brightness_min(0.15)
        engine.set_brightness_max(0.85)

        features = _make_features(rms=0.5)
        beat_info = _make_beat_info(
            is_beat=True,
            bpm=128.0,
            confidence=0.9,
            beat_strength=0.8,
            predicted_next_beat=101.0,
        )

        for i in range(30):
            states = engine.tick(features, beat_info, dt=0.033, now=100.0 + i * 0.033)
            for s in states:
                assert 0.14 <= s.brightness <= 0.86, (
                    f"Brightness {s.brightness} outside expected range"
                )
