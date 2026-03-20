"""Tests for predictive beat triggering (Task 0.4).

Verifies that EffectEngine fires light commands early when PLL confidence is
high, falls back to reactive when confidence is low, and prevents double-
triggering when both predictive and reactive beats fire for the same period.
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


class TestPredictiveTriggerBasic:
    """Test that predictive triggering fires beats early."""

    def test_predictive_fires_before_beat(self):
        """Engine should fire a flash when now >= predicted_next_beat - latency_comp."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        # Simulate: predicted beat at now + 0.05s, compensation is 0.08s
        # So fire_at = predicted - 0.08 = now - 0.03, meaning NOW > fire_at => should fire
        now = 1000.0
        predicted = now + 0.05  # Beat expected 50ms from now

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,  # No reactive beat yet
            confidence=0.8,
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        # Check that flash was triggered (flash_brightness > 0 on lights)
        assert any(light.flash_brightness > 0 for light in engine._lights), \
            "Predictive trigger should fire a flash when within latency window"

    def test_predictive_does_not_fire_too_early(self):
        """Engine should NOT fire if predicted beat is too far in the future."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        predicted = now + 0.5  # Beat expected 500ms from now — way too early

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Should not fire predictive beat when predicted beat is far in the future"

    def test_predictive_advances_rotation(self):
        """Predictive beat should advance the palette rotation just like reactive beats."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        initial_beat_count = engine._beat_count

        now = 1000.0
        predicted = now + 0.05

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert engine._beat_count == initial_beat_count + 1, \
            "Predictive beat should advance the beat counter"


class TestConfidenceGating:
    """Test that predictive triggering respects confidence threshold."""

    def test_low_confidence_falls_back_to_reactive(self):
        """With low PLL confidence, predictive trigger should NOT fire."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.6,
        )

        now = 1000.0
        predicted = now + 0.05  # Within firing window

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.3,  # Below threshold
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Low confidence should prevent predictive triggering"

    def test_reactive_still_works_at_low_confidence(self):
        """With low confidence, reactive beats should still fire normally."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.6,
        )

        now = 1000.0
        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=True,
            confidence=0.3,  # Below predictive threshold
            beat_strength=0.9,
            predicted_next_beat=0.0,  # No prediction
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert any(light.flash_brightness > 0 for light in engine._lights), \
            "Reactive beats should work regardless of confidence threshold"

    def test_confidence_at_exact_threshold(self):
        """At exactly the threshold, predictive triggering should work."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.6,
        )

        now = 1000.0
        predicted = now + 0.05

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.6,  # Exactly at threshold
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert any(light.flash_brightness > 0 for light in engine._lights), \
            "At exactly the confidence threshold, predictive should fire"


class TestDoubleTriggerPrevention:
    """Test that the same beat period doesn't trigger both predictive and reactive."""

    def test_reactive_skipped_after_predictive(self):
        """If predictive trigger fired, a subsequent reactive beat should be suppressed."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        predicted = now + 0.05  # Within latency window

        features = _make_features()

        # Tick 1: Predictive fires
        beat_info_1 = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=predicted,
        )
        engine.tick(features, beat_info_1, dt=0.033, now=now)

        # Verify predictive fired
        assert any(light.flash_brightness > 0 for light in engine._lights)
        beat_count_after_predictive = engine._beat_count

        # Let flash decay a bit so we can detect a new flash
        for light in engine._lights:
            light.flash_brightness = 0.0

        # Tick 2: Reactive beat arrives for the same predicted beat (~50ms later)
        now_2 = now + 0.05
        beat_info_2 = _make_beat_info(
            is_beat=True,
            confidence=0.8,
            beat_strength=0.9,
            predicted_next_beat=predicted,  # Same prediction
        )
        engine.tick(features, beat_info_2, dt=0.033, now=now_2)

        # The reactive beat should be suppressed — no new flash
        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Reactive beat should be suppressed when predictive already fired for this period"
        assert engine._beat_count == beat_count_after_predictive, \
            "Beat count should not advance again for the same beat period"

    def test_new_beat_fires_after_predictive_period_passes(self):
        """After the predicted beat period passes, a new beat should fire normally."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        predicted_1 = now + 0.05

        features = _make_features()

        # Tick 1: Predictive fires for beat 1
        beat_info_1 = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=predicted_1,
        )
        engine.tick(features, beat_info_1, dt=0.033, now=now)
        assert any(light.flash_brightness > 0 for light in engine._lights)

        # Clear flash state
        for light in engine._lights:
            light.flash_brightness = 0.0

        # Tick 2: New prediction for next beat (different predicted_next_beat)
        predicted_2 = predicted_1 + 0.469  # 128 BPM = ~469ms between beats
        now_2 = now + 0.033  # Just one tick later
        beat_info_2 = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=predicted_2,
        )

        # fire_at = predicted_2 - 0.08 = now + 0.05 + 0.469 - 0.08 = now + 0.439
        # now_2 = now + 0.033, which is < fire_at, so no fire yet.
        engine.tick(features, beat_info_2, dt=0.033, now=now_2)

        # The predictive_beat_fired flag should be reset for the new period
        assert not engine._predictive_beat_fired, \
            "Predictive state should reset when a new beat period starts"


class TestDisabledPrediction:
    """Test behavior when prediction is disabled."""

    def test_zero_latency_comp_disables_prediction(self):
        """With latency_compensation_ms=0, only reactive beats should work."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=0.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        predicted = now + 0.05

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.9,
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "With zero latency compensation, predictive triggering should be disabled"

    def test_no_prediction_timestamp_uses_reactive(self):
        """Without predicted_next_beat, reactive beats work normally."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=True,
            confidence=0.9,
            beat_strength=0.8,
            predicted_next_beat=0.0,  # No prediction available
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        assert any(light.flash_brightness > 0 for light in engine._lights), \
            "Reactive beats should work when no prediction is available"


class TestPredictiveBeatStrength:
    """Test beat strength handling for predictive triggers."""

    def test_predictive_uses_default_strength_when_no_strength(self):
        """When beat_strength is 0, predictive trigger uses a reasonable default."""
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        predicted = now + 0.05

        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            beat_strength=0.0,  # No strength info yet
            predicted_next_beat=predicted,
        )

        engine.tick(features, beat_info, dt=0.033, now=now)

        # Should still fire with default strength (0.7)
        max_flash = max(light.flash_brightness for light in engine._lights)
        assert max_flash > 0.5, \
            f"Predictive beat should use default strength ~0.7, got {max_flash}"


class TestSafetyWithPredictive:
    """Test that safety limits still apply to predictive beats."""

    def test_flash_rate_limit_applies_to_predictive(self):
        """Max flash rate should still limit predictive triggers."""
        engine = EffectEngine(
            num_lights=3,
            max_flash_hz=3.0,  # Max 3 flashes per second
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        now = 1000.0
        features = _make_features()

        # Fire first predictive beat
        beat_info_1 = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=now + 0.05,
        )
        engine.tick(features, beat_info_1, dt=0.033, now=now)
        assert any(light.flash_brightness > 0 for light in engine._lights)

        # Immediately try to fire another (different predicted beat, within rate limit)
        for light in engine._lights:
            light.flash_brightness = 0.0

        now_2 = now + 0.033  # Just one tick later
        beat_info_2 = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=now + 0.15,  # Different beat, but too soon
        )
        engine.tick(features, beat_info_2, dt=0.033, now=now_2)

        # Flash rate limiter should prevent the second flash
        # (min interval = 1/3 = 333ms, but we're only ~33ms later)
        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Flash rate limiter should still apply to predictive beats"


class TestSetterMethods:
    """Test the new setter methods for configuration."""

    def test_set_latency_compensation(self):
        engine = EffectEngine(num_lights=3, latency_compensation_ms=80.0)
        assert abs(engine._latency_compensation_sec - 0.08) < 1e-6

        engine.set_latency_compensation(120.0)
        assert abs(engine._latency_compensation_sec - 0.12) < 1e-6

        engine.set_latency_compensation(0.0)
        assert engine._latency_compensation_sec == 0.0

        # Negative should be clamped to 0
        engine.set_latency_compensation(-50.0)
        assert engine._latency_compensation_sec == 0.0

    def test_set_predictive_confidence_threshold(self):
        engine = EffectEngine(num_lights=3, predictive_confidence_threshold=0.6)
        assert engine._predictive_confidence_threshold == 0.6

        engine.set_predictive_confidence_threshold(0.8)
        assert engine._predictive_confidence_threshold == 0.8

        # Clamp to [0, 1]
        engine.set_predictive_confidence_threshold(1.5)
        assert engine._predictive_confidence_threshold == 1.0

        engine.set_predictive_confidence_threshold(-0.1)
        assert engine._predictive_confidence_threshold == 0.0


class TestResetClearsPredictiveState:
    """Test that reset() clears all predictive state."""

    def test_reset_clears_predictive_state(self):
        engine = EffectEngine(
            num_lights=3,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        # Trigger a predictive beat to set state
        now = 1000.0
        features = _make_features()
        beat_info = _make_beat_info(
            is_beat=False,
            confidence=0.8,
            predicted_next_beat=now + 0.05,
        )
        engine.tick(features, beat_info, dt=0.033, now=now)

        assert engine._predictive_beat_fired
        assert engine._last_predictive_beat_target > 0

        engine.reset()

        assert not engine._predictive_beat_fired
        assert engine._last_predictive_beat_target == 0.0


class TestEndToEndPredictive:
    """Integration-style tests simulating real beat patterns."""

    def test_steady_128bpm_predictive_beats(self):
        """Simulate steady 128 BPM with prediction, verify beats fire early."""
        engine = EffectEngine(
            num_lights=6,
            latency_compensation_ms=80.0,
            predictive_confidence_threshold=0.5,
        )

        bpm = 128.0
        beat_period = 60.0 / bpm  # ~0.469s
        dt = 0.033  # ~30 Hz tick rate

        predictive_fires = 0
        reactive_fires = 0
        total_ticks = 0

        # Simulate 5 seconds of 128 BPM using a fixed base time.
        # We pass simulated `now` to tick() so the engine sees consistent time.
        base_time = 1000.0  # Arbitrary large base to avoid zero-time edge cases
        t = 0.0
        next_beat = beat_period  # First beat at t=beat_period
        last_beat_t = 0.0

        while t < 5.0:
            now_sim = base_time + t
            is_beat = False
            beat_strength = 0.0

            # Check if a reactive beat happens at this tick
            if t >= next_beat:
                is_beat = True
                beat_strength = 0.8
                last_beat_t = t
                next_beat += beat_period

            predicted_next = last_beat_t + beat_period if last_beat_t > 0 else 0.0
            # Convert to simulated monotonic time
            predicted_next_mono = base_time + predicted_next if predicted_next > 0 else 0.0

            features = _make_features()
            beat_info = _make_beat_info(
                is_beat=is_beat,
                bpm=bpm,
                confidence=0.85,
                beat_strength=beat_strength,
                predicted_next_beat=predicted_next_mono,
                time_since_beat=t - last_beat_t if last_beat_t > 0 else 0.0,
            )

            # Track flash state before tick
            had_flash_before = any(light.flash_brightness > 0.5 for light in engine._lights)

            engine.tick(features, beat_info, dt=dt, now=now_sim)
            total_ticks += 1

            # Check if a new flash was triggered
            has_flash_now = any(light.flash_brightness > 0.5 for light in engine._lights)
            if has_flash_now and not had_flash_before:
                if not is_beat:
                    predictive_fires += 1
                else:
                    reactive_fires += 1

            t += dt

        # With 5 seconds at 128 BPM, expect ~10 beats.
        # Most should be predictive since confidence is high.
        total_fires = predictive_fires + reactive_fires
        assert total_fires >= 5, \
            f"Expected at least 5 beat fires in 5s at 128 BPM, got {total_fires}"
        assert predictive_fires > 0, \
            f"Expected some predictive fires, got 0 (reactive={reactive_fires})"
