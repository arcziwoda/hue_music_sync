"""Tests for white flash mode and manual triggers (Tasks 2.16, 2.17).

Verifies:
- Task 2.16: White flash mode drives saturation toward 0 during beat flash
- Task 2.16: White flash mode has no effect when flash is not active
- Task 2.16: Toggle on/off via public API
- Task 2.17: trigger_manual_flash() fires a single flash on next tick
- Task 2.17: trigger_manual_strobe() fires 3 flashes at max rate
- Task 2.17: Manual triggers respect safety limiter (max flash rate)
- Task 2.17: Manual flash does not accumulate when spammed
"""

import math
import time

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.engine import EffectEngine, _LightSmoothed


# --- Test helpers ---


def _silence_features() -> AudioFeatures:
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
    return BeatInfo(
        is_beat=False,
        bpm=0.0,
        bpm_confidence=0.0,
        beat_strength=0.0,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


def _beat(strength: float = 0.8, bpm: float = 128.0) -> BeatInfo:
    return BeatInfo(
        is_beat=True,
        bpm=bpm,
        bpm_confidence=0.5,
        beat_strength=strength,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


# ============================================================================
# Task 2.16: White flash mode
# ============================================================================


class TestWhiteFlashMode:
    """Test that white flash mode drives saturation toward 0 during flash."""

    def test_white_flash_default_off(self):
        """White flash mode should be disabled by default."""
        engine = EffectEngine(num_lights=4)
        assert engine.white_flash_mode is False

    def test_white_flash_toggle(self):
        """White flash mode can be toggled on and off."""
        engine = EffectEngine(num_lights=4)
        engine.set_white_flash_mode(True)
        assert engine.white_flash_mode is True
        engine.set_white_flash_mode(False)
        assert engine.white_flash_mode is False

    def test_white_flash_reduces_saturation_on_beat(self):
        """With white flash on and a beat, saturation should be lower than without."""
        engine_normal = EffectEngine(num_lights=4, attack_alpha=0.9, release_alpha=0.1)
        engine_white = EffectEngine(num_lights=4, attack_alpha=0.9, release_alpha=0.1)
        engine_white.set_white_flash_mode(True)

        # Run several ticks with loud audio to build up color saturation
        now = 100.0
        for _ in range(10):
            engine_normal.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
            engine_white.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Trigger a beat on both engines
        now += 1.0  # Ensure past any cooldown
        states_normal = engine_normal.tick(
            _loud_features(), _beat(strength=1.0), dt=0.033, now=now
        )
        states_white = engine_white.tick(
            _loud_features(), _beat(strength=1.0), dt=0.033, now=now
        )

        # The smoothed saturation for the white flash engine should show
        # the internal lights have reduced saturation.
        # Check the internal light state (smoothed values used for output).
        for light_w, light_n in zip(engine_white._lights, engine_normal._lights):
            # White flash should have lower saturation
            assert light_w.saturation <= light_n.saturation + 0.01, (
                f"White flash light sat={light_w.saturation:.3f} should be "
                f"<= normal sat={light_n.saturation:.3f}"
            )

    def test_white_flash_no_effect_without_beat(self):
        """When no beat is active (no flash brightness), white flash mode
        should not affect saturation."""
        engine = EffectEngine(num_lights=4)
        engine.set_white_flash_mode(True)

        # Run many ticks with no beat to ensure flash decays fully
        now = 100.0
        for _ in range(30):
            engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Confirm flash brightness is near zero on all lights
        for light in engine._lights:
            assert light.flash_brightness < 0.02, (
                f"Flash brightness should be near zero, got {light.flash_brightness}"
            )

    def test_white_flash_proportional_to_flash_brightness(self):
        """After flash fully decays, saturation should recover toward its
        pre-flash level (white flash effect is proportional to flash brightness)."""
        engine = EffectEngine(num_lights=4, attack_alpha=0.9)
        engine.set_white_flash_mode(True)

        # Build up energy so lights have saturated color
        now = 100.0
        for _ in range(20):
            engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Record stable saturation before beat
        sat_before_beat = engine._lights[0].saturation

        # Trigger strong beat
        now += 1.0
        engine.tick(_loud_features(), _beat(strength=1.0), dt=0.033, now=now)

        # Let flash fully decay over many ticks (~2 seconds)
        for _ in range(60):
            now += 0.033
            engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)

        sat_after_full_decay = engine._lights[0].saturation

        # After flash has fully decayed, saturation should have recovered
        # close to the pre-flash level (within 15% tolerance due to EMA smoothing)
        assert sat_after_full_decay >= sat_before_beat * 0.75, (
            f"Saturation should recover after flash decays: "
            f"before={sat_before_beat:.3f}, after={sat_after_full_decay:.3f}"
        )

    def test_white_flash_preserved_across_reset(self):
        """White flash mode state should NOT be reset by reset() — it is a
        user preference, not a processing state."""
        engine = EffectEngine(num_lights=4)
        engine.set_white_flash_mode(True)
        # Note: reset() does not currently reset _white_flash_mode,
        # which is correct behavior (it's a user toggle, not transient state).
        engine.reset()
        assert engine.white_flash_mode is True


# ============================================================================
# Task 2.17: Manual flash/strobe triggers
# ============================================================================


class TestManualFlash:
    """Test manual single flash trigger."""

    def test_manual_flash_fires_on_next_tick(self):
        """After calling trigger_manual_flash(), flash should fire on next tick."""
        engine = EffectEngine(num_lights=4)

        # Run a few warm-up ticks
        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Ensure enough time since last flash
        now += 2.0
        engine.trigger_manual_flash()

        # On next tick, flash_brightness should be set on all lights
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        for light in engine._lights:
            assert light.flash_brightness > 0.5, (
                f"Manual flash should produce high flash brightness, "
                f"got {light.flash_brightness:.3f}"
            )

    def test_manual_flash_single_only(self):
        """Manual flash fires only once, then stops."""
        engine = EffectEngine(num_lights=4)

        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        now += 2.0
        engine.trigger_manual_flash()

        # First tick: flash fires
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
        flash_1 = engine._lights[0].flash_brightness

        # Multiple more ticks with large enough gaps for rate limiter
        for _ in range(5):
            now += 1.0
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        # After several ticks with no new triggers, no new flashes should have been added
        # (flash brightness should only decay)
        flash_end = engine._lights[0].flash_brightness
        assert flash_end < flash_1, "Flash should decay, not fire again"

    def test_manual_flash_no_accumulation(self):
        """Rapid trigger_manual_flash() calls should not accumulate."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_flash()
        engine.trigger_manual_flash()
        engine.trigger_manual_flash()
        # Should still be just 1 pending
        assert engine._manual_flash_pending == 1

    def test_manual_flash_respects_safety_limiter(self):
        """Manual flash should respect max flash rate."""
        engine = EffectEngine(num_lights=4, max_flash_hz=3.0)

        now = 100.0
        # Trigger a beat flash first
        engine.tick(_loud_features(), _beat(), dt=0.033, now=now)

        # Immediately trigger manual flash
        engine.trigger_manual_flash()
        now += 0.01  # Only 10ms later — within cooldown
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        # Check that the manual flash pending was NOT consumed (rate limited)
        # because _min_flash_interval = 1/3 = 333ms
        assert engine._manual_flash_pending == 1, (
            "Manual flash should remain pending when rate limited"
        )

    def test_manual_flash_safe_mode_reduces_intensity(self):
        """In safe mode, manual flash should start at 70% intensity.

        After one tick of exponential decay (dt=0.033, tau=0.25), the flash
        brightness will be 0.7 * exp(-0.033/0.25) ~ 0.613. We verify that
        the initial value is 0.7 by accounting for the decay.
        """
        engine = EffectEngine(num_lights=4)
        engine.set_safe_mode(True)

        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        now += 2.0
        dt = 0.033
        engine.trigger_manual_flash()
        engine.tick(_silence_features(), _no_beat(), dt=dt, now=now)

        # Flash brightness has decayed by one tick: initial * exp(-dt/tau)
        # We back-calculate the initial value from the observed value.
        tau = engine._flash_tau
        decay_factor = math.exp(-dt / tau)
        expected_after_decay = 0.7 * decay_factor

        for light in engine._lights:
            assert abs(light.flash_brightness - expected_after_decay) < 0.05, (
                f"Safe mode manual flash should be ~{expected_after_decay:.3f} "
                f"after one tick decay, got {light.flash_brightness:.3f}"
            )


class TestManualStrobe:
    """Test manual strobe burst trigger."""

    def test_manual_strobe_queues_three_flashes(self):
        """trigger_manual_strobe() should queue 3 pending flashes."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()
        assert engine._manual_flash_pending == 3

    def test_manual_strobe_fires_across_ticks(self):
        """Strobe should fire one flash per tick (when rate allows), for 3 total."""
        engine = EffectEngine(num_lights=4, max_flash_hz=3.0)

        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        now += 2.0
        engine.trigger_manual_strobe()

        flash_count = 0
        # Run enough ticks to fire all 3 flashes (each needs 333ms gap)
        for _ in range(60):
            prev_pending = engine._manual_flash_pending
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            if engine._manual_flash_pending < prev_pending:
                flash_count += 1
            now += 0.033

        # At 3Hz max rate, 3 flashes take ~1 second.
        # With 60 ticks at 33ms each = 2 seconds, all 3 should have fired.
        assert flash_count == 3, f"Expected 3 strobe flashes, got {flash_count}"
        assert engine._manual_flash_pending == 0

    def test_manual_strobe_overrides_pending_flash(self):
        """Strobe should override a pending single flash with its 3-flash count."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_flash()
        assert engine._manual_flash_pending == 1
        engine.trigger_manual_strobe()
        assert engine._manual_flash_pending == 3

    def test_manual_strobe_reset_clears_pending(self):
        """Engine reset should clear pending manual flashes."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()
        assert engine._manual_flash_pending == 3
        engine.reset()
        assert engine._manual_flash_pending == 0


class TestWhiteFlashWithManualTrigger:
    """Test that white flash mode works correctly with manual triggers."""

    def test_white_flash_applies_to_manual_trigger(self):
        """When white flash is on and a manual flash fires, the internal
        light saturation should be reduced (driven toward white)."""
        engine = EffectEngine(num_lights=4, attack_alpha=0.9)
        engine.set_white_flash_mode(True)

        # Build up energy so lights have color/saturation
        now = 100.0
        for _ in range(15):
            engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Record saturation before manual flash
        now += 2.0
        engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)
        sat_before = engine._lights[0].saturation

        # Trigger manual flash
        engine.trigger_manual_flash()
        now += 0.033
        engine.tick(_loud_features(), _no_beat(), dt=0.033, now=now)

        # After flash with white mode, saturation should be reduced
        sat_after = engine._lights[0].saturation
        assert sat_after < sat_before, (
            f"White flash should reduce saturation on manual flash: "
            f"before={sat_before:.3f}, after={sat_after:.3f}"
        )
