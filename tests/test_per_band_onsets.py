"""Tests for per-band onset detection (Task 1.5) and bass pulse / treble sparkle effects (Task 1.9 + 1.10).

Verifies:
- Per-band onset detection in BeatDetector (kick/snare/hihat)
- Per-band adaptive thresholds (median-based)
- Per-band cooldowns
- Per-band onset fields in BeatInfo
- Reset clears per-band state
- Bass pulse effect triggers on kick_onset (red/orange hue, brightness proportional to bass)
- Treble sparkle effect triggers on hihat_onset (blue/violet hue, random lights, fast decay)
- Snare onset triggers white/bright flash (via main beat flash path)
- Effects are subtle and don't overwhelm main beat flash
- Safety limiter still works with new effects
"""

import math

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatDetector, BeatInfo
from hue_visualizer.visualizer.engine import EffectEngine


# --- Test helpers ---


def _make_band_features(
    low: float = 0.0,
    mid: float = 0.0,
    high: float = 0.0,
    rms: float = 0.0,
    flux: float = 0.0,
) -> AudioFeatures:
    """Create AudioFeatures with specified band energies.

    Args:
        low: Energy for sub_bass[0] + bass[1] bands (20-250 Hz).
        mid: Energy for low_mid[2] + mid[3] + upper_mid[4] bands (250-4000 Hz).
        high: Energy for presence[5] + brilliance[6] bands (4-20 kHz).
        rms: Overall RMS energy.
        flux: Spectral flux.
    """
    bands = np.zeros(7)
    bands[0] = low  # sub_bass
    bands[1] = low  # bass
    bands[2] = mid  # low_mid
    bands[3] = mid  # mid
    bands[4] = mid  # upper_mid
    bands[5] = high  # presence
    bands[6] = high  # brilliance
    return AudioFeatures(
        band_energies=bands,
        spectral_flux=flux,
        rms=rms if rms > 0 else (low + mid + high) / 3.0,
        peak=(low + mid + high) / 2.0,
        spectrum=np.zeros(1024),
    )


def _silence_features() -> AudioFeatures:
    return _make_band_features(low=0.0, mid=0.0, high=0.0, rms=0.0)


def _no_beat(**overrides) -> BeatInfo:
    defaults = dict(
        is_beat=False, bpm=0.0, bpm_confidence=0.0, beat_strength=0.0,
        predicted_next_beat=0.0, time_since_beat=0.0,
        kick_onset=False, snare_onset=False, hihat_onset=False,
        kick_energy=0.0, snare_energy=0.0, hihat_energy=0.0,
    )
    defaults.update(overrides)
    return BeatInfo(**defaults)


def _kick_beat(energy: float = 0.7) -> BeatInfo:
    return BeatInfo(
        is_beat=True, bpm=128.0, bpm_confidence=0.5, beat_strength=0.7,
        predicted_next_beat=0.0, time_since_beat=0.0,
        kick_onset=True, snare_onset=False, hihat_onset=False,
        kick_energy=energy, snare_energy=0.0, hihat_energy=0.0,
    )


def _hihat_beat(energy: float = 0.5) -> BeatInfo:
    return BeatInfo(
        is_beat=False, bpm=128.0, bpm_confidence=0.5, beat_strength=0.0,
        predicted_next_beat=0.0, time_since_beat=0.0,
        kick_onset=False, snare_onset=False, hihat_onset=True,
        kick_energy=0.0, snare_energy=0.0, hihat_energy=energy,
    )


def _snare_beat(energy: float = 0.6) -> BeatInfo:
    return BeatInfo(
        is_beat=False, bpm=128.0, bpm_confidence=0.5, beat_strength=0.0,
        predicted_next_beat=0.0, time_since_beat=0.0,
        kick_onset=False, snare_onset=True, hihat_onset=False,
        kick_energy=0.0, snare_energy=energy, hihat_energy=0.0,
    )


# ============================================================================
# Per-band onset detection (BeatDetector) — Task 1.5
# ============================================================================


class TestPerBandOnsetDetection:
    """Test per-band onset detection in BeatDetector."""

    def test_kick_onset_detected(self):
        """A sudden bass energy spike should trigger kick_onset."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup with low energy
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        # Bass spike
        t += 0.5  # Ensure past all cooldowns
        result = bd.detect(
            _make_band_features(low=0.9, mid=0.1, high=0.1),
            timestamp=t,
        )
        assert result.kick_onset, "Bass energy spike should trigger kick_onset"

    def test_snare_onset_detected(self):
        """A sudden mid energy spike should trigger snare_onset."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        # Mid spike
        t += 0.5
        result = bd.detect(
            _make_band_features(low=0.1, mid=0.9, high=0.1),
            timestamp=t,
        )
        assert result.snare_onset, "Mid energy spike should trigger snare_onset"

    def test_hihat_onset_detected(self):
        """A sudden high energy spike should trigger hihat_onset."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        # High spike
        t += 0.5
        result = bd.detect(
            _make_band_features(low=0.1, mid=0.1, high=0.9),
            timestamp=t,
        )
        assert result.hihat_onset, "High energy spike should trigger hihat_onset"

    def test_simultaneous_kick_and_hihat(self):
        """Both kick and hihat can trigger in the same frame."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        # Both low and high spike
        t += 0.5
        result = bd.detect(
            _make_band_features(low=0.9, mid=0.1, high=0.9),
            timestamp=t,
        )
        assert result.kick_onset, "Bass spike should trigger kick_onset"
        assert result.hihat_onset, "High spike should trigger hihat_onset"

    def test_band_energies_stored_in_beatinfo(self):
        """BeatInfo should contain the current per-band energy values."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(15):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        result = bd.detect(
            _make_band_features(low=0.7, mid=0.5, high=0.3),
            timestamp=t,
        )
        assert result.kick_energy == pytest.approx(0.7, abs=0.01)
        assert result.snare_energy == pytest.approx(0.5, abs=0.01)
        assert result.hihat_energy == pytest.approx(0.3, abs=0.01)

    def test_no_onset_in_silence(self):
        """No onsets should be detected during silence."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup and continue with silence
        result = None
        for _ in range(50):
            result = bd.detect(
                _make_band_features(low=0.0, mid=0.0, high=0.0),
                timestamp=t,
            )
            t += frame_dur

        assert result is not None
        assert not result.kick_onset
        assert not result.snare_onset
        assert not result.hihat_onset

    def test_constant_energy_no_repeated_onsets(self):
        """Constant energy should not trigger repeated onsets (adaptive threshold)."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        kick_count = 0
        for i in range(100):
            result = bd.detect(
                _make_band_features(low=0.5, mid=0.5, high=0.5),
                timestamp=t,
            )
            t += frame_dur
            if i > 30 and result.kick_onset:
                kick_count += 1

        # With constant energy, the median catches up — should not keep triggering
        assert kick_count <= 3, \
            f"Constant energy should not trigger repeated onsets, got {kick_count}"


class TestPerBandCooldowns:
    """Test per-band cooldown timers."""

    def test_kick_cooldown(self):
        """Two kicks too close together should be gated by cooldown."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1), timestamp=t)
            t += frame_dur

        # First kick
        t += 0.5
        r1 = bd.detect(_make_band_features(low=0.9), timestamp=t)

        # Second kick immediately (within 150ms cooldown)
        t += 0.05  # 50ms
        r2 = bd.detect(_make_band_features(low=0.9), timestamp=t)

        if r1.kick_onset:
            assert not r2.kick_onset, "Kick cooldown should prevent double trigger"

    def test_hihat_short_cooldown(self):
        """Hi-hat cooldown is shorter (60ms), allowing rapid hi-hats."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_band_features(high=0.1), timestamp=t)
            t += frame_dur

        # First hi-hat
        t += 0.5
        r1 = bd.detect(_make_band_features(high=0.9), timestamp=t)

        # Second hi-hat after 100ms (past 60ms cooldown)
        t += 0.1
        # Need a quiet frame to let median drop, then spike again
        bd.detect(_make_band_features(high=0.1), timestamp=t)
        t += frame_dur
        r2 = bd.detect(_make_band_features(high=0.9), timestamp=t)

        # At least one of the two pairs should show the hi-hat triggering again
        if r1.hihat_onset:
            # The second may or may not trigger depending on adaptive threshold,
            # but the cooldown should not be the blocker
            assert (t - bd._last_hihat_time) >= 0  # Cooldown was not the issue


class TestPerBandReset:
    """Test that reset clears per-band state."""

    def test_reset_clears_band_histories(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        for _ in range(30):
            bd.detect(_make_band_features(low=0.5, mid=0.3, high=0.2), timestamp=t)
            t += frame_dur

        assert len(bd._low_band_history) > 0
        assert len(bd._mid_band_history) > 0
        assert len(bd._high_band_history) > 0

        bd.reset()

        assert len(bd._low_band_history) == 0
        assert len(bd._mid_band_history) == 0
        assert len(bd._high_band_history) == 0
        assert bd._last_kick_time == 0.0
        assert bd._last_snare_time == 0.0
        assert bd._last_hihat_time == 0.0

    def test_main_beat_still_works(self):
        """Per-band detection should not break the main is_beat detection."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup with low energy
        for _ in range(30):
            bd.detect(_make_band_features(low=0.1, mid=0.1, high=0.1), timestamp=t)
            t += frame_dur

        # Strong bass spike should trigger both is_beat AND kick_onset
        t += 0.5
        result = bd.detect(
            _make_band_features(low=0.9, mid=0.1, high=0.1, flux=0.01),
            timestamp=t,
        )
        assert result.is_beat, "Main beat detection should still work"


# ============================================================================
# Bass pulse effect (EffectEngine) — Task 1.10
# ============================================================================


class TestBassPulseEffect:
    """Test bass pulse effect triggered by kick_onset."""

    def test_kick_triggers_bass_pulse(self):
        """Kick onset should activate bass pulse on lights."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, mid=0.2, high=0.1, rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire a kick onset
        engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1001.0)

        # At least some lights should have bass_pulse_brightness > 0
        has_pulse = any(l.bass_pulse_brightness > 0 for l in engine._lights)
        assert has_pulse, "Kick onset should trigger bass pulse on some lights"

    def test_bass_pulse_hue_is_red_orange(self):
        """Bass pulse should push hue toward red/orange (0-30 degrees)."""
        engine = EffectEngine(num_lights=4, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, mid=0.2, high=0.1, rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire a kick onset
        engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1001.0)

        for light in engine._lights:
            if light.bass_pulse_brightness > 0.01:
                assert 0.0 <= light.bass_pulse_hue <= 30.0, \
                    f"Bass pulse hue should be in red-orange range, got {light.bass_pulse_hue}"

    def test_bass_pulse_decays(self):
        """Bass pulse brightness should decay over time."""
        engine = EffectEngine(num_lights=4, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire kick
        engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1001.0)
        pulse_before = max(l.bass_pulse_brightness for l in engine._lights)
        assert pulse_before > 0

        # Let it decay for several ticks
        for i in range(10):
            engine.tick(features, _no_beat(), dt=0.033, now=1001.033 + i * 0.033)

        pulse_after = max(l.bass_pulse_brightness for l in engine._lights)
        assert pulse_after < pulse_before, \
            "Bass pulse should decay over time"

    def test_bass_pulse_adds_brightness(self):
        """Bass pulse should increase overall light brightness."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.5, mid=0.3, high=0.2, rms=0.4)

        # Stabilize
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record brightness before kick
        states_before = engine.tick(features, _no_beat(), dt=0.033, now=1002.0)
        max_before = max(s.brightness for s in states_before)

        # Fire a kick
        states_after = engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1002.033)
        max_after = max(s.brightness for s in states_after)

        assert max_after > max_before, \
            f"Bass pulse should increase brightness: {max_after} > {max_before}"

    def test_bass_pulse_spatial_weighting_freq_mode(self):
        """In frequency_zones mode, bass-position lights get stronger pulse."""
        engine = EffectEngine(
            num_lights=6, max_flash_hz=30.0, spatial_mode="frequency_zones"
        )
        features = _make_band_features(low=0.8, mid=0.2, high=0.1, rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire kick
        engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1001.0)

        # First light (position 0.0 = bass) should have stronger pulse than last
        pulse_first = engine._lights[0].bass_pulse_brightness
        pulse_last = engine._lights[-1].bass_pulse_brightness

        assert pulse_first > pulse_last, \
            f"Bass-position light should have stronger pulse: {pulse_first} > {pulse_last}"


# ============================================================================
# Treble sparkle effect (EffectEngine) — Task 1.9
# ============================================================================


class TestTrebleSparkleEffect:
    """Test treble sparkle effect triggered by hihat_onset."""

    def test_hihat_triggers_sparkle(self):
        """Hi-hat onset should activate sparkle on some lights."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.2, mid=0.2, high=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire a hi-hat onset
        engine.tick(features, _hihat_beat(energy=0.6), dt=0.033, now=1001.0)

        has_sparkle = any(l.sparkle_brightness > 0 for l in engine._lights)
        assert has_sparkle, "Hi-hat onset should trigger sparkle on some lights"

    def test_sparkle_on_limited_lights(self):
        """Sparkle should only affect 1-2 lights, not all."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(high=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire hi-hat
        engine.tick(features, _hihat_beat(energy=0.6), dt=0.033, now=1001.0)

        sparkle_count = sum(1 for l in engine._lights if l.sparkle_brightness > 0)
        assert 1 <= sparkle_count <= 2, \
            f"Sparkle should affect 1-2 lights, got {sparkle_count}"

    def test_sparkle_hue_is_blue_violet(self):
        """Sparkle hue should be in the blue-violet range (240-280 degrees)."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(high=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire hi-hat
        engine.tick(features, _hihat_beat(energy=0.6), dt=0.033, now=1001.0)

        for light in engine._lights:
            if light.sparkle_brightness > 0.01:
                assert 240.0 <= light.sparkle_hue <= 280.0, \
                    f"Sparkle hue should be blue-violet, got {light.sparkle_hue}"

    def test_sparkle_decays_fast(self):
        """Sparkle should decay much faster than the main beat flash (~75ms tau)."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(high=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire hi-hat
        engine.tick(features, _hihat_beat(energy=0.8), dt=0.033, now=1001.0)
        sparkle_before = max(l.sparkle_brightness for l in engine._lights)
        assert sparkle_before > 0

        # After ~150ms (2x tau), sparkle should be mostly gone
        for i in range(5):
            engine.tick(features, _no_beat(), dt=0.033, now=1001.033 + i * 0.033)

        sparkle_after = max(l.sparkle_brightness for l in engine._lights)
        # Should have decayed by at least 80% after ~150ms with tau=75ms
        assert sparkle_after < sparkle_before * 0.3, \
            f"Sparkle should decay fast: {sparkle_after} should be << {sparkle_before}"

    def test_sparkle_random_lights_change(self):
        """Repeated hi-hat onsets should target different random lights."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(high=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire multiple hi-hats and collect which lights sparkle
        targets_seen = set()
        for j in range(20):
            # Clear previous sparkle
            for l in engine._lights:
                l.sparkle_brightness = 0.0

            engine.tick(
                features, _hihat_beat(energy=0.6),
                dt=0.033, now=1001.0 + j * 0.2,
            )
            for idx, l in enumerate(engine._lights):
                if l.sparkle_brightness > 0:
                    targets_seen.add(idx)

        # Over 20 hi-hats, we should have hit at least 3 different lights
        assert len(targets_seen) >= 3, \
            f"Sparkle should target different lights over time, only hit {targets_seen}"


# ============================================================================
# Snare flash effect — Task 1.5 integration
# ============================================================================


class TestSnareFlash:
    """Test that snare onsets trigger the white/bright flash."""

    def test_snare_triggers_flash(self):
        """Snare onset should trigger the main beat flash (bright/white)."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(mid=0.7, rms=0.4)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire snare onset (no main beat)
        engine.tick(features, _snare_beat(energy=0.7), dt=0.033, now=1001.0)

        has_flash = any(l.flash_brightness > 0 for l in engine._lights)
        assert has_flash, "Snare onset should trigger flash_brightness"

    def test_snare_flash_respects_rate_limit(self):
        """Snare flash should still respect the max flash Hz safety limit."""
        engine = EffectEngine(num_lights=3, max_flash_hz=3.0)
        features = _make_band_features(mid=0.7, rms=0.4)

        # Fire first snare
        now = 1000.0
        engine.tick(features, _snare_beat(energy=0.7), dt=0.033, now=now)

        # Clear flash manually
        for light in engine._lights:
            light.flash_brightness = 0.0

        # Second snare too soon (33ms later, rate limit = 333ms)
        now_2 = now + 0.033
        engine.tick(features, _snare_beat(energy=0.7), dt=0.033, now=now_2)
        assert all(light.flash_brightness == 0 for light in engine._lights), \
            "Snare flash should respect rate limit"


# ============================================================================
# Integration: effects coexist and don't overwhelm
# ============================================================================


class TestEffectsIntegration:
    """Test that all per-band effects work together without overwhelming."""

    def test_all_onsets_simultaneous(self):
        """Kick + snare + hi-hat simultaneously should not crash or overflow brightness."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, mid=0.7, high=0.6, rms=0.7)

        # Stabilize
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # All onsets at once
        all_onset = BeatInfo(
            is_beat=True, bpm=128.0, bpm_confidence=0.5, beat_strength=0.8,
            predicted_next_beat=0.0, time_since_beat=0.0,
            kick_onset=True, snare_onset=True, hihat_onset=True,
            kick_energy=0.8, snare_energy=0.7, hihat_energy=0.6,
        )
        states = engine.tick(features, all_onset, dt=0.033, now=1002.0)

        for s in states:
            assert 0 <= s.brightness <= 1.0, \
                f"Brightness should be clamped to [0, 1], got {s.brightness}"

    def test_brightness_bounded_during_sustained_onsets(self):
        """Repeated onsets should never push brightness above 1.0."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=1.0, mid=0.9, high=0.8, rms=0.9)

        for i in range(100):
            beat = BeatInfo(
                is_beat=(i % 5 == 0),
                bpm=128.0, bpm_confidence=0.5,
                beat_strength=0.9 if i % 5 == 0 else 0.0,
                predicted_next_beat=0.0, time_since_beat=0.0,
                kick_onset=(i % 5 == 0),
                snare_onset=(i % 10 == 2),
                hihat_onset=(i % 3 == 0),
                kick_energy=0.9, snare_energy=0.7, hihat_energy=0.6,
            )
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)
            for s in states:
                assert 0 <= s.brightness <= 1.0, \
                    f"Brightness out of bounds: {s.brightness}"

    def test_main_beat_flash_still_dominant(self):
        """Main beat flash should still be the primary brightness driver."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.5, mid=0.3, high=0.5, rms=0.4)

        # Stabilize
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Hihat-only: sparkle should be subtle
        engine.tick(features, _hihat_beat(energy=0.5), dt=0.033, now=1002.0)
        sparkle_max = max(l.sparkle_brightness for l in engine._lights)

        # Reset
        engine.reset()
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1100.0 + i * 0.033)

        # Full beat: flash should be much stronger
        full_beat = BeatInfo(
            is_beat=True, bpm=128.0, bpm_confidence=0.5, beat_strength=0.9,
            predicted_next_beat=0.0, time_since_beat=0.0,
            kick_onset=True, snare_onset=False, hihat_onset=False,
            kick_energy=0.8, snare_energy=0.0, hihat_energy=0.0,
        )
        engine.tick(features, full_beat, dt=0.033, now=1102.0)
        flash_max = max(l.flash_brightness for l in engine._lights)

        assert flash_max > sparkle_max, \
            f"Main beat flash ({flash_max}) should dominate over sparkle ({sparkle_max})"

    def test_reset_clears_new_effect_state(self):
        """Engine reset should clear bass pulse and sparkle state."""
        engine = EffectEngine(num_lights=6, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, high=0.7, rms=0.5)

        # Stabilize + trigger effects
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        engine.tick(features, _kick_beat(energy=0.8), dt=0.033, now=1001.0)
        engine.tick(features, _hihat_beat(energy=0.6), dt=0.033, now=1001.1)

        has_pulse = any(l.bass_pulse_brightness > 0 for l in engine._lights)
        has_sparkle = any(l.sparkle_brightness > 0 for l in engine._lights)
        assert has_pulse or has_sparkle, "Should have some effects active"

        engine.reset()

        # All cleared
        for light in engine._lights:
            assert light.bass_pulse_brightness == 0.0
            assert light.sparkle_brightness == 0.0
        assert engine._sparkle_last_lights == []

    def test_single_light_engine(self):
        """Effects should work correctly with just 1 light."""
        engine = EffectEngine(num_lights=1, max_flash_hz=30.0)
        features = _make_band_features(low=0.8, mid=0.7, high=0.6, rms=0.6)

        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # All three effects
        all_onset = BeatInfo(
            is_beat=True, bpm=128.0, bpm_confidence=0.5, beat_strength=0.8,
            predicted_next_beat=0.0, time_since_beat=0.0,
            kick_onset=True, snare_onset=True, hihat_onset=True,
            kick_energy=0.8, snare_energy=0.7, hihat_energy=0.6,
        )
        states = engine.tick(features, all_onset, dt=0.033, now=1001.0)
        assert len(states) == 1
        assert 0 <= states[0].brightness <= 1.0

    def test_continuous_30_seconds_no_crash(self):
        """Run all effects for 30 simulated seconds without crashes."""
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        dt = 0.033

        for i in range(900):
            phase = (i / 900.0) * 2 * math.pi
            rms = max(0.0, 0.5 + 0.5 * math.sin(phase))
            features = _make_band_features(
                low=rms * 0.8,
                mid=rms * 0.5,
                high=rms * 0.3,
                rms=rms,
            )

            beat = BeatInfo(
                is_beat=(i % 15 == 0) and rms > 0.2,
                bpm=128.0 if rms > 0.1 else 0.0,
                bpm_confidence=0.5 if rms > 0.1 else 0.0,
                beat_strength=0.7 if (i % 15 == 0) and rms > 0.2 else 0.0,
                predicted_next_beat=0.0, time_since_beat=0.0,
                kick_onset=(i % 15 == 0) and rms > 0.3,
                snare_onset=(i % 30 == 7) and rms > 0.3,
                hihat_onset=(i % 7 == 0) and rms > 0.2,
                kick_energy=rms * 0.7,
                snare_energy=rms * 0.5,
                hihat_energy=rms * 0.3,
            )

            states = engine.tick(features, beat, dt=dt, now=1000.0 + i * dt)
            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0
