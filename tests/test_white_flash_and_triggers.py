"""Tests for strobe system and manual triggers.

Verifies:
- Manual flash: single colored flash fires and decays
- Manual strobe burst: triggers proper strobe with dark phases
- Auto strobe: toggle on/off, fires on high energy
- Strobe safety: respects max frequency, safe mode clamps to 2 Hz
- Strobe lifecycle: activates, ticks on/off cycles, deactivates
"""

import math

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
# Manual flash tests
# ============================================================================


class TestManualFlash:
    """Test manual single flash trigger."""

    def test_manual_flash_fires_on_next_tick(self):
        engine = EffectEngine(num_lights=4)
        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        now += 2.0
        engine.trigger_manual_flash()
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        for light in engine._lights:
            assert light.flash_brightness > 0.5

    def test_manual_flash_single_only(self):
        engine = EffectEngine(num_lights=4)
        now = 100.0
        for _ in range(5):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        now += 2.0
        engine.trigger_manual_flash()
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
        flash_1 = engine._lights[0].flash_brightness

        for _ in range(5):
            now += 1.0
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        flash_end = engine._lights[0].flash_brightness
        assert flash_end < flash_1, "Flash should decay, not fire again"

    def test_manual_flash_no_accumulation(self):
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_flash()
        engine.trigger_manual_flash()
        engine.trigger_manual_flash()
        assert engine._manual_flash_pending == 1

    def test_manual_flash_respects_safety_limiter(self):
        engine = EffectEngine(num_lights=4, max_flash_hz=3.0)
        now = 100.0
        engine.tick(_loud_features(), _beat(), dt=0.033, now=now)
        engine.trigger_manual_flash()
        now += 0.01  # Only 10ms later
        engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
        assert engine._manual_flash_pending == 1

    def test_manual_flash_safe_mode_reduces_intensity(self):
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

        tau = engine._flash_tau
        decay_factor = math.exp(-dt / tau)
        expected_after_decay = 0.7 * decay_factor

        for light in engine._lights:
            assert abs(light.flash_brightness - expected_after_decay) < 0.05


# ============================================================================
# Strobe burst tests
# ============================================================================


class TestStrobeBurst:
    """Test the strobe burst system (manual and auto)."""

    def test_manual_strobe_activates_strobe(self):
        """trigger_manual_strobe() should activate the strobe state machine."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()
        assert engine.strobe_active is True
        assert engine._strobe_remaining_cycles == engine._strobe_manual_cycles

    def test_strobe_outputs_white_on_phase(self):
        """During strobe ON phase, lights should be white (near D65)."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()

        # First tick: should be in ON phase (phase starts at 0)
        states = engine.tick(_silence_features(), _no_beat(), dt=0.01, now=100.0)

        for s in states:
            # White chromaticity: x≈0.3127, y≈0.3290
            assert abs(s.x - 0.3127) < 0.05, f"Expected white x, got {s.x}"
            assert s.brightness > 0.5, f"ON phase should be bright, got {s.brightness}"

    def test_strobe_outputs_dark_off_phase(self):
        """During strobe OFF phase, lights should be at brightness_min (dark)."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()

        # Advance past the 50% phase mark (at 2.5 Hz, half cycle = 200ms)
        now = 100.0
        # Tick through the ON phase
        for _ in range(7):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033
        # Now at ~231ms into the cycle — should be in OFF phase
        states = engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)

        for s in states:
            assert s.brightness <= engine._brightness_min + 0.01, \
                f"OFF phase should be dark, got {s.brightness}"

    def test_strobe_deactivates_after_cycles(self):
        """Strobe should deactivate after all cycles complete."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()
        cycles = engine._strobe_manual_cycles
        freq = engine._strobe_frequency

        # Total duration = cycles / freq
        total_duration = cycles / freq
        now = 100.0

        # Tick through the entire strobe duration
        ticks = int(total_duration / 0.033) + 20  # extra ticks for safety
        for _ in range(ticks):
            engine.tick(_silence_features(), _no_beat(), dt=0.033, now=now)
            now += 0.033

        assert engine.strobe_active is False, "Strobe should deactivate after all cycles"

    def test_strobe_reset_clears_state(self):
        """Engine reset should clear strobe state."""
        engine = EffectEngine(num_lights=4)
        engine.trigger_manual_strobe()
        assert engine.strobe_active is True
        engine.reset()
        assert engine.strobe_active is False
        assert engine._strobe_remaining_cycles == 0


# ============================================================================
# Auto strobe tests
# ============================================================================


class TestAutoStrobe:
    """Test automatic strobe triggers (energy-based)."""

    def test_strobe_enabled_default_off(self):
        engine = EffectEngine(num_lights=4)
        assert engine.strobe_enabled is False

    def test_strobe_enabled_toggle(self):
        engine = EffectEngine(num_lights=4)
        engine.set_strobe_enabled(True)
        assert engine.strobe_enabled is True
        engine.set_strobe_enabled(False)
        assert engine.strobe_enabled is False

    def test_no_auto_strobe_when_disabled(self):
        """With strobe disabled, high energy should not trigger strobe."""
        engine = EffectEngine(num_lights=4)
        engine.set_strobe_enabled(False)

        now = 100.0
        # Feed high energy to exceed threshold
        for _ in range(100):
            engine.tick(_loud_features(rms=0.9), _no_beat(), dt=0.033, now=now)
            now += 0.033

        assert engine.strobe_active is False



# ============================================================================
# Strobe safety tests
# ============================================================================


class TestStrobeSafety:
    """Test strobe safety limits."""

    def test_strobe_frequency_clamped_to_physical_max(self):
        """Strobe frequency should not exceed physical max (8 Hz)."""
        engine = EffectEngine(num_lights=4)
        engine.set_strobe_frequency(20.0)  # Way above max
        assert engine._strobe_frequency <= 8.0

    def test_safe_mode_clamps_strobe_frequency(self):
        """Safe mode should clamp strobe frequency to 2 Hz."""
        engine = EffectEngine(num_lights=4)
        engine._strobe_frequency = 6.0
        engine.set_safe_mode(True)
        assert engine._strobe_frequency <= 2.0
        assert engine._strobe_max_frequency == 2.0

    def test_normal_mode_allows_fast_strobe(self):
        """Normal mode should allow fast strobe (up to 8 Hz)."""
        engine = EffectEngine(num_lights=4)
        engine.set_safe_mode(False)
        assert engine._strobe_max_frequency == 8.0
        engine.set_strobe_frequency(6.0)
        assert engine._strobe_frequency == 6.0

    def test_safe_mode_burst_uses_clamped_frequency(self):
        """Strobe burst in safe mode should use clamped frequency."""
        engine = EffectEngine(num_lights=4)
        engine.set_safe_mode(True)
        engine._strobe_frequency = 6.0  # Above safe limit
        engine.trigger_strobe_burst(4)
        assert engine._strobe_frequency <= 2.0

    def test_manual_strobe_works_when_auto_disabled(self):
        """Manual strobe should work even when auto-strobe is disabled."""
        engine = EffectEngine(num_lights=4)
        engine.set_strobe_enabled(False)
        engine.trigger_manual_strobe()
        assert engine.strobe_active is True
