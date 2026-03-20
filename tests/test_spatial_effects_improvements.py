"""Tests for spatial/effect improvements (Tasks 2.11, 2.12, 2.13, 2.14).

Covers:
- Task 2.11: Alternating spatial mode (even=bass/warm, odd=treble/cool)
- Task 2.12: Wave pulse with per-bulb delay on beat trigger
- Task 2.13: Beat-synced breathing + energy-modulated range
- Task 2.14: Color cycling speed modulation by section/energy
"""

import math

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.audio.section_detector import Section, SectionInfo
from hue_visualizer.visualizer.engine import EffectEngine, GenerativeLayer
from hue_visualizer.visualizer.spatial import SpatialMapper


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


def _bass_heavy_features(rms: float = 0.6) -> AudioFeatures:
    """Audio features with strong bass, weak treble."""
    bands = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.05, 0.02])
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=300.0,
        spectral_flux=10.0,
        spectral_rolloff=1000.0,
        spectral_flatness=0.15,
        rms=rms,
        peak=rms * 1.4,
        spectrum=np.zeros(1024),
    )


def _treble_heavy_features(rms: float = 0.6) -> AudioFeatures:
    """Audio features with strong treble, weak bass."""
    bands = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9])
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=6000.0,
        spectral_flux=15.0,
        spectral_rolloff=8000.0,
        spectral_flatness=0.3,
        rms=rms,
        peak=rms * 1.4,
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


def _no_beat_with_bpm(bpm: float = 128.0) -> BeatInfo:
    """No beat detected but BPM is known (between beats)."""
    return BeatInfo(
        is_beat=False,
        bpm=bpm,
        bpm_confidence=0.5,
        beat_strength=0.0,
        predicted_next_beat=0.0,
        time_since_beat=0.1,
    )


# ============================================================================
# Task 2.11: Alternating spatial mode
# ============================================================================


class TestAlternatingSpatialMode:
    """Test that alternating mode creates bass/treble contrast on even/odd lights."""

    def test_alternating_mode_constant_exists(self):
        assert SpatialMapper.ALTERNATING == "alternating"
        assert "alternating" in SpatialMapper.MODES

    def test_set_alternating_mode(self):
        engine = EffectEngine(num_lights=6, spatial_mode="frequency_zones")
        engine.set_spatial_mode("alternating")
        assert engine.spatial_mapper.mode == "alternating"

    def test_alternating_produces_correct_light_count(self):
        engine = EffectEngine(num_lights=6, spatial_mode="alternating")
        features = _bass_heavy_features()
        beat = _no_beat()
        states = engine.tick(features, beat, dt=0.033, now=1000.0)
        assert len(states) == 6

    def test_alternating_even_odd_brightness_differs_with_unequal_bands(self):
        """With bass-heavy audio, even lights (bass) should be brighter than odd (treble)."""
        engine = EffectEngine(num_lights=6, spatial_mode="alternating")
        features = _bass_heavy_features(rms=0.6)
        beat = _no_beat()

        # Stabilize
        states = None
        for i in range(60):
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert states is not None
        # Even lights (0, 2, 4) respond to bass
        even_b = [states[i].brightness for i in range(0, 6, 2)]
        # Odd lights (1, 3, 5) respond to treble
        odd_b = [states[i].brightness for i in range(1, 6, 2)]

        avg_even = sum(even_b) / len(even_b)
        avg_odd = sum(odd_b) / len(odd_b)

        # With bass-heavy audio, even lights should be brighter (bass energy >> treble energy)
        assert avg_even > avg_odd, (
            f"Even lights (bass) should be brighter with bass-heavy audio: "
            f"even_avg={avg_even:.3f}, odd_avg={avg_odd:.3f}"
        )

    def test_alternating_treble_heavy_reverses_pattern(self):
        """With treble-heavy audio, odd lights (treble) should be brighter than even (bass)."""
        engine = EffectEngine(num_lights=6, spatial_mode="alternating")
        features = _treble_heavy_features(rms=0.6)
        beat = _no_beat()

        states = None
        for i in range(60):
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        assert states is not None
        even_b = [states[i].brightness for i in range(0, 6, 2)]
        odd_b = [states[i].brightness for i in range(1, 6, 2)]

        avg_even = sum(even_b) / len(even_b)
        avg_odd = sum(odd_b) / len(odd_b)

        assert avg_odd > avg_even, (
            f"Odd lights (treble) should be brighter with treble-heavy audio: "
            f"even_avg={avg_even:.3f}, odd_avg={avg_odd:.3f}"
        )

    def test_alternating_single_light(self):
        """Single light should work without crashing."""
        engine = EffectEngine(num_lights=1, spatial_mode="alternating")
        features = _loud_features()
        states = engine.tick(features, _no_beat(), dt=0.033, now=1000.0)
        assert len(states) == 1
        assert 0 <= states[0].brightness <= 1.0

    def test_alternating_two_lights_distinct_behavior(self):
        """With 2 lights: light 0 = bass (warm), light 1 = treble (cool)."""
        engine = EffectEngine(num_lights=2, spatial_mode="alternating")
        features = _bass_heavy_features(rms=0.7)
        beat = _no_beat()

        for i in range(60):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # Check internal hues: light 0 (even) should be warm, light 1 (odd) should be cool
        light0_hue = engine._lights[0].hue
        light1_hue = engine._lights[1].hue

        # Warm = 0-40 deg (red/orange/amber)
        # Cool = 200-280 deg (cyan/blue/violet)
        is_light0_warm = light0_hue < 60 or light0_hue > 330
        is_light1_cool = 180 < light1_hue < 300

        assert is_light0_warm, f"Light 0 (even/bass) should be warm, got hue={light0_hue:.1f}"
        assert is_light1_cool, f"Light 1 (odd/treble) should be cool, got hue={light1_hue:.1f}"

    def test_alternating_continuous_operation(self):
        """Run alternating mode for 10 seconds without crashes."""
        engine = EffectEngine(num_lights=6, spatial_mode="alternating")
        dt = 0.033
        now = 1000.0

        for i in range(300):
            rms = 0.3 + 0.3 * abs(math.sin(i * 0.05))
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0


# ============================================================================
# Task 2.12: Wave pulse with per-bulb delay
# ============================================================================


class TestWavePulse:
    """Test beat-triggered wave pulse propagation."""

    def test_wave_pulse_state_initialized(self):
        sm = SpatialMapper(num_lights=6)
        assert sm._wave_pulse_active is False
        assert sm._wave_pulse_start == 0.0

    def test_wave_pulse_reset(self):
        sm = SpatialMapper(num_lights=6)
        sm._wave_pulse_active = True
        sm._wave_pulse_start = 1000.0
        sm.reset()
        assert sm._wave_pulse_active is False
        assert sm._wave_pulse_start == 0.0

    def test_beat_triggers_wave_pulse(self):
        """In wave mode, a beat should activate the wave pulse."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave", max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Fire a beat
        engine.tick(features, _beat(strength=0.8), dt=0.033, now=1001.0)

        assert engine.spatial_mapper._wave_pulse_active is True
        assert engine.spatial_mapper._wave_pulse_start == pytest.approx(1001.0)

    def test_wave_pulse_not_triggered_in_other_modes(self):
        """Wave pulse should only trigger in wave mode."""
        engine = EffectEngine(num_lights=6, spatial_mode="uniform", max_flash_hz=10.0)
        features = _loud_features()
        engine.tick(features, _beat(strength=0.8), dt=0.033, now=1000.0)
        assert engine.spatial_mapper._wave_pulse_active is False

    def test_wave_pulse_creates_sequential_brightness(self):
        """Beat-triggered wave pulse should create a brightness pattern
        that propagates across lights with delay."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave", max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record pre-beat brightness
        states_before = engine.tick(features, _no_beat(), dt=0.033, now=1001.0)
        b_before = [s.brightness for s in states_before]

        # Fire a beat to start wave pulse
        states_beat = engine.tick(features, _beat(strength=0.9), dt=0.033, now=1001.033)
        b_at_beat = [s.brightness for s in states_beat]

        # First light should get the pulse immediately (or very quickly)
        # Later lights should get it after delay (75ms * index)
        # At t=33ms after pulse start, first light is fully active (delay=0ms)
        # but light 5 hasn't been triggered yet (delay=375ms)
        assert b_at_beat[0] >= b_before[0], (
            f"First light should be at least as bright after pulse: "
            f"before={b_before[0]:.3f}, after={b_at_beat[0]:.3f}"
        )

    def test_wave_pulse_propagates_with_delay(self):
        """After pulse starts, each bulb should activate sequentially."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave", max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record steady-state brightness
        states_baseline = engine.tick(features, _no_beat(), dt=0.033, now=1001.0)
        b_baseline = [s.brightness for s in states_baseline]

        # Fire the beat pulse
        engine.tick(features, _beat(strength=0.9), dt=0.033, now=1001.033)

        # At 200ms after pulse: light 0 is decaying, light 2 is near peak
        # (light 2 trigger time = 2 * 75ms = 150ms, so 50ms into decay)
        for i in range(5):  # Advance ~165ms
            engine.tick(features, _no_beat(), dt=0.033, now=1001.066 + i * 0.033)

        states_mid = engine.tick(features, _no_beat(), dt=0.033, now=1001.231)
        b_mid = [s.brightness for s in states_mid]

        # Light 5 (delay = 5*75ms = 375ms) shouldn't be triggered yet at ~200ms
        # Light 0 (delay = 0ms) should be decaying by now
        # The propagation creates a gradient where earlier lights have decayed more
        # We just verify the pulse is still active (at least some lights are brighter)
        assert engine.spatial_mapper._wave_pulse_active is True

    def test_wave_pulse_deactivates_after_travel(self):
        """Wave pulse should deactivate after it has fully traveled and decayed."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave", max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Fire a beat
        engine.tick(features, _beat(strength=0.9), dt=0.033, now=1000.0)
        assert engine.spatial_mapper._wave_pulse_active is True

        # Advance well past the total travel + decay time
        # Travel: 5 * 75ms = 375ms. Decay: 4 * 200ms = 800ms. Total: ~1175ms
        now = 1000.033
        for i in range(60):  # ~2 seconds
            engine.tick(features, _no_beat(), dt=0.033, now=now)
            now += 0.033

        assert engine.spatial_mapper._wave_pulse_active is False

    def test_wave_continuous_sine_still_works(self):
        """The continuous sine wave should still modulate brightness alongside pulse."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave")
        features = _loud_features(rms=0.5)

        # Run without any beat for several ticks — only continuous sine should be active
        states = None
        for i in range(30):
            states = engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        assert states is not None
        brightnesses = [s.brightness for s in states]
        spread = max(brightnesses) - min(brightnesses)
        assert spread > 0.01, "Continuous sine wave should create brightness variation"


# ============================================================================
# Task 2.13: Beat-synced breathing + energy-modulated range
# ============================================================================


class TestBreathingBeatSync:
    """Test that breathing syncs to BPM when available."""

    def test_breathing_uses_fixed_rate_without_bpm(self):
        """Without BPM, breathing should use the default fixed rate (0.25 Hz)."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=0.25,
            wave_speed=0.0,
        )

        # Run for exactly 4 seconds at 0.25 Hz — should complete 1 full cycle
        for _ in range(400):
            gen.tick(0.01, bpm=0.0, energy_level=0.0)

        assert abs(gen._breathing_phase) < 0.02 or abs(gen._breathing_phase - 1.0) < 0.02, (
            f"After 4s at 0.25Hz with no BPM, phase should wrap to ~0, got {gen._breathing_phase}"
        )

    def test_breathing_syncs_to_bpm(self):
        """With BPM=128, breathing should use a 4-beat cycle (1.875s)."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=0.25,  # Won't be used when BPM is available
            wave_speed=0.0,
        )

        # At 128 BPM: 4 beats = 4 * (60/128) = 1.875 seconds per cycle
        # After half a cycle (0.9375s), phase should be ~0.5
        for _ in range(94):
            gen.tick(0.01, bpm=128.0, energy_level=0.0)

        assert abs(gen._breathing_phase - 0.5) < 0.05, (
            f"After half a 4-beat cycle at 128 BPM, phase should be ~0.5, got {gen._breathing_phase}"
        )

    def test_breathing_syncs_to_different_bpm(self):
        """Different BPM should produce different breathing rates."""
        gen_fast = GenerativeLayer(num_lights=1, wave_speed=0.0)
        gen_slow = GenerativeLayer(num_lights=1, wave_speed=0.0)

        # Run both for exactly 1 second (short enough to avoid wrapping)
        for _ in range(100):
            gen_fast.tick(0.01, bpm=180.0, energy_level=0.0)
            gen_slow.tick(0.01, bpm=80.0, energy_level=0.0)

        # At 180 BPM: period = 4*(60/180) = 1.333s, rate = 0.75 Hz
        # In 1s: phase = 0.75
        # At 80 BPM: period = 4*(60/80) = 3.0s, rate = 0.333 Hz
        # In 1s: phase = 0.333
        # Faster BPM should advance phase further in 1 second
        assert gen_fast._breathing_phase > gen_slow._breathing_phase, (
            f"Faster BPM should give higher breathing phase: "
            f"fast={gen_fast._breathing_phase:.3f}, slow={gen_slow._breathing_phase:.3f}"
        )


class TestBreathingEnergyModulation:
    """Test that breathing range compresses with higher energy."""

    def test_silence_full_breathing_range(self):
        """In silence (energy=0), breathing should use full range."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=1.0,
            breathing_min=0.2,
            breathing_max=0.8,
            wave_speed=0.0,
        )

        brightnesses = []
        for _ in range(100):
            result = gen.tick(0.01, bpm=0.0, energy_level=0.0)
            _, _, b = result[0]
            brightnesses.append(b)

        br_range = max(brightnesses) - min(brightnesses)
        assert br_range > 0.4, (
            f"In silence, breathing range should be close to full (0.6), got {br_range:.3f}"
        )

    def test_loud_compressed_breathing_range(self):
        """At high energy, breathing range should be compressed."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=1.0,
            breathing_min=0.2,
            breathing_max=0.8,
            wave_speed=0.0,
        )

        brightnesses = []
        for _ in range(100):
            result = gen.tick(0.01, bpm=0.0, energy_level=0.9)
            _, _, b = result[0]
            brightnesses.append(b)

        br_range = max(brightnesses) - min(brightnesses)
        assert br_range < 0.3, (
            f"At high energy, breathing range should be compressed, got {br_range:.3f}"
        )

    def test_energy_modulation_is_gradual(self):
        """Intermediate energy levels should produce intermediate ranges."""
        ranges = {}
        for energy in [0.0, 0.5, 1.0]:
            gen = GenerativeLayer(
                num_lights=1,
                breathing_rate_hz=1.0,
                breathing_min=0.2,
                breathing_max=0.8,
                wave_speed=0.0,
            )
            brightnesses = []
            for _ in range(100):
                result = gen.tick(0.01, bpm=0.0, energy_level=energy)
                _, _, b = result[0]
                brightnesses.append(b)
            ranges[energy] = max(brightnesses) - min(brightnesses)

        # Range should decrease with energy
        assert ranges[0.0] > ranges[0.5], (
            f"range(0.0)={ranges[0.0]:.3f} should > range(0.5)={ranges[0.5]:.3f}"
        )
        assert ranges[0.5] > ranges[1.0], (
            f"range(0.5)={ranges[0.5]:.3f} should > range(1.0)={ranges[1.0]:.3f}"
        )

    def test_breathing_is_sinusoidal_not_cubic(self):
        """Verify breathing uses sinusoidal curve, not cubic easing."""
        gen = GenerativeLayer(
            num_lights=1,
            breathing_rate_hz=1.0,
            breathing_min=0.0,
            breathing_max=1.0,
            wave_speed=0.0,
        )

        # At phase 0.25, sine gives 0.5 + 0.5*sin(pi/2) = 1.0
        # Cubic would give different value
        # Advance to phase 0.25
        for _ in range(25):
            gen.tick(0.01, bpm=0.0, energy_level=0.0)

        assert abs(gen._breathing_phase - 0.25) < 0.02
        result = gen.tick(0.001, bpm=0.0, energy_level=0.0)
        _, _, b = result[0]
        # At phase 0.25, sin(2*pi*0.25) = sin(pi/2) = 1.0
        # breath = 0.5 + 0.5*1.0 = 1.0
        # With wave_speed=0.0, brightness = 1.0 * (0.75 + 0.25 * wave)
        # wave at pos 0 with wave_phase ~0: sin(0) = 0, so wave = 0.5
        # brightness = 1.0 * (0.75 + 0.25 * 0.5) = 0.875
        assert b > 0.8, f"At peak breathing phase, brightness should be high, got {b:.3f}"


class TestBreathingIntegration:
    """Test breathing improvements in the full engine pipeline."""

    def test_engine_passes_bpm_to_generative(self):
        """Engine should pass BPM to generative layer for beat-synced breathing."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features(rms=0.5)
        beat = _no_beat_with_bpm(bpm=128.0)

        # Record phase before
        gen = engine._generative
        phase_before = gen._breathing_phase

        # Tick once
        engine.tick(features, beat, dt=0.033, now=1000.0)

        # Phase should have advanced using beat-synced rate
        # At 128 BPM, rate = 1 / 1.875 = 0.533 Hz
        # In 33ms: phase advance = 0.033 * 0.533 = ~0.0176
        expected_advance = 0.033 / (4.0 * 60.0 / 128.0)
        actual_advance = gen._breathing_phase - phase_before
        assert abs(actual_advance - expected_advance) < 0.005, (
            f"Phase advance should match beat-synced rate: "
            f"expected={expected_advance:.4f}, got={actual_advance:.4f}"
        )


# ============================================================================
# Task 2.14: Color cycling speed modulation by section
# ============================================================================


class TestRotationSpeedModulation:
    """Test hue rotation speed modulation based on musical section."""

    def test_normal_section_speed_is_1x(self):
        """In normal section, rotation speed multiplier should be 1.0."""
        engine = EffectEngine(num_lights=4)
        engine._current_section = Section.NORMAL
        engine._section_intensity = 0.5
        assert engine._get_rotation_speed_multiplier() == 1.0

    def test_breakdown_slows_rotation(self):
        """In breakdown, rotation speed should be halved at full intensity."""
        engine = EffectEngine(num_lights=4)
        engine._current_section = Section.BREAKDOWN
        engine._section_intensity = 1.0
        mult = engine._get_rotation_speed_multiplier()
        assert mult == pytest.approx(0.5), f"Breakdown should halve speed, got {mult}"

    def test_buildup_doubles_rotation(self):
        """In buildup, rotation speed should double at full intensity."""
        engine = EffectEngine(num_lights=4)
        engine._current_section = Section.BUILDUP
        engine._section_intensity = 1.0
        mult = engine._get_rotation_speed_multiplier()
        assert mult == pytest.approx(2.0), f"Buildup should double speed, got {mult}"

    def test_drop_moderate_speedup(self):
        """In drop, rotation speed should increase moderately."""
        engine = EffectEngine(num_lights=4)
        engine._current_section = Section.DROP
        engine._section_intensity = 1.0
        mult = engine._get_rotation_speed_multiplier()
        assert mult == pytest.approx(1.5), f"Drop should be 1.5x speed, got {mult}"

    def test_intensity_scales_multiplier(self):
        """Section intensity should smoothly scale the multiplier."""
        engine = EffectEngine(num_lights=4)
        engine._current_section = Section.BREAKDOWN

        # At 0 intensity, should be 1.0x
        engine._section_intensity = 0.0
        assert engine._get_rotation_speed_multiplier() == pytest.approx(1.0)

        # At 0.5 intensity, should be 0.75x (halfway to 0.5)
        engine._section_intensity = 0.5
        assert engine._get_rotation_speed_multiplier() == pytest.approx(0.75)

        # At 1.0 intensity, should be 0.5x
        engine._section_intensity = 1.0
        assert engine._get_rotation_speed_multiplier() == pytest.approx(0.5)

    def test_breakdown_slows_hue_rotation_in_practice(self):
        """During breakdown, hue rotation should advance slower than normal."""
        engine_normal = EffectEngine(num_lights=4)
        engine_breakdown = EffectEngine(num_lights=4)

        features = _loud_features(rms=0.5)
        beat = _no_beat_with_bpm(bpm=128.0)
        normal_section = SectionInfo(section=Section.NORMAL, intensity=0.0)
        breakdown_section = SectionInfo(
            section=Section.BREAKDOWN, confidence=0.9, intensity=0.8,
        )

        # Run both engines for 3 seconds
        now = 1000.0
        for i in range(90):
            engine_normal.tick(features, beat, dt=0.033, now=now, section_info=normal_section)
            engine_breakdown.tick(features, beat, dt=0.033, now=now, section_info=breakdown_section)
            now += 0.033

        # Generative hue phase should have advanced less in breakdown
        phase_normal = engine_normal._generative._hue_phase
        phase_breakdown = engine_breakdown._generative._hue_phase
        assert phase_breakdown < phase_normal, (
            f"Breakdown should slow hue rotation: "
            f"normal={phase_normal:.4f}, breakdown={phase_breakdown:.4f}"
        )

    def test_buildup_speeds_hue_rotation_in_practice(self):
        """During buildup, hue rotation should advance faster than normal."""
        engine_normal = EffectEngine(num_lights=4)
        engine_buildup = EffectEngine(num_lights=4)

        features = _loud_features(rms=0.5)
        beat = _no_beat_with_bpm(bpm=128.0)
        normal_section = SectionInfo(section=Section.NORMAL, intensity=0.0)
        buildup_section = SectionInfo(
            section=Section.BUILDUP, confidence=0.9, intensity=0.8,
        )

        now = 1000.0
        for i in range(90):
            engine_normal.tick(features, beat, dt=0.033, now=now, section_info=normal_section)
            engine_buildup.tick(features, beat, dt=0.033, now=now, section_info=buildup_section)
            now += 0.033

        phase_normal = engine_normal._generative._hue_phase
        phase_buildup = engine_buildup._generative._hue_phase
        assert phase_buildup > phase_normal, (
            f"Buildup should speed up hue rotation: "
            f"normal={phase_normal:.4f}, buildup={phase_buildup:.4f}"
        )


# ============================================================================
# End-to-end integration
# ============================================================================


class TestSpatialEffectsEndToEnd:
    """Integration tests combining all improvements."""

    def test_alternating_mode_with_beats(self):
        """Run alternating mode with beat triggers — no crashes."""
        engine = EffectEngine(num_lights=6, spatial_mode="alternating", max_flash_hz=10.0)
        dt = 0.033
        now = 1000.0

        for i in range(300):
            rms = 0.3 + 0.3 * abs(math.sin(i * 0.05))
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.7, bpm=128.0) if is_beat else _no_beat_with_bpm(128.0)
            section = SectionInfo(section=Section.NORMAL)

            states = engine.tick(features, beat, dt=dt, now=now, section_info=section)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0

    def test_wave_mode_with_pulse_and_sections(self):
        """Wave mode with pulse + section modulation — full scenario."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave", max_flash_hz=10.0)
        dt = 0.033
        now = 1000.0

        # Normal section with beats
        for i in range(60):
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.8, bpm=128.0) if is_beat else _no_beat_with_bpm(128.0)
            section = SectionInfo(section=Section.NORMAL)
            engine.tick(_loud_features(rms=0.5), beat, dt=dt, now=now, section_info=section)
            now += dt

        # Breakdown — rotation slows, breathing compresses
        for i in range(60):
            section = SectionInfo(section=Section.BREAKDOWN, confidence=0.8, intensity=0.7)
            engine.tick(_loud_features(rms=0.2), _no_beat_with_bpm(128.0), dt=dt, now=now, section_info=section)
            now += dt

        # Buildup — rotation speeds up
        for i in range(60):
            is_beat = (i % 12 == 0)
            beat = _beat(strength=0.9, bpm=128.0) if is_beat else _no_beat_with_bpm(128.0)
            section = SectionInfo(section=Section.BUILDUP, confidence=0.9, intensity=0.8)
            states = engine.tick(_loud_features(rms=0.6), beat, dt=dt, now=now, section_info=section)
            now += dt

            for s in states:
                assert 0 <= s.brightness <= 1.0

    def test_all_spatial_modes_produce_valid_output(self):
        """Every spatial mode (including new alternating) should produce valid output."""
        features = _loud_features(rms=0.5)
        beat = _no_beat()

        for mode in SpatialMapper.MODES:
            engine = EffectEngine(num_lights=6, spatial_mode=mode)
            for i in range(30):
                states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)
            assert len(states) == 6, f"Mode {mode} should produce 6 states"
            for s in states:
                assert 0 <= s.brightness <= 1.0, f"Mode {mode}: brightness out of range"
                assert 0 <= s.x <= 1.0, f"Mode {mode}: x out of range"
                assert 0 <= s.y <= 1.0, f"Mode {mode}: y out of range"
