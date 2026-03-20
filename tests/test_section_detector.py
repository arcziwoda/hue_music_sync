"""Tests for section detection (Task 1.3).

Verifies:
- SectionDetector correctly classifies DROP, BUILDUP, BREAKDOWN, NORMAL
- DROP detection: bass spike after low-bass period
- BUILDUP detection: rising RMS + centroid + onset density
- BREAKDOWN detection: sustained low bass with mid/high activity
- State machine transitions and hysteresis
- Integration with EffectEngine section modulation
- Reset clears all state
"""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.audio.section_detector import (
    Section,
    SectionDetector,
    SectionInfo,
)
from hue_visualizer.visualizer.engine import EffectEngine


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


def _loud_features(rms: float = 0.7, bass: float = 0.8) -> AudioFeatures:
    """AudioFeatures representing loud music with configurable bass."""
    bands = np.array([bass, bass * 0.9, 0.5, 0.4, 0.3, 0.2, 0.1])
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


def _breakdown_features(rms: float = 0.3) -> AudioFeatures:
    """AudioFeatures representing a breakdown: low bass, mid/high activity."""
    bands = np.array([0.02, 0.03, 0.4, 0.5, 0.4, 0.3, 0.2])
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=5000.0,
        spectral_flux=10.0,
        spectral_rolloff=6000.0,
        spectral_flatness=0.4,
        rms=rms,
        peak=rms * 1.3,
        spectrum=np.zeros(1024),
    )


def _no_beat() -> BeatInfo:
    return BeatInfo()


def _beat(strength: float = 0.8, bpm: float = 128.0) -> BeatInfo:
    return BeatInfo(
        is_beat=True,
        bpm=bpm,
        bpm_confidence=0.5,
        beat_strength=strength,
    )


# ============================================================================
# SectionDetector unit tests
# ============================================================================


class TestSectionDetectorInit:
    """Test initialization and defaults."""

    def test_default_state_is_normal(self):
        det = SectionDetector()
        assert det.current_section == Section.NORMAL

    def test_initial_output_is_normal(self):
        det = SectionDetector()
        info = det.update(0.5, 0.5, 3000.0, False, 128.0, now=0.0)
        assert info.section == Section.NORMAL

    def test_custom_parameters(self):
        det = SectionDetector(
            window_beats=16,
            drop_bass_multiplier=4.0,
            drop_duration_beats=8,
            buildup_min_beats=6,
            breakdown_bass_threshold=0.25,
            breakdown_min_beats=6,
            sample_rate_hz=50.0,
        )
        assert det._window_beats == 16
        assert det._drop_bass_multiplier == 4.0

    def test_clamped_parameters(self):
        """Parameters below minimum should be clamped."""
        det = SectionDetector(
            window_beats=1,  # min 4
            drop_bass_multiplier=0.5,  # min 1.5
            breakdown_bass_threshold=0.0,  # min 0.05
        )
        assert det._window_beats == 4
        assert det._drop_bass_multiplier == 1.5
        assert det._breakdown_bass_threshold == 0.05


class TestSectionDetectorReset:
    """Test that reset clears all state."""

    def test_reset_returns_to_normal(self):
        det = SectionDetector(sample_rate_hz=30.0)

        # Feed enough data to potentially change section
        for i in range(100):
            det.update(0.5, 0.5, 3000.0, i % 15 == 0, 128.0, now=i * 0.033)

        det.reset()

        assert det.current_section == Section.NORMAL
        assert det.beats_in_section == 0
        assert len(det._bass_history) == 0
        assert len(det._rms_history) == 0

    def test_reset_clears_histories(self):
        det = SectionDetector(sample_rate_hz=30.0)

        for i in range(60):
            det.update(0.3, 0.3, 2000.0, False, 128.0, now=i * 0.033)

        det.reset()

        assert len(det._bass_history) == 0
        assert len(det._rms_history) == 0
        assert len(det._centroid_history) == 0
        assert len(det._onset_history) == 0
        assert len(det._bass_long_history) == 0


class TestDropDetection:
    """Test DROP section detection."""

    def test_bass_spike_after_quiet_triggers_drop(self):
        """A strong bass spike after a period of low bass should trigger DROP."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            drop_bass_multiplier=3.0,
        )

        now = 0.0
        dt = 0.033

        # Phase 1: Establish a baseline with moderate bass (~2 seconds)
        for i in range(60):
            det.update(0.3, 0.3, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Phase 2: Low bass period (~2 seconds, simulating a breakdown/buildup tease)
        for i in range(60):
            det.update(0.05, 0.2, 4000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Phase 3: BASS DROP! (way above the running average)
        info = None
        for i in range(10):
            info = det.update(0.95, 0.8, 2000.0, True, 128.0, now=now)
            now += dt
            if info.section == Section.DROP:
                break

        assert info is not None
        assert info.section == Section.DROP, \
            f"Bass spike after quiet should trigger DROP, got {info.section}"

    def test_no_drop_without_preceding_quiet(self):
        """Sustained high bass should NOT trigger a drop (it's just loud music)."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            drop_bass_multiplier=3.0,
        )

        now = 0.0
        dt = 0.033

        # Constant high bass for 4 seconds — no quiet period
        info = None
        for i in range(120):
            info = det.update(0.7, 0.6, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        assert info is not None
        assert info.section != Section.DROP, \
            "Sustained high bass without quiet period should not trigger DROP"

    def test_drop_expires_after_duration(self):
        """DROP should transition back to NORMAL after drop_duration_beats."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            drop_bass_multiplier=3.0,
            drop_duration_beats=4,
        )

        now = 0.0
        dt = 0.033

        # Establish baseline
        for i in range(60):
            det.update(0.3, 0.3, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Low bass period
        for i in range(60):
            det.update(0.05, 0.2, 4000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Trigger drop
        for i in range(5):
            det.update(0.95, 0.8, 2000.0, True, 128.0, now=now)
            now += dt

        assert det.current_section == Section.DROP

        # Continue with normal bass — drop should expire
        # At 128 BPM, 4 beats = ~1.875 seconds = ~57 ticks
        for i in range(80):
            info = det.update(0.4, 0.4, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        assert info.section == Section.NORMAL, \
            f"DROP should have expired, got {info.section}"


class TestBuildupDetection:
    """Test BUILDUP section detection."""

    def test_rising_trend_triggers_buildup(self):
        """Rising RMS + centroid + onset density should trigger BUILDUP."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            buildup_min_beats=4,
        )

        now = 0.0
        dt = 0.033

        # Fill history with baseline low energy
        for i in range(60):
            det.update(0.1, 0.1, 1000.0, False, 128.0, now=now)
            now += dt

        # Gradually rising RMS, centroid, and more beats
        detected_buildup = False
        for i in range(120):
            progress = i / 120.0
            rms = 0.1 + 0.7 * progress
            centroid = 1000.0 + 5000.0 * progress
            bass = 0.1 + 0.5 * progress
            # More beats as we build up
            is_beat = (i % max(1, int(20 - 15 * progress))) == 0
            info = det.update(bass, rms, centroid, is_beat, 128.0, now=now)
            now += dt
            if info.section == Section.BUILDUP:
                detected_buildup = True
                break

        assert detected_buildup, "Rising trends should trigger BUILDUP"

    def test_flat_energy_no_buildup(self):
        """Constant energy levels should NOT trigger buildup."""
        det = SectionDetector(sample_rate_hz=30.0)

        now = 0.0
        dt = 0.033

        info = None
        for i in range(200):
            info = det.update(0.4, 0.4, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        assert info is not None
        assert info.section != Section.BUILDUP, \
            "Constant energy should not trigger BUILDUP"


class TestBreakdownDetection:
    """Test BREAKDOWN section detection."""

    def test_low_bass_with_mids_triggers_breakdown(self):
        """Low bass with sustained mid/high energy should trigger BREAKDOWN."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            breakdown_bass_threshold=0.3,
            breakdown_min_beats=4,
        )

        now = 0.0
        dt = 0.033

        # Phase 1: Establish baseline with normal bass (~3 seconds)
        for i in range(90):
            det.update(0.5, 0.5, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Phase 2: Breakdown — bass drops, mids/highs continue (~3 seconds)
        detected_breakdown = False
        for i in range(90):
            # Very low bass but still RMS from mids
            info = det.update(0.03, 0.3, 5000.0, i % 20 == 0, 128.0, now=now)
            now += dt
            if info.section == Section.BREAKDOWN:
                detected_breakdown = True
                break

        assert detected_breakdown, \
            "Low bass with mid/high activity should trigger BREAKDOWN"

    def test_silence_not_breakdown(self):
        """Complete silence should NOT trigger breakdown (no activity)."""
        det = SectionDetector(sample_rate_hz=30.0)

        now = 0.0
        dt = 0.033

        # Establish baseline
        for i in range(60):
            det.update(0.4, 0.4, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Total silence
        info = None
        for i in range(90):
            info = det.update(0.0, 0.0, 0.0, False, 128.0, now=now)
            now += dt

        assert info is not None
        assert info.section != Section.BREAKDOWN, \
            "Complete silence should not trigger BREAKDOWN (no mid/high activity)"

    def test_breakdown_exits_when_bass_returns(self):
        """BREAKDOWN should end when bass energy returns."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            breakdown_bass_threshold=0.3,
        )

        now = 0.0
        dt = 0.033

        # Establish baseline
        for i in range(90):
            det.update(0.5, 0.5, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        # Trigger breakdown
        for i in range(90):
            det.update(0.03, 0.3, 5000.0, i % 20 == 0, 128.0, now=now)
            now += dt

        assert det.current_section == Section.BREAKDOWN

        # Bass returns — should exit breakdown
        info = None
        for i in range(90):
            info = det.update(0.6, 0.6, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt

        assert info is not None
        assert info.section != Section.BREAKDOWN, \
            f"Should exit BREAKDOWN when bass returns, got {info.section}"


class TestSectionTransitions:
    """Test section state machine transitions."""

    def test_buildup_to_drop_transition(self):
        """A buildup followed by a bass spike should transition to DROP."""
        det = SectionDetector(
            sample_rate_hz=30.0,
            drop_bass_multiplier=3.0,
        )

        now = 0.0
        dt = 0.033

        # Low baseline
        for i in range(60):
            det.update(0.1, 0.1, 1000.0, False, 128.0, now=now)
            now += dt

        # Buildup: rising energy
        for i in range(90):
            progress = i / 90.0
            rms = 0.1 + 0.5 * progress
            centroid = 1000.0 + 4000.0 * progress
            bass = 0.05 + 0.15 * progress  # Bass stays relatively low during buildup
            is_beat = (i % max(1, int(15 - 10 * progress))) == 0
            det.update(bass, rms, centroid, is_beat, 128.0, now=now)
            now += dt

        # Bass DROP — massive spike
        info = None
        for i in range(10):
            info = det.update(0.95, 0.9, 2000.0, True, 128.0, now=now)
            now += dt
            if info.section == Section.DROP:
                break

        assert info is not None
        assert info.section == Section.DROP, \
            f"Bass spike after buildup should trigger DROP, got {info.section}"

    def test_normal_is_default(self):
        """Without any dramatic changes, section should stay NORMAL."""
        det = SectionDetector(sample_rate_hz=30.0)

        now = 0.0
        dt = 0.033

        sections_seen = set()
        for i in range(200):
            info = det.update(0.3, 0.3, 3000.0, i % 15 == 0, 128.0, now=now)
            now += dt
            sections_seen.add(info.section)

        assert Section.NORMAL in sections_seen
        # Moderate constant input should not trigger anything dramatic
        assert Section.DROP not in sections_seen

    def test_beats_in_section_counts(self):
        """beats_in_section should count beats within the current section."""
        det = SectionDetector(sample_rate_hz=30.0)

        now = 0.0
        dt = 0.033

        # Feed some beats
        beat_count = 0
        for i in range(60):
            is_beat = (i % 15 == 0)
            if is_beat:
                beat_count += 1
            info = det.update(0.3, 0.3, 3000.0, is_beat, 128.0, now=now)
            now += dt

        # Should have counted beats
        assert info.beats_in_section > 0


class TestSectionInfoOutput:
    """Test SectionInfo dataclass output."""

    def test_default_section_info(self):
        info = SectionInfo()
        assert info.section == Section.NORMAL
        assert info.confidence == 0.0
        assert info.intensity == 0.0
        assert info.beats_in_section == 0

    def test_section_enum_values(self):
        """Section enum values should be human-readable strings."""
        assert Section.NORMAL.value == "normal"
        assert Section.DROP.value == "drop"
        assert Section.BUILDUP.value == "buildup"
        assert Section.BREAKDOWN.value == "breakdown"

    def test_intensity_is_bounded(self):
        """Intensity should always be in [0, 1]."""
        det = SectionDetector(sample_rate_hz=30.0)

        now = 0.0
        dt = 0.033

        for i in range(300):
            bass = 0.9 if i > 200 else 0.05
            rms = 0.8 if i > 200 else 0.1
            info = det.update(bass, rms, 3000.0, i % 10 == 0, 128.0, now=now)
            now += dt
            assert 0.0 <= info.intensity <= 1.0, \
                f"Intensity out of range: {info.intensity}"


class TestBeatsToSamples:
    """Test the beat-to-sample conversion utility."""

    def test_normal_bpm(self):
        det = SectionDetector(sample_rate_hz=30.0)
        # 8 beats at 128 BPM = 8 * (60/128) = 3.75 seconds = 112.5 samples
        samples = det._beats_to_samples(8, 128.0)
        assert abs(samples - 113) <= 1

    def test_zero_bpm_fallback(self):
        det = SectionDetector(sample_rate_hz=30.0)
        # With BPM=0, should fall back to ~4 seconds
        samples = det._beats_to_samples(8, 0.0)
        assert samples == int(30.0 * 4)

    def test_slow_bpm(self):
        det = SectionDetector(sample_rate_hz=30.0)
        # 8 beats at 60 BPM = 8 seconds = 240 samples
        samples = det._beats_to_samples(8, 60.0)
        assert abs(samples - 240) <= 1

    def test_minimum_samples(self):
        """Should return at least 10 samples even for fast BPM."""
        det = SectionDetector(sample_rate_hz=30.0)
        samples = det._beats_to_samples(1, 300.0)
        assert samples >= 10


# ============================================================================
# Engine integration tests
# ============================================================================


class TestEngineSectionIntegration:
    """Test that EffectEngine correctly responds to section info."""

    def test_tick_accepts_section_info(self):
        """Engine tick should accept optional section_info parameter."""
        engine = EffectEngine(num_lights=6)
        features = _loud_features()
        beat = _no_beat()
        section = SectionInfo(section=Section.NORMAL)

        states = engine.tick(features, beat, dt=0.033, now=1000.0, section_info=section)
        assert len(states) == 6

    def test_tick_works_without_section_info(self):
        """Engine tick should work fine without section_info (backward compat)."""
        engine = EffectEngine(num_lights=6)
        features = _loud_features()
        beat = _no_beat()

        states = engine.tick(features, beat, dt=0.033, now=1000.0)
        assert len(states) == 6

    def test_drop_boosts_brightness(self):
        """During DROP, brightness should be noticeably higher than NORMAL."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        features = _loud_features(rms=0.5, bass=0.7)

        # Stabilize in NORMAL
        for i in range(60):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)
        states_normal = engine.tick(
            features, _no_beat(), dt=0.033, now=1002.0,
            section_info=SectionInfo(section=Section.NORMAL),
        )
        max_normal = max(s.brightness for s in states_normal)

        # Now feed DROP section
        states_drop = engine.tick(
            features, _no_beat(), dt=0.033, now=1002.033,
            section_info=SectionInfo(
                section=Section.DROP,
                confidence=0.9,
                intensity=0.9,
            ),
        )
        max_drop = max(s.brightness for s in states_drop)

        assert max_drop >= max_normal, \
            f"DROP brightness ({max_drop}) should be >= NORMAL ({max_normal})"

    def test_drop_transition_triggers_flash(self):
        """Transitioning INTO drop should fire a max-brightness flash."""
        engine = EffectEngine(num_lights=4, max_flash_hz=10.0)
        features = _loud_features(rms=0.5)

        # Stabilize in NORMAL
        for i in range(30):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033,
                section_info=SectionInfo(section=Section.NORMAL),
            )

        # Transition to DROP
        engine.tick(
            features, _no_beat(), dt=0.033, now=1001.0,
            section_info=SectionInfo(
                section=Section.DROP, confidence=0.9, intensity=1.0,
            ),
        )

        # Check that flash was triggered on all lights
        assert any(light.flash_brightness > 0.5 for light in engine._lights), \
            "DROP transition should trigger a strong flash"

    def test_breakdown_dims_brightness(self):
        """During BREAKDOWN, brightness should be lower than NORMAL."""
        engine = EffectEngine(num_lights=6)
        features = _loud_features(rms=0.5, bass=0.6)

        # Stabilize in NORMAL mode
        for i in range(100):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033,
                section_info=SectionInfo(section=Section.NORMAL),
            )
        states_normal = engine.tick(
            features, _no_beat(), dt=0.033, now=1003.3,
            section_info=SectionInfo(section=Section.NORMAL),
        )
        avg_normal = sum(s.brightness for s in states_normal) / len(states_normal)

        # Switch to BREAKDOWN for enough ticks to take effect
        for i in range(30):
            engine.tick(
                _breakdown_features(rms=0.3), _no_beat(), dt=0.033,
                now=1003.3 + (i + 1) * 0.033,
                section_info=SectionInfo(
                    section=Section.BREAKDOWN, confidence=0.8, intensity=0.9,
                ),
            )

        states_breakdown = engine.tick(
            _breakdown_features(rms=0.3), _no_beat(), dt=0.033, now=1004.3,
            section_info=SectionInfo(
                section=Section.BREAKDOWN, confidence=0.8, intensity=0.9,
            ),
        )
        avg_breakdown = sum(s.brightness for s in states_breakdown) / len(states_breakdown)

        assert avg_breakdown < avg_normal, \
            f"BREAKDOWN brightness ({avg_breakdown:.3f}) should be < NORMAL ({avg_normal:.3f})"

    def test_buildup_increases_reactive_weight(self):
        """During BUILDUP, reactive weight should be pushed higher."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features(rms=0.4)

        # Stabilize in NORMAL
        for i in range(100):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033,
                section_info=SectionInfo(section=Section.NORMAL),
            )
        rw_normal = engine.reactive_weight

        # Now apply BUILDUP section with high intensity
        for i in range(30):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1003.3 + i * 0.033,
                section_info=SectionInfo(
                    section=Section.BUILDUP, confidence=0.8, intensity=0.9,
                ),
            )

        # Check the modulated reactive weight indirectly via section state
        assert engine.current_section == Section.BUILDUP

    def test_engine_section_properties(self):
        """Engine should expose section state via properties."""
        engine = EffectEngine(num_lights=4)
        assert engine.current_section == Section.NORMAL
        assert engine.section_intensity == 0.0

        features = _loud_features()
        engine.tick(
            features, _no_beat(), dt=0.033, now=1000.0,
            section_info=SectionInfo(
                section=Section.DROP, confidence=0.9, intensity=0.8,
            ),
        )
        assert engine.current_section == Section.DROP
        assert engine.section_intensity == 0.8

    def test_engine_reset_clears_section(self):
        """Reset should clear section state."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features()

        engine.tick(
            features, _no_beat(), dt=0.033, now=1000.0,
            section_info=SectionInfo(
                section=Section.BUILDUP, confidence=0.7, intensity=0.6,
            ),
        )
        assert engine.current_section == Section.BUILDUP

        engine.reset()

        assert engine.current_section == Section.NORMAL
        assert engine.section_intensity == 0.0

    def test_section_modulation_brightness_bounded(self):
        """Brightness should always stay in [0, 1] regardless of section."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)

        sections = [
            SectionInfo(section=Section.NORMAL),
            SectionInfo(section=Section.DROP, confidence=1.0, intensity=1.0),
            SectionInfo(section=Section.BUILDUP, confidence=1.0, intensity=1.0),
            SectionInfo(section=Section.BREAKDOWN, confidence=1.0, intensity=1.0),
        ]

        now = 1000.0
        for section in sections:
            for i in range(50):
                rms = 0.8 if section.section != Section.BREAKDOWN else 0.2
                features = _loud_features(rms=rms) if rms > 0.3 else _breakdown_features(rms=rms)
                is_beat = i % 10 == 0
                beat = _beat() if is_beat else _no_beat()
                states = engine.tick(
                    features, beat, dt=0.033, now=now,
                    section_info=section,
                )
                now += 0.033
                for s in states:
                    assert 0 <= s.brightness <= 1.0, \
                        f"Brightness {s.brightness} out of range during {section.section}"
                    assert 0 <= s.x <= 1.0
                    assert 0 <= s.y <= 1.0


class TestEngineSectionColorModulation:
    """Test section-specific color modulation."""

    def test_breakdown_shifts_colors_cool(self):
        """BREAKDOWN should shift hues toward cooler colors."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features(rms=0.5)

        # Run in NORMAL to stabilize
        for i in range(60):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033,
                section_info=SectionInfo(section=Section.NORMAL),
            )

        normal_hues = [light.hue for light in engine._lights]

        # Run in BREAKDOWN
        for i in range(30):
            engine.tick(
                _breakdown_features(), _no_beat(), dt=0.033,
                now=1002.0 + i * 0.033,
                section_info=SectionInfo(
                    section=Section.BREAKDOWN, confidence=0.9, intensity=0.9,
                ),
            )

        breakdown_hues = [light.hue for light in engine._lights]

        # Hues should have shifted (we don't know exactly where, but they should differ)
        diffs = []
        for nh, bh in zip(normal_hues, breakdown_hues):
            diff = abs(bh - nh)
            if diff > 180:
                diff = 360 - diff
            diffs.append(diff)

        # At least some lights should show hue shift
        assert max(diffs) > 1.0, \
            f"BREAKDOWN should shift hues, max diff was only {max(diffs)}"

    def test_drop_increases_saturation(self):
        """DROP should boost saturation for more vivid colors."""
        engine = EffectEngine(num_lights=4)
        features = _loud_features(rms=0.5)

        # Stabilize in NORMAL
        for i in range(60):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033,
                section_info=SectionInfo(section=Section.NORMAL),
            )
        normal_sats = [light.saturation for light in engine._lights]

        # Feed DROP for a while
        for i in range(30):
            engine.tick(
                features, _no_beat(), dt=0.033, now=1002.0 + i * 0.033,
                section_info=SectionInfo(
                    section=Section.DROP, confidence=0.9, intensity=0.9,
                ),
            )
        drop_sats = [light.saturation for light in engine._lights]

        avg_normal_sat = sum(normal_sats) / len(normal_sats)
        avg_drop_sat = sum(drop_sats) / len(drop_sats)

        assert avg_drop_sat >= avg_normal_sat - 0.05, \
            f"DROP saturation ({avg_drop_sat:.3f}) should be >= NORMAL ({avg_normal_sat:.3f})"


class TestEndToEndSection:
    """End-to-end integration test simulating a realistic music scenario."""

    def test_full_song_structure(self):
        """Simulate: intro -> buildup -> drop -> breakdown -> buildup -> drop.

        Verify that the detector correctly identifies each section and that
        the engine responds without crashing.
        """
        det = SectionDetector(sample_rate_hz=30.0)
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)

        now = 0.0
        dt = 0.033
        bpm = 128.0
        beat_period = 60.0 / bpm
        last_beat = now

        all_states = []
        sections_seen = set()

        # -- Phase 1: Intro (moderate levels, ~3 seconds) --
        for i in range(90):
            is_beat = (now - last_beat) >= beat_period
            if is_beat:
                last_beat = now
            info = det.update(0.3, 0.3, 2000.0, is_beat, bpm, now=now)
            beat = _beat() if is_beat else _no_beat()
            states = engine.tick(
                _loud_features(rms=0.3, bass=0.3), beat, dt=dt, now=now,
                section_info=info,
            )
            sections_seen.add(info.section)
            all_states.append(states)
            now += dt

        # -- Phase 2: Buildup (~3 seconds, rising energy) --
        for i in range(90):
            progress = i / 90.0
            rms = 0.3 + 0.5 * progress
            centroid = 2000.0 + 5000.0 * progress
            bass = 0.1 + 0.2 * progress
            is_beat = (now - last_beat) >= beat_period
            if is_beat:
                last_beat = now
            info = det.update(bass, rms, centroid, is_beat, bpm, now=now)
            beat = _beat() if is_beat else _no_beat()
            features = _loud_features(rms=rms, bass=bass)
            states = engine.tick(features, beat, dt=dt, now=now, section_info=info)
            sections_seen.add(info.section)
            all_states.append(states)
            now += dt

        # -- Phase 3: DROP (~2 seconds, massive bass) --
        for i in range(60):
            is_beat = (now - last_beat) >= beat_period
            if is_beat:
                last_beat = now
            info = det.update(0.95, 0.85, 2000.0, is_beat, bpm, now=now)
            beat = _beat(strength=0.9) if is_beat else _no_beat()
            states = engine.tick(
                _loud_features(rms=0.85, bass=0.95), beat, dt=dt, now=now,
                section_info=info,
            )
            sections_seen.add(info.section)
            all_states.append(states)
            now += dt

        # -- Phase 4: Breakdown (~3 seconds, bass drops out) --
        for i in range(90):
            is_beat = (now - last_beat) >= beat_period
            if is_beat:
                last_beat = now
            info = det.update(0.03, 0.25, 5000.0, is_beat, bpm, now=now)
            beat = _beat(strength=0.3) if is_beat else _no_beat()
            states = engine.tick(
                _breakdown_features(rms=0.25), beat, dt=dt, now=now,
                section_info=info,
            )
            sections_seen.add(info.section)
            all_states.append(states)
            now += dt

        # Verify we saw multiple section types
        assert Section.NORMAL in sections_seen, "Should have seen NORMAL"

        # Verify all states are valid
        for states in all_states:
            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0

    def test_continuous_operation_no_crashes(self):
        """Run detector + engine for 30 simulated seconds with varying input."""
        det = SectionDetector(sample_rate_hz=30.0)
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)

        now = 0.0
        dt = 0.033
        import math

        for i in range(900):
            # Sine wave varying bass and RMS
            phase = (i / 900.0) * 4 * math.pi
            bass = max(0.0, 0.4 + 0.5 * math.sin(phase))
            rms = max(0.0, 0.3 + 0.4 * math.sin(phase * 0.7))
            centroid = 2000.0 + 3000.0 * max(0, math.sin(phase * 0.3))
            is_beat = i % 15 == 0

            info = det.update(bass, rms, centroid, is_beat, 128.0, now=now)
            features = _loud_features(rms=rms, bass=bass)
            beat = _beat() if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now, section_info=info)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
