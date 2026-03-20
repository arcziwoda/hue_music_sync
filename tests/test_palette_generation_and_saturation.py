"""Tests for algorithmic palette generation (Task 2.10) and saturation slider (Task 2.19)."""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.visualizer.presets import (
    generate_palette,
    PALETTE_ALGO_MODES,
)
from hue_visualizer.visualizer.color_mapper import ColorMapper, COLOR_MODE_PALETTE
from hue_visualizer.visualizer.engine import EffectEngine


def _features(centroid=1000.0, rms=0.3, flatness=0.2, flux=0.0) -> AudioFeatures:
    return AudioFeatures(
        spectral_centroid=centroid,
        rms=rms,
        spectral_flatness=flatness,
        spectral_flux=flux,
        band_energies=np.zeros(7),
    )


# =========================================================================
# Task 2.10: Algorithmic palette generation
# =========================================================================


class TestGeneratePaletteComplementary:
    """Complementary palettes: base + opposite + two accents."""

    def test_returns_four_hues(self):
        palette = generate_palette("complementary", 0.0)
        assert len(palette) == 4

    def test_base_hue_is_first(self):
        palette = generate_palette("complementary", 120.0)
        assert palette[0] == 120.0

    def test_opposite_is_180_apart(self):
        palette = generate_palette("complementary", 60.0)
        # Second hue should be 60 + 180 = 240
        assert abs(palette[1] - 240.0) < 0.01

    def test_accents_at_80_and_260(self):
        palette = generate_palette("complementary", 0.0)
        assert abs(palette[2] - 80.0) < 0.01
        assert abs(palette[3] - 260.0) < 0.01

    def test_wraps_around_360(self):
        palette = generate_palette("complementary", 300.0)
        # base=300, opposite=300+180=480%360=120
        assert abs(palette[0] - 300.0) < 0.01
        assert abs(palette[1] - 120.0) < 0.01
        # accents: 300+80=380%360=20, 300+260=560%360=200
        assert abs(palette[2] - 20.0) < 0.01
        assert abs(palette[3] - 200.0) < 0.01

    def test_all_hues_in_valid_range(self):
        for base in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
            palette = generate_palette("complementary", float(base))
            for h in palette:
                assert 0.0 <= h < 360.0, f"Hue {h} out of range for base={base}"


class TestGeneratePaletteTriadic:
    """Triadic palettes: three hues 120 degrees apart."""

    def test_returns_three_hues(self):
        palette = generate_palette("triadic", 0.0)
        assert len(palette) == 3

    def test_evenly_spaced_120_degrees(self):
        palette = generate_palette("triadic", 30.0)
        assert abs(palette[0] - 30.0) < 0.01
        assert abs(palette[1] - 150.0) < 0.01
        assert abs(palette[2] - 270.0) < 0.01

    def test_wraps_correctly(self):
        palette = generate_palette("triadic", 300.0)
        assert abs(palette[0] - 300.0) < 0.01
        assert abs(palette[1] - 60.0) < 0.01  # 300+120=420%360=60
        assert abs(palette[2] - 180.0) < 0.01  # 300+240=540%360=180

    def test_all_hues_in_valid_range(self):
        for base in range(0, 360, 30):
            palette = generate_palette("triadic", float(base))
            for h in palette:
                assert 0.0 <= h < 360.0


class TestGeneratePaletteAnalogous:
    """Analogous palettes: three hues 30 degrees apart (tight cluster)."""

    def test_returns_three_hues(self):
        palette = generate_palette("analogous", 180.0)
        assert len(palette) == 3

    def test_centered_around_base(self):
        palette = generate_palette("analogous", 180.0)
        assert abs(palette[0] - 150.0) < 0.01  # base - 30
        assert abs(palette[1] - 180.0) < 0.01  # base
        assert abs(palette[2] - 210.0) < 0.01  # base + 30

    def test_wraps_near_zero(self):
        palette = generate_palette("analogous", 10.0)
        assert abs(palette[0] - 340.0) < 0.01  # 10-30=-20 -> 340
        assert abs(palette[1] - 10.0) < 0.01
        assert abs(palette[2] - 40.0) < 0.01

    def test_all_hues_in_valid_range(self):
        for base in range(0, 360, 15):
            palette = generate_palette("analogous", float(base))
            for h in palette:
                assert 0.0 <= h < 360.0

    def test_hues_are_close_together(self):
        """Analogous palettes should have hues within 60-degree span."""
        palette = generate_palette("analogous", 200.0)
        # All hues should be within 60 degrees of each other
        for i in range(len(palette)):
            for j in range(i + 1, len(palette)):
                diff = abs(palette[i] - palette[j])
                if diff > 180:
                    diff = 360 - diff
                assert diff <= 60


class TestGeneratePaletteValidation:
    """Input validation and edge cases."""

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown palette algorithm"):
            generate_palette("invalid_mode", 0.0)

    def test_negative_hue_normalized(self):
        palette = generate_palette("triadic", -60.0)
        # -60 % 360 = 300
        assert abs(palette[0] - 300.0) < 0.01

    def test_hue_above_360_normalized(self):
        palette = generate_palette("triadic", 720.0)
        # 720 % 360 = 0
        assert abs(palette[0] - 0.0) < 0.01

    def test_all_modes_are_valid(self):
        """Every mode listed in PALETTE_ALGO_MODES should work."""
        for mode in PALETTE_ALGO_MODES:
            palette = generate_palette(mode, 180.0)
            assert len(palette) >= 2
            for h in palette:
                assert 0.0 <= h < 360.0

    def test_different_base_hues_produce_different_palettes(self):
        p1 = generate_palette("triadic", 0.0)
        p2 = generate_palette("triadic", 90.0)
        assert p1 != p2


class TestGeneratePaletteHarmony:
    """Verify that generated palettes are aesthetically harmonious."""

    def test_complementary_has_high_contrast(self):
        """Complementary pairs should span a wide hue range."""
        palette = generate_palette("complementary", 60.0)
        # base (60) and opposite (240) are 180 degrees apart
        diff = abs(palette[0] - palette[1])
        if diff > 180:
            diff = 360 - diff
        assert diff == 180.0

    def test_triadic_equal_spacing(self):
        """All gaps between triadic hues should be exactly 120 degrees."""
        palette = generate_palette("triadic", 45.0)
        for i in range(3):
            diff = (palette[(i + 1) % 3] - palette[i]) % 360
            assert abs(diff - 120.0) < 0.01

    def test_analogous_tight_range(self):
        """Analogous palettes should feel cohesive (within 60 degrees)."""
        palette = generate_palette("analogous", 90.0)
        # Total span: base-30 to base+30 = 60 degrees
        h_min = min(palette)
        h_max = max(palette)
        span = h_max - h_min
        if span > 180:
            span = 360 - span
        assert span == 60.0


# =========================================================================
# Task 2.19: Saturation slider (ColorMapper)
# =========================================================================


class TestSaturationBoostColorMapper:
    """Tests for saturation_boost multiplier on ColorMapper."""

    def test_default_boost_is_1(self):
        mapper = ColorMapper()
        assert mapper.saturation_boost == 1.0

    def test_constructor_boost(self):
        mapper = ColorMapper(saturation_boost=0.5)
        assert mapper.saturation_boost == 0.5

    def test_constructor_clamps_above_1(self):
        mapper = ColorMapper(saturation_boost=1.5)
        assert mapper.saturation_boost == 1.0

    def test_constructor_clamps_below_0(self):
        mapper = ColorMapper(saturation_boost=-0.5)
        assert mapper.saturation_boost == 0.0

    def test_set_saturation_boost(self):
        mapper = ColorMapper()
        mapper.set_saturation_boost(0.7)
        assert mapper.saturation_boost == 0.7

    def test_set_saturation_boost_clamps(self):
        mapper = ColorMapper()
        mapper.set_saturation_boost(2.0)
        assert mapper.saturation_boost == 1.0
        mapper.set_saturation_boost(-1.0)
        assert mapper.saturation_boost == 0.0

    def test_zero_boost_gives_zero_saturation(self):
        mapper = ColorMapper(saturation_boost=0.0)
        for _ in range(30):
            _, sat, _ = mapper.map(_features(flatness=0.0, rms=0.5))
        assert sat == 0.0

    def test_full_boost_gives_normal_saturation(self):
        mapper = ColorMapper(saturation_boost=1.0)
        for _ in range(50):
            _, sat, _ = mapper.map(_features(flatness=0.0, rms=0.3))
        assert sat > 0.7  # tonal -> high saturation

    def test_half_boost_reduces_saturation(self):
        # Full boost
        full = ColorMapper(saturation_boost=1.0)
        for _ in range(50):
            _, sat_full, _ = full.map(_features(flatness=0.1, rms=0.3))

        # Half boost
        half = ColorMapper(saturation_boost=0.5)
        for _ in range(50):
            _, sat_half, _ = half.map(_features(flatness=0.1, rms=0.3))

        # Half should be approximately half of full
        assert sat_half < sat_full
        assert abs(sat_half - sat_full * 0.5) < 0.05

    def test_boost_does_not_affect_brightness(self):
        mapper_full = ColorMapper(saturation_boost=1.0)
        mapper_zero = ColorMapper(saturation_boost=0.0)

        for _ in range(50):
            _, _, b_full = mapper_full.map(_features(rms=0.5))
            _, _, b_zero = mapper_zero.map(_features(rms=0.5))

        # Brightness should be the same regardless of saturation boost
        assert abs(b_full - b_zero) < 0.01

    def test_boost_does_not_affect_hue(self):
        mapper_full = ColorMapper(saturation_boost=1.0)
        mapper_zero = ColorMapper(saturation_boost=0.0)

        for _ in range(50):
            h_full, _, _ = mapper_full.map(_features(centroid=3000.0))
            h_zero, _, _ = mapper_zero.map(_features(centroid=3000.0))

        assert abs(h_full - h_zero) < 0.01

    def test_dynamic_boost_change(self):
        """Changing boost mid-stream should immediately affect output."""
        mapper = ColorMapper(saturation_boost=1.0)

        # Run a few frames at full saturation
        for _ in range(30):
            _, sat_full, _ = mapper.map(_features(flatness=0.0, rms=0.3))

        # Now reduce to zero
        mapper.set_saturation_boost(0.0)
        _, sat_zero, _ = mapper.map(_features(flatness=0.0, rms=0.3))
        assert sat_zero == 0.0


# =========================================================================
# Task 2.19: Saturation slider (EffectEngine integration)
# =========================================================================


class TestSaturationBoostEngine:
    """Tests for saturation_boost integration with EffectEngine."""

    @pytest.fixture
    def engine(self):
        return EffectEngine(num_lights=4, gamma=2.2)

    def test_default_saturation_boost(self, engine):
        assert engine.saturation_boost == 1.0

    def test_set_saturation_boost(self, engine):
        engine.set_saturation_boost(0.5)
        assert engine.saturation_boost == 0.5

    def test_saturation_boost_clamps(self, engine):
        engine.set_saturation_boost(2.0)
        assert engine.saturation_boost == 1.0
        engine.set_saturation_boost(-1.0)
        assert engine.saturation_boost == 0.0

    def test_saturation_boost_delegates_to_color_mapper(self, engine):
        engine.set_saturation_boost(0.3)
        assert engine.color_mapper.saturation_boost == 0.3


# =========================================================================
# Integration: palette generation used with EffectEngine
# =========================================================================


class TestPaletteAlgoWithEngine:
    """Integration tests: generated palettes work with EffectEngine.set_palette()."""

    @pytest.fixture
    def engine(self):
        return EffectEngine(num_lights=6, gamma=2.2)

    def test_complementary_palette_accepted(self, engine):
        palette = generate_palette("complementary", 200.0)
        engine.set_palette(palette)
        assert engine._palette == palette

    def test_triadic_palette_accepted(self, engine):
        palette = generate_palette("triadic", 120.0)
        engine.set_palette(palette)
        assert engine._palette == palette

    def test_analogous_palette_accepted(self, engine):
        palette = generate_palette("analogous", 60.0)
        engine.set_palette(palette)
        assert engine._palette == palette

    def test_generated_palette_propagates_to_generative_layer(self, engine):
        palette = generate_palette("triadic", 90.0)
        engine.set_palette(palette)
        assert engine._generative._palette == palette

    def test_engine_ticks_with_algo_palette(self, engine):
        """Engine should produce valid light states with a generated palette."""
        from hue_visualizer.audio.beat_detector import BeatInfo

        palette = generate_palette("complementary", 270.0)
        engine.set_palette(palette)

        features = _features(rms=0.5, centroid=2000.0)
        beat = BeatInfo()

        states = engine.tick(features, beat, dt=0.033)
        assert len(states) == 6
        for s in states:
            assert 0.0 <= s.brightness <= 1.0
            assert 0.0 <= s.x <= 1.0
            assert 0.0 <= s.y <= 1.0
