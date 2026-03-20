"""Tests for ColorMapper."""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.visualizer.color_mapper import (
    ColorMapper,
    _ema,
    _smooth_hue,
    _smooth_hue_range,
    centroid_to_hue,
    COLOR_MODE_PALETTE,
    COLOR_MODE_CENTROID,
)


@pytest.fixture
def mapper():
    return ColorMapper(gamma=2.2)


def _features(centroid=1000.0, rms=0.3, flatness=0.2, flux=0.0) -> AudioFeatures:
    return AudioFeatures(
        spectral_centroid=centroid,
        rms=rms,
        spectral_flatness=flatness,
        spectral_flux=flux,
        band_energies=np.zeros(7),
    )


class TestColorMapper:
    def test_returns_three_values(self, mapper):
        offset, s, b = mapper.map(_features())
        assert -25 <= offset <= 25  # hue offset
        assert 0 <= s <= 1
        assert 0 <= b <= 1

    def test_low_centroid_negative_offset(self, mapper):
        for _ in range(50):
            offset, _, _ = mapper.map(_features(centroid=100))
        assert offset < 0  # warm shift

    def test_high_centroid_positive_offset(self, mapper):
        for _ in range(50):
            offset, _, _ = mapper.map(_features(centroid=8000))
        assert offset > 0  # cool shift

    def test_high_rms_bright(self, mapper):
        for _ in range(50):
            _, _, b = mapper.map(_features(rms=0.8))
        assert b > 0.3

    def test_silence_dark(self, mapper):
        for _ in range(50):
            _, _, b = mapper.map(_features(rms=0.0))
        assert b < 0.05

    def test_tonal_sound_high_saturation(self, mapper):
        for _ in range(50):
            _, s, _ = mapper.map(_features(flatness=0.0))
        assert s > 0.8

    def test_noisy_sound_low_saturation(self, mapper):
        for _ in range(50):
            _, s, _ = mapper.map(_features(flatness=0.9))
        assert s < 0.5

    def test_high_flux_faster_offset_change(self, mapper):
        # Low flux: offset changes slowly
        for _ in range(20):
            mapper.map(_features(centroid=100, flux=0.0))
        offset_low_flux = mapper._hue_offset

        mapper.reset()

        # High flux: offset changes faster
        for _ in range(20):
            mapper.map(_features(centroid=100, flux=100.0))
        offset_high_flux = mapper._hue_offset

        assert abs(offset_high_flux) > abs(offset_low_flux)

    def test_reset(self, mapper):
        mapper.map(_features(centroid=5000, rms=0.5))
        mapper.reset()
        assert mapper._brightness == 0.0
        assert mapper._hue_offset == 0.0


class TestEma:
    def test_ema_moves_toward_target(self):
        assert _ema(0.0, 1.0, 0.5) == 0.5
        assert _ema(1.0, 0.0, 0.5) == 0.5

    def test_ema_alpha_zero_no_change(self):
        assert _ema(0.5, 1.0, 0.0) == 0.5

    def test_ema_alpha_one_immediate(self):
        assert _ema(0.0, 1.0, 1.0) == 1.0


class TestSmoothHue:
    def test_shortest_path_forward(self):
        result = _smooth_hue(10.0, 50.0, 1.0)
        assert abs(result - 50.0) < 0.1

    def test_shortest_path_wraps(self):
        result = _smooth_hue(350.0, 10.0, 1.0)
        assert abs(result - 10.0) < 0.1

    def test_partial_step(self):
        result = _smooth_hue(0.0, 100.0, 0.5)
        assert abs(result - 50.0) < 0.1


class TestCentroidToHue:
    """Tests for the centroid_to_hue log-scale mapping function."""

    def test_low_frequency_maps_to_red(self):
        # 100 Hz -> 0 degrees (red)
        assert centroid_to_hue(100.0) == 0.0

    def test_high_frequency_maps_to_violet(self):
        # 10000 Hz -> 300 degrees (violet)
        assert centroid_to_hue(10000.0) == 300.0

    def test_below_min_clamped_to_zero(self):
        assert centroid_to_hue(50.0) == 0.0

    def test_above_max_clamped_to_300(self):
        assert centroid_to_hue(20000.0) == 300.0

    def test_mid_frequency_warm_range(self):
        # ~316 Hz (geometric mean of 100 and 1000) should be ~25% through range
        hue = centroid_to_hue(316.0)
        assert 60 < hue < 90  # roughly 75 degrees

    def test_1000hz_is_midrange(self):
        # 1000 Hz: log(1000) - log(100) / (log(10000) - log(100))
        # = (6.9 - 4.6) / (9.2 - 4.6) = 2.3 / 4.6 = 0.5 -> 150 degrees
        hue = centroid_to_hue(1000.0)
        assert 145 < hue < 155  # ~150 degrees

    def test_monotonically_increasing(self):
        freqs = [100, 200, 500, 1000, 2000, 5000, 10000]
        hues = [centroid_to_hue(f) for f in freqs]
        for i in range(len(hues) - 1):
            assert hues[i] < hues[i + 1]


class TestCentroidMode:
    """Tests for the centroid color mode in ColorMapper."""

    @pytest.fixture
    def centroid_mapper(self):
        return ColorMapper(gamma=2.2, color_mode=COLOR_MODE_CENTROID)

    def test_default_mode_is_palette(self, mapper):
        assert mapper.color_mode == COLOR_MODE_PALETTE

    def test_centroid_mode_constructor(self, centroid_mapper):
        assert centroid_mapper.color_mode == COLOR_MODE_CENTROID

    def test_set_color_mode(self, mapper):
        mapper.set_color_mode(COLOR_MODE_CENTROID)
        assert mapper.color_mode == COLOR_MODE_CENTROID
        mapper.set_color_mode(COLOR_MODE_PALETTE)
        assert mapper.color_mode == COLOR_MODE_PALETTE

    def test_invalid_mode_ignored(self, mapper):
        mapper.set_color_mode("invalid")
        assert mapper.color_mode == COLOR_MODE_PALETTE

    def test_centroid_mode_returns_hue_not_offset(self, centroid_mapper):
        """In centroid mode, first return value is a direct hue (0-300), not offset."""
        for _ in range(50):
            hue, s, b = centroid_mapper.map(_features(centroid=1000.0, rms=0.5))
        # 1000 Hz -> ~150 degrees
        assert 100 < hue < 200

    def test_centroid_mode_low_freq_warm(self, centroid_mapper):
        """Low centroid (bass) should produce warm/red hue."""
        for _ in range(80):
            hue, _, _ = centroid_mapper.map(_features(centroid=150.0, rms=0.3))
        assert hue < 60  # warm: red/orange/yellow range

    def test_centroid_mode_high_freq_cool(self, centroid_mapper):
        """High centroid (treble) should produce cool/violet hue."""
        for _ in range(80):
            hue, _, _ = centroid_mapper.map(_features(centroid=8000.0, rms=0.3))
        assert hue > 200  # cool: blue/violet range

    def test_centroid_mode_brightness_still_works(self, centroid_mapper):
        """Brightness should still respond to RMS in centroid mode."""
        for _ in range(50):
            _, _, b = centroid_mapper.map(_features(rms=0.8))
        assert b > 0.3

    def test_centroid_mode_saturation_still_works(self, centroid_mapper):
        """Saturation should still respond to flatness in centroid mode."""
        for _ in range(50):
            _, s, _ = centroid_mapper.map(_features(flatness=0.0))
        assert s > 0.8

    def test_centroid_mode_high_flux_faster_change(self, centroid_mapper):
        """Higher spectral flux should cause faster hue changes in centroid mode."""
        # Low flux
        for _ in range(20):
            centroid_mapper.map(_features(centroid=8000, flux=0.0))
        hue_slow = centroid_mapper._centroid_hue

        centroid_mapper.reset()

        # High flux
        for _ in range(20):
            centroid_mapper.map(_features(centroid=8000, flux=100.0))
        hue_fast = centroid_mapper._centroid_hue

        # Both should be heading toward ~270 deg, but high flux gets there faster
        # Starting from reset (180 deg), fast should be further from 180
        assert abs(hue_fast - 180) > abs(hue_slow - 180)

    def test_reset_clears_centroid_hue(self, centroid_mapper):
        centroid_mapper.map(_features(centroid=5000))
        centroid_mapper.reset()
        assert centroid_mapper._centroid_hue == 180.0

    def test_switching_modes_preserves_state(self, mapper):
        """Switching between modes shouldn't cause jumps."""
        # Run some frames in palette mode
        for _ in range(20):
            mapper.map(_features(centroid=1000, rms=0.5))

        # Switch to centroid mode
        mapper.set_color_mode(COLOR_MODE_CENTROID)
        assert mapper.color_mode == COLOR_MODE_CENTROID

        # Should still produce valid values
        hue, s, b = mapper.map(_features(centroid=1000, rms=0.5))
        assert 0 <= hue <= 360
        assert 0 <= s <= 1
        assert 0 <= b <= 1


class TestSmoothHueRange:
    """Tests for the _smooth_hue_range helper."""

    def test_basic_interpolation(self):
        result = _smooth_hue_range(0.0, 100.0, 1.0, 360.0)
        assert abs(result - 100.0) < 0.1

    def test_wraps_around(self):
        # 350 -> 10 should go forward (shortest path) on 360 scale
        result = _smooth_hue_range(350.0, 10.0, 1.0, 360.0)
        assert abs(result - 10.0) < 0.1

    def test_partial_step(self):
        result = _smooth_hue_range(0.0, 100.0, 0.5, 360.0)
        assert abs(result - 50.0) < 0.1

    def test_custom_max_val(self):
        # On a 300 degree scale
        result = _smooth_hue_range(0.0, 150.0, 1.0, 300.0)
        assert abs(result - 150.0) < 0.1
