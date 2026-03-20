"""Tests for AudioAnalyzer."""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioAnalyzer, AudioFeatures


@pytest.fixture
def analyzer():
    return AudioAnalyzer(sample_rate=44100, fft_size=2048, bass_boost=2.0)


def _sine(freq: float, duration_samples: int = 2048, sr: int = 44100) -> np.ndarray:
    t = np.arange(duration_samples) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


class TestAudioAnalyzer:
    def test_silence_returns_zeros(self, analyzer):
        frame = np.zeros(1024, dtype=np.float32)
        f = analyzer.analyze(frame)
        assert f.rms == 0.0 or f.rms < 0.01
        assert f.peak < 0.01

    def test_sine_wave_has_energy(self, analyzer):
        frame = _sine(440, 2048)
        f = analyzer.analyze(frame)
        assert f.peak > 0.1
        assert f.spectral_centroid > 0

    def test_bass_sine_centroid_low(self, analyzer):
        frame = _sine(100, 2048)
        f = analyzer.analyze(frame)
        assert f.spectral_centroid < 500

    def test_treble_sine_centroid_high(self, analyzer):
        frame = _sine(8000, 2048)
        f = analyzer.analyze(frame)
        assert f.spectral_centroid > 3000

    def test_band_energies_shape(self, analyzer):
        frame = _sine(440, 2048)
        f = analyzer.analyze(frame)
        assert f.band_energies.shape == (7,)
        assert all(0 <= v <= 2.0 for v in f.band_energies)

    def test_spectral_flux_increases_on_change(self, analyzer):
        # First frame: silence
        analyzer.analyze(np.zeros(2048, dtype=np.float32))
        # Second frame: loud sine -> high flux
        f = analyzer.analyze(_sine(1000, 2048))
        assert f.spectral_flux > 0

    def test_auto_gain_normalizes_rms(self, analyzer):
        # Feed consistent quiet signal
        for _ in range(100):
            f = analyzer.analyze(np.random.randn(1024).astype(np.float32) * 0.01)
        # Now a louder frame should normalize near 1.0
        f = analyzer.analyze(np.random.randn(1024).astype(np.float32) * 0.1)
        assert f.rms > 0.5

    def test_spectrum_is_db_scale(self, analyzer):
        f = analyzer.analyze(_sine(440, 2048))
        assert len(f.spectrum) > 0
        # dB values should be negative for most bins
        assert np.mean(f.spectrum) < 0

    def test_reset_clears_state(self, analyzer):
        analyzer.analyze(_sine(440, 2048))
        analyzer.reset()
        assert analyzer._prev_frame is None
        assert analyzer._prev_magnitude is None
        assert len(analyzer._rms_history) == 0
