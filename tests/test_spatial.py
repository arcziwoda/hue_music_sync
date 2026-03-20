"""Tests for SpatialMapper and EffectEngine spatial distribution.

SpatialMapper holds positions, mode constants, and wave state.
EffectEngine._distribute() implements the actual spatial logic.
These tests verify both layers.
"""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.spatial import SpatialMapper
from hue_visualizer.visualizer.engine import EffectEngine


@pytest.fixture
def features():
    return AudioFeatures(
        band_energies=np.array([0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]),
        rms=0.3,
    )


@pytest.fixture
def beat_info():
    return BeatInfo(
        is_beat=False,
        bpm=128.0,
        bpm_confidence=0.0,
        beat_strength=0.0,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


class TestSpatialMapperData:
    """Test SpatialMapper as a data/state container."""

    def test_positions_linear(self):
        sm = SpatialMapper(num_lights=4)
        assert len(sm._positions) == 4
        assert sm._positions[0] == pytest.approx(0.0)
        assert sm._positions[-1] == pytest.approx(1.0)

    def test_single_light_position(self):
        sm = SpatialMapper(num_lights=1)
        assert len(sm._positions) == 1
        assert sm._positions[0] == pytest.approx(0.0)

    def test_modes_list(self):
        assert "uniform" in SpatialMapper.MODES
        assert "frequency_zones" in SpatialMapper.MODES
        assert "wave" in SpatialMapper.MODES
        assert "mirror" in SpatialMapper.MODES

    def test_mode_constants(self):
        assert SpatialMapper.UNIFORM == "uniform"
        assert SpatialMapper.FREQUENCY_ZONES == "frequency_zones"
        assert SpatialMapper.WAVE == "wave"
        assert SpatialMapper.MIRROR == "mirror"

    def test_default_mode(self):
        sm = SpatialMapper(num_lights=4)
        assert sm.mode == "frequency_zones"

    def test_wave_phase_reset(self):
        sm = SpatialMapper(num_lights=4)
        sm._wave_phase = 0.5
        sm.reset()
        assert sm._wave_phase == 0.0

    def test_min_one_light(self):
        sm = SpatialMapper(num_lights=0)
        assert sm.num_lights == 1


class TestEngineDistribute:
    """Test EffectEngine spatial distribution via tick()."""

    def test_uniform_all_same_brightness(self, features, beat_info):
        engine = EffectEngine(num_lights=4, spatial_mode="uniform")
        # Tick multiple times so energy level stabilizes and reactive layer
        # dominates (the generative layer's spatial wave adds small per-light
        # variation that diminishes as reactive weight increases).
        states = None
        for i in range(30):
            states = engine.tick(features, beat_info, dt=0.033, now=1000.0 + i * 0.033)
        assert states is not None
        assert len(states) == 4
        # In uniform mode, the reactive layer gives all lights the same brightness.
        # The generative layer adds spatial wave variation (~25% of base), which
        # blends in proportionally to (1 - reactive_weight). With RMS=0.3 over
        # 30 ticks, reactive_weight stabilizes around 0.5, so generative spatial
        # variation contributes moderately. Tolerance of 0.08 accounts for this.
        brightnesses = [s.brightness for s in states]
        for b in brightnesses:
            assert abs(b - brightnesses[0]) < 0.08

    def test_frequency_zones_returns_correct_count(self, features, beat_info):
        engine = EffectEngine(num_lights=6, spatial_mode="frequency_zones")
        states = engine.tick(features, beat_info, dt=0.033, now=1000.0)
        assert len(states) == 6

    def test_frequency_zones_brightness_varies(self, features, beat_info):
        """With varying band energies, different lights should have different brightness."""
        engine = EffectEngine(num_lights=6, spatial_mode="frequency_zones")
        # Tick multiple times to let smoothing converge
        states = None
        for i in range(30):
            states = engine.tick(features, beat_info, dt=0.033, now=1000.0 + i * 0.033)
        assert states is not None
        brightnesses = [s.brightness for s in states]
        # First light (bass-heavy) should differ from last (treble-heavy)
        # because band_energies are unequal
        assert brightnesses[0] != pytest.approx(brightnesses[-1], abs=0.01)

    def test_wave_returns_correct_count(self, features, beat_info):
        engine = EffectEngine(num_lights=4, spatial_mode="wave")
        states = engine.tick(features, beat_info, dt=0.033, now=1000.0)
        assert len(states) == 4

    def test_wave_brightness_varies(self, features, beat_info):
        """Wave mode should create brightness variation across lights."""
        engine = EffectEngine(num_lights=6, spatial_mode="wave")
        # Tick multiple times to let wave phase advance and smoothing converge
        states = None
        for i in range(30):
            states = engine.tick(features, beat_info, dt=0.033, now=1000.0 + i * 0.033)
        assert states is not None
        brightnesses = [s.brightness for s in states]
        # Not all brightnesses should be equal (wave creates variation)
        assert max(brightnesses) - min(brightnesses) > 0.01

    def test_mirror_symmetric(self, features, beat_info):
        """Mirror mode should produce symmetric brightness pattern."""
        engine = EffectEngine(num_lights=6, spatial_mode="mirror")
        # Tick enough times for smoothing to converge
        states = None
        for i in range(60):
            states = engine.tick(features, beat_info, dt=0.033, now=1000.0 + i * 0.033)
        assert states is not None
        assert abs(states[0].brightness - states[5].brightness) < 0.05
        assert abs(states[1].brightness - states[4].brightness) < 0.05

    def test_single_light(self, features, beat_info):
        engine = EffectEngine(num_lights=1, spatial_mode="frequency_zones")
        states = engine.tick(features, beat_info, dt=0.033, now=1000.0)
        assert len(states) == 1

    def test_set_spatial_mode(self):
        engine = EffectEngine(num_lights=4, spatial_mode="uniform")
        assert engine.spatial_mapper.mode == "uniform"
        engine.set_spatial_mode("wave")
        assert engine.spatial_mapper.mode == "wave"

    def test_set_spatial_mode_invalid_ignored(self):
        engine = EffectEngine(num_lights=4, spatial_mode="uniform")
        engine.set_spatial_mode("invalid_mode")
        assert engine.spatial_mapper.mode == "uniform"
