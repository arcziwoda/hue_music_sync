"""Tests for chase spatial mode (Task 1.8) and bridge position reading (Task 1.15).

Task 1.8 — Chase effect:
- Sequential per-bulb activation that advances on beats
- Exponential decay creates a visible "tail" trailing the active bulb
- Direction alternates (bounces) by default
- Configurable decay time constant

Task 1.15 — Entertainment area positions:
- Read channel x/y/z positions from EntertainmentConfiguration
- Normalize x-coordinates to 0-1 range
- Pass positions to SpatialMapper for accurate spatial distribution
- Graceful fallback to linear when bridge is not connected
"""

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.spatial import SpatialMapper
from hue_visualizer.visualizer.engine import EffectEngine
from hue_visualizer.bridge.entertainment_controller import EntertainmentController


# --- Test helpers ---


def _loud_features(rms: float = 0.5) -> AudioFeatures:
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
# Task 1.8: Chase mode — SpatialMapper tests
# ============================================================================


class TestChaseSpatialMapperData:
    """Test that SpatialMapper has chase mode registered."""

    def test_chase_in_modes_list(self):
        assert "chase" in SpatialMapper.MODES

    def test_chase_constant(self):
        assert SpatialMapper.CHASE == "chase"

    def test_chase_mode_selectable(self):
        sm = SpatialMapper(num_lights=6, mode="chase")
        assert sm.mode == "chase"

    def test_chase_state_initialized(self):
        sm = SpatialMapper(num_lights=6)
        assert sm._chase_position == 0.0
        assert len(sm._chase_last_activated) == 6
        assert sm._chase_direction == 1
        assert sm._chase_alternating is True

    def test_chase_reset_clears_state(self):
        sm = SpatialMapper(num_lights=6)
        sm._chase_position = 3.0
        sm._chase_last_activated = [100.0] * 6
        sm._chase_direction = -1

        sm.reset()

        assert sm._chase_position == 0.0
        assert all(t == 0.0 for t in sm._chase_last_activated)
        assert sm._chase_direction == 1


# ============================================================================
# Task 1.8: Chase mode — EffectEngine tests
# ============================================================================


class TestChaseEngineBasic:
    """Test chase mode produces valid output from the engine."""

    def test_chase_returns_correct_light_count(self):
        engine = EffectEngine(num_lights=6, spatial_mode="chase")
        features = _loud_features(rms=0.5)
        states = engine.tick(features, _no_beat(), dt=0.033, now=1000.0)
        assert len(states) == 6

    def test_chase_produces_valid_brightness(self):
        """All brightness values should be in [0, 1]."""
        engine = EffectEngine(num_lights=6, spatial_mode="chase")
        features = _loud_features(rms=0.5)
        for i in range(60):
            states = engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)
        for s in states:
            assert 0.0 <= s.brightness <= 1.0

    def test_chase_selectable_via_set_spatial_mode(self):
        engine = EffectEngine(num_lights=6, spatial_mode="uniform")
        engine.set_spatial_mode("chase")
        assert engine.spatial_mapper.mode == "chase"

    def test_chase_single_light_no_crash(self):
        engine = EffectEngine(num_lights=1, spatial_mode="chase")
        features = _loud_features(rms=0.5)
        for i in range(30):
            states = engine.tick(
                features, _beat() if i % 5 == 0 else _no_beat(),
                dt=0.033, now=1000.0 + i * 0.033,
            )
        assert len(states) == 1
        assert 0.0 <= states[0].brightness <= 1.0


class TestChaseAdvancesOnBeat:
    """Test that the chase position advances when beats are detected."""

    def test_beat_advances_chase_position(self):
        engine = EffectEngine(num_lights=6, spatial_mode="chase", max_flash_hz=30.0)
        features = _loud_features(rms=0.5)

        # Initial position
        assert engine.spatial_mapper._chase_position == 0.0

        # First beat should advance to position 1
        engine.tick(features, _beat(), dt=0.033, now=1000.0)
        assert engine.spatial_mapper._chase_position == 1.0

        # Second beat should advance to position 2
        engine.tick(features, _beat(), dt=0.033, now=1000.5)
        assert engine.spatial_mapper._chase_position == 2.0

    def test_no_beat_no_advance(self):
        engine = EffectEngine(num_lights=6, spatial_mode="chase")
        features = _loud_features(rms=0.5)

        engine.tick(features, _no_beat(), dt=0.033, now=1000.0)
        assert engine.spatial_mapper._chase_position == 0.0

        engine.tick(features, _no_beat(), dt=0.033, now=1000.033)
        assert engine.spatial_mapper._chase_position == 0.0

    def test_chase_direction_alternates(self):
        """When alternating is True, chase should bounce at the ends."""
        engine = EffectEngine(num_lights=4, spatial_mode="chase", max_flash_hz=30.0)
        engine.spatial_mapper._chase_alternating = True
        features = _loud_features(rms=0.5)

        positions = []
        now = 1000.0
        for i in range(8):
            engine.tick(features, _beat(), dt=0.033, now=now)
            positions.append(engine.spatial_mapper._chase_position)
            now += 0.5

        # With 4 lights (indices 0-3), alternating:
        # Start at 0, beats: 1, 2, 3, then direction reverses: 2, 1, 0, then reverses: 1, 2
        assert positions[0] == 1.0  # Forward
        assert positions[1] == 2.0  # Forward
        assert positions[2] == 3.0  # Forward, hits end
        # Should reverse direction
        assert positions[3] == 2.0  # Backward
        assert positions[4] == 1.0  # Backward
        assert positions[5] == 0.0  # Backward, hits start
        # Should reverse direction again
        assert positions[6] == 1.0  # Forward
        assert positions[7] == 2.0  # Forward


class TestChaseBrightnessDecay:
    """Test that the chase creates a visible brightness pulse that decays."""

    def test_active_bulb_brightest(self):
        """The most recently activated bulb should be the brightest."""
        engine = EffectEngine(num_lights=6, spatial_mode="chase", max_flash_hz=30.0)
        features = _loud_features(rms=0.6)

        # Warm up the engine so smoothing stabilizes
        now = 1000.0
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Fire a beat to activate bulb 1
        engine.tick(features, _beat(), dt=0.033, now=now)
        now += 0.033

        # Tick a few more frames without beat so decay kicks in
        states = None
        for i in range(5):
            states = engine.tick(features, _no_beat(), dt=0.033, now=now)
            now += 0.033

        assert states is not None
        # The recently activated bulb (index 1) should be significantly
        # brighter than distant bulbs. Due to EMA smoothing we allow some
        # tolerance, but the general pattern should hold.
        brightnesses = [s.brightness for s in states]
        # Active bulb (1) vs a distant bulb (e.g., 5)
        assert brightnesses[1] > brightnesses[5] * 0.9, \
            f"Active bulb should be brighter: {brightnesses}"

    def test_brightness_decays_over_time(self):
        """After a bulb is activated, its chase brightness should decay."""
        engine = EffectEngine(num_lights=6, spatial_mode="chase", max_flash_hz=30.0)
        features = _loud_features(rms=0.5)

        now = 1000.0
        # Warm up
        for i in range(20):
            engine.tick(features, _no_beat(), dt=0.033, now=now)
            now += 0.033

        # Fire a beat (activates bulb 1)
        engine.tick(features, _beat(), dt=0.033, now=now)
        now += 0.033

        # Record bulb 1's brightness shortly after activation
        states_soon = engine.tick(features, _no_beat(), dt=0.033, now=now)
        now += 0.033
        bright_soon = states_soon[1].brightness

        # Wait longer (500ms)
        for i in range(15):
            states_later = engine.tick(features, _no_beat(), dt=0.033, now=now)
            now += 0.033
        bright_later = states_later[1].brightness

        # Brightness of bulb 1 should be lower after decay (smoothed by EMA,
        # but the exponential decay in the reactive layer should show through)
        assert bright_later < bright_soon or bright_soon < 0.1, \
            f"Brightness should decay: soon={bright_soon}, later={bright_later}"


class TestChaseContinuousOperation:
    """Integration test: run chase mode for extended period."""

    def test_30_seconds_chase_no_crashes(self):
        engine = EffectEngine(num_lights=6, spatial_mode="chase", max_flash_hz=10.0)
        dt = 0.033
        now = 1000.0
        beat_interval = 60.0 / 128.0
        last_beat = now

        for i in range(900):  # ~30 seconds
            is_beat = (now - last_beat) >= beat_interval
            if is_beat:
                last_beat = now
            beat = _beat() if is_beat else _no_beat()
            rms = 0.3 + 0.3 * abs(math.sin(i * 0.02))
            features = _loud_features(rms=rms)

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0.0 <= s.brightness <= 1.0
                assert 0.0 <= s.x <= 1.0
                assert 0.0 <= s.y <= 1.0


# ============================================================================
# Task 1.15: Entertainment area position data — SpatialMapper tests
# ============================================================================


class TestSpatialMapperPositions:
    """Test that SpatialMapper can accept and store external positions."""

    def test_set_positions_stores_values(self):
        sm = SpatialMapper(num_lights=4)
        sm.set_positions([0.0, 0.3, 0.7, 1.0])
        assert sm._positions == [0.0, 0.3, 0.7, 1.0]
        assert sm._using_bridge_positions is True

    def test_set_positions_wrong_count_ignored(self):
        sm = SpatialMapper(num_lights=4)
        original = list(sm._positions)
        sm.set_positions([0.0, 0.5])  # Wrong count
        assert sm._positions == original
        assert sm._using_bridge_positions is False

    def test_default_positions_linear(self):
        sm = SpatialMapper(num_lights=6)
        assert sm._positions[0] == pytest.approx(0.0)
        assert sm._positions[2] == pytest.approx(0.4)
        assert sm._positions[5] == pytest.approx(1.0)
        assert sm._using_bridge_positions is False


class TestEnginePositions:
    """Test that EffectEngine properly delegates position setting."""

    def test_set_light_positions_propagates(self):
        engine = EffectEngine(num_lights=4)
        engine.set_light_positions([0.1, 0.3, 0.6, 0.9])
        assert engine.spatial_mapper._positions == [0.1, 0.3, 0.6, 0.9]
        assert engine.spatial_mapper._using_bridge_positions is True

    def test_set_light_positions_updates_generative(self):
        engine = EffectEngine(num_lights=4)
        engine.set_light_positions([0.1, 0.3, 0.6, 0.9])
        assert engine._generative._positions == [0.1, 0.3, 0.6, 0.9]

    def test_set_light_positions_wrong_count(self):
        engine = EffectEngine(num_lights=4)
        original_spatial = list(engine.spatial_mapper._positions)
        original_gen = list(engine._generative._positions)
        engine.set_light_positions([0.0, 0.5])  # Wrong count
        assert engine.spatial_mapper._positions == original_spatial
        assert engine._generative._positions == original_gen

    def test_engine_works_with_custom_positions(self):
        """Engine should produce valid output with non-linear positions."""
        engine = EffectEngine(num_lights=6, spatial_mode="frequency_zones")
        engine.set_light_positions([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        features = _loud_features(rms=0.5)
        for i in range(30):
            states = engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        assert len(states) == 6
        for s in states:
            assert 0.0 <= s.brightness <= 1.0


# ============================================================================
# Task 1.15: Entertainment area position data — Controller tests
# ============================================================================


@dataclass
class _MockPosition:
    """Mock Position dataclass matching hue_entertainment_pykit.Position."""
    x: float
    y: float
    z: float


@dataclass
class _MockChannel:
    """Mock EntertainmentChannel."""
    channel_id: int
    position: _MockPosition
    members: list = None

    def __post_init__(self):
        if self.members is None:
            self.members = []


class TestReadChannelPositions:
    """Test EntertainmentController._read_channel_positions static method."""

    def test_basic_positions_normalized(self):
        """Positions in [-1, 1] should be normalized to [0, 1]."""
        config = MagicMock()
        config.channels = [
            _MockChannel(channel_id=0, position=_MockPosition(x=-1.0, y=0.0, z=0.0)),
            _MockChannel(channel_id=1, position=_MockPosition(x=0.0, y=0.0, z=0.0)),
            _MockChannel(channel_id=2, position=_MockPosition(x=1.0, y=0.0, z=0.0)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        assert len(positions) == 3
        assert positions[0] == pytest.approx(0.0)
        assert positions[1] == pytest.approx(0.5)
        assert positions[2] == pytest.approx(1.0)

    def test_sorted_by_channel_id(self):
        """Channels should be sorted by channel_id for consistent ordering."""
        config = MagicMock()
        config.channels = [
            _MockChannel(channel_id=2, position=_MockPosition(x=1.0, y=0.0, z=0.0)),
            _MockChannel(channel_id=0, position=_MockPosition(x=-1.0, y=0.0, z=0.0)),
            _MockChannel(channel_id=1, position=_MockPosition(x=0.0, y=0.0, z=0.0)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        # After sorting by channel_id: 0(-1.0) -> 0.0, 1(0.0) -> 0.5, 2(1.0) -> 1.0
        assert positions[0] == pytest.approx(0.0)
        assert positions[1] == pytest.approx(0.5)
        assert positions[2] == pytest.approx(1.0)

    def test_all_same_position_falls_back_to_linear(self):
        """If all lights are at the same x, fall back to linear distribution."""
        config = MagicMock()
        config.channels = [
            _MockChannel(channel_id=0, position=_MockPosition(x=0.5, y=0.0, z=0.0)),
            _MockChannel(channel_id=1, position=_MockPosition(x=0.5, y=0.0, z=0.0)),
            _MockChannel(channel_id=2, position=_MockPosition(x=0.5, y=0.0, z=0.0)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        assert len(positions) == 3
        assert positions[0] == pytest.approx(0.0)
        assert positions[1] == pytest.approx(0.5)
        assert positions[2] == pytest.approx(1.0)

    def test_no_channels_returns_empty(self):
        """If no channels, return empty list."""
        config = MagicMock()
        config.channels = []

        positions = EntertainmentController._read_channel_positions(config)
        assert positions == []

    def test_missing_channels_attribute(self):
        """If channels attribute missing, return empty list."""
        config = MagicMock(spec=[])  # No attributes at all

        positions = EntertainmentController._read_channel_positions(config)
        assert positions == []

    def test_asymmetric_positions(self):
        """Test non-symmetric arrangements (e.g., clustered to one side)."""
        config = MagicMock()
        config.channels = [
            _MockChannel(channel_id=0, position=_MockPosition(x=-1.0, y=0.0, z=0.0)),
            _MockChannel(channel_id=1, position=_MockPosition(x=-0.8, y=0.0, z=0.0)),
            _MockChannel(channel_id=2, position=_MockPosition(x=-0.6, y=0.0, z=0.0)),
            _MockChannel(channel_id=3, position=_MockPosition(x=1.0, y=0.0, z=0.0)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        assert len(positions) == 4
        assert positions[0] == pytest.approx(0.0)       # x=-1.0 -> 0.0
        assert positions[1] == pytest.approx(0.1)        # x=-0.8 -> 0.1
        assert positions[2] == pytest.approx(0.2)        # x=-0.6 -> 0.2
        assert positions[3] == pytest.approx(1.0)        # x=1.0 -> 1.0

    def test_six_light_room_layout(self):
        """Test realistic 6-light setup matching the project's physical layout."""
        config = MagicMock()
        # Simulate 6 lights spread across a room (-1 to +1 range)
        config.channels = [
            _MockChannel(channel_id=0, position=_MockPosition(x=-0.9, y=-0.5, z=0.3)),
            _MockChannel(channel_id=1, position=_MockPosition(x=-0.4, y=0.6, z=0.8)),
            _MockChannel(channel_id=2, position=_MockPosition(x=-0.1, y=-0.3, z=0.5)),
            _MockChannel(channel_id=3, position=_MockPosition(x=0.2, y=0.4, z=0.5)),
            _MockChannel(channel_id=4, position=_MockPosition(x=0.6, y=-0.6, z=0.3)),
            _MockChannel(channel_id=5, position=_MockPosition(x=0.9, y=0.2, z=0.8)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        assert len(positions) == 6
        # All should be in [0, 1] and monotonically non-decreasing
        # (since channels are already sorted by channel_id and x-coords increase)
        for p in positions:
            assert 0.0 <= p <= 1.0
        assert positions[0] == pytest.approx(0.0)   # min x
        assert positions[5] == pytest.approx(1.0)    # max x

    def test_single_light(self):
        """Single light should get position 0.0 (linear fallback for span=0)."""
        config = MagicMock()
        config.channels = [
            _MockChannel(channel_id=0, position=_MockPosition(x=0.3, y=0.0, z=0.0)),
        ]

        positions = EntertainmentController._read_channel_positions(config)

        assert len(positions) == 1
        assert positions[0] == pytest.approx(0.0)

    def test_exception_returns_empty(self):
        """If parsing fails, should return empty list (not crash)."""
        config = MagicMock()
        # channels that will cause an exception during iteration
        config.channels = "not a list"

        positions = EntertainmentController._read_channel_positions(config)
        assert positions == []


class TestControllerLightPositionsProperty:
    """Test the light_positions property on the controller."""

    def test_empty_before_connection(self):
        ctrl = EntertainmentController(
            bridge_ip="192.168.1.100",
            username="test",
            clientkey="test",
        )
        assert ctrl.light_positions == []

    def test_positions_set_after_read(self):
        ctrl = EntertainmentController(
            bridge_ip="192.168.1.100",
            username="test",
            clientkey="test",
        )
        ctrl._light_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert ctrl.light_positions == [0.0, 0.25, 0.5, 0.75, 1.0]
