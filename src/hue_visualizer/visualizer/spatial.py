"""Spatial mapper — holds light positions, mode constants, and wave state.

The actual spatial distribution logic lives in EffectEngine._distribute(),
which integrates palette rotation (chase effect) with spatial modes.
SpatialMapper provides the shared data and state that the engine uses.

Modes:
- uniform: all lights same color
- frequency_zones: each light responds to different frequency band
- wave: brightness/hue wave propagates across lights
- mirror: symmetric pattern from center
- chase: sequential per-bulb activation with beat-synced travel
"""


class SpatialMapper:
    """Holds light positions, spatial mode, and wave phase state.

    The per-light distribution logic is in EffectEngine._distribute(),
    which uses this class for positions, mode constants, and wave state.
    """

    UNIFORM = "uniform"
    FREQUENCY_ZONES = "frequency_zones"
    WAVE = "wave"
    MIRROR = "mirror"
    CHASE = "chase"

    MODES = [UNIFORM, FREQUENCY_ZONES, WAVE, MIRROR, CHASE]

    def __init__(self, num_lights: int, mode: str = "frequency_zones"):
        self.num_lights = max(num_lights, 1)
        self.mode = mode
        self._wave_phase = 0.0

        # Default: linear arrangement 0.0 -> 1.0
        self._positions = [
            i / max(self.num_lights - 1, 1) for i in range(self.num_lights)
        ]
        self._using_bridge_positions = False

        # Chase state: which bulb is currently "lit" (fractional for smooth travel)
        self._chase_position: float = 0.0
        # Per-bulb activation timestamps (monotonic seconds) for decay calculation
        self._chase_last_activated: list[float] = [0.0] * self.num_lights
        # Chase direction: +1 forward, -1 backward
        self._chase_direction: int = 1
        # Whether to alternate direction on each beat cycle
        self._chase_alternating: bool = True

    def set_positions(self, positions: list[float]) -> None:
        """Set light positions from bridge entertainment area data.

        Args:
            positions: List of normalized 0-1 positions, one per light.
                       Must have exactly num_lights entries.
        """
        if len(positions) == self.num_lights:
            self._positions = list(positions)
            self._using_bridge_positions = True

    def reset(self) -> None:
        self._wave_phase = 0.0
        self._chase_position = 0.0
        self._chase_last_activated = [0.0] * self.num_lights
        self._chase_direction = 1
