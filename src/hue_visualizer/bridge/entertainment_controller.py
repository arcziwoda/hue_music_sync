"""
Entertainment Controller - High-level interface for Hue Entertainment API.

This module provides a clean, Pythonic interface for controlling Hue lights
via the Entertainment API with minimal latency.
"""

import logging
import warnings
from typing import Optional, Tuple
from dataclasses import dataclass

import requests
import urllib3
from hue_entertainment_pykit import create_bridge, Entertainment, Streaming

# Hue bridges use self-signed certificates — suppress the expected warning.
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class LightState:
    """Represents the state of a light in XY color space."""
    x: float  # CIE x coordinate (0.0 - 1.0)
    y: float  # CIE y coordinate (0.0 - 1.0)
    brightness: float  # Brightness (0.0 - 1.0)
    light_id: int = 0  # Light ID (0 = all lights in some implementations)

    def to_xyb_tuple(self) -> Tuple[float, float, float, int]:
        """Convert to tuple format expected by streaming API."""
        return (self.x, self.y, self.brightness, self.light_id)


class EntertainmentController:
    """
    High-level controller for Hue Entertainment API.

    Features:
    - Automatic connection management
    - Simple color setting API (RGB, HSV, XY)
    - Support for individual lights or all lights
    - Context manager support for clean resource management

    Example:
        with EntertainmentController(bridge_ip, username, clientkey) as controller:
            controller.set_color_rgb(255, 0, 0)  # Red
            controller.set_brightness(0.5)
    """

    def __init__(
        self,
        bridge_ip: str,
        username: str,
        clientkey: str,
        entertainment_area_id: Optional[str] = None,
    ):
        """
        Initialize Entertainment Controller.

        Args:
            bridge_ip: IP address of Hue Bridge
            username: Hue API username
            clientkey: Entertainment API client key
            entertainment_area_id: Optional specific entertainment area ID
        """
        self.bridge_ip = bridge_ip
        self.username = username
        self.clientkey = clientkey
        self.entertainment_area_id = entertainment_area_id

        self._bridge = None
        self._entertainment_service = None
        self._streaming = None
        self._is_streaming = False
        self._ent_config = None
        self._num_lights = 0  # Will be set after connection
        self._light_positions: list[float] = []  # Normalized 0-1 x-positions from bridge

        logger.info(f"EntertainmentController initialized for bridge at {bridge_ip}")

    def connect(self) -> None:
        """
        Connect to the Hue Bridge and start Entertainment streaming.

        Raises:
            ConnectionError: If connection fails
            ValueError: If no entertainment areas found
        """
        logger.info("Connecting to Hue Bridge...")

        # Get bridge configuration
        config = self._get_bridge_config()
        logger.debug(f"Bridge config: {config.get('name')} - {config.get('bridgeid')}")

        # Create bridge instance
        self._bridge = create_bridge(
            identification=config.get('bridgeid'),
            rid=config.get('bridgeid'),
            ip_address=self.bridge_ip,
            swversion=int(config.get('swversion', 0)),
            username=self.username,
            hue_app_id="hue-visualizer",
            clientkey=self.clientkey,
            name=config.get('name', 'Hue Bridge')
        )
        logger.debug("Bridge instance created")

        # Initialize Entertainment service
        self._entertainment_service = Entertainment(self._bridge)
        ent_configs = self._entertainment_service.get_entertainment_configs()

        if not ent_configs:
            raise ValueError(
                "No entertainment configurations found. "
                "Please create an Entertainment Area in the Hue app."
            )

        # Select entertainment config
        if self.entertainment_area_id and self.entertainment_area_id in ent_configs:
            self._ent_config = ent_configs[self.entertainment_area_id]
        else:
            # Use first available config
            self._ent_config = list(ent_configs.values())[0]

        # Get number of lights in entertainment area
        self._num_lights = len(self._ent_config.channels) if hasattr(self._ent_config, 'channels') else 0
        logger.info(f"Using entertainment configuration with {self._num_lights} lights")

        # Read light positions from entertainment area channels (Task 1.15)
        self._light_positions = self._read_channel_positions(self._ent_config)
        if self._light_positions:
            logger.info(
                f"Light positions from bridge: "
                f"{[round(p, 3) for p in self._light_positions]}"
            )

        # Start streaming
        self._streaming = Streaming(
            self._bridge,
            self._ent_config,
            self._entertainment_service.get_ent_conf_repo()
        )
        self._streaming.start_stream()
        self._streaming.set_color_space("xyb")  # Use XYB color space
        self._is_streaming = True

        logger.info("✓ Entertainment streaming started")

    def disconnect(self) -> None:
        """Stop streaming and disconnect from bridge."""
        if self._streaming and self._is_streaming:
            logger.info("Stopping Entertainment stream...")
            try:
                self._streaming.stop_stream()
                self._is_streaming = False
                logger.info("✓ Stream stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")

    def set_light_state(self, state: LightState) -> None:
        """
        Set light state directly using XY color space.

        Args:
            state: LightState with x, y, brightness, and light_id

        Raises:
            RuntimeError: If not connected
        """
        if not self._is_streaming or self._streaming is None:
            raise RuntimeError("Not connected. Call connect() first.")

        self._streaming.set_input(state.to_xyb_tuple())

    def set_light_states_batch(self, states: list[LightState]) -> None:
        """
        Set multiple light states in a single call.

        Each state is passed to the streaming library's set_input which
        buffers inputs internally. This avoids per-light Python call overhead
        and makes it explicit that all states belong to the same frame.

        Args:
            states: List of LightState objects to send as one batch.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._is_streaming or self._streaming is None:
            raise RuntimeError("Not connected. Call connect() first.")

        for state in states:
            self._streaming.set_input(state.to_xyb_tuple())

    def set_color_xy(
        self,
        x: float,
        y: float,
        brightness: float = 1.0,
        light_id: Optional[int] = None
    ) -> None:
        """
        Set light color using CIE XY color space.

        Args:
            x: CIE x coordinate (0.0 - 1.0)
            y: CIE y coordinate (0.0 - 1.0)
            brightness: Brightness (0.0 - 1.0)
            light_id: Light ID (None for all lights, or specific light index)
        """
        if light_id is None:
            # Set all lights
            if self._num_lights > 0:
                for idx in range(self._num_lights):
                    state = LightState(x=x, y=y, brightness=brightness, light_id=idx)
                    self.set_light_state(state)
            else:
                # Fallback if num_lights not set
                state = LightState(x=x, y=y, brightness=brightness, light_id=0)
                self.set_light_state(state)
        else:
            state = LightState(x=x, y=y, brightness=brightness, light_id=light_id)
            self.set_light_state(state)

    def set_color_rgb(
        self,
        r: int,
        g: int,
        b: int,
        brightness: float = 1.0,
        light_id: Optional[int] = None
    ) -> None:
        """
        Set light color using RGB values.

        Args:
            r: Red (0-255)
            g: Green (0-255)
            b: Blue (0-255)
            brightness: Brightness (0.0 - 1.0)
            light_id: Light ID (None for all lights, or specific light index)
        """
        from ..utils.color_conversion import rgb_to_xy
        x, y = rgb_to_xy(r, g, b)
        self.set_color_xy(x, y, brightness, light_id)

    def set_color_hsv(
        self,
        h: float,
        s: float,
        v: float,
        light_id: Optional[int] = None
    ) -> None:
        """
        Set light color using HSV values.

        Args:
            h: Hue (0.0 - 360.0 degrees)
            s: Saturation (0.0 - 1.0)
            v: Value/Brightness (0.0 - 1.0)
            light_id: Light ID (None for all lights, or specific light index)
        """
        from ..utils.color_conversion import hsv_to_xy
        x, y = hsv_to_xy(h, s, v)
        self.set_color_xy(x, y, v, light_id)

    def set_brightness(self, brightness: float, light_id: Optional[int] = None) -> None:
        """
        Set brightness while maintaining current color.

        Args:
            brightness: Brightness (0.0 - 1.0)
            light_id: Light ID (None for all lights, or specific light index)

        Note: This uses white color. For colored brightness changes,
              use set_color_xy/rgb/hsv with brightness parameter.
        """
        # White color in XY space
        self.set_color_xy(0.3127, 0.3290, brightness, light_id)

    def turn_off(self, light_id: Optional[int] = None) -> None:
        """
        Turn off lights (set brightness to 0).

        Args:
            light_id: Light ID (None for all lights, or specific light index)
        """
        self.set_brightness(0.0, light_id)

    @property
    def light_positions(self) -> list[float]:
        """Normalized 0-1 x-positions of lights from the entertainment area.

        Returns an empty list if positions are not available (e.g. not connected
        or the entertainment config has no channel position data).
        """
        return self._light_positions

    @staticmethod
    def _read_channel_positions(ent_config) -> list[float]:
        """Extract and normalize light x-positions from entertainment area channels.

        The Hue Entertainment API provides per-channel 3D positions with x in
        range [-1, 1]. We normalize x to [0, 1] for the spatial mapper.

        Channels are sorted by channel_id to ensure consistent ordering that
        matches the streaming light indices.

        Args:
            ent_config: EntertainmentConfiguration instance from hue_entertainment_pykit.

        Returns:
            List of normalized 0-1 x-positions sorted by channel_id.
            Empty list if channels or positions are unavailable.
        """
        try:
            channels = getattr(ent_config, 'channels', None)
            if not channels:
                return []

            # Sort channels by channel_id for consistent ordering
            sorted_channels = sorted(channels, key=lambda ch: ch.channel_id)

            raw_x = []
            for ch in sorted_channels:
                pos = getattr(ch, 'position', None)
                if pos is not None:
                    raw_x.append(pos.x)
                else:
                    raw_x.append(0.0)

            if not raw_x:
                return []

            # Normalize from [-1, 1] range to [0, 1]
            min_x = min(raw_x)
            max_x = max(raw_x)
            span = max_x - min_x

            if span < 1e-6:
                # All lights at the same x position — distribute linearly
                n = len(raw_x)
                return [i / max(n - 1, 1) for i in range(n)]

            return [(x - min_x) / span for x in raw_x]

        except Exception as e:
            logger.warning(f"Failed to read channel positions: {e}")
            return []

    def _get_bridge_config(self) -> dict:
        """Get bridge configuration via REST API."""
        url = f"https://{self.bridge_ip}/api/{self.username}/config"
        try:
            response = requests.get(url, timeout=5, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to get bridge config: {e}")

    def __enter__(self):
        """Context manager entry - connect to bridge."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from bridge."""
        self.disconnect()
        return False

    @property
    def is_connected(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
