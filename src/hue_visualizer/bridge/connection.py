"""Basic REST API connection to Hue Bridge."""

import warnings

import requests
from typing import Dict, Any, List, Optional
import urllib3

from ..core.exceptions import BridgeConnectionError

# Hue bridges use self-signed certificates — suppress the expected warning.
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


class HueBridge:
    """
    Simple wrapper for Hue Bridge REST API.

    This handles basic HTTP communication for setup and configuration.
    For real-time light control, use EntertainmentAPI instead.
    """

    def __init__(self, bridge_ip: str, username: str):
        """
        Initialize bridge connection.

        Args:
            bridge_ip: IP address of the bridge
            username: API username/token
        """
        self.bridge_ip = bridge_ip
        self.username = username
        self.base_url = f"https://{bridge_ip}/api/{username}"

    def _get(self, endpoint: str) -> Any:
        """Make GET request to the bridge."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=5, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise BridgeConnectionError(f"GET {endpoint} failed: {e}")

    def _put(self, endpoint: str, data: Dict[str, Any]) -> Any:
        """Make PUT request to the bridge."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.put(url, json=data, timeout=5, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise BridgeConnectionError(f"PUT {endpoint} failed: {e}")

    def get_lights(self) -> Dict[str, Any]:
        """
        Get all lights connected to the bridge.

        Returns:
            Dict mapping light IDs to light info
        """
        return self._get("lights")

    def get_light(self, light_id: str) -> Dict[str, Any]:
        """Get info about a specific light."""
        return self._get(f"lights/{light_id}")

    def set_light_state(self, light_id: str, state: Dict[str, Any]) -> Any:
        """
        Set the state of a light (on/off, brightness, color).

        Args:
            light_id: ID of the light
            state: State dictionary (e.g., {"on": True, "bri": 254, "hue": 10000})

        Returns:
            Response from the bridge
        """
        return self._put(f"lights/{light_id}/state", state)

    def get_groups(self) -> Dict[str, Any]:
        """Get all groups (including entertainment areas)."""
        return self._get("groups")

    def get_group(self, group_id: str) -> Dict[str, Any]:
        """Get info about a specific group."""
        return self._get(f"groups/{group_id}")

    def get_entertainment_areas(self) -> List[Dict[str, Any]]:
        """
        Get all entertainment areas configured on the bridge.

        Returns:
            List of entertainment area info dicts
        """
        groups = self.get_groups()
        entertainment_areas = []

        for group_id, group_info in groups.items():
            if group_info.get("type") == "Entertainment":
                entertainment_areas.append(
                    {
                        "id": group_id,
                        "name": group_info.get("name"),
                        "lights": group_info.get("lights", []),
                        "class": group_info.get("class"),
                    }
                )

        return entertainment_areas

    def test_connection(self) -> bool:
        """
        Test if the connection to the bridge is working.

        Returns:
            True if connection is valid

        Raises:
            BridgeConnectionError: If connection fails
        """
        try:
            self.get_lights()
            return True
        except BridgeConnectionError:
            raise
