"""Hue Bridge discovery and pairing utilities."""

import requests
from typing import Optional

from ..core.exceptions import BridgeDiscoveryError, BridgeConnectionError


def discover_bridge() -> str:
    """
    Discover Hue Bridge on the local network using Philips discovery service.

    Returns:
        str: IP address of the discovered bridge

    Raises:
        BridgeDiscoveryError: If no bridge is found or discovery fails
    """
    try:
        response = requests.get("https://discovery.meethue.com/", timeout=5)
        response.raise_for_status()
        bridges = response.json()

        if not bridges:
            raise BridgeDiscoveryError("No Hue Bridge found on the network")

        # Return first bridge IP
        bridge_ip = bridges[0]["internalipaddress"]
        return bridge_ip

    except requests.RequestException as e:
        raise BridgeDiscoveryError(f"Failed to discover bridge: {e}")
    except (KeyError, IndexError) as e:
        raise BridgeDiscoveryError(f"Invalid discovery response format: {e}")


def create_user(bridge_ip: str, app_name: str = "hue-visualizer") -> str:
    """
    Create a new user on the Hue Bridge (requires physical button press).

    Args:
        bridge_ip: IP address of the bridge
        app_name: Application name for the username

    Returns:
        str: The created username/API token

    Raises:
        BridgeConnectionError: If user creation fails
    """
    url = f"http://{bridge_ip}/api"
    payload = {"devicetype": f"{app_name}#python"}

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            result = data[0]

            # Check for error (button not pressed)
            if "error" in result:
                error_type = result["error"].get("type")
                if error_type == 101:
                    raise BridgeConnectionError(
                        "Link button not pressed. Please press the button on the bridge and try again."
                    )
                raise BridgeConnectionError(f"Bridge error: {result['error'].get('description')}")

            # Success
            if "success" in result:
                username = result["success"]["username"]
                return username

        raise BridgeConnectionError(f"Unexpected response format: {data}")

    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to create user: {e}")


def verify_connection(bridge_ip: str, username: str) -> bool:
    """
    Verify that the connection to the bridge works with the given credentials.

    Args:
        bridge_ip: IP address of the bridge
        username: API username/token

    Returns:
        bool: True if connection is valid

    Raises:
        BridgeConnectionError: If verification fails
    """
    url = f"http://{bridge_ip}/api/{username}/lights"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Check for error response
        if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
            raise BridgeConnectionError(f"Invalid credentials: {data[0]['error'].get('description')}")

        return True

    except requests.RequestException as e:
        raise BridgeConnectionError(f"Failed to verify connection: {e}")
