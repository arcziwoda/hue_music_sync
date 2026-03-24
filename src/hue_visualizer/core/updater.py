"""Auto-update mechanism — check GitHub Releases, download, and apply updates.

Supports macOS (.dmg) and Windows (.zip) desktop builds.
In dev mode (not frozen), check works but apply is disabled.
"""

import enum
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import requests

from .exceptions import UpdateError
from .persistence import _config_dir

logger = logging.getLogger(__name__)

GITHUB_REPO = "arcziwoda/vslzr"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CHECK_TIMEOUT = 10
DOWNLOAD_CHUNK_SIZE = 64 * 1024


class UpdateState(enum.Enum):
    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    READY = "ready"
    APPLYING = "applying"
    ERROR = "error"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse version string like '1.3.0' or 'v1.3.0' into comparable tuple."""
    return tuple(int(x) for x in v.lstrip("v").split(".")[:3])


def _is_newer(latest: str, current: str) -> bool:
    """Return True if latest version is newer than current."""
    try:
        return _parse_version(latest) > _parse_version(current)
    except (ValueError, IndexError):
        return False


def _find_platform_asset(assets: list[dict]) -> dict | None:
    """Find the download asset matching the current platform."""
    if sys.platform == "darwin":
        suffix = "macos-arm64.dmg"
    elif sys.platform == "win32":
        suffix = "windows-x64.zip"
    else:
        return None

    for asset in assets:
        if asset.get("name", "").endswith(suffix):
            return asset
    return None


def _get_app_path() -> Path | None:
    """Get path to the current app installation.

    macOS: the .app bundle (walks up from sys.executable).
    Windows: parent folder of VSLZR.exe.
    """
    if not getattr(sys, "frozen", False):
        return None

    exe = Path(sys.executable)

    if sys.platform == "darwin":
        p = exe
        while p.parent != p:
            if p.suffix == ".app":
                return p
            p = p.parent
        return None
    elif sys.platform == "win32":
        return exe.parent

    return None


def _updates_dir() -> Path:
    """Return the updates staging directory in the user's config dir."""
    d = _config_dir() / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_old_updates() -> None:
    """Remove leftover update files from previous sessions."""
    d = _config_dir() / "updates"
    if d.exists():
        try:
            shutil.rmtree(d)
            logger.info("Cleaned up old update files")
        except OSError as e:
            logger.warning(f"Failed to clean update dir: {e}")


class Updater:
    """Manages the full update lifecycle: check → download → apply."""

    def __init__(self):
        self.state = UpdateState.IDLE
        self.info: dict | None = None  # GitHub release info
        self.download_progress: tuple[int, int] = (0, 0)
        self.error: str | None = None
        self._download_path: Path | None = None
        self._lock = threading.Lock()

    def _set_state(self, state: UpdateState, error: str | None = None):
        self.state = state
        self.error = error

    def check(self) -> dict | None:
        """Check GitHub for a newer release. Thread-safe, blocking."""
        with self._lock:
            self._set_state(UpdateState.CHECKING)

        try:
            resp = requests.get(
                GITHUB_API_URL,
                timeout=CHECK_TIMEOUT,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Update check failed: {e}")
            with self._lock:
                self._set_state(UpdateState.ERROR, str(e))
            return None

        from hue_visualizer import __version__

        tag = data.get("tag_name", "")
        current = __version__

        if not _is_newer(tag, current):
            with self._lock:
                self._set_state(UpdateState.UP_TO_DATE)
                self.info = {"current_version": current, "latest_version": tag}
            return None

        asset = _find_platform_asset(data.get("assets", []))
        if not asset:
            logger.warning(f"No platform asset found for {sys.platform}")
            with self._lock:
                self._set_state(UpdateState.ERROR, "No download available for this platform")
            return None

        info = {
            "current_version": current,
            "latest_version": tag,
            "release_url": data.get("html_url", ""),
            "download_url": asset["browser_download_url"],
            "download_size": asset.get("size", 0),
            "asset_name": asset["name"],
        }

        with self._lock:
            self.info = info
            self._set_state(UpdateState.AVAILABLE)

        logger.info(f"Update available: {current} -> {tag}")
        return info

    def download(self) -> Path:
        """Download the update asset. Blocking, run in executor."""
        with self._lock:
            if self.state != UpdateState.AVAILABLE or not self.info:
                raise UpdateError("No update available to download")
            self._set_state(UpdateState.DOWNLOADING)
            self.download_progress = (0, self.info["download_size"])

        url = self.info["download_url"]
        filename = self.info["asset_name"]
        expected_size = self.info["download_size"]
        dest = _updates_dir() / filename

        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    with self._lock:
                        self.download_progress = (downloaded, expected_size)

            # Verify size if server reported it
            if expected_size > 0 and downloaded != expected_size:
                dest.unlink(missing_ok=True)
                raise UpdateError(
                    f"Download incomplete: {downloaded}/{expected_size} bytes"
                )

            with self._lock:
                self._download_path = dest
                self._set_state(UpdateState.READY)

            logger.info(f"Update downloaded: {dest} ({downloaded} bytes)")
            return dest

        except UpdateError:
            raise
        except Exception as e:
            dest.unlink(missing_ok=True)
            with self._lock:
                self._set_state(UpdateState.ERROR, str(e))
            raise UpdateError(f"Download failed: {e}") from e

    def apply(self) -> None:
        """Apply the downloaded update. Platform-specific."""
        with self._lock:
            if self.state != UpdateState.READY or not self._download_path:
                raise UpdateError("No update ready to apply")
            self._set_state(UpdateState.APPLYING)

        if not self.can_self_update():
            with self._lock:
                self._set_state(UpdateState.ERROR, "Cannot self-update in this mode")
            raise UpdateError("Cannot self-update in this mode")

        try:
            if sys.platform == "darwin":
                self._apply_macos()
            elif sys.platform == "win32":
                self._apply_windows()
            else:
                raise UpdateError(f"Unsupported platform: {sys.platform}")
        except UpdateError:
            raise
        except Exception as e:
            with self._lock:
                self._set_state(UpdateState.ERROR, str(e))
            raise UpdateError(f"Apply failed: {e}") from e

    def _apply_macos(self) -> None:
        """macOS: mount DMG, copy .app, unmount, relaunch."""
        assert self._download_path is not None
        dmg_path = self._download_path
        app_path = _get_app_path()
        if not app_path:
            raise UpdateError("Cannot determine app location")

        mount_point = Path(tempfile.mkdtemp(prefix="vslzr_mount_"))

        try:
            # Mount DMG
            result = subprocess.run(
                [
                    "hdiutil", "attach", str(dmg_path),
                    "-mountpoint", str(mount_point),
                    "-nobrowse", "-readonly",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise UpdateError(f"Failed to mount DMG: {result.stderr.strip()}")

            # Find .app in mounted volume
            apps = list(mount_point.glob("*.app"))
            if not apps:
                raise UpdateError("No .app found in DMG")

            new_app = apps[0]

            # Replace old .app with new one
            if app_path.exists():
                shutil.rmtree(app_path)
            shutil.copytree(str(new_app), str(app_path), symlinks=True)

            logger.info(f"Updated {app_path}")

            # Unmount
            subprocess.run(
                ["hdiutil", "detach", str(mount_point)],
                capture_output=True, timeout=30,
            )

            # Clean up downloaded DMG
            dmg_path.unlink(missing_ok=True)

            # Relaunch
            subprocess.Popen(["open", "-n", str(app_path)])

        except UpdateError:
            raise
        except Exception as e:
            # Try to unmount on error
            subprocess.run(
                ["hdiutil", "detach", str(mount_point)],
                capture_output=True,
            )
            raise UpdateError(f"macOS update failed: {e}") from e

    def _apply_windows(self) -> None:
        """Windows: extract ZIP, write batch updater, launch it detached."""
        assert self._download_path is not None
        zip_path = self._download_path
        app_path = _get_app_path()
        if not app_path:
            raise UpdateError("Cannot determine app location")

        staging = _updates_dir() / "staging"
        if staging.exists():
            shutil.rmtree(staging)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(staging)

            # The ZIP contains files directly (VSLZR.exe, etc.) or in a VSLZR/ subfolder
            extracted = staging / "VSLZR"
            if not extracted.exists():
                # Files are directly in staging
                extracted = staging

            exe_path = app_path / "VSLZR.exe"
            bat_path = _updates_dir() / "update.bat"
            log_path = _updates_dir() / "update.log"

            # ping -n N waits N-1 seconds (no console required, unlike timeout)
            # taskkill ensures the old process is fully dead before xcopy
            bat_content = (
                '@echo off\n'
                f'echo Updating VSLZR... > "{log_path}"\n'
                'ping -n 4 127.0.0.1 >nul\n'
                f'taskkill /f /im VSLZR.exe >nul 2>&1\n'
                'ping -n 2 127.0.0.1 >nul\n'
                f'xcopy /s /y /q "{extracted}\\*" "{app_path}\\" >> "{log_path}" 2>&1\n'
                f'if errorlevel 1 (echo XCOPY FAILED >> "{log_path}") '
                f'else (echo XCOPY OK >> "{log_path}")\n'
                f'start "" "{exe_path}"\n'
                f'rd /s /q "{staging}"\n'
                'del "%~f0"\n'
            )
            bat_path.write_text(bat_content)

            # CREATE_NO_WINDOW: hidden console (batch commands work)
            # CREATE_NEW_PROCESS_GROUP: survives parent exit
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                ["cmd", "/c", str(bat_path)],
                creationflags=CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            )

            logger.info(f"Update batch script launched: {bat_path}")

        except UpdateError:
            raise
        except Exception as e:
            raise UpdateError(f"Windows update failed: {e}") from e

    def can_self_update(self) -> bool:
        """Check if the app can perform a self-update."""
        if not getattr(sys, "frozen", False):
            return False

        app_path = _get_app_path()
        if not app_path:
            return False

        # macOS: can't update if running from mounted DMG
        if sys.platform == "darwin" and str(app_path).startswith("/Volumes/"):
            return False

        # Check write permission
        try:
            test_path = app_path.parent if sys.platform == "win32" else app_path.parent
            return os.access(test_path, os.W_OK)
        except OSError:
            return False

    def dismiss(self) -> None:
        """User dismissed the update notification."""
        with self._lock:
            if self.state in (UpdateState.AVAILABLE, UpdateState.UP_TO_DATE, UpdateState.ERROR):
                self._set_state(UpdateState.IDLE)

    def get_status(self) -> dict:
        """Return JSON-serializable status dict for API responses."""
        from hue_visualizer import __version__

        with self._lock:
            result = {
                "state": self.state.value,
                "current_version": __version__,
                "is_frozen": getattr(sys, "frozen", False),
                "can_install": self.can_self_update(),
                "error": self.error,
            }

            if self.info:
                result["latest_version"] = self.info.get("latest_version")
                result["release_url"] = self.info.get("release_url")
                result["download_size"] = self.info.get("download_size", 0)

            if self.state == UpdateState.DOWNLOADING:
                downloaded, total = self.download_progress
                result["progress_downloaded"] = downloaded
                result["progress_total"] = total

            return result
