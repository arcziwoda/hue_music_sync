"""Tests for bridge config persistence module."""

import json
import os
import threading
from pathlib import Path

import pytest

from hue_visualizer.core import persistence


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Redirect config directory to a temp path for test isolation."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Reset any cached state (module-level lock is fine, directory changes via env)
    yield


class TestConfigPaths:
    """Verify config directory and file path logic."""

    def test_config_dir_uses_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "custom"))
        d = persistence._config_dir()
        assert d == tmp_path / "custom" / "hue-visualizer"

    def test_config_dir_default_when_no_xdg(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        d = persistence._config_dir()
        assert d == Path.home() / ".config" / "hue-visualizer"

    def test_config_path_returns_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        p = persistence._config_path()
        assert p.name == "config.json"
        assert "hue-visualizer" in str(p)


class TestLoadBridgeConfig:
    """Test loading bridge config."""

    def test_returns_none_values_when_no_file(self):
        cfg = persistence.load_bridge_config()
        assert cfg == {
            "ip": None,
            "username": None,
            "clientkey": None,
            "entertainment_area_id": None,
        }

    def test_returns_saved_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Write config directly
        config_dir = tmp_path / "hue-visualizer"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({
            "bridge": {
                "ip": "192.168.1.50",
                "username": "testuser",
                "clientkey": "testkey",
                "entertainment_area_id": "2",
            }
        }))

        cfg = persistence.load_bridge_config()
        assert cfg["ip"] == "192.168.1.50"
        assert cfg["username"] == "testuser"
        assert cfg["clientkey"] == "testkey"
        assert cfg["entertainment_area_id"] == "2"

    def test_handles_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "hue-visualizer"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text("not valid json{{{")

        cfg = persistence.load_bridge_config()
        assert cfg["ip"] is None

    def test_handles_missing_bridge_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "hue-visualizer"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(json.dumps({"other": "stuff"}))

        cfg = persistence.load_bridge_config()
        assert cfg["ip"] is None


class TestSaveBridgeConfig:
    """Test saving bridge config."""

    def test_creates_directory_and_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "new_dir"))

        persistence.save_bridge_config(
            ip="10.0.0.1",
            username="user123",
            clientkey="key456",
            area_id="3",
        )

        config_file = tmp_path / "new_dir" / "hue-visualizer" / "config.json"
        assert config_file.exists()

        data = json.loads(config_file.read_text())
        assert data["bridge"]["ip"] == "10.0.0.1"
        assert data["bridge"]["username"] == "user123"
        assert data["bridge"]["clientkey"] == "key456"
        assert data["bridge"]["entertainment_area_id"] == "3"

    def test_preserves_other_config_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "hue-visualizer"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({"other": "preserved", "bridge": {"ip": "old"}})
        )

        persistence.save_bridge_config(
            ip="10.0.0.2", username="u", clientkey="k"
        )

        data = json.loads((config_dir / "config.json").read_text())
        assert data["other"] == "preserved"
        assert data["bridge"]["ip"] == "10.0.0.2"

    def test_area_id_optional(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        persistence.save_bridge_config(
            ip="10.0.0.3", username="u", clientkey="k"
        )

        cfg = persistence.load_bridge_config()
        assert cfg["ip"] == "10.0.0.3"
        assert cfg["entertainment_area_id"] is None

    def test_roundtrip(self):
        persistence.save_bridge_config(
            ip="192.168.0.1", username="abc", clientkey="def", area_id="1"
        )
        cfg = persistence.load_bridge_config()
        assert cfg["ip"] == "192.168.0.1"
        assert cfg["username"] == "abc"
        assert cfg["clientkey"] == "def"
        assert cfg["entertainment_area_id"] == "1"


class TestClearBridgeConfig:
    """Test clearing bridge config."""

    def test_clears_saved_config(self):
        persistence.save_bridge_config(
            ip="192.168.0.1", username="abc", clientkey="def"
        )
        persistence.clear_bridge_config()
        cfg = persistence.load_bridge_config()
        assert cfg["ip"] is None
        assert cfg["username"] is None

    def test_clear_when_nothing_saved(self):
        # Should not raise
        persistence.clear_bridge_config()
        cfg = persistence.load_bridge_config()
        assert cfg["ip"] is None

    def test_preserves_other_keys_on_clear(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "hue-visualizer"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps({"other": "keep", "bridge": {"ip": "gone"}})
        )

        persistence.clear_bridge_config()

        data = json.loads((config_dir / "config.json").read_text())
        assert data["other"] == "keep"
        assert "bridge" not in data


class TestThreadSafety:
    """Verify concurrent access doesn't corrupt data."""

    def test_concurrent_writes(self):
        errors = []

        def writer(n):
            try:
                persistence.save_bridge_config(
                    ip=f"10.0.0.{n}",
                    username=f"user{n}",
                    clientkey=f"key{n}",
                    area_id=str(n),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Final config should be valid JSON with one of the written values
        cfg = persistence.load_bridge_config()
        assert cfg["ip"] is not None
        assert cfg["ip"].startswith("10.0.0.")


class TestGetConfigPath:
    """Test the diagnostic helper."""

    def test_returns_string(self):
        path = persistence.get_config_path()
        assert isinstance(path, str)
        assert "config.json" in path
