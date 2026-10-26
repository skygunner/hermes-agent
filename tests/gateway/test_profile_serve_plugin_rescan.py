"""#144336: a served profile's plugin directories are re-scanned when it is (re)served.

``hermes -p <p> gateway restart`` re-runs ``_load_secondary_profile_config`` for the profile, and the
adapter it rebuilds must come from a user platform plugin dropped over the bundled key after boot (and
from the bundled copy again once that override is removed). A serve with nothing changed keeps the
live registration instead of unloading and re-importing it.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import textwrap
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_plugin_rewire import GatewayPluginRewireMixin
from hermes_constants import hermes_home_key

_PLUGIN_YAML = "name: irc-platform\nlabel: IRC\nkind: platform\nversion: 1.0.0\ndescription: test platform\n"

_PLUGIN_INIT = textwrap.dedent('''
    class _Adapter:
        def __init__(self, origin):
            self.origin = origin

    def register(ctx):
        ctx.register_platform(
            "irc", "IRC", adapter_factory=lambda cfg: _Adapter({origin!r}), check_fn=lambda: True,
        )
''')


def _write_platform_plugin(plugins_root: Path, origin: str) -> Path:
    """A ``platforms/irc`` plugin whose adapter factory stamps which copy it came from."""
    plugin_dir = plugins_root / "platforms" / "irc"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(_PLUGIN_YAML, encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN_INIT.format(origin=origin), encoding="utf-8")
    return plugin_dir


class _ServeRunner(GatewayAdapterLifecycleMixin, GatewayPluginRewireMixin):
    """``_load_secondary_profile_config`` only: the busy-mode snapshot and config-hook registration
    belong to the other startup phases."""

    def _snapshot_profile_busy_modes(self, profile_name, config):
        pass

    @staticmethod
    def _register_config_hooks(fail_fmt, *fail_args, level=logging.DEBUG):
        pass


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    _write_platform_plugin(bundled, origin="bundled")
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("plugins:\n  enabled: [platforms/irc]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled)
    return profile


def _registration(profile_home: Path):
    """``(entry, deferred loader)`` this profile's runtime scope holds for the platform."""
    return platform_registry.snapshot_registration("irc", scope=hermes_home_key(profile_home))


def test_hot_serve_rebuilds_adapters_from_the_current_plugin_directories(profile_home):
    runner = _ServeRunner()

    async def serve():
        await runner._load_secondary_profile_config("work", profile_home)

    async def scenario():
        await serve()  # boot: only the bundled copy is on disk, and it registers lazily
        entry, loader = _registration(profile_home)
        assert entry is None and loader is not None

        _write_platform_plugin(profile_home / "plugins", origin="user")  # the override lands later
        await serve()
        entry, loader = _registration(profile_home)
        assert loader is None
        assert entry is not None
        assert entry.adapter_factory(PlatformConfig(enabled=True)).origin == "user"
        served_entry = entry

        await serve()  # nothing changed: the live registration survives the serve
        assert _registration(profile_home)[0] is served_entry

        shutil.rmtree(profile_home / "plugins" / "platforms" / "irc")  # rollback
        await serve()
        entry, loader = _registration(profile_home)
        assert entry is None and loader is not None

    asyncio.run(scenario())
