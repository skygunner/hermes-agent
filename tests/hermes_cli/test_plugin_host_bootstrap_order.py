"""Regression for #144976: ``hermes_bootstrap`` must be imported before the plugin host captures
the protocol streams, because bootstrap may ``os.execv`` into the managed interpreter and the
duplicated fds are non-inheritable."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent(
    """
    import json, os

    with open(os.environ["BOOTSTRAP_PROBE"], "w", encoding="utf-8") as handle:
        # _take_protocol_streams() ends with dup2(2, 1), so fd 1 and fd 2 are the same open file
        # once it has run. Before it runs they are the two separate pipes the parent gave us.
        json.dump({"stdout_is_stderr": os.path.samestat(os.fstat(1), os.fstat(2))}, handle)
    """
)

PLUGIN_INIT = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="bootstrap-probe"))
    """
)


def test_extract_profiles_imports_bootstrap_before_capturing_the_protocol(tmp_path):
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    (stub_root / "hermes_bootstrap.py").write_text(PROBE, encoding="utf-8")
    plugin_dir = tmp_path / "probe-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")
    probe_path = tmp_path / "probe.json"

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "home"),
        "PYTHONPATH": f"{stub_root}:{REPO_ROOT}",
        "BOOTSTRAP_PROBE": str(probe_path),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
    }
    done = subprocess.run(
        [sys.executable, "-P", "-m", "hermes_cli.plugin_host_child",
         "--extract-profiles", str(plugin_dir), "probe_plugin"],
        capture_output=True, text=True, timeout=120, check=False, env=env, cwd=str(tmp_path),
    )

    assert probe_path.exists(), f"bootstrap never imported: {done.stderr[-800:]}"
    seen = json.loads(probe_path.read_text(encoding="utf-8"))
    assert seen["stdout_is_stderr"] is False, (
        "bootstrap ran after the protocol capture: fd 1 already pointed at stderr, so a re-exec "
        f"would lose the channel. stderr tail: {done.stderr[-400:]}"
    )

    payload = json.loads(done.stdout)
    assert [p["name"] for p in payload["profiles"]] == ["bootstrap-probe"], done.stderr[-800:]
