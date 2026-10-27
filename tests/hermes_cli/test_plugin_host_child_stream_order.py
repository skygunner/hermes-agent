"""Bootstrap must be imported before the plugin host redirects the protocol streams.

The plugin host child imports ``hermes_bootstrap`` for its runtime selection, and that
import can re-exec the process (``os.execv``). If the protocol streams were already
taken (fd 0 -> /dev/null, fd 1 -> stderr, both dup'd handles non-inheritable), the
re-exec loses the protocol channel: the persistent host sees EOF and
``--extract-profiles`` prints nothing. So at the moment ``hermes_bootstrap`` is
imported, fd 0 must still be the caller-provided stream — verified here by a stub
bootstrap module that snapshots fd 0's seekability at import time (a pipe is not
seekable; /dev/null is).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDER_PLUGIN = '''
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(name="fdprobe", base_url="https://fdprobe.example/v1"))
'''

STUB_BOOTSTRAP = '''
import os
from pathlib import Path

try:
    os.lseek(0, 0, os.SEEK_CUR)
    Path(os.environ["HERMES_TEST_FD0_STATE"]).write_text("already-redirected")
except OSError:
    Path(os.environ["HERMES_TEST_FD0_STATE"]).write_text("still-pipe")
'''


@pytest.mark.platforms("any")
def test_bootstrap_imports_before_the_protocol_streams_are_taken(tmp_path):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "hermes_bootstrap.py").write_text(STUB_BOOTSTRAP, encoding="utf-8")
    fd_state = tmp_path / "fd0-state.txt"

    plugin_dir = tmp_path / "plugins" / "model-providers" / "fdprobe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(PROVIDER_PLUGIN, encoding="utf-8")

    child = tmp_path / "child.py"
    child.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(stub_dir)!r})\n"
        "from hermes_cli.plugin_host_child import extract_profiles_main\n"
        "rc = extract_profiles_main(" + repr(str(plugin_dir)) + ", 'fdprobe')\n"
        "assert rc == 0\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(child)],
        check=True,
        capture_output=True,
        stdin=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO_ROOT),
            "HERMES_TEST_FD0_STATE": str(fd_state),
        },
        timeout=120,
    )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert [p["name"] for p in payload["profiles"]] == ["fdprobe"]
    assert fd_state.read_text() == "still-pipe"
