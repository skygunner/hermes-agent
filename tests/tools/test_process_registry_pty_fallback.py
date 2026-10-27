"""A live PTY child must not be duplicated by the pipe fallback when tracking fails.

Regression for #145163: ``spawn_local(use_pty=True)`` spawns the PTY child, then calls
``_track_started``. If tracking raised, ``spawn_local`` fell through to the pipe path and
launched the same command a second time while the first child was still alive. Real PTY,
real process: the marker is written by whatever launches the command, so a duplicate
launch is observable on disk.
"""

import shutil
import sys

import pytest

import tools.process_registry as module
from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry

# getattr keeps the behavior test runnable (red, not a collection error) on a base
# revision that predates the exception class.
PtyTrackingFailed = getattr(module, "PtyTrackingFailed", RuntimeError)

_POSIX_PTY = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None, reason="POSIX PTY + sh")


@pytest.fixture
def unscoped(monkeypatch):
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)


@_POSIX_PTY
def test_tracking_failure_after_pty_spawn_refuses_pipe_fallback(tmp_path, monkeypatch, unscoped):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    real_track = registry._track_started

    def failing_track(session, reader_target, reader_name, *extra_args):
        if "pty-reader" in reader_name:
            raise RuntimeError("injected tracking failure")
        return real_track(session, reader_target, reader_name, *extra_args)

    monkeypatch.setattr(registry, "_track_started", failing_track)

    marker = tmp_path / "launched"
    command = f"echo ran >> {marker}; exec sleep 60"

    with pytest.raises(PtyTrackingFailed):
        registry.spawn_local(command, cwd=str(tmp_path), use_pty=True)

    assert not marker.exists(), "the command was launched after PTY tracking failed"


@_POSIX_PTY
def test_pipe_fallback_still_runs_when_no_pty_child_was_created(tmp_path, monkeypatch, unscoped):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()

    def raise_missing_backend(session, safe_command, env_vars):
        raise ImportError("injected: no PTY backend")

    monkeypatch.setattr(registry, "_spawn_local_pty", raise_missing_backend)

    marker = tmp_path / "launched"
    session = registry.spawn_local(f"echo ran >> {marker}", cwd=str(tmp_path), use_pty=True)
    try:
        assert wait_until(lambda: marker.exists(), timeout=10)
    finally:
        registry.kill_process(session.id)
    assert marker.exists(), "pipe fallback must still run when no PTY child was created"
