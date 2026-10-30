"""Regression for a registered but idle launchd gateway (#142741)."""

import subprocess
from types import SimpleNamespace

from hermes_cli import gateway as gw


def test_bootstrap_kickstarts_registered_idle_job(monkeypatch):
    commands = []
    running = False

    def launchctl(cmd, **kwargs):
        nonlocal running
        commands.append(cmd)
        if cmd[1] == "kickstart":
            running = True
        output = '{\n\t"PID" = 4242;\n};' if running else '{\n\t"Label" = "ai.hermes.gateway";\n};'
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(gw.subprocess, "run", launchctl)
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda message: None)

    assert gw._retry_launchctl_bootstrap_until_registered(
        "gui/501", "/tmp/gateway.plist", "ai.hermes.gateway", deadline=0
    ) is True
    assert ["launchctl", "kickstart", "gui/501/ai.hermes.gateway"] in commands


def test_failed_bootstrap_is_not_kickstarted(monkeypatch):
    commands = []

    def launchctl(cmd, **kwargs):
        commands.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(gw.subprocess, "run", launchctl)
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda message: None)

    assert gw._retry_launchctl_bootstrap_until_registered(
        "gui/501", "/tmp/gateway.plist", "ai.hermes.gateway", deadline=0
    ) is False
    assert all(cmd[1] != "kickstart" for cmd in commands)
