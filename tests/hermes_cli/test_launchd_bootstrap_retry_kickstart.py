"""`_retry_launchctl_bootstrap_until_registered` kickstarts a job launchd holds but does not run.

Regression for #142741. A plist whose job launchd registers without starting it (an on-demand job
with no RunAtLoad/KeepAlive, or a per-user `gui/<uid>` domain that does not start jobs at bootstrap)
used to keep the retry loop re-bootstrapping an already-loaded label: `bootstrap` fails EIO into a
bootout, the label stays registered and idle, and the loop runs to its deadline and returns False,
so `gateway start`/`restart` reports failure while one `launchctl kickstart` would have started it.
A label launchd already knows is kickstarted; only a label launchd does not know is bootstrapped.
"""
from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli

DOMAIN = "gui/501"
LABEL = "ai.hermes.gateway"
PLIST = "/tmp/ai.hermes.gateway.plist"
KICKSTART = ["launchctl", "kickstart", f"{DOMAIN}/{LABEL}"]

# `launchctl list <label>` for a job launchd runs: a PID means supervision, exit 0 alone does not.
RUNNING_LIST = f'{{\n\t"PID" = 4242;\n\t"Label" = "{LABEL}";\n}};'
# A definition launchd holds without running it (macOS 26+ `state = not running`): exit 0, no PID.
IDLE_LIST = f'{{\n\t"Label" = "{LABEL}";\n}};'


def _retry(monkeypatch, *, list_outputs, deadline=None):
    """Drive the retry loop against a fake launchctl and return ``(ok, recorded_commands)``.

    ``list_outputs`` yields one ``(returncode, stdout)`` pair per `launchctl list`, so a test decides
    when the job starts. The expired-by-default deadline keeps the un-fixed loop to a single attempt
    instead of spinning; the success path never reaches the deadline check.
    """
    calls = SimpleNamespace(commands=[], bootstrap=0, kickstart=0)
    listings = iter(list_outputs)

    def fake_run(cmd, check=False, **kwargs):
        calls.commands.append(list(cmd))
        if cmd[:2] == ["launchctl", "list"]:
            rc, stdout = next(listings)
            return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")
        if cmd[:2] == ["launchctl", "kickstart"]:
            calls.kickstart += 1
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["launchctl", "bootstrap"]:
            calls.bootstrap += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway_cli.time, "sleep", lambda *_a, **_k: None)

    ok = gateway_cli._retry_launchctl_bootstrap_until_registered(
        DOMAIN, PLIST, LABEL,
        deadline=gateway_cli.time.monotonic() - 1 if deadline is None else deadline,
    )
    return ok, calls


def test_registered_but_idle_job_is_kickstarted_not_rebootstrap(monkeypatch):
    """An already-loaded label is kickstarted; re-bootstrapping it only EIOs into a bootout."""
    ok, calls = _retry(monkeypatch, list_outputs=[(0, IDLE_LIST), (0, RUNNING_LIST)])

    assert ok is True
    assert KICKSTART in calls.commands
    assert calls.kickstart == 1
    assert calls.bootstrap == 0


def test_unloaded_job_is_still_bootstrapped(monkeypatch):
    """The other half of the boundary: a label launchd does not know is bootstrapped, not kickstarted."""
    ok, calls = _retry(
        monkeypatch,
        list_outputs=[(1, ""), (0, RUNNING_LIST)],
        deadline=gateway_cli.time.monotonic() + 60,
    )

    assert ok is True
    assert calls.bootstrap >= 1
    assert KICKSTART not in calls.commands
