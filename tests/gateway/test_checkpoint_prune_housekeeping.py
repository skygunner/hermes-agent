"""Checkpoint store pruning rides the gateway housekeeping tick, not the constructor.

Its ``git gc`` repacks the whole store (tens of seconds on a GB store); run at construction it
delayed the control socket, adapters and the code_sha stamp, so the first restart of the day (the
``hermes update`` one) looked hung and failed fleet verification.
"""

import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_runs_the_checkpoint_prune(monkeypatch):
    import tools.checkpoint_maintenance as cm

    calls = []
    monkeypatch.setattr(cm, "auto_prune_from_config", lambda: calls.append(True) or {"skipped": False})

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [True]


def test_gateway_housekeeping_prunes_each_served_profile(monkeypatch, tmp_path):
    """Under a multiplexed gateway the prune tick runs once per served profile inside that
    profile's runtime scope, so each profile's own store and marker advance (#145267)."""
    import tools.checkpoint_maintenance as cm
    from contextlib import contextmanager
    from types import SimpleNamespace

    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    launch_home = tmp_path / "launch"
    sec_home = tmp_path / "secondary"
    launch_home.mkdir()
    sec_home.mkdir()

    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes",
                        lambda config: [("default", str(launch_home)), ("secondary", str(sec_home))])

    @contextmanager
    def fake_scope(home, *args, **kwargs):
        token = set_hermes_home_override(str(home))
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)

    calls = []
    monkeypatch.setattr(cm, "auto_prune_from_config",
                        lambda: calls.append(get_hermes_home()) or {"skipped": False})

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0, runner=runner)

    assert calls == [launch_home, sec_home]
