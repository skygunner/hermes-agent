"""Regression for #145010: the browser orphan reap is driven by gateway startup, not by browser usage.

``start_cleanup_thread`` used to be reached only from ``get_session_info``, so the janitor (and the
orphan reap in its cycle 0) existed only in a process that had already served a browser request. A
gateway that was force-killed while holding a Chromium tree leaked it until some later process
happened to open a browser; the exit-time reap never runs after ``kill -9``.
"""

from __future__ import annotations

import asyncio
import threading
import types

import gateway.run as gw_run
from tools import browser_session
from tools import browser_session_lifecycle as lifecycle


class _Ticker:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def restart_if_dead(self) -> bool:
        return False


def test_gateway_startup_reaps_orphans_without_serving_a_browser(monkeypatch):
    reaped = threading.Event()

    def _reap() -> int:
        reaped.set()
        return 0

    monkeypatch.setattr(lifecycle, "reap_orphaned_browser_sessions", _reap)
    monkeypatch.setattr(gw_run, "_cron_tick_profile_homes", lambda _cfg: [])
    monkeypatch.setattr(gw_run, "_start_gateway_housekeeping", lambda *_a, **_k: None)
    from cron import scheduler_provider, scheduler_thread

    monkeypatch.setattr(
        scheduler_provider, "scheduler_for_profile_mode",
        lambda _provider, multiplex_profiles=False: scheduler_provider.InProcessCronScheduler())
    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _Ticker)

    runner = types.SimpleNamespace(
        config={}, adapters={}, _profile_adapters=None, _primary_profile_name="default",
        _draining=False, _external_drain_active=False)

    assert not browser_session._active_sessions, "precondition: no browser session was ever created"
    async def _start_gateway() -> None:
        gw_run._start_gateway_start_cron_and_housekeeping(runner)

    try:
        asyncio.run(_start_gateway())
        assert reaped.wait(10.0), "gateway startup never started the browser orphan reaper"
    finally:
        lifecycle.stop_cleanup_thread()
