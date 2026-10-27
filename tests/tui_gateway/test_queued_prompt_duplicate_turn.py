"""A queued copy of a prompt that already ran must not drain as a second turn.

Regression for #145251. A prompt admitted while the turn was still being built (or a client
retry of an interrupted, lease-reclaimed turn) is queued even though the session is already
running — or has just finished running — the same text. The drain then dispatches the copy, so
a completed user message starts a fresh agent turn and repeats work.
"""

import threading
import types

from tui_gateway import server


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def _run_turn(session: dict, text: str, dispatched: list[str]) -> None:
    dispatched.append(text)
    with session["history_lock"]:
        server._start_inflight_turn(session, text)
        server._clear_inflight_turn(session)
        session["running"] = False


def _stub_dispatch(monkeypatch, dispatched: list[str]) -> None:
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: _run_turn(session, text, dispatched),
    )


def test_drain_does_not_rerun_the_prompt_the_turn_already_ran(monkeypatch):
    """RED for #145251: the completed prompt's queued copy drained as a fresh turn."""
    dispatched: list[str] = []
    _stub_dispatch(monkeypatch, dispatched)
    session = _session()
    server._start_inflight_turn(session, "release it")
    server._clear_inflight_turn(session)
    session["queued_prompt"] = {"text": "release it", "transport": None}

    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert dispatched == []
    assert session.get("queued_prompt") is None


def test_drain_keeps_a_followup_that_differs_from_the_last_turn(monkeypatch):
    dispatched: list[str] = []
    _stub_dispatch(monkeypatch, dispatched)
    session = _session()
    server._start_inflight_turn(session, "release it")
    server._clear_inflight_turn(session)
    session["queued_prompt"] = {"text": "then update the changelog", "transport": None}

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert dispatched == ["then update the changelog"]


def test_drain_drops_a_second_queued_copy_of_the_dispatched_prompt(monkeypatch):
    dispatched: list[str] = []
    _stub_dispatch(monkeypatch, dispatched)
    session = _session(
        queued_prompt={"text": "release it", "transport": None},
        queued_prompts=[
            {"text": "release it", "transport": None},
            {"text": "then update the changelog", "transport": None},
        ],
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert server._drain_queued_prompt("r2", "sid", session) is True
    assert dispatched == ["release it", "then update the changelog"]
    assert server._drain_queued_prompt("r3", "sid", session) is False


def test_drain_keeps_the_followup_of_a_merged_self_copy(monkeypatch):
    """A slot merged to ``"{live}\\n\\n{follow-up}"`` keeps the follow-up, not the live original."""
    dispatched: list[str] = []
    _stub_dispatch(monkeypatch, dispatched)
    session = _session(
        queued_prompt={"text": "release it", "transport": None},
        queued_prompts=[{"text": "release it\n\nfix the typo", "transport": None}],
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert server._drain_queued_prompt("r2", "sid", session) is True
    assert dispatched == ["release it", "fix the typo"]
