"""Regression for #143239: a Group Chat Send that stored its message must not answer with an error."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def test_send_never_answers_with_an_error_after_storing_the_message(
    tmp_path: Path, monkeypatch
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    payload = {"text": "Hello", "thread_id": "thread-1"}

    # A room this gateway cannot bind is refused before anything is stored.
    with monkeypatch.context() as patch:
        patch.setattr(service, "bindings", lambda: ())
        with pytest.raises(hosted_rooms.RoomNotFoundError):
            service.send(room_id="room-1", event_id="user-1", payload=payload)
    assert [
        event for event in service._events("room-1") if event["kind"] == "message.user"
    ] == []

    def storage_full(_binding) -> None:
        raise hosted_rooms.HostedRoomError(
            "Group Chat storage is full on this host. Delete an old Group Chat and try again."
        )

    # A failed preparation after the commit is reported separately, not as a failed Send.
    with monkeypatch.context() as patch:
        patch.setattr(service, "prepare_room", storage_full)
        event = service.send(room_id="room-1", event_id="user-2", payload=payload)
    assert event["payload"]["text"] == "Hello"
    assert [
        stored["payload"]["text"]
        for stored in service._events("room-1")
        if stored["kind"] == "message.user"
    ] == ["Hello"]
