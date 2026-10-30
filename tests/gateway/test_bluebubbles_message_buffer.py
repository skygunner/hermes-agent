"""Regression for rapid BlueBubbles messages (#30989)."""
import asyncio
import json

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter


class Request:
    def __init__(self, message_id, text):
        self.query = {"password": "secret"}
        self.headers = {}
        self.body = json.dumps({"type": "new-message", "data": {
            "guid": message_id, "text": text, "handle": {"address": "sender"},
            "chats": [{"guid": "chat"}],
        }}).encode()

    async def read(self):
        return self.body


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("first", "second"), (None, None)])
async def test_rapid_distinct_messages_are_both_delivered(ids, monkeypatch):
    adapter = BlueBubblesAdapter(PlatformConfig(extra={"password": "secret", "send_read_receipts": False}))
    timer_started = asyncio.Event()
    release_timer = asyncio.Event()
    delivered = []

    async def wait_for_release(_delay):
        timer_started.set()
        await release_timer.wait()

    async def handle(event):
        delivered.append(event.text)

    monkeypatch.setattr(asyncio, "sleep", wait_for_release)
    adapter.handle_message = handle
    response = await adapter._handle_webhook(Request(ids[0], "first text"))
    assert response.status == 200
    await asyncio.wait_for(timer_started.wait(), timeout=2)
    first_task = adapter._bluebubble_tasks["chat"]
    try:
        response = await adapter._handle_webhook(Request(ids[1], "second text"))
        assert response.status == 200
        release_timer.set()
        await asyncio.wait_for(asyncio.gather(*list(adapter._bluebubble_tasks.values())), timeout=2)
        assert delivered == ["first text", "second text"]
    finally:
        release_timer.set()
        await asyncio.gather(first_task, return_exceptions=True)
